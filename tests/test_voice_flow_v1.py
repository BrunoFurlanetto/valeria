from pathlib import Path

import pytest

from app.assistant.cli import main
from app.assistant.config import AssistantConfig, AssistantConfigError
from app.assistant.groq_stt_client import GroqSttClient, SttClientError
from app.assistant.elevenlabs_tts_client import ElevenLabsTtsClient, TtsClientError
from app.assistant.voice_loop import VoiceLoop
import app.assistant.voice_loop as voice_loop_module
from app.assistant.audio.playback import PlaybackError
from app.assistant.audio.recorder import RecordingError


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeWakeWordEngine:
    def __init__(self):
        self.callback = None

    def run(self, callback):
        self.callback = callback


class FakeAssistant:
    def __init__(self, config):
        self.config = config

    def respond(self, transcript):
        return f"resposta para {transcript}"


class FakeRecorder:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.calls = []

    def record(self, output_dir, duration_seconds):
        self.calls.append((Path(output_dir), duration_seconds))
        return Path(output_dir) / "command.wav"


class FakeGroqSttClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def transcribe(self, audio_path):
        self.calls.append(audio_path)
        return "acenda a luz"


class FakeElevenLabsTtsClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def synthesize_speech(self, text, output_dir):
        self.calls.append((text, Path(output_dir)))
        return Path(output_dir) / "response.wav"


class FakeFeedbackPlayer:
    def __init__(self, player=None):
        self.player = player
        self.earcon_calls = []
        self.phrase_calls = []

    def play_earcon(self, kind):
        self.earcon_calls.append(kind)

    def play_fallback_phrase(self, kind):
        self.phrase_calls.append(kind)


class FakeAudioPlayer:
    def __init__(self):
        self.calls = []

    def play(self, audio_path):
        self.calls.append(audio_path)


class TempFileRecorder(FakeRecorder):
    """Recorder que realmente cria o arquivo no disco para testes de cleanup."""

    def record(self, output_dir, duration_seconds):
        path = Path(output_dir) / "command.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"command")
        return path


class TempFileTtsClient(FakeElevenLabsTtsClient):
    """TTS client que realmente cria o arquivo no disco para testes de cleanup."""

    def synthesize_speech(self, text, output_dir):
        path = Path(output_dir) / "response.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"response")
        return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(tmp_path):
    return AssistantConfig(
        groq_api_key="fake-groq-key",
        elevenlabs_api_key="fake-elevenlabs-key",
        elevenlabs_voice_id="fake-voice-id",
        audio_output_dir=tmp_path,
        command_record_seconds=1.25,
        command_sample_rate=8000,
    )


def patch_voice_loop_dependencies(monkeypatch):
    """Substitui todas as dependencias externas do VoiceLoop por fakes."""
    monkeypatch.setattr(voice_loop_module, "Assistant", FakeAssistant)
    monkeypatch.setattr(voice_loop_module, "CommandRecorder", FakeRecorder)
    monkeypatch.setattr(voice_loop_module, "GroqSttClient", FakeGroqSttClient)
    monkeypatch.setattr(voice_loop_module, "ElevenLabsTtsClient", FakeElevenLabsTtsClient)
    monkeypatch.setattr(voice_loop_module, "FeedbackPlayer", FakeFeedbackPlayer)
    monkeypatch.setattr(voice_loop_module, "AudioPlayer", FakeAudioPlayer)


# ---------------------------------------------------------------------------
# Testes de fluxo de turno
# ---------------------------------------------------------------------------

def test_voice_loop_runs_one_turn_from_wake_word_to_playback(tmp_path, monkeypatch, capsys):
    patch_voice_loop_dependencies(monkeypatch)
    engine = FakeWakeWordEngine()
    loop = VoiceLoop(make_config(tmp_path), engine)

    loop.run()
    engine.callback(0)  # nao aciona turno (predicao != 1)
    engine.callback(1)  # aciona turno

    captured = capsys.readouterr()

    turn_dir = loop.recorder.calls[0][0]

    assert loop.recorder.sample_rate == 8000
    assert loop.recorder.calls == [(turn_dir, 1.25)]
    assert turn_dir.parent == tmp_path
    assert turn_dir.name.startswith("turn-")

    # STT: transcribe recebeu o arquivo de comando
    assert loop.stt.calls == [turn_dir / "command.wav"]

    # FeedbackPlayer: earcon positivo tocado apos resposta gerada
    assert loop.feedback.earcon_calls == ["positive"]
    assert loop.feedback.phrase_calls == []

    # TTS: synthesize chamado com texto da resposta
    assert loop.tts.calls == [("resposta para acenda a luz", turn_dir)]

    # Player: tocou o audio gerado
    assert loop.player.calls == [turn_dir / "response.wav"]

    assert "Wake word detected. Recording command..." in captured.out
    assert "Command transcribed." in captured.out
    assert "Assistant response generated." in captured.out
    assert "Audio response played." in captured.out


def test_voice_loop_ignores_nested_wake_word_while_turn_is_running(tmp_path, monkeypatch):
    patch_voice_loop_dependencies(monkeypatch)
    loop = VoiceLoop(make_config(tmp_path), FakeWakeWordEngine())
    calls = []

    def handle_turn():
        calls.append("start")
        loop._on_prediction(1)
        calls.append("end")

    monkeypatch.setattr(loop, "handle_turn", handle_turn)

    loop._on_prediction(1)

    assert calls == ["start", "end"]
    assert loop._handling_turn is False


# ---------------------------------------------------------------------------
# Testes de recuperacao de erros por etapa
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("failing_dependency", "error", "expected_output"),
    [
        ("recorder", RecordingError("microfone indisponivel"), "Voice error: microfone indisponivel"),
        ("stt", SttClientError("api indisponivel"), "STT error: api indisponivel"),
        ("player", PlaybackError("saida de audio indisponivel"), "Playback error: saida de audio indisponivel"),
        (
            "tts",
            TtsClientError("elevenlabs down", "elevenlabs_tts_error"),
            "TTS error (elevenlabs_tts_error): elevenlabs down",
        ),
    ],
)
def test_voice_loop_recovers_from_turn_errors(
    tmp_path,
    monkeypatch,
    capsys,
    failing_dependency,
    error,
    expected_output,
):
    patch_voice_loop_dependencies(monkeypatch)
    engine = FakeWakeWordEngine()
    loop = VoiceLoop(make_config(tmp_path), engine)

    if failing_dependency == "recorder":
        monkeypatch.setattr(loop.recorder, "record", lambda *_a, **_kw: (_ for _ in ()).throw(error))
    elif failing_dependency == "stt":
        monkeypatch.setattr(loop.stt, "transcribe", lambda *_a, **_kw: (_ for _ in ()).throw(error))
    elif failing_dependency == "tts":
        monkeypatch.setattr(loop.tts, "synthesize_speech", lambda *_a, **_kw: (_ for _ in ()).throw(error))
    else:
        monkeypatch.setattr(loop.player, "play", lambda *_a, **_kw: (_ for _ in ()).throw(error))

    loop._on_prediction(1)

    assert loop._handling_turn is False
    assert expected_output in capsys.readouterr().out


def test_stt_error_triggers_error_earcon_and_fallback_phrase(tmp_path, monkeypatch, capsys):
    """Quando STT falha: earcon de erro + frase de erro devem ser acionados."""
    patch_voice_loop_dependencies(monkeypatch)
    loop = VoiceLoop(make_config(tmp_path), FakeWakeWordEngine())
    stt_error = SttClientError("groq api down")
    monkeypatch.setattr(loop.stt, "transcribe", lambda *_a, **_kw: (_ for _ in ()).throw(stt_error))

    loop._on_prediction(1)

    assert loop.feedback.earcon_calls == ["error"]
    assert loop.feedback.phrase_calls == ["error"]


def test_tts_error_triggers_positive_fallback_phrase_and_prints_response(tmp_path, monkeypatch, capsys):
    """Quando TTS falha: earcon positivo ja tocou; frase positiva estatica + resposta em texto."""
    patch_voice_loop_dependencies(monkeypatch)
    loop = VoiceLoop(make_config(tmp_path), FakeWakeWordEngine())
    tts_error = TtsClientError("quota exceeded", "elevenlabs_quota_exceeded")
    monkeypatch.setattr(loop.tts, "synthesize_speech", lambda *_a, **_kw: (_ for _ in ()).throw(tts_error))

    loop._on_prediction(1)

    captured = capsys.readouterr()
    assert loop.feedback.earcon_calls == ["positive"]
    assert loop.feedback.phrase_calls == ["positive"]
    assert "Entendi seu pedido. A resposta esta no terminal." in captured.out
    assert "TTS error (elevenlabs_quota_exceeded)" in captured.out


def test_tts_none_plays_positive_fallback_phrase(tmp_path, monkeypatch, capsys):
    """Quando TTS nao esta configurado (sem ELEVENLABS_API_KEY): earcon + frase positiva estatica."""
    patch_voice_loop_dependencies(monkeypatch)
    config = AssistantConfig(
        groq_api_key="fake-groq-key",
        elevenlabs_api_key=None,  # TTS desabilitado
        elevenlabs_voice_id=None,
        audio_output_dir=tmp_path,
        command_record_seconds=1.25,
        command_sample_rate=8000,
    )
    loop = VoiceLoop(config, FakeWakeWordEngine())

    loop._on_prediction(1)

    captured = capsys.readouterr()
    assert loop.tts is None
    assert loop.feedback.earcon_calls == ["positive"]
    assert loop.feedback.phrase_calls == ["positive"]
    assert "Entendi seu pedido. A resposta esta no terminal." in captured.out


# ---------------------------------------------------------------------------
# Testes de seguranca (SEC-03-001)
# ---------------------------------------------------------------------------

def test_voice_loop_does_not_print_full_transcript_or_response(tmp_path, monkeypatch, capsys):
    """SEC-03-001: transcript e resposta nao devem aparecer no stdout/stderr."""
    secret_transcript = "codigo confidencial 123456"
    secret_response = "resposta privada completa"

    class SensitiveAssistant(FakeAssistant):
        def respond(self, _transcript):
            return secret_response

    class SensitiveStt(FakeGroqSttClient):
        def transcribe(self, _audio_path):
            return secret_transcript

    class WritingTts(TempFileTtsClient):
        pass

    monkeypatch.setattr(voice_loop_module, "Assistant", SensitiveAssistant)
    monkeypatch.setattr(voice_loop_module, "CommandRecorder", TempFileRecorder)
    monkeypatch.setattr(voice_loop_module, "GroqSttClient", SensitiveStt)
    monkeypatch.setattr(voice_loop_module, "ElevenLabsTtsClient", WritingTts)
    monkeypatch.setattr(voice_loop_module, "FeedbackPlayer", FakeFeedbackPlayer)
    monkeypatch.setattr(voice_loop_module, "AudioPlayer", FakeAudioPlayer)
    loop = VoiceLoop(make_config(tmp_path), FakeWakeWordEngine())

    loop._on_prediction(1)

    captured = capsys.readouterr()
    combined_output = captured.out + captured.err
    assert secret_transcript not in combined_output
    assert secret_response not in combined_output


# ---------------------------------------------------------------------------
# Testes de limpeza de arquivos temporarios
# ---------------------------------------------------------------------------

def test_voice_loop_removes_temp_dirs_after_success(tmp_path, monkeypatch):
    """TemporaryDirectory deve ser limpo pelo context manager apos turno bem-sucedido."""
    monkeypatch.setattr(voice_loop_module, "Assistant", FakeAssistant)
    monkeypatch.setattr(voice_loop_module, "CommandRecorder", TempFileRecorder)
    monkeypatch.setattr(voice_loop_module, "GroqSttClient", FakeGroqSttClient)
    monkeypatch.setattr(voice_loop_module, "ElevenLabsTtsClient", TempFileTtsClient)
    monkeypatch.setattr(voice_loop_module, "FeedbackPlayer", FakeFeedbackPlayer)
    monkeypatch.setattr(voice_loop_module, "AudioPlayer", FakeAudioPlayer)
    loop = VoiceLoop(make_config(tmp_path), FakeWakeWordEngine())

    loop._on_prediction(1)

    assert not any(p.name.startswith("turn-") for p in tmp_path.iterdir())


@pytest.mark.parametrize("failure_stage", ["stt", "tts", "playback"])
def test_voice_loop_removes_temp_dirs_after_failures(tmp_path, monkeypatch, failure_stage):
    """TemporaryDirectory deve ser limpo mesmo em caso de falha parcial."""

    class FailingStt(FakeGroqSttClient):
        def transcribe(self, audio_path):
            if failure_stage == "stt":
                raise SttClientError("api indisponivel")
            return super().transcribe(audio_path)

    class FailingTts(TempFileTtsClient):
        def synthesize_speech(self, text, output_dir):
            if failure_stage == "tts":
                raise TtsClientError("elevenlabs down", "elevenlabs_tts_error")
            return super().synthesize_speech(text, output_dir)

    class FailingPlayer(FakeAudioPlayer):
        def play(self, audio_path):
            if failure_stage == "playback":
                raise PlaybackError("playback indisponivel")
            super().play(audio_path)

    monkeypatch.setattr(voice_loop_module, "Assistant", FakeAssistant)
    monkeypatch.setattr(voice_loop_module, "CommandRecorder", TempFileRecorder)
    monkeypatch.setattr(voice_loop_module, "GroqSttClient", FailingStt)
    monkeypatch.setattr(voice_loop_module, "ElevenLabsTtsClient", FailingTts)
    monkeypatch.setattr(voice_loop_module, "FeedbackPlayer", FakeFeedbackPlayer)
    monkeypatch.setattr(voice_loop_module, "AudioPlayer", FailingPlayer)
    loop = VoiceLoop(make_config(tmp_path), FakeWakeWordEngine())

    loop._on_prediction(1)

    assert not any(p.name.startswith("turn-") for p in tmp_path.iterdir())


# ---------------------------------------------------------------------------
# Testes de configuracao por variaveis de ambiente
# ---------------------------------------------------------------------------

def test_voice_config_reads_groq_and_elevenlabs_environment(monkeypatch, tmp_path):
    """AssistantConfig.from_env deve ler variaveis Groq e ElevenLabs corretamente."""
    monkeypatch.setattr(AssistantConfig, "_load_dotenv_if_available", lambda: None)
    monkeypatch.setenv("VALERIA_ASSISTANT_NAME", "Aurora")
    monkeypatch.setenv("GROQ_API_KEY", "groq-env-key")
    monkeypatch.setenv("VALERIA_GROQ_CHAT_MODEL", "groq-chat-env")
    monkeypatch.setenv("VALERIA_GROQ_STT_MODEL", "whisper-env")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-env-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice-env-id")
    monkeypatch.setenv("VALERIA_ELEVENLABS_MODEL", "eleven_multilingual_v2")
    audio_dir = "tmp/valeria-audio/test-env"
    monkeypatch.setenv("VALERIA_AUDIO_OUTPUT_DIR", audio_dir)
    monkeypatch.setenv("VALERIA_COMMAND_RECORD_SECONDS", "2.5")
    monkeypatch.setenv("VALERIA_COMMAND_SAMPLE_RATE", "22050")

    config = AssistantConfig.from_env()

    assert config.assistant_name == "Aurora"
    assert config.groq_api_key == "groq-env-key"
    assert config.groq_chat_model == "groq-chat-env"
    assert config.groq_stt_model == "whisper-env"
    assert config.elevenlabs_api_key == "eleven-env-key"
    assert config.elevenlabs_voice_id == "voice-env-id"
    assert config.elevenlabs_model == "eleven_multilingual_v2"
    assert config.command_record_seconds == 2.5
    assert config.command_sample_rate == 22050


def test_audio_output_dir_outside_allowed_root_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(AssistantConfig, "_load_dotenv_if_available", lambda: None)
    outside_root = tmp_path.parent / "outside-valeria-audio"
    monkeypatch.setenv("VALERIA_AUDIO_OUTPUT_DIR", str(outside_root))

    with pytest.raises(AssistantConfigError, match="VALERIA_AUDIO_OUTPUT_DIR"):
        AssistantConfig.from_env()


@pytest.mark.parametrize("seconds", ["abc", "-1", "9999"])
def test_invalid_command_record_seconds_fails_controlled(monkeypatch, seconds):
    monkeypatch.setattr(AssistantConfig, "_load_dotenv_if_available", lambda: None)
    monkeypatch.setenv("VALERIA_COMMAND_RECORD_SECONDS", seconds)

    with pytest.raises(AssistantConfigError, match="VALERIA_COMMAND_RECORD_SECONDS"):
        AssistantConfig.from_env()


def test_missing_pyaudio_does_not_break_text_mode(monkeypatch, capsys):
    """Sem pyaudio, o modo texto deve funcionar sem excecao."""
    monkeypatch.setitem(__import__("sys").modules, "pyaudio", None)
    monkeypatch.setattr(AssistantConfig, "_load_dotenv_if_available", lambda: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    exit_code = main(["text", "--once", "ola"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Recebi seu pedido, mas a inteligencia online nao esta disponivel agora." in captured.out
    assert "Valeria recebeu: ola" not in captured.out


def test_stt_client_requires_groq_api_key():
    """GroqSttClient deve levantar SttClientError se api_key for vazia."""
    with pytest.raises(SttClientError, match="GROQ_API_KEY"):
        GroqSttClient(api_key="")

    with pytest.raises(SttClientError, match="GROQ_API_KEY"):
        GroqSttClient(api_key=None)


def test_tts_client_requires_api_key_and_voice_id():
    """ElevenLabsTtsClient deve levantar TtsClientError se api_key ou voice_id vazios."""
    with pytest.raises(TtsClientError):
        ElevenLabsTtsClient(api_key="", voice_id="voice-id")

    with pytest.raises(TtsClientError):
        ElevenLabsTtsClient(api_key="api-key", voice_id="")
