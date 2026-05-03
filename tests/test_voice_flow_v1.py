from pathlib import Path
from types import SimpleNamespace

import pytest

from app.assistant.cli import main
from app.assistant.config import AssistantConfig, AssistantConfigError
from app.assistant.gemini_client import GeminiClient, GeminiClientError
from app.assistant.voice_loop import VoiceLoop
import app.assistant.voice_loop as voice_loop_module
from app.assistant.audio.playback import PlaybackError
from app.assistant.audio.recorder import RecordingError


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


class FakeGeminiClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def transcribe_audio(self, audio_path):
        self.calls.append(("transcribe", audio_path))
        return "acenda a luz"

    def synthesize_speech(self, text, output_dir):
        self.calls.append(("synthesize", text, Path(output_dir)))
        return Path(output_dir) / "response.wav"


class FakeAudioPlayer:
    def __init__(self):
        self.calls = []

    def play(self, audio_path):
        self.calls.append(audio_path)


class TempFileRecorder(FakeRecorder):
    def record(self, output_dir, duration_seconds):
        path = Path(output_dir) / "command.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"command")
        return path


class TempFileGeminiClient(FakeGeminiClient):
    def synthesize_speech(self, text, output_dir):
        path = Path(output_dir) / "response.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"response")
        return path


def make_config(tmp_path):
    return AssistantConfig(
        google_api_key="fake-key",
        audio_output_dir=tmp_path,
        command_record_seconds=1.25,
        command_sample_rate=8000,
    )


def patch_voice_loop_dependencies(monkeypatch):
    monkeypatch.setattr(voice_loop_module, "Assistant", FakeAssistant)
    monkeypatch.setattr(voice_loop_module, "CommandRecorder", FakeRecorder)
    monkeypatch.setattr(voice_loop_module, "GeminiClient", FakeGeminiClient)
    monkeypatch.setattr(voice_loop_module, "AudioPlayer", FakeAudioPlayer)


def test_voice_loop_runs_one_turn_from_wake_word_to_playback(tmp_path, monkeypatch, capsys):
    patch_voice_loop_dependencies(monkeypatch)
    engine = FakeWakeWordEngine()
    loop = VoiceLoop(make_config(tmp_path), engine)

    loop.run()
    engine.callback(0)
    engine.callback(1)

    captured = capsys.readouterr()

    turn_dir = loop.recorder.calls[0][0]

    assert loop.recorder.sample_rate == 8000
    assert loop.recorder.calls == [(turn_dir, 1.25)]
    assert turn_dir.parent == tmp_path
    assert turn_dir.name.startswith("turn-")
    assert loop.gemini.calls == [
        ("transcribe", turn_dir / "command.wav"),
        ("synthesize", "resposta para acenda a luz", turn_dir),
    ]
    assert loop.player.calls == [turn_dir / "response.wav"]
    assert "Wake word detected. Recording command..." in captured.out
    assert "Command transcribed." in captured.out
    assert "Assistant response generated." in captured.out
    assert "Audio response played." in captured.out


@pytest.mark.parametrize(
    ("failing_dependency", "error", "expected_output"),
    [
        ("recorder", RecordingError("microfone indisponivel"), "Voice error: microfone indisponivel"),
        ("gemini", GeminiClientError("api indisponivel"), "Gemini error: api indisponivel"),
        ("player", PlaybackError("saida de audio indisponivel"), "Playback error: saida de audio indisponivel"),
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
        monkeypatch.setattr(loop.recorder, "record", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
    elif failing_dependency == "gemini":
        monkeypatch.setattr(loop.gemini, "transcribe_audio", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
    else:
        monkeypatch.setattr(loop.player, "play", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))

    loop._on_prediction(1)

    assert loop._handling_turn is False
    assert expected_output in capsys.readouterr().out


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


def test_voice_config_reads_voice_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("VALERIA_ASSISTANT_NAME", "Aurora")
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
    monkeypatch.setenv("VALERIA_GEMINI_MODEL", "gemini-env")
    monkeypatch.setenv("VALERIA_TRANSCRIPTION_MODEL", "transcribe-env")
    monkeypatch.setenv("VALERIA_TTS_MODEL", "tts-env")
    monkeypatch.setenv("VALERIA_TTS_VOICE", "Leda")
    audio_dir = Path("tmp/valeria-audio/test-env")
    monkeypatch.setenv("VALERIA_AUDIO_OUTPUT_DIR", str(audio_dir))
    monkeypatch.setenv("VALERIA_COMMAND_RECORD_SECONDS", "2.5")
    monkeypatch.setenv("VALERIA_COMMAND_SAMPLE_RATE", "22050")

    config = AssistantConfig.from_env()

    assert config.assistant_name == "Aurora"
    assert config.google_api_key == "env-key"
    assert config.gemini_model == "gemini-env"
    assert config.transcription_model == "transcribe-env"
    assert config.tts_model == "tts-env"
    assert config.tts_voice == "Leda"
    assert config.audio_output_dir == audio_dir
    assert config.command_record_seconds == 2.5
    assert config.command_sample_rate == 22050


def test_missing_google_api_key_is_controlled_error():
    with pytest.raises(GeminiClientError, match="GOOGLE_API_KEY"):
        GeminiClient(
            api_key=None,
            transcription_model="transcribe",
            tts_model="tts",
            tts_voice="Kore",
        )


def test_generate_content_error_becomes_sanitized_gemini_client_error(tmp_path):
    audio_path = tmp_path / "command.wav"
    audio_path.write_bytes(b"audio")
    leaked_detail = "raw transcript with token=secret"

    class FakePart:
        @staticmethod
        def from_bytes(data, mime_type):
            return SimpleNamespace(data=data, mime_type=mime_type)

    class FakeTypes:
        Part = FakePart

    class FakeModels:
        def generate_content(self, **_kwargs):
            raise RuntimeError(leaked_detail)

    client = GeminiClient(
        api_key="fake-key",
        transcription_model="transcribe",
        tts_model="tts",
        tts_voice="Kore",
    )
    client._client = SimpleNamespace(models=FakeModels())
    client._types = FakeTypes

    with pytest.raises(GeminiClientError) as exc_info:
        client.transcribe_audio(audio_path)

    assert "transcription request failed." in str(exc_info.value)
    assert leaked_detail not in str(exc_info.value)


def test_voice_loop_does_not_print_full_transcript_or_response(tmp_path, monkeypatch, capsys):
    secret_transcript = "codigo confidencial 123456"
    secret_response = "resposta privada completa"

    class SensitiveAssistant(FakeAssistant):
        def respond(self, _transcript):
            return secret_response

    class SensitiveGemini(FakeGeminiClient):
        def transcribe_audio(self, _audio_path):
            return secret_transcript

        def synthesize_speech(self, _text, output_dir):
            path = Path(output_dir) / "response.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"response")
            return path

    monkeypatch.setattr(voice_loop_module, "Assistant", SensitiveAssistant)
    monkeypatch.setattr(voice_loop_module, "CommandRecorder", TempFileRecorder)
    monkeypatch.setattr(voice_loop_module, "GeminiClient", SensitiveGemini)
    monkeypatch.setattr(voice_loop_module, "AudioPlayer", FakeAudioPlayer)
    loop = VoiceLoop(make_config(tmp_path), FakeWakeWordEngine())

    loop._on_prediction(1)

    captured = capsys.readouterr()
    combined_output = captured.out + captured.err
    assert secret_transcript not in combined_output
    assert secret_response not in combined_output


def test_voice_loop_removes_temp_files_after_success(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_loop_module, "Assistant", FakeAssistant)
    monkeypatch.setattr(voice_loop_module, "CommandRecorder", TempFileRecorder)
    monkeypatch.setattr(voice_loop_module, "GeminiClient", TempFileGeminiClient)
    monkeypatch.setattr(voice_loop_module, "AudioPlayer", FakeAudioPlayer)
    loop = VoiceLoop(make_config(tmp_path), FakeWakeWordEngine())

    loop._on_prediction(1)

    assert not (tmp_path / "command.wav").exists()
    assert not (tmp_path / "response.wav").exists()


@pytest.mark.parametrize("failure_stage", ["gemini", "playback"])
def test_voice_loop_removes_temp_files_after_failures(tmp_path, monkeypatch, failure_stage):
    class FailingGemini(TempFileGeminiClient):
        def transcribe_audio(self, audio_path):
            if failure_stage == "gemini":
                raise GeminiClientError("api indisponivel")
            return super().transcribe_audio(audio_path)

    class FailingPlayer(FakeAudioPlayer):
        def play(self, audio_path):
            if failure_stage == "playback":
                raise PlaybackError("playback indisponivel")
            super().play(audio_path)

    monkeypatch.setattr(voice_loop_module, "Assistant", FakeAssistant)
    monkeypatch.setattr(voice_loop_module, "CommandRecorder", TempFileRecorder)
    monkeypatch.setattr(voice_loop_module, "GeminiClient", FailingGemini)
    monkeypatch.setattr(voice_loop_module, "AudioPlayer", FailingPlayer)
    loop = VoiceLoop(make_config(tmp_path), FakeWakeWordEngine())

    loop._on_prediction(1)

    assert not (tmp_path / "command.wav").exists()
    assert not (tmp_path / "response.wav").exists()


def test_audio_output_dir_outside_allowed_root_is_rejected(monkeypatch, tmp_path):
    outside_root = tmp_path.parent / "outside-valeria-audio"
    monkeypatch.setenv("VALERIA_AUDIO_OUTPUT_DIR", str(outside_root))

    with pytest.raises(AssistantConfigError, match="VALERIA_AUDIO_OUTPUT_DIR"):
        AssistantConfig.from_env()


@pytest.mark.parametrize("seconds", ["abc", "-1", "9999"])
def test_invalid_command_record_seconds_fails_controlled(monkeypatch, seconds):
    monkeypatch.setenv("VALERIA_COMMAND_RECORD_SECONDS", seconds)

    with pytest.raises(AssistantConfigError, match="VALERIA_COMMAND_RECORD_SECONDS"):
        AssistantConfig.from_env()


def test_missing_pyaudio_does_not_break_text_mode(monkeypatch, capsys):
    monkeypatch.setitem(__import__("sys").modules, "pyaudio", None)

    exit_code = main(["text", "--once", "ola"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Recebi seu pedido, mas a inteligencia online nao esta disponivel agora." in captured.out
    assert "Valeria recebeu: ola" not in captured.out


def test_gemini_client_extracts_audio_from_mocked_response_parts():
    audio_bytes = b"wav-data"
    response = SimpleNamespace(
        parts=[SimpleNamespace(inline_data=SimpleNamespace(data=audio_bytes))],
        candidates=[],
    )
    client = GeminiClient(
        api_key="fake-key",
        transcription_model="transcribe",
        tts_model="tts",
        tts_voice="Kore",
    )

    assert client._extract_audio(response) == (audio_bytes, None)


def test_gemini_client_extracts_audio_from_mocked_candidate_content():
    audio_bytes = b"candidate-wav-data"
    response = SimpleNamespace(
        parts=[],
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(inline_data=SimpleNamespace(data=audio_bytes))]
                )
            )
        ],
    )
    client = GeminiClient(
        api_key="fake-key",
        transcription_model="transcribe",
        tts_model="tts",
        tts_voice="Kore",
    )

    assert client._extract_audio(response) == (audio_bytes, None)
