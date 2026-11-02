"""Testes para os novos clientes de providers: GroqChatClient, GroqSttClient, ElevenLabsTtsClient.

Todos os testes sao mockados — zero acesso a rede.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.assistant.groq_chat_client import GroqChatClient, ChatClientError
from app.assistant.groq_stt_client import GroqSttClient, SttClientError
from app.assistant.elevenlabs_tts_client import ElevenLabsTtsClient, TtsClientError


# ---------------------------------------------------------------------------
# GroqChatClient
# ---------------------------------------------------------------------------

def _make_chat_response(text, model="openai/gpt-oss-20b", total_tokens=50):
    """Monta resposta fake no formato do SDK Groq."""
    usage = SimpleNamespace(total_tokens=total_tokens)
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_groq_chat_client(model="openai/gpt-oss-20b", fallback_model="qwen/qwen3-32b"):
    client = GroqChatClient(
        api_key="fake-key",
        chat_model=model,
        fallback_model=fallback_model,
    )
    return client


class FakeGroqCompletions:
    """Fake de client.chat.completions.create para GroqChatClient."""

    def __init__(self, responses=None, errors=None):
        """
        responses: lista de textos a retornar em sequencia.
        errors: lista de excecoes a levantar em sequencia (None = usar response).
        """
        self._responses = list(responses or [])
        self._errors = list(errors or [])
        self.calls = []

    def create(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        if self._errors:
            err = self._errors.pop(0)
            if err is not None:
                raise err
        text = self._responses.pop(0) if self._responses else ""
        return _make_chat_response(text, model=model)


def _inject_fake_completions(chat_client, completions):
    """Injeta um fake de completions no GroqChatClient ignorando lazy import."""
    chat_client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )


class TestGroqChatClientGenerate:
    def test_generate_returns_text_from_primary_model(self):
        client = _make_groq_chat_client()
        completions = FakeGroqCompletions(responses=["Ola, como posso ajudar?"])
        _inject_fake_completions(client, completions)

        result = client.generate("system prompt", [{"role": "user", "content": "ola"}])

        assert result == "Ola, como posso ajudar?"
        assert completions.calls[0]["model"] == "openai/gpt-oss-20b"

    def test_generate_falls_back_to_second_model_when_primary_fails(self):
        client = _make_groq_chat_client()
        completions = FakeGroqCompletions(
            responses=["resposta do fallback"],
            errors=[RuntimeError("primary down"), None],
        )
        _inject_fake_completions(client, completions)

        result = client.generate("prompt", [{"role": "user", "content": "teste"}])

        assert result == "resposta do fallback"
        assert completions.calls[0]["model"] == "openai/gpt-oss-20b"
        assert completions.calls[1]["model"] == "qwen/qwen3-32b"

    def test_generate_raises_chat_client_error_when_both_models_fail(self):
        client = _make_groq_chat_client()
        completions = FakeGroqCompletions(
            errors=[RuntimeError("primary down"), RuntimeError("fallback down")],
        )
        _inject_fake_completions(client, completions)

        with pytest.raises(ChatClientError, match="modelos de chat falharam"):
            client.generate("prompt", [{"role": "user", "content": "teste"}])

    def test_generate_strips_thinking_tags_from_output(self):
        client = _make_groq_chat_client()
        text_with_thinking = "<think>raciocinio interno\nmultiline</think>Resposta visivel."
        completions = FakeGroqCompletions(responses=[text_with_thinking])
        _inject_fake_completions(client, completions)

        result = client.generate("prompt", [])

        assert result == "Resposta visivel."
        assert "<think>" not in result
        assert "raciocinio interno" not in result

    def test_generate_with_metadata_returns_text_model_and_tokens(self):
        client = _make_groq_chat_client()
        completions = FakeGroqCompletions(responses=["resposta com metadata"])
        _inject_fake_completions(client, completions)

        text, model, tokens = client.generate_with_metadata(
            "prompt", [{"role": "user", "content": "teste"}]
        )

        assert text == "resposta com metadata"
        assert model == "openai/gpt-oss-20b"
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_generate_with_metadata_uses_fallback_model_name(self):
        client = _make_groq_chat_client()
        completions = FakeGroqCompletions(
            responses=["fallback response"],
            errors=[RuntimeError("primary down"), None],
        )
        _inject_fake_completions(client, completions)

        text, model, tokens = client.generate_with_metadata("prompt", [])

        assert model == "qwen/qwen3-32b"

    def test_generate_with_metadata_raises_on_both_models_failing(self):
        client = _make_groq_chat_client()
        completions = FakeGroqCompletions(
            errors=[RuntimeError("p1"), RuntimeError("p2")],
        )
        _inject_fake_completions(client, completions)

        with pytest.raises(ChatClientError):
            client.generate_with_metadata("prompt", [])

    def test_reasoning_format_hidden_applied_to_gpt_oss_model(self):
        """gpt-oss models devem receber reasoning_format=hidden."""
        client = _make_groq_chat_client(model="openai/gpt-oss-20b")
        completions = FakeGroqCompletions(responses=["ok"])
        _inject_fake_completions(client, completions)

        client.generate("prompt", [])

        assert completions.calls[0].get("reasoning_format") == "hidden"

    def test_reasoning_format_not_applied_to_qwen_model(self):
        """Modelos sem 'gpt-oss' nao devem receber reasoning_format."""
        client = _make_groq_chat_client(model="qwen/qwen3-32b")
        completions = FakeGroqCompletions(responses=["ok"])
        _inject_fake_completions(client, completions)

        client.generate("prompt", [])

        assert "reasoning_format" not in completions.calls[0]

    def test_raises_on_empty_api_key(self):
        with pytest.raises(ChatClientError, match="GROQ_API_KEY"):
            GroqChatClient(api_key="")

    def test_raises_on_none_api_key(self):
        with pytest.raises(ChatClientError, match="GROQ_API_KEY"):
            GroqChatClient(api_key=None)

    def test_build_messages_converts_model_role_to_assistant(self):
        """Groq usa 'assistant' em vez de 'model' para mensagens do assistente."""
        client = _make_groq_chat_client()
        messages = [
            {"role": "user", "content": "oi"},
            {"role": "model", "content": "ola"},  # legado — deve ser convertido
        ]
        completions = FakeGroqCompletions(responses=["ok"])
        _inject_fake_completions(client, completions)

        client.generate("system prompt", messages)

        sent_messages = completions.calls[0]["messages"]
        roles = [m["role"] for m in sent_messages]
        assert "model" not in roles
        assert "assistant" in roles


# ---------------------------------------------------------------------------
# GroqSttClient
# ---------------------------------------------------------------------------

def _make_transcription_response(text):
    return SimpleNamespace(text=text)


class TestGroqSttClient:
    def test_transcribe_returns_text(self, tmp_path):
        audio_path = tmp_path / "command.wav"
        audio_path.write_bytes(b"fake-wav-data")

        client = GroqSttClient(api_key="fake-key")

        # Injeta fake do SDK Groq
        fake_transcriptions = SimpleNamespace(
            create=lambda model, file, language: _make_transcription_response("acenda a luz")
        )
        client._client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=fake_transcriptions)
        )

        result = client.transcribe(audio_path)

        assert result == "acenda a luz"

    def test_transcribe_raises_when_audio_path_not_found(self):
        client = GroqSttClient(api_key="fake-key")

        with pytest.raises(SttClientError, match="nao encontrado"):
            client.transcribe(Path("nao_existe.wav"))

    def test_transcribe_raises_on_api_error(self, tmp_path):
        audio_path = tmp_path / "command.wav"
        audio_path.write_bytes(b"fake-wav-data")

        client = GroqSttClient(api_key="fake-key")

        def fail_create(**_kwargs):
            raise RuntimeError("api error")

        client._client = SimpleNamespace(
            audio=SimpleNamespace(
                transcriptions=SimpleNamespace(create=fail_create)
            )
        )

        with pytest.raises(SttClientError, match="Transcricao falhou"):
            client.transcribe(audio_path)

    def test_transcribe_raises_on_empty_response(self, tmp_path):
        audio_path = tmp_path / "command.wav"
        audio_path.write_bytes(b"fake-wav-data")

        client = GroqSttClient(api_key="fake-key")
        client._client = SimpleNamespace(
            audio=SimpleNamespace(
                transcriptions=SimpleNamespace(
                    create=lambda **_kw: _make_transcription_response("   ")
                )
            )
        )

        with pytest.raises(SttClientError, match="vazio"):
            client.transcribe(audio_path)

    def test_transcribe_raises_on_none_text(self, tmp_path):
        audio_path = tmp_path / "command.wav"
        audio_path.write_bytes(b"fake-wav-data")

        client = GroqSttClient(api_key="fake-key")
        client._client = SimpleNamespace(
            audio=SimpleNamespace(
                transcriptions=SimpleNamespace(
                    create=lambda **_kw: SimpleNamespace(text=None)
                )
            )
        )

        with pytest.raises(SttClientError, match="vazio"):
            client.transcribe(audio_path)

    def test_raises_on_empty_api_key(self):
        with pytest.raises(SttClientError, match="GROQ_API_KEY"):
            GroqSttClient(api_key="")

    def test_raises_on_none_api_key(self):
        with pytest.raises(SttClientError, match="GROQ_API_KEY"):
            GroqSttClient(api_key=None)


# ---------------------------------------------------------------------------
# ElevenLabsTtsClient
# ---------------------------------------------------------------------------

def _make_tts_client(api_key="fake-key", voice_id="fake-voice"):
    return ElevenLabsTtsClient(api_key=api_key, voice_id=voice_id)


def _inject_fake_tts(client, chunks=None, error=None):
    """Injeta fake do SDK ElevenLabs no client."""
    wav_bytes = chunks if chunks is not None else [b"RIFF", b"fake", b"wav"]

    def fake_convert(voice_id, text, model_id, output_format):
        if error is not None:
            raise error
        return iter(wav_bytes)

    client._client = SimpleNamespace(
        text_to_speech=SimpleNamespace(convert=fake_convert)
    )


class TestElevenLabsTtsClient:
    def test_synthesize_speech_returns_wav_path(self, tmp_path):
        client = _make_tts_client()
        _inject_fake_tts(client, chunks=[b"RIFF", b"data"])

        path = client.synthesize_speech("ola", tmp_path)

        assert path.exists()
        assert path.suffix == ".wav"
        assert path.parent == tmp_path
        assert path.read_bytes() == b"RIFFdata"

    def test_synthesize_creates_output_dir_if_needed(self, tmp_path):
        client = _make_tts_client()
        _inject_fake_tts(client)
        nested_dir = tmp_path / "a" / "b" / "c"

        path = client.synthesize_speech("teste", nested_dir)

        assert path.exists()

    def test_synthesize_raises_tts_client_error_on_api_failure(self, tmp_path):
        client = _make_tts_client()
        _inject_fake_tts(client, error=RuntimeError("network down"))

        with pytest.raises(TtsClientError):
            client.synthesize_speech("ola", tmp_path)

    def test_synthesize_raises_on_empty_audio(self, tmp_path):
        client = _make_tts_client()
        _inject_fake_tts(client, chunks=[b""])  # bytes vazios

        with pytest.raises(TtsClientError, match="vazio"):
            client.synthesize_speech("ola", tmp_path)

    @pytest.mark.parametrize(
        ("exc_type_name", "status_code", "expected_error_type"),
        [
            ("UnauthorizedError", None, "elevenlabs_auth_error"),
            ("ForbiddenError", None, "elevenlabs_auth_error"),
            ("ApiError", 401, "elevenlabs_auth_error"),
            ("ApiError", 402, "elevenlabs_quota_exceeded"),
            ("ApiError", 429, "elevenlabs_rate_limited"),
            ("ApiError", 500, "elevenlabs_tts_error"),
        ],
    )
    def test_synthesize_classifies_error_type_correctly(
        self, tmp_path, exc_type_name, status_code, expected_error_type
    ):
        client = _make_tts_client()

        # Cria excecao fake com o nome e status_code corretos
        exc_class = type(exc_type_name, (Exception,), {})
        exc = exc_class("api error")
        if status_code is not None:
            exc.status_code = status_code

        _inject_fake_tts(client, error=exc)

        with pytest.raises(TtsClientError) as exc_info:
            client.synthesize_speech("ola", tmp_path)

        assert exc_info.value.error_type == expected_error_type

    def test_raises_on_empty_api_key(self):
        with pytest.raises(TtsClientError):
            ElevenLabsTtsClient(api_key="", voice_id="voice-id")

    def test_raises_on_none_api_key(self):
        with pytest.raises(TtsClientError):
            ElevenLabsTtsClient(api_key=None, voice_id="voice-id")

    def test_raises_on_empty_voice_id(self):
        with pytest.raises(TtsClientError):
            ElevenLabsTtsClient(api_key="api-key", voice_id="")

    def test_raises_on_none_voice_id(self):
        with pytest.raises(TtsClientError):
            ElevenLabsTtsClient(api_key="api-key", voice_id=None)

    def test_error_type_attribute_accessible(self, tmp_path):
        """TtsClientError deve expor error_type para log no VoiceLoop."""
        client = _make_tts_client()
        _inject_fake_tts(client, error=RuntimeError("any"))

        with pytest.raises(TtsClientError) as exc_info:
            client.synthesize_speech("ola", tmp_path)

        assert hasattr(exc_info.value, "error_type")
        assert isinstance(exc_info.value.error_type, str)
