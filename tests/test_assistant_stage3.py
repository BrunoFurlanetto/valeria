import json
from types import SimpleNamespace

import pytest

from app.assistant.config import AssistantConfig
from app.assistant.core import Assistant, AssistantCore, AssistantResponse
from app.assistant.event_logger import EventLogger
from app.assistant.gemini_client import GeminiClient, GeminiClientError


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class MemoryEventLogger:
    def __init__(self):
        self.events = []

    def log(self, event):
        self.events.append(event)


class FakeGeminiClient:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []

    def generate(self, prompt, messages):
        self.calls.append((prompt, list(messages)))
        if self.error:
            raise self.error
        return self.responses.pop(0)


def make_config(tmp_path, api_key=None):
    return AssistantConfig(
        google_api_key=api_key,
        event_log_path=tmp_path / "events.jsonl",
    )


def test_assistant_config_from_env_uses_current_gemini_default(monkeypatch):
    monkeypatch.setattr(AssistantConfig, "_load_dotenv_if_available", lambda: None)
    for name in (
        "GOOGLE_API_KEY",
        "VALERIA_GEMINI_MODEL",
        "VALERIA_TRANSCRIPTION_MODEL",
        "VALERIA_TTS_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    config = AssistantConfig.from_env()

    assert config.google_api_key is None
    assert config.gemini_model == DEFAULT_GEMINI_MODEL
    assert config.transcription_model == DEFAULT_GEMINI_MODEL


def test_assistant_core_respond_returns_assistant_response(tmp_path):
    logger = MemoryEventLogger()
    client = FakeGeminiClient(responses=["  Ola, posso ajudar.  "])
    core = AssistantCore(
        make_config(tmp_path, api_key="fake-key"),
        gemini_client=client,
        event_logger=logger,
    )

    response = core.respond("ola")

    assert isinstance(response, AssistantResponse)
    assert response.text == "Ola, posso ajudar."
    assert response.tool_calls == []
    assert response.metadata["fallback"] is False
    assert response.metadata["model"] == DEFAULT_GEMINI_MODEL
    assert logger.events[-1]["status"] == "ok"
    assert logger.events[-1]["input_length"] == 3
    assert logger.events[-1]["response_length"] == len(response.text)


def test_assistant_respond_keeps_string_contract(tmp_path):
    assistant = Assistant(make_config(tmp_path))

    response = assistant.respond("ola")

    assert isinstance(response, str)
    assert response == "Recebi seu pedido, mas a inteligencia online nao esta disponivel agora."
    assert "ola" not in response


def test_assistant_core_limits_short_history(tmp_path):
    client = FakeGeminiClient(responses=["resposta 1", "resposta 2", "resposta 3"])
    core = AssistantCore(
        make_config(tmp_path, api_key="fake-key"),
        gemini_client=client,
        event_logger=MemoryEventLogger(),
        max_history_messages=4,
    )

    core.respond("primeira")
    core.respond("segunda")
    core.respond("terceira")

    assert core.history == [
        {"role": "user", "content": "segunda"},
        {"role": "model", "content": "resposta 2"},
        {"role": "user", "content": "terceira"},
        {"role": "model", "content": "resposta 3"},
    ]
    assert [message["content"] for message in client.calls[-1][1]] == [
        "primeira",
        "resposta 1",
        "segunda",
        "resposta 2",
        "terceira",
    ]


def test_assistant_core_empty_input_uses_local_fallback_without_history(tmp_path):
    logger = MemoryEventLogger()
    core = AssistantCore(
        make_config(tmp_path, api_key="fake-key"),
        gemini_client=FakeGeminiClient(responses=["nao usado"]),
        event_logger=logger,
    )

    response = core.respond("   ")

    assert response.text == "Pode repetir o que voce precisa?"
    assert response.metadata["fallback"] is True
    assert response.metadata["error_type"] == "empty_input"
    assert core.history == []
    assert logger.events[-1]["status"] == "fallback"
    assert logger.events[-1]["history_size"] == 0


def test_assistant_core_gemini_unavailable_uses_local_fallback(tmp_path):
    core = AssistantCore(
        make_config(tmp_path),
        gemini_client=None,
        event_logger=MemoryEventLogger(),
    )

    response = core.respond("abrir agenda")

    assert response.metadata["fallback"] is True
    assert response.metadata["error_type"] == "gemini_unavailable"
    assert response.text == "Recebi seu pedido, mas a inteligencia online nao esta disponivel agora."
    assert "abrir agenda" not in response.text
    assert len(core.history) == 2


def test_assistant_core_gemini_error_uses_local_fallback(tmp_path):
    core = AssistantCore(
        make_config(tmp_path, api_key="fake-key"),
        gemini_client=FakeGeminiClient(error=GeminiClientError("generation request failed.")),
        event_logger=MemoryEventLogger(),
    )

    response = core.respond("abrir agenda")

    assert response.metadata["fallback"] is True
    assert response.metadata["error_type"] == "gemini_error"
    assert response.text == "Recebi seu pedido, mas a inteligencia online nao esta disponivel agora."
    assert "abrir agenda" not in response.text
    assert len(core.history) == 2


def test_assistant_core_empty_gemini_response_uses_local_fallback(tmp_path):
    core = AssistantCore(
        make_config(tmp_path, api_key="fake-key"),
        gemini_client=FakeGeminiClient(responses=["  \n  "]),
        event_logger=MemoryEventLogger(),
    )

    response = core.respond("abrir agenda")

    assert response.metadata["fallback"] is True
    assert response.metadata["error_type"] == "empty_response"
    assert response.text == "Recebi seu pedido, mas a inteligencia online nao esta disponivel agora."
    assert "abrir agenda" not in response.text
    assert len(core.history) == 2


def test_event_logger_appends_jsonl_without_transcript_response_or_secrets(tmp_path):
    log_path = tmp_path / "events.jsonl"
    logger = EventLogger(log_path)

    logger.log(
        {
            "event": "assistant.respond",
            "status": "ok",
            "model": DEFAULT_GEMINI_MODEL,
            "fallback": False,
            "latency_ms": 10.7,
            "input_length": 12,
            "response_length": 20,
            "history_size": 2,
            "transcript": "codigo secreto 123",
            "response": "resposta privada",
            "GOOGLE_API_KEY": "AIza-secret",
            "api_key": "secret-token",
        }
    )
    logger.log({"event": "assistant.respond", "status": "fallback", "error_type": "empty_input"})

    lines = log_path.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines]
    combined = "\n".join(lines)

    assert len(lines) == 2
    assert payloads[0]["event"] == "assistant.respond"
    assert payloads[0]["latency_ms"] == 10.7
    assert payloads[1]["status"] == "fallback"
    assert "codigo secreto 123" not in combined
    assert "resposta privada" not in combined
    assert "AIza-secret" not in combined
    assert "secret-token" not in combined
    assert "transcript" not in payloads[0]
    assert "response" not in payloads[0]
    assert "api_key" not in payloads[0]


def test_event_logger_failure_does_not_break_caller(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("already a file", encoding="utf-8")
    logger = EventLogger(blocker / "events.jsonl")

    logger.log({"event": "assistant.respond", "status": "ok"})


def test_gemini_generate_error_is_sanitized_without_raw_detail():
    leaked_detail = "provider stack trace token=secret raw prompt"

    class FakePart:
        @staticmethod
        def from_text(text):
            return SimpleNamespace(text=text)

    class FakeTypes:
        Part = FakePart

        @staticmethod
        def Content(role, parts):
            return SimpleNamespace(role=role, parts=parts)

        @staticmethod
        def GenerateContentConfig(**kwargs):
            return SimpleNamespace(**kwargs)

    class FakeModels:
        def generate_content(self, **_kwargs):
            raise RuntimeError(leaked_detail)

    client = GeminiClient(
        api_key="fake-key",
        transcription_model="transcribe",
        tts_model="tts",
        tts_voice="Kore",
        chat_model="chat",
    )
    client._client = SimpleNamespace(models=FakeModels())
    client._types = FakeTypes

    with pytest.raises(GeminiClientError) as exc_info:
        client.generate("system prompt", [{"role": "user", "content": "ola"}])

    assert str(exc_info.value) == "generation request failed."
    assert leaked_detail not in str(exc_info.value)
