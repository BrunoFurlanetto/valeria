import json

import pytest

from app.assistant.config import AssistantConfig
from app.assistant.core import Assistant, AssistantCore, AssistantResponse
from app.assistant.event_logger import EventLogger
from app.assistant.groq_chat_client import GroqChatClient, ChatClientError


DEFAULT_GROQ_CHAT_MODEL = "openai/gpt-oss-20b"


class MemoryEventLogger:
    def __init__(self):
        self.events = []

    def log(self, event):
        self.events.append(event)


class FakeGroqChatClient:
    """Fake de GroqChatClient que implementa generate_with_metadata sem rede."""

    def __init__(self, responses=None, error=None, model="openai/gpt-oss-20b", tokens=42):
        self.responses = list(responses or [])
        self.error = error
        self.model = model
        self.tokens = tokens
        self.calls = []

    def generate_with_metadata(self, prompt, messages):
        self.calls.append((prompt, list(messages)))
        if self.error:
            raise self.error
        text = self.responses.pop(0)
        return text, self.model, self.tokens

    def generate(self, prompt, messages):
        text, _model, _tokens = self.generate_with_metadata(prompt, messages)
        return text


def make_config(tmp_path, api_key=None):
    return AssistantConfig(
        groq_api_key=api_key,
        event_log_path=tmp_path / "events.jsonl",
    )


def test_assistant_config_from_env_uses_groq_defaults(monkeypatch):
    monkeypatch.setattr(AssistantConfig, "_load_dotenv_if_available", lambda: None)
    for name in (
        "GROQ_API_KEY",
        "VALERIA_GROQ_CHAT_MODEL",
        "VALERIA_GROQ_CHAT_FALLBACK_MODEL",
        "VALERIA_GROQ_STT_MODEL",
        "ELEVENLABS_API_KEY",
        "ELEVENLABS_VOICE_ID",
        "VALERIA_ELEVENLABS_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    config = AssistantConfig.from_env()

    assert config.groq_api_key is None
    assert config.groq_chat_model == DEFAULT_GROQ_CHAT_MODEL
    assert config.groq_chat_fallback_model == "qwen/qwen3-32b"
    assert config.groq_stt_model == "whisper-large-v3-turbo"
    assert config.elevenlabs_api_key is None
    assert config.elevenlabs_voice_id is None
    assert config.elevenlabs_model == "eleven_flash_v2_5"


def test_assistant_config_from_env_reads_groq_and_elevenlabs_keys(monkeypatch):
    monkeypatch.setattr(AssistantConfig, "_load_dotenv_if_available", lambda: None)
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.setenv("VALERIA_GROQ_CHAT_MODEL", "openai/custom-model")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-test-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice-id-abc")
    monkeypatch.setenv("VALERIA_ELEVENLABS_MODEL", "eleven_multilingual_v2")

    config = AssistantConfig.from_env()

    assert config.groq_api_key == "groq-test-key"
    assert config.groq_chat_model == "openai/custom-model"
    assert config.elevenlabs_api_key == "eleven-test-key"
    assert config.elevenlabs_voice_id == "voice-id-abc"
    assert config.elevenlabs_model == "eleven_multilingual_v2"


def test_assistant_core_respond_returns_assistant_response(tmp_path):
    logger = MemoryEventLogger()
    client = FakeGroqChatClient(responses=["  Ola, posso ajudar.  "])
    core = AssistantCore(
        make_config(tmp_path, api_key="fake-key"),
        chat_client=client,
        event_logger=logger,
    )

    response = core.respond("ola")

    assert isinstance(response, AssistantResponse)
    assert response.text == "Ola, posso ajudar."
    assert response.tool_calls == []
    assert response.metadata["fallback"] is False
    assert response.metadata["model"] == "openai/gpt-oss-20b"
    assert response.metadata["tokens_used"] == 42
    assert logger.events[-1]["status"] == "ok"
    assert logger.events[-1]["input_length"] == 3
    assert logger.events[-1]["response_length"] == len(response.text)
    assert logger.events[-1]["tokens_used"] == 42


def test_assistant_respond_keeps_string_contract(tmp_path):
    assistant = Assistant(make_config(tmp_path))

    response = assistant.respond("ola")

    assert isinstance(response, str)
    assert response == "Recebi seu pedido, mas a inteligencia online nao esta disponivel agora."
    assert "ola" not in response


def test_assistant_core_limits_short_history(tmp_path):
    client = FakeGroqChatClient(responses=["resposta 1", "resposta 2", "resposta 3"])
    core = AssistantCore(
        make_config(tmp_path, api_key="fake-key"),
        chat_client=client,
        event_logger=MemoryEventLogger(),
        max_history_messages=4,
    )

    core.respond("primeira")
    core.respond("segunda")
    core.respond("terceira")

    # History usa role "assistant" (padrao Groq), nao "model"
    assert core.history == [
        {"role": "user", "content": "segunda"},
        {"role": "assistant", "content": "resposta 2"},
        {"role": "user", "content": "terceira"},
        {"role": "assistant", "content": "resposta 3"},
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
        chat_client=FakeGroqChatClient(responses=["nao usado"]),
        event_logger=logger,
    )

    response = core.respond("   ")

    assert response.text == "Pode repetir o que voce precisa?"
    assert response.metadata["fallback"] is True
    assert response.metadata["error_type"] == "empty_input"
    assert core.history == []
    assert logger.events[-1]["status"] == "fallback"
    assert logger.events[-1]["history_size"] == 0


def test_assistant_core_groq_unavailable_uses_local_fallback(tmp_path):
    """Quando chat_client=None (sem GROQ_API_KEY), deve usar fallback groq_unavailable."""
    core = AssistantCore(
        make_config(tmp_path),  # groq_api_key=None
        chat_client=None,
        event_logger=MemoryEventLogger(),
    )

    response = core.respond("abrir agenda")

    assert response.metadata["fallback"] is True
    assert response.metadata["error_type"] == "groq_unavailable"
    assert response.text == "Recebi seu pedido, mas a inteligencia online nao esta disponivel agora."
    assert "abrir agenda" not in response.text
    assert len(core.history) == 2


def test_assistant_core_groq_error_uses_local_fallback(tmp_path):
    """Quando GroqChatClient levanta ChatClientError, deve usar fallback groq_error."""
    core = AssistantCore(
        make_config(tmp_path, api_key="fake-key"),
        chat_client=FakeGroqChatClient(error=ChatClientError("Todos os modelos falharam.")),
        event_logger=MemoryEventLogger(),
    )

    response = core.respond("abrir agenda")

    assert response.metadata["fallback"] is True
    assert response.metadata["error_type"] == "groq_error"
    assert response.text == "Recebi seu pedido, mas a inteligencia online nao esta disponivel agora."
    assert "abrir agenda" not in response.text
    assert len(core.history) == 2


def test_assistant_core_empty_groq_response_uses_local_fallback(tmp_path):
    """Quando o modelo retorna texto vazio, deve usar fallback empty_response."""
    core = AssistantCore(
        make_config(tmp_path, api_key="fake-key"),
        chat_client=FakeGroqChatClient(responses=["  \n  "]),
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
            "model": DEFAULT_GROQ_CHAT_MODEL,
            "fallback": False,
            "latency_ms": 10.7,
            "input_length": 12,
            "response_length": 20,
            "tokens_used": 87,
            "history_size": 2,
            # campos sensiveis que NAO devem aparecer no log
            "transcript": "codigo secreto 123",
            "response": "resposta privada",
            "GROQ_API_KEY": "gsk_secret",
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
    assert "gsk_secret" not in combined
    assert "secret-token" not in combined
    assert "transcript" not in payloads[0]
    assert "response" not in payloads[0]
    assert "api_key" not in payloads[0]


def test_event_logger_includes_tokens_used(tmp_path):
    """EventLogger deve preservar tokens_used no payload quando presente."""
    log_path = tmp_path / "events.jsonl"
    logger = EventLogger(log_path)

    logger.log(
        {
            "event": "assistant.respond",
            "status": "ok",
            "model": DEFAULT_GROQ_CHAT_MODEL,
            "tokens_used": 120,
            "fallback": False,
        }
    )

    payload = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert payload["tokens_used"] == 120


def test_event_logger_failure_does_not_break_caller(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("already a file", encoding="utf-8")
    logger = EventLogger(blocker / "events.jsonl")

    logger.log({"event": "assistant.respond", "status": "ok"})


def test_groq_chat_client_raises_on_empty_api_key():
    """GroqChatClient deve levantar ChatClientError se api_key for vazia ou None."""
    with pytest.raises(ChatClientError, match="GROQ_API_KEY"):
        GroqChatClient(api_key="")

    with pytest.raises(ChatClientError, match="GROQ_API_KEY"):
        GroqChatClient(api_key=None)


def test_sec_03_001_fallback_does_not_echo_user_input(tmp_path):
    """SEC-03-001: nenhum texto do usuario deve aparecer na resposta de fallback."""
    sensitive_inputs = [
        "minha senha eh abc123",
        "numero do cartao 1234 5678",
        "codigo secreto ultraconfidencial",
    ]
    for user_text in sensitive_inputs:
        core = AssistantCore(
            make_config(tmp_path),
            chat_client=None,
            event_logger=MemoryEventLogger(),
        )
        response = core.respond(user_text)
        assert user_text not in response.text, (
            f"Fallback ecoou entrada do usuario: '{user_text}' apareceu em '{response.text}'"
        )
