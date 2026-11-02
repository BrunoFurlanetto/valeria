import time
from dataclasses import dataclass, field

from app.assistant.config import AssistantConfig
from app.assistant.event_logger import EventLogger
from app.assistant.groq_chat_client import GroqChatClient, ChatClientError
from app.assistant.prompts import VALERIA_PERSONA_PROMPT


@dataclass(frozen=True)
class AssistantResponse:
    text: str
    tool_calls: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class AssistantCore:
    def __init__(
        self,
        config: AssistantConfig,
        chat_client=None,
        event_logger=None,
        max_history_messages=6,
    ):
        self.config = config
        self.chat_client = chat_client or self._build_chat_client()
        self.event_logger = event_logger or EventLogger(config.event_log_path)
        self.max_history_messages = max(2, int(max_history_messages))
        self.history = []

    def respond(self, text):
        started = time.monotonic()
        cleaned = (text or "").strip()
        if not cleaned:
            response = self._fallback_response("empty_input")
            self._log_turn(started, cleaned, response, fallback=True, error_type="empty_input")
            return response

        if self.chat_client is None:
            response = self._fallback_response("groq_unavailable")
            self._remember(cleaned, response.text)
            self._log_turn(started, cleaned, response, fallback=True, error_type="groq_unavailable")
            return response

        messages = [*self.history, {"role": "user", "content": cleaned}]
        try:
            generated, model_used, tokens_used = self.chat_client.generate_with_metadata(
                VALERIA_PERSONA_PROMPT, messages
            )
        except ChatClientError:
            response = self._fallback_response("groq_error")
            self._remember(cleaned, response.text)
            self._log_turn(started, cleaned, response, fallback=True, error_type="groq_error")
            return response

        generated = generated.strip()
        if not generated:
            response = self._fallback_response("empty_response")
            self._remember(cleaned, response.text)
            self._log_turn(started, cleaned, response, fallback=True, error_type="empty_response")
            return response

        response = AssistantResponse(
            text=self._compact_for_voice(generated),
            tool_calls=[],
            metadata={
                "model": model_used,
                "fallback": False,
                "latency_ms": self._latency_ms(started),
                "tokens_used": tokens_used,
            },
        )
        self._remember(cleaned, response.text)
        self._log_turn(started, cleaned, response, fallback=False)
        return response

    def _build_chat_client(self):
        if not self.config.groq_api_key:
            return None
        try:
            return GroqChatClient(
                api_key=self.config.groq_api_key,
                chat_model=self.config.groq_chat_model,
                fallback_model=self.config.groq_chat_fallback_model,
            )
        except ChatClientError:
            return None

    def _fallback_response(self, reason):
        text_by_reason = {
            "empty_input": "Pode repetir o que voce precisa?",
            "empty_response": "Recebi seu pedido, mas a inteligencia online nao esta disponivel agora.",
            "groq_error": "Recebi seu pedido, mas a inteligencia online nao esta disponivel agora.",
            "groq_unavailable": "Recebi seu pedido, mas a inteligencia online nao esta disponivel agora.",
        }
        return AssistantResponse(
            text=text_by_reason.get(reason, text_by_reason["groq_error"]),
            tool_calls=[],
            metadata={
                "model": self.config.groq_chat_model,
                "fallback": True,
                "error_type": reason,
                "tokens_used": None,
            },
        )

    def _remember(self, user_text, assistant_text):
        self.history.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        )
        self.history = self.history[-self.max_history_messages :]

    def _compact_for_voice(self, text):
        cleaned = " ".join(text.split())
        return cleaned[:600].strip()

    def _log_turn(self, started, user_text, response, fallback, error_type=None):
        metadata = response.metadata or {}
        self.event_logger.log(
            {
                "event": "assistant.respond",
                "status": "fallback" if fallback else "ok",
                "model": metadata.get("model", self.config.groq_chat_model),
                "fallback": fallback,
                "error_type": error_type or metadata.get("error_type"),
                "latency_ms": self._latency_ms(started),
                "tokens_used": metadata.get("tokens_used"),
                "input_length": len(user_text),
                "response_length": len(response.text),
                "history_size": len(self.history),
            }
        )

    def _latency_ms(self, started):
        return int((time.monotonic() - started) * 1000)


class Assistant:
    def __init__(self, config: AssistantConfig):
        self.core = AssistantCore(config)

    def respond(self, user_input):
        return self.core.respond(user_input).text
