"""Cliente de chat via Groq API.

Usa gpt-oss-20b como modelo primario (reasoning, reasoning_format=hidden para
suprimir chain-of-thought visivel) com fallback automatico para qwen/qwen3-32b.
"""

import re


class ChatClientError(Exception):
    """Raised when the Groq chat request fails on both models."""


class GroqChatClient:
    # Expressao para remover tags de raciocinio remanescentes se reasoning_format nao suprimir
    _THINKING_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

    def __init__(
        self,
        api_key: str,
        chat_model: str = "openai/gpt-oss-20b",
        fallback_model: str = "qwen/qwen3-32b",
        max_tokens: int = 160,
        temperature: float = 0.4,
    ):
        if not api_key:
            raise ChatClientError("GROQ_API_KEY e obrigatorio para o chat.")
        self._api_key = api_key
        self.chat_model = chat_model
        self.fallback_model = fallback_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def generate(self, prompt: str, messages: list) -> str:
        """Gera resposta de texto para a lista de mensagens.

        Args:
            prompt: System prompt com persona do assistente.
            messages: Historico de mensagens no formato [{role, content}].

        Returns:
            Texto da resposta gerado pelo modelo.

        Raises:
            ChatClientError: Se ambos os modelos falharem.
        """
        client = self._get_client()
        chat_messages = self._build_messages(prompt, messages)

        try:
            response, model_used = self._call(client, self.chat_model, chat_messages)
        except Exception:
            try:
                response, model_used = self._call(client, self.fallback_model, chat_messages)
            except Exception as exc:
                raise ChatClientError("Todos os modelos de chat falharam.") from exc

        text = self._extract_text(response)
        return self._strip_thinking_tags(text).strip()

    def generate_with_metadata(self, prompt: str, messages: list) -> tuple[str, str, int | None]:
        """Como generate(), mas retorna (texto, modelo_usado, tokens_usados).

        Returns:
            Tupla (texto_resposta, nome_modelo, tokens_totais_ou_None)
        """
        client = self._get_client()
        chat_messages = self._build_messages(prompt, messages)

        try:
            response, model_used = self._call(client, self.chat_model, chat_messages)
        except Exception:
            try:
                response, model_used = self._call(client, self.fallback_model, chat_messages)
            except Exception as exc:
                raise ChatClientError("Todos os modelos de chat falharam.") from exc

        text = self._extract_text(response)
        tokens = self._extract_tokens(response)
        return self._strip_thinking_tags(text).strip(), model_used, tokens

    # -------------------------------------------------------------------------
    # Metodos privados
    # -------------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
            except ImportError as exc:
                raise ChatClientError("SDK groq nao instalado.") from exc
            self._client = Groq(api_key=self._api_key)
        return self._client

    def _call(self, client, model: str, messages: list):
        """Faz uma chamada de chat completion. Retorna (response, model_name)."""
        kwargs = dict(
            model=model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        # reasoning_format=hidden suprime CoT visivel em modelos de raciocinio
        if "gpt-oss" in model or "reasoning" in model:
            kwargs["reasoning_format"] = "hidden"

        response = client.chat.completions.create(**kwargs)
        return response, model

    def _build_messages(self, prompt: str, messages: list) -> list:
        """Monta lista de mensagens com system prompt."""
        chat_messages = [{"role": "system", "content": prompt}]
        for msg in messages:
            role = msg.get("role", "user")
            # Groq usa "assistant" em vez de "model"
            if role == "model":
                role = "assistant"
            content = str(msg.get("content", "")).strip()
            if content:
                chat_messages.append({"role": role, "content": content})
        return chat_messages

    @staticmethod
    def _extract_text(response) -> str:
        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            return ""

    @staticmethod
    def _extract_tokens(response) -> int | None:
        try:
            usage = response.usage
            if usage is not None:
                return getattr(usage, "total_tokens", None)
        except AttributeError:
            pass
        return None

    @classmethod
    def _strip_thinking_tags(cls, text: str) -> str:
        """Remove tags <think>...</think> remanescentes do output."""
        return cls._THINKING_RE.sub("", text)
