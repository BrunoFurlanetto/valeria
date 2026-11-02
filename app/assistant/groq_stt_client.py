"""Cliente de transcricao de audio via Groq Whisper API."""

from pathlib import Path


class SttClientError(Exception):
    """Raised when the Groq STT transcription fails."""


class GroqSttClient:
    def __init__(
        self,
        api_key: str,
        model: str = "whisper-large-v3-turbo",
        language: str = "pt",
    ):
        if not api_key:
            raise SttClientError("GROQ_API_KEY e obrigatorio para transcricao.")
        self._api_key = api_key
        self.model = model
        self.language = language
        self._client = None

    def transcribe(self, audio_path) -> str:
        """Transcreve arquivo de audio WAV para texto em pt-BR.

        Args:
            audio_path: Caminho para o arquivo WAV de entrada.

        Returns:
            Texto transcrito.

        Raises:
            SttClientError: Se a transcricao falhar ou retornar vazio.
        """
        path = Path(audio_path)
        if not path.exists():
            raise SttClientError(f"arquivo de audio nao encontrado: {path}")

        client = self._get_client()
        try:
            with open(path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=self.language,
                )
        except Exception as exc:
            raise SttClientError("Transcricao falhou.") from exc

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise SttClientError("Transcricao retornou texto vazio.")
        return text

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
            except ImportError as exc:
                raise SttClientError("SDK groq nao instalado.") from exc
            self._client = Groq(api_key=self._api_key)
        return self._client
