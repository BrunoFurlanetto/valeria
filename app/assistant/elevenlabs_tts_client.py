"""Cliente TTS via ElevenLabs API.

Mantem a interface synthesize_speech(text, output_dir) -> Path para
compatibilidade com VoiceLoop.
"""

import time
from pathlib import Path


class TtsClientError(Exception):
    """Raised when ElevenLabs TTS fails."""

    def __init__(self, message: str, error_type: str = "tts_error"):
        super().__init__(message)
        self.error_type = error_type


class ElevenLabsTtsClient:
    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_flash_v2_5",
        output_format: str = "wav_24000",
    ):
        if not api_key:
            raise TtsClientError("ELEVENLABS_API_KEY e obrigatorio para TTS.", "tts_config_error")
        if not voice_id:
            raise TtsClientError("ELEVENLABS_VOICE_ID e obrigatorio para TTS.", "tts_config_error")
        self._api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self._client = None

    def synthesize_speech(self, text: str, output_dir) -> Path:
        """Sintetiza texto para fala e salva como WAV.

        Args:
            text: Texto a ser convertido em fala.
            output_dir: Diretorio onde o arquivo WAV sera salvo.

        Returns:
            Path do arquivo WAV gerado.

        Raises:
            TtsClientError: Se a sintese falhar. O campo error_type indica
                o tipo de erro para logging:
                - elevenlabs_auth_error     (401)
                - elevenlabs_quota_exceeded (402)
                - elevenlabs_rate_limited   (429)
                - elevenlabs_tts_error      (outros)
        """
        client = self._get_client()
        output_path = self._output_path(output_dir)

        try:
            audio_chunks = client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id=self.model_id,
                output_format=self.output_format,
            )
            audio_bytes = b"".join(audio_chunks)
        except Exception as exc:
            error_type = self._classify_error(exc)
            raise TtsClientError(f"TTS falhou: {type(exc).__name__}", error_type) from exc

        if not audio_bytes:
            raise TtsClientError("TTS retornou audio vazio.", "tts_empty_response")

        output_path.write_bytes(audio_bytes)
        return output_path

    # -------------------------------------------------------------------------
    # Metodos privados
    # -------------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            try:
                from elevenlabs import ElevenLabs
            except ImportError as exc:
                raise TtsClientError("SDK elevenlabs nao instalado.", "tts_import_error") from exc
            self._client = ElevenLabs(api_key=self._api_key)
        return self._client

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        """Mapeia excecao do ElevenLabs SDK para tipo de erro de log."""
        exc_type = type(exc).__name__
        # Erros especificos do SDK ElevenLabs
        if exc_type == "UnauthorizedError":
            return "elevenlabs_auth_error"
        if exc_type == "ForbiddenError":
            return "elevenlabs_auth_error"
        # ApiError tem status_code para 402 (quota) e 429 (rate limit)
        status = getattr(exc, "status_code", None)
        if status == 401:
            return "elevenlabs_auth_error"
        if status == 402:
            return "elevenlabs_quota_exceeded"
        if status == 429:
            return "elevenlabs_rate_limited"
        return "elevenlabs_tts_error"

    @staticmethod
    def _output_path(output_dir) -> Path:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        return directory / f"response-{stamp}.wav"
