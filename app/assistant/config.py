import os
from pathlib import Path
from dataclasses import dataclass
from typing import ClassVar


class AssistantConfigError(Exception):
    """Raised when environment configuration is invalid."""


@dataclass(frozen=True)
class AssistantConfig:
    assistant_name: str = "Valeria"

    # Groq — chat e STT
    groq_api_key: str | None = None
    groq_chat_model: str = "openai/gpt-oss-20b"
    groq_chat_fallback_model: str = "qwen/qwen3-32b"
    groq_stt_model: str = "whisper-large-v3-turbo"

    # ElevenLabs — TTS
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = None
    elevenlabs_model: str = "eleven_flash_v2_5"

    # Audio local
    audio_output_dir: Path = Path("tmp/valeria-audio")
    event_log_path: Path = Path("tmp/valeria-logs/events.jsonl")
    command_record_seconds: float = 5.0
    command_sample_rate: int = 16000
    tts_sample_rate: int = 24000

    _AUDIO_BASE_DIR: ClassVar[Path] = Path("tmp/valeria-audio")
    _DURATION_RANGE: ClassVar[tuple[float, float]] = (0.5, 30.0)
    _SAMPLE_RATE_RANGE: ClassVar[tuple[int, int]] = (8000, 48000)

    @classmethod
    def from_env(cls):
        cls._load_dotenv_if_available()
        assistant_name = os.getenv("VALERIA_ASSISTANT_NAME", "Valeria").strip() or "Valeria"

        # Groq
        groq_api_key = os.getenv("GROQ_API_KEY")
        groq_chat_model = (
            os.getenv("VALERIA_GROQ_CHAT_MODEL", "openai/gpt-oss-20b").strip()
            or "openai/gpt-oss-20b"
        )
        groq_chat_fallback_model = (
            os.getenv("VALERIA_GROQ_CHAT_FALLBACK_MODEL", "qwen/qwen3-32b").strip()
            or "qwen/qwen3-32b"
        )
        groq_stt_model = (
            os.getenv("VALERIA_GROQ_STT_MODEL", "whisper-large-v3-turbo").strip()
            or "whisper-large-v3-turbo"
        )

        # ElevenLabs
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID")
        elevenlabs_model = (
            os.getenv("VALERIA_ELEVENLABS_MODEL", "eleven_flash_v2_5").strip()
            or "eleven_flash_v2_5"
        )

        audio_output_dir = cls._audio_dir_from_env()
        command_record_seconds = cls._float_from_env(
            "VALERIA_COMMAND_RECORD_SECONDS",
            5.0,
            cls._DURATION_RANGE[0],
            cls._DURATION_RANGE[1],
        )
        command_sample_rate = cls._int_from_env(
            "VALERIA_COMMAND_SAMPLE_RATE",
            16000,
            cls._SAMPLE_RATE_RANGE[0],
            cls._SAMPLE_RATE_RANGE[1],
        )
        tts_sample_rate = cls._int_from_env(
            "VALERIA_TTS_SAMPLE_RATE",
            24000,
            cls._SAMPLE_RATE_RANGE[0],
            cls._SAMPLE_RATE_RANGE[1],
        )

        return cls(
            assistant_name=assistant_name,
            groq_api_key=groq_api_key,
            groq_chat_model=groq_chat_model,
            groq_chat_fallback_model=groq_chat_fallback_model,
            groq_stt_model=groq_stt_model,
            elevenlabs_api_key=elevenlabs_api_key,
            elevenlabs_voice_id=elevenlabs_voice_id,
            elevenlabs_model=elevenlabs_model,
            audio_output_dir=audio_output_dir,
            command_record_seconds=command_record_seconds,
            command_sample_rate=command_sample_rate,
            tts_sample_rate=tts_sample_rate,
        )

    @classmethod
    def _audio_dir_from_env(cls):
        raw_path = os.getenv("VALERIA_AUDIO_OUTPUT_DIR", str(cls._AUDIO_BASE_DIR)).strip()
        candidate = Path(raw_path or cls._AUDIO_BASE_DIR)
        base = cls._AUDIO_BASE_DIR.resolve()
        resolved = candidate.resolve()
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise AssistantConfigError(
                "VALERIA_AUDIO_OUTPUT_DIR must be tmp/valeria-audio or a subdirectory."
            ) from exc
        return candidate

    @staticmethod
    def _load_dotenv_if_available():
        try:
            from dotenv import load_dotenv
        except ImportError:
            return
        load_dotenv()

    @staticmethod
    def _float_from_env(name, default, minimum, maximum):
        value = os.getenv(name)
        if value is None or not value.strip():
            return default
        try:
            parsed = float(value)
        except ValueError as exc:
            raise AssistantConfigError(f"{name} must be a number.") from exc
        if not minimum <= parsed <= maximum:
            raise AssistantConfigError(f"{name} must be between {minimum} and {maximum}.")
        return parsed

    @staticmethod
    def _int_from_env(name, default, minimum, maximum):
        value = os.getenv(name)
        if value is None or not value.strip():
            return default
        try:
            parsed = int(value)
        except ValueError as exc:
            raise AssistantConfigError(f"{name} must be an integer.") from exc
        if not minimum <= parsed <= maximum:
            raise AssistantConfigError(f"{name} must be between {minimum} and {maximum}.")
        return parsed
