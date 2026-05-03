import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class AssistantConfig:
    assistant_name: str = "Valeria"
    google_api_key: str | None = None
    gemini_model: str = "gemini-pro"

    @classmethod
    def from_env(cls):
        load_dotenv()
        assistant_name = os.getenv("VALERIA_ASSISTANT_NAME", "Valeria").strip() or "Valeria"
        gemini_model = os.getenv("VALERIA_GEMINI_MODEL", "gemini-pro").strip() or "gemini-pro"

        return cls(
            assistant_name=assistant_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            gemini_model=gemini_model,
        )

