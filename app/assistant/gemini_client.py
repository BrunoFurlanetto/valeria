import time
import wave
from pathlib import Path


class GeminiClientError(Exception):
    """Raised when a Gemini voice operation fails."""


class GeminiClient:
    def __init__(self, api_key, transcription_model, tts_model, tts_voice, tts_sample_rate=24000):
        if not api_key:
            raise GeminiClientError("GOOGLE_API_KEY is required for voice mode.")

        self.api_key = api_key
        self.transcription_model = transcription_model
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.tts_sample_rate = tts_sample_rate
        self._client = None
        self._types = None

    def transcribe_audio(self, audio_path):
        client, types = self._load_sdk()
        path = Path(audio_path)
        try:
            audio_bytes = path.read_bytes()
        except OSError as exc:
            raise GeminiClientError(f"could not read audio file: {path}") from exc

        response = self._generate_content(
            client,
            model=self.transcription_model,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                "Transcribe this audio. Return only the spoken text in Brazilian Portuguese.",
            ],
            operation="transcription",
        )
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise GeminiClientError("transcription returned empty text.")
        return text

    def synthesize_speech(self, text, output_dir):
        client, types = self._load_sdk()
        output_path = self._output_path(output_dir)

        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.tts_voice,
                    )
                )
            ),
        )
        response = self._generate_content(
            client,
            model=self.tts_model,
            contents=f"Say this in Brazilian Portuguese: {text}",
            config=config,
            operation="TTS",
        )

        audio_bytes, mime_type = self._extract_audio(response)
        if not audio_bytes:
            raise GeminiClientError("TTS returned no audio.")

        if self._is_wave_audio(audio_bytes, mime_type):
            output_path.write_bytes(audio_bytes)
        else:
            self._write_pcm_as_wave(output_path, audio_bytes)
        return output_path

    def _load_sdk(self):
        if self._client is not None and self._types is not None:
            return self._client, self._types

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GeminiClientError("google-genai is not installed.") from exc

        self._client = genai.Client(api_key=self.api_key)
        self._types = types
        return self._client, self._types

    def _generate_content(self, client, operation, **kwargs):
        try:
            return client.models.generate_content(**kwargs)
        except Exception as exc:
            raise GeminiClientError(f"{operation} request failed.") from exc

    def _extract_audio(self, response):
        parts = list(getattr(response, "parts", []) or [])
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            parts.extend(getattr(content, "parts", []) or [])

        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            data = getattr(inline_data, "data", None)
            if data:
                return data, getattr(inline_data, "mime_type", None)

        return None, None

    def _is_wave_audio(self, audio_bytes, mime_type):
        return mime_type == "audio/wav" or audio_bytes.startswith(b"RIFF")

    def _write_pcm_as_wave(self, output_path, audio_bytes):
        with wave.open(str(output_path), "wb") as wave_file:
            wave_file.setnchannels(1)
            wave_file.setsampwidth(2)
            wave_file.setframerate(self.tts_sample_rate)
            wave_file.writeframes(audio_bytes)

    def _output_path(self, output_dir):
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        return directory / f"response-{stamp}.wav"
