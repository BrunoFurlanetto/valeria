import wave
from pathlib import Path


class PlaybackError(Exception):
    """Raised when generated audio cannot be played locally."""


class AudioPlayer:
    def play(self, audio_path):
        path = Path(audio_path)
        if not path.exists():
            raise PlaybackError(f"audio file not found: {path}")

        try:
            import pyaudio
        except ImportError as exc:
            raise PlaybackError(f"PyAudio is unavailable: {exc}") from exc

        try:
            with wave.open(str(path), "rb") as wave_file:
                audio = pyaudio.PyAudio()
                stream = None

                try:
                    stream = audio.open(
                        format=audio.get_format_from_width(wave_file.getsampwidth()),
                        channels=wave_file.getnchannels(),
                        rate=wave_file.getframerate(),
                        output=True,
                    )
                    data = wave_file.readframes(1024)
                    while data:
                        stream.write(data)
                        data = wave_file.readframes(1024)
                finally:
                    if stream is not None:
                        stream.stop_stream()
                        stream.close()
                    audio.terminate()
        except wave.Error as exc:
            raise PlaybackError(f"invalid wave audio: {path}") from exc
        except OSError as exc:
            raise PlaybackError(f"audio playback failed: {exc}") from exc
