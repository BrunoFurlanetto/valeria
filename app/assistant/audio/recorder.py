import time
import wave
from pathlib import Path


class RecordingError(Exception):
    """Raised when command audio cannot be recorded."""


class CommandRecorder:
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

    def record(self, output_dir, duration_seconds=5.0):
        output_path = self._output_path(output_dir)

        try:
            import pyaudio
        except ImportError as exc:
            raise RecordingError(f"PyAudio is unavailable: {exc}") from exc

        audio = pyaudio.PyAudio()
        stream = None
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )
            frames = self._read_frames(stream, duration_seconds)
            self._save_wave(output_path, frames, audio.get_sample_size(pyaudio.paInt16))
        except OSError as exc:
            raise RecordingError(f"microphone recording failed: {exc}") from exc
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            audio.terminate()

        return output_path

    def _read_frames(self, stream, duration_seconds):
        frames = []
        total_chunks = max(1, int(self.sample_rate / self.chunk_size * duration_seconds))
        for _ in range(total_chunks):
            frames.append(stream.read(self.chunk_size, exception_on_overflow=False))
        return frames

    def _save_wave(self, output_path, frames, sample_width):
        with wave.open(str(output_path), "wb") as wave_file:
            wave_file.setnchannels(1)
            wave_file.setsampwidth(sample_width)
            wave_file.setframerate(self.sample_rate)
            wave_file.writeframes(b"".join(frames))

    def _output_path(self, output_dir):
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        return directory / f"command-{stamp}.wav"
