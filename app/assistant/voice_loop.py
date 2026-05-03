import tempfile

from app.assistant.audio.playback import AudioPlayer, PlaybackError
from app.assistant.audio.recorder import CommandRecorder, RecordingError
from app.assistant.core import Assistant
from app.assistant.gemini_client import GeminiClient, GeminiClientError


class VoiceLoop:
    def __init__(self, config, wake_word_engine):
        self.config = config
        self.wake_word_engine = wake_word_engine
        self.assistant = Assistant(config)
        self.recorder = CommandRecorder(sample_rate=config.command_sample_rate)
        self.player = AudioPlayer()
        self.gemini = GeminiClient(
            api_key=config.google_api_key,
            transcription_model=config.transcription_model,
            tts_model=config.tts_model,
            tts_voice=config.tts_voice,
            tts_sample_rate=config.tts_sample_rate,
        )
        self._handling_turn = False

    def run(self):
        self.wake_word_engine.run(self._on_prediction)

    def _on_prediction(self, prediction):
        if prediction != 1 or self._handling_turn:
            return

        self._handling_turn = True
        try:
            self.handle_turn()
        finally:
            self._handling_turn = False

    def handle_turn(self):
        print("Wake word detected. Recording command...")
        self.config.audio_output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.config.audio_output_dir, prefix="turn-") as turn_dir:
            try:
                command_audio = self.recorder.record(
                    turn_dir,
                    duration_seconds=self.config.command_record_seconds,
                )
                transcript = self.gemini.transcribe_audio(command_audio)
                print("Command transcribed.")
                response_text = self.assistant.respond(transcript)
                print("Assistant response generated.")
                speech_path = self.gemini.synthesize_speech(response_text, turn_dir)
                self.player.play(speech_path)
                print("Audio response played.")
            except RecordingError as exc:
                print(f"Voice error: {exc}")
            except GeminiClientError as exc:
                print(f"Gemini error: {exc}")
            except PlaybackError as exc:
                print(f"Playback error: {exc}")
