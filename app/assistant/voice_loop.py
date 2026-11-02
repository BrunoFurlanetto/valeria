import tempfile

from app.assistant.audio.playback import AudioPlayer, PlaybackError
from app.assistant.audio.recorder import CommandRecorder, RecordingError
from app.assistant.core import Assistant
from app.assistant.elevenlabs_tts_client import ElevenLabsTtsClient, TtsClientError
from app.assistant.feedback_player import FeedbackPlayer
from app.assistant.groq_stt_client import GroqSttClient, SttClientError
from app.assistant.messages import FALLBACK_ERROR_TEXT, FALLBACK_POSITIVE_TEXT


class VoiceLoop:
    def __init__(self, config, wake_word_engine):
        self.config = config
        self.wake_word_engine = wake_word_engine
        self.assistant = Assistant(config)
        self.recorder = CommandRecorder(sample_rate=config.command_sample_rate)
        self.player = AudioPlayer()
        self.feedback = FeedbackPlayer(player=self.player)
        self.stt = GroqSttClient(
            api_key=config.groq_api_key or "",
            model=config.groq_stt_model,
        )
        self.tts = self._build_tts_client(config)
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
            except RecordingError as exc:
                print(f"Voice error: {exc}")
                return

            # STT
            try:
                transcript = self.stt.transcribe(command_audio)
            except SttClientError as exc:
                print(f"STT error: {exc}")
                self.feedback.play_earcon("error")
                self.feedback.play_fallback_phrase("error")
                print(FALLBACK_ERROR_TEXT)
                return

            print("Command transcribed.")

            # Chat
            response_text = self.assistant.respond(transcript)
            print("Assistant response generated.")

            # TTS
            self.feedback.play_earcon("positive")

            if self.tts is None:
                # ElevenLabs nao configurado — usar frase estatica
                self.feedback.play_fallback_phrase("positive")
                print(FALLBACK_POSITIVE_TEXT)
                print(f"Response: {response_text}")
                return

            try:
                speech_path = self.tts.synthesize_speech(response_text, turn_dir)
                try:
                    self.player.play(speech_path)
                    print("Audio response played.")
                except PlaybackError as exc:
                    print(f"Playback error: {exc}")
                    self.feedback.play_fallback_phrase("positive")
                    print(FALLBACK_POSITIVE_TEXT)
                    print(f"Response: {response_text}")
            except TtsClientError as exc:
                print(f"TTS error ({exc.error_type}): {exc}")
                self.feedback.play_fallback_phrase("positive")
                print(FALLBACK_POSITIVE_TEXT)
                print(f"Response: {response_text}")

    @staticmethod
    def _build_tts_client(config):
        if not config.elevenlabs_api_key or not config.elevenlabs_voice_id:
            return None
        try:
            return ElevenLabsTtsClient(
                api_key=config.elevenlabs_api_key,
                voice_id=config.elevenlabs_voice_id,
                model_id=config.elevenlabs_model,
            )
        except TtsClientError:
            return None
