"""FeedbackPlayer: reproduz earcons e frases de fallback estaticas.

Assets esperados em app/assistant/assets/sounds/:
- earcon_positive.wav
- earcon_error.wav
- fallback_positive.wav
- fallback_error.wav

Qualquer ausencia ou erro de audio e tratado silenciosamente.
"""

from pathlib import Path

_ASSETS_DIR = Path(__file__).parent / "assets" / "sounds"

_EARCON_FILES = {
    "positive": _ASSETS_DIR / "earcon_positive.wav",
    "error": _ASSETS_DIR / "earcon_error.wav",
}

_FALLBACK_FILES = {
    "positive": _ASSETS_DIR / "fallback_positive.wav",
    "error": _ASSETS_DIR / "fallback_error.wav",
}


class FeedbackPlayer:
    """Reproduz sons de feedback usando o AudioPlayer existente."""

    def __init__(self, player=None):
        """
        Args:
            player: Instancia de AudioPlayer. Se None, cria uma nova.
        """
        self._player = player

    def play_earcon(self, kind: str) -> None:
        """Toca o earcon correspondente ao tipo de evento.

        Args:
            kind: "positive" ou "error". Silencioso se tipo desconhecido ou arquivo ausente.
        """
        path = _EARCON_FILES.get(kind)
        self._play_silent(path)

    def play_fallback_phrase(self, kind: str) -> None:
        """Toca a frase de fallback estatica correspondente ao tipo de evento.

        Args:
            kind: "positive" ou "error". Silencioso se tipo desconhecido ou arquivo ausente.
        """
        path = _FALLBACK_FILES.get(kind)
        self._play_silent(path)

    def _play_silent(self, path: Path | None) -> None:
        """Tenta reproduzir o arquivo WAV. Silencioso em caso de qualquer erro."""
        if path is None or not path.exists():
            return
        try:
            player = self._get_player()
            player.play(path)
        except Exception:
            # FeedbackPlayer nunca propaga excecao — feedback e best-effort
            pass

    def _get_player(self):
        if self._player is None:
            from app.assistant.audio.playback import AudioPlayer
            self._player = AudioPlayer()
        return self._player
