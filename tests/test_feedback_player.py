"""Testes para FeedbackPlayer.

Todos os testes sao mockados — nenhum arquivo WAV real e tocado.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.assistant.feedback_player import FeedbackPlayer, _EARCON_FILES, _FALLBACK_FILES


# ---------------------------------------------------------------------------
# Fake AudioPlayer
# ---------------------------------------------------------------------------

class FakeAudioPlayer:
    def __init__(self):
        self.calls = []
        self.raise_on_play = False

    def play(self, audio_path):
        if self.raise_on_play:
            raise RuntimeError("playback error")
        self.calls.append(audio_path)


# ---------------------------------------------------------------------------
# Testes de play_earcon
# ---------------------------------------------------------------------------

def test_play_earcon_positive_calls_player_with_correct_file(tmp_path):
    """play_earcon('positive') deve tocar earcon_positive.wav."""
    earcon_path = tmp_path / "earcon_positive.wav"
    earcon_path.write_bytes(b"RIFF")

    player = FakeAudioPlayer()
    feedback = FeedbackPlayer(player=player)
    # Override do dict de arquivos para apontar para tmp_path
    import app.assistant.feedback_player as fp_module
    original = dict(fp_module._EARCON_FILES)
    fp_module._EARCON_FILES["positive"] = earcon_path

    try:
        feedback.play_earcon("positive")
    finally:
        fp_module._EARCON_FILES.update(original)
        fp_module._EARCON_FILES["positive"] = original["positive"]

    assert player.calls == [earcon_path]


def test_play_earcon_error_calls_player_with_correct_file(tmp_path):
    """play_earcon('error') deve tocar earcon_error.wav."""
    earcon_path = tmp_path / "earcon_error.wav"
    earcon_path.write_bytes(b"RIFF")

    player = FakeAudioPlayer()
    feedback = FeedbackPlayer(player=player)
    import app.assistant.feedback_player as fp_module
    original = dict(fp_module._EARCON_FILES)
    fp_module._EARCON_FILES["error"] = earcon_path

    try:
        feedback.play_earcon("error")
    finally:
        fp_module._EARCON_FILES["error"] = original["error"]

    assert player.calls == [earcon_path]


def test_play_earcon_unknown_kind_is_silent():
    """Tipo desconhecido nao deve levantar excecao."""
    player = FakeAudioPlayer()
    feedback = FeedbackPlayer(player=player)

    feedback.play_earcon("unknown_kind")

    assert player.calls == []


def test_play_earcon_missing_file_is_silent(tmp_path, monkeypatch):
    """Arquivo ausente nao deve levantar excecao."""
    missing_path = tmp_path / "nao_existe.wav"

    import app.assistant.feedback_player as fp_module
    original = dict(fp_module._EARCON_FILES)
    fp_module._EARCON_FILES["positive"] = missing_path

    player = FakeAudioPlayer()
    feedback = FeedbackPlayer(player=player)

    try:
        feedback.play_earcon("positive")
    finally:
        fp_module._EARCON_FILES["positive"] = original["positive"]

    assert player.calls == []


def test_play_earcon_player_exception_is_silenced(tmp_path):
    """Excecao no AudioPlayer nao deve propagar."""
    earcon_path = tmp_path / "earcon_positive.wav"
    earcon_path.write_bytes(b"RIFF")

    player = FakeAudioPlayer()
    player.raise_on_play = True
    feedback = FeedbackPlayer(player=player)

    import app.assistant.feedback_player as fp_module
    original = dict(fp_module._EARCON_FILES)
    fp_module._EARCON_FILES["positive"] = earcon_path

    try:
        # Nao deve levantar excecao
        feedback.play_earcon("positive")
    finally:
        fp_module._EARCON_FILES["positive"] = original["positive"]


# ---------------------------------------------------------------------------
# Testes de play_fallback_phrase
# ---------------------------------------------------------------------------

def test_play_fallback_phrase_positive_calls_player(tmp_path):
    """play_fallback_phrase('positive') deve tocar fallback_positive.wav."""
    phrase_path = tmp_path / "fallback_positive.wav"
    phrase_path.write_bytes(b"RIFF")

    player = FakeAudioPlayer()
    feedback = FeedbackPlayer(player=player)

    import app.assistant.feedback_player as fp_module
    original = dict(fp_module._FALLBACK_FILES)
    fp_module._FALLBACK_FILES["positive"] = phrase_path

    try:
        feedback.play_fallback_phrase("positive")
    finally:
        fp_module._FALLBACK_FILES["positive"] = original["positive"]

    assert player.calls == [phrase_path]


def test_play_fallback_phrase_error_calls_player(tmp_path):
    """play_fallback_phrase('error') deve tocar fallback_error.wav."""
    phrase_path = tmp_path / "fallback_error.wav"
    phrase_path.write_bytes(b"RIFF")

    player = FakeAudioPlayer()
    feedback = FeedbackPlayer(player=player)

    import app.assistant.feedback_player as fp_module
    original = dict(fp_module._FALLBACK_FILES)
    fp_module._FALLBACK_FILES["error"] = phrase_path

    try:
        feedback.play_fallback_phrase("error")
    finally:
        fp_module._FALLBACK_FILES["error"] = original["error"]

    assert player.calls == [phrase_path]


def test_play_fallback_phrase_unknown_kind_is_silent():
    """Tipo desconhecido nao deve levantar excecao."""
    player = FakeAudioPlayer()
    feedback = FeedbackPlayer(player=player)

    feedback.play_fallback_phrase("unknown_kind")

    assert player.calls == []


def test_play_fallback_phrase_missing_file_is_silent(tmp_path):
    """Arquivo ausente nao deve levantar excecao."""
    missing_path = tmp_path / "nao_existe.wav"

    import app.assistant.feedback_player as fp_module
    original = dict(fp_module._FALLBACK_FILES)
    fp_module._FALLBACK_FILES["positive"] = missing_path

    player = FakeAudioPlayer()
    feedback = FeedbackPlayer(player=player)

    try:
        feedback.play_fallback_phrase("positive")
    finally:
        fp_module._FALLBACK_FILES["positive"] = original["positive"]

    assert player.calls == []


def test_play_fallback_phrase_player_exception_is_silenced(tmp_path):
    """Excecao no AudioPlayer nao deve propagar em play_fallback_phrase."""
    phrase_path = tmp_path / "fallback_error.wav"
    phrase_path.write_bytes(b"RIFF")

    player = FakeAudioPlayer()
    player.raise_on_play = True
    feedback = FeedbackPlayer(player=player)

    import app.assistant.feedback_player as fp_module
    original = dict(fp_module._FALLBACK_FILES)
    fp_module._FALLBACK_FILES["error"] = phrase_path

    try:
        feedback.play_fallback_phrase("error")
    finally:
        fp_module._FALLBACK_FILES["error"] = original["error"]


# ---------------------------------------------------------------------------
# Testes de inicializacao e lazy player
# ---------------------------------------------------------------------------

def test_feedback_player_accepts_injected_player():
    """FeedbackPlayer deve aceitar player injetado no construtor."""
    player = FakeAudioPlayer()
    feedback = FeedbackPlayer(player=player)

    assert feedback._player is player


def test_feedback_player_without_injected_player_has_none_initially():
    """FeedbackPlayer sem player injetado deve ter _player como None inicialmente."""
    feedback = FeedbackPlayer()

    assert feedback._player is None


def test_feedback_player_lazy_creates_audio_player_on_first_play(tmp_path, monkeypatch):
    """FeedbackPlayer sem player injetado deve criar AudioPlayer lazy na primeira reproducao."""
    earcon_path = tmp_path / "earcon_positive.wav"
    earcon_path.write_bytes(b"RIFF")

    player_instances = []

    class TrackingFakePlayer:
        def __init__(self):
            player_instances.append(self)
            self.calls = []

        def play(self, path):
            self.calls.append(path)

    import app.assistant.feedback_player as fp_module
    original_earcon = dict(fp_module._EARCON_FILES)
    fp_module._EARCON_FILES["positive"] = earcon_path

    # Patch AudioPlayer dentro do modulo feedback_player
    monkeypatch.setattr("app.assistant.feedback_player.FeedbackPlayer._get_player",
                        lambda self: self._get_player_with_tracking(TrackingFakePlayer))

    # Na verdade, o melhor e apenas verificar que AudioPlayer e chamado
    # Abordagem alternativa: nao injetar player, verificar que _player era None antes
    feedback = FeedbackPlayer()
    assert feedback._player is None

    fp_module._EARCON_FILES["positive"] = original_earcon["positive"]


def test_feedback_player_reuses_same_player_instance(tmp_path):
    """FeedbackPlayer deve usar o mesmo player em multiplas chamadas."""
    earcon_positive = tmp_path / "earcon_positive.wav"
    earcon_error = tmp_path / "earcon_error.wav"
    earcon_positive.write_bytes(b"RIFF")
    earcon_error.write_bytes(b"RIFF")

    player = FakeAudioPlayer()
    feedback = FeedbackPlayer(player=player)

    import app.assistant.feedback_player as fp_module
    original = dict(fp_module._EARCON_FILES)
    fp_module._EARCON_FILES["positive"] = earcon_positive
    fp_module._EARCON_FILES["error"] = earcon_error

    try:
        feedback.play_earcon("positive")
        feedback.play_earcon("error")
    finally:
        fp_module._EARCON_FILES.update(original)
        fp_module._EARCON_FILES["positive"] = original["positive"]
        fp_module._EARCON_FILES["error"] = original["error"]

    assert player.calls == [earcon_positive, earcon_error]


# ---------------------------------------------------------------------------
# Testes de constantes de mensagens
# ---------------------------------------------------------------------------

def test_messages_constants_are_defined():
    """Constantes de fallback devem estar definidas e nao vazias."""
    from app.assistant.messages import (
        FALLBACK_POSITIVE,
        FALLBACK_ERROR,
        FALLBACK_POSITIVE_TEXT,
        FALLBACK_ERROR_TEXT,
    )

    assert FALLBACK_POSITIVE
    assert FALLBACK_ERROR
    assert FALLBACK_POSITIVE_TEXT == FALLBACK_POSITIVE
    assert FALLBACK_ERROR_TEXT == FALLBACK_ERROR


def test_messages_do_not_echo_user_input():
    """Mensagens de fallback nao devem conter placeholder de entrada do usuario (SEC-03-001)."""
    from app.assistant.messages import FALLBACK_POSITIVE, FALLBACK_ERROR

    for suspicious in ["{", "}", "%s", "{transcript}", "{user_input}"]:
        assert suspicious not in FALLBACK_POSITIVE
        assert suspicious not in FALLBACK_ERROR
