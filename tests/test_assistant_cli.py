import sys
import threading
import types

from app.assistant.cli import main


def test_text_mode_once_does_not_require_microphone(capsys):
    exit_code = main(["text", "--once", "ola"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Valeria recebeu: ola" in captured.out


def test_run_requires_model_file_without_no_wake_word(capsys):
    exit_code = main(["run"])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "--model-file is required" in captured.err


def test_run_reports_missing_model_file(capsys):
    exit_code = main(["run", "--model-file", "missing-model.zip"])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "model file not found" in captured.err


def test_run_no_wake_word_starts_text_mode_without_model(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _prompt: "exit")

    exit_code = main(["run", "--no-wake-word"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Valeria text mode" in captured.out


def test_run_reports_wake_word_engine_runtime_failure_after_import(tmp_path, monkeypatch, capsys):
    model_file = tmp_path / "valeria_optimized.zip"
    model_file.write_text("fake model")

    fake_exec = types.ModuleType("app.network.exec")

    class FailingWakeWordEngine:
        def __init__(self, _model_file):
            raise RuntimeError("audio backend unavailable")

    fake_exec.WakeWordEngine = FailingWakeWordEngine
    monkeypatch.setitem(sys.modules, "app.network.exec", fake_exec)

    exit_code = main(["run", "--model-file", str(model_file)])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "voice loop could not start" in captured.err
    assert "audio backend unavailable" in captured.err


def test_run_uses_voice_loop_with_mocked_engine(tmp_path, monkeypatch, capsys):
    model_file = tmp_path / "valeria_optimized.zip"
    model_file.write_text("fake model")
    calls = []

    fake_exec = types.ModuleType("app.network.exec")
    fake_voice_loop = types.ModuleType("app.assistant.voice_loop")

    class FakeWakeWordEngine:
        def __init__(self, model_file_path):
            calls.append(("engine_init", model_file_path))

    class FakeVoiceLoop:
        def __init__(self, config, engine):
            calls.append(("voice_loop_init", config, engine))

        def run(self):
            calls.append(("voice_loop_run", None))

    class FakeEvent:
        def wait(self):
            calls.append(("wait", None))

    fake_exec.WakeWordEngine = FakeWakeWordEngine
    fake_voice_loop.VoiceLoop = FakeVoiceLoop
    monkeypatch.setitem(sys.modules, "app.network.exec", fake_exec)
    monkeypatch.setitem(sys.modules, "app.assistant.voice_loop", fake_voice_loop)
    monkeypatch.setattr(threading, "Event", FakeEvent)

    exit_code = main(["run", "--model-file", str(model_file)])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls[0] == ("engine_init", str(model_file))
    assert calls[1][0] == "voice_loop_init"
    assert isinstance(calls[1][2], FakeWakeWordEngine)
    assert calls[2] == ("voice_loop_run", None)
    assert calls[3] == ("wait", None)
    assert "Starting Valeria wake-word mode" in captured.out


def test_run_wake_word_happy_path_does_not_block(tmp_path, monkeypatch, capsys):
    model_file = tmp_path / "valeria_optimized.zip"
    model_file.write_text("fake model")
    calls = []

    fake_exec = types.ModuleType("app.network.exec")
    fake_voice_loop = types.ModuleType("app.assistant.voice_loop")

    class FakeWakeWordEngine:
        def __init__(self, model_file_path):
            calls.append(("init", model_file_path))

    class FakeVoiceLoop:
        def __init__(self, config, engine):
            calls.append(("voice_loop_init", config, engine))

        def run(self):
            calls.append(("voice_loop_run", None))

    class FakeEvent:
        def wait(self):
            calls.append(("wait", None))

    fake_exec.WakeWordEngine = FakeWakeWordEngine
    fake_voice_loop.VoiceLoop = FakeVoiceLoop
    monkeypatch.setitem(sys.modules, "app.network.exec", fake_exec)
    monkeypatch.setitem(sys.modules, "app.assistant.voice_loop", fake_voice_loop)
    monkeypatch.setattr(threading, "Event", FakeEvent)

    exit_code = main(["run", "--model-file", str(model_file)])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls[0] == ("init", str(model_file))
    assert calls[1][0] == "voice_loop_init"
    assert isinstance(calls[1][2], FakeWakeWordEngine)
    assert calls[2] == ("voice_loop_run", None)
    assert calls[3] == ("wait", None)
    assert "Starting Valeria wake-word mode" in captured.out


def test_invalid_config_returns_controlled_cli_error(monkeypatch, capsys):
    monkeypatch.setenv("VALERIA_AUDIO_OUTPUT_DIR", "..")

    exit_code = main(["text", "--once", "ola"])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Configuration error:" in captured.err
    assert "VALERIA_AUDIO_OUTPUT_DIR" in captured.err
