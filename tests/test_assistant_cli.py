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
    assert "wake-word engine could not start" in captured.err
    assert "audio backend unavailable" in captured.err


def test_run_wake_word_happy_path_does_not_block(tmp_path, monkeypatch, capsys):
    model_file = tmp_path / "valeria_optimized.zip"
    model_file.write_text("fake model")
    calls = []

    fake_exec = types.ModuleType("app.network.exec")

    class FakeWakeWordEngine:
        def __init__(self, model_file_path):
            calls.append(("init", model_file_path))

        def run(self, callback):
            calls.append(("run", callback))
            callback(1)

    class FakeEvent:
        def wait(self):
            calls.append(("wait", None))

    fake_exec.WakeWordEngine = FakeWakeWordEngine
    monkeypatch.setitem(sys.modules, "app.network.exec", fake_exec)
    monkeypatch.setattr(threading, "Event", FakeEvent)

    exit_code = main(["run", "--model-file", str(model_file)])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls[0] == ("init", str(model_file))
    assert calls[1][0] == "run"
    assert callable(calls[1][1])
    assert calls[2] == ("wait", None)
    assert "Starting Valeria wake-word mode" in captured.out
    assert "Wake word detected." in captured.out
