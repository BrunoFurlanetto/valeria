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

