# Repository Guidelines

## Project Structure & Module Organization
Core code lives under `app/`:
- `app/network/`: wake-word model pipeline (`dataset.py`, `model.py`, `training.py`, `optimized_model.py`, `exec.py`).
- `app/utils/`: audio data preparation scripts (`collect_audios.py`, `mp3_to_wav.py`, `processing_audios.py`, etc.).
- `app/requirements.txt`: pinned Python dependencies.

Repository root includes `README.md`, `Dockerfile`, and `.env` support. Model artifacts currently live in `app/network/save_model/`. Keep generated files (`.pt`, `.zip`, `__pycache__/`) out of commits unless intentionally versioned.

## Build, Test, and Development Commands
Use Python 3.10+ with a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r app/requirements.txt
```

Key scripts:
- `python app/network/training.py --train_data_json <train.json> --test_data_json <test.json> --save_checkpoint_path app/network/save_model --model_name valeria_wake`
Trains and saves the best checkpoint.
- `python app/network/optimized_model.py --model_checkpoint app/network/save_model/valeria_wake.pt --save_path app/network/save_model/valeria_optimized.zip`
Exports a TorchScript-optimized model.
- `python app/network/exec.py --model_file app/network/save_model/valeria_optimized.zip`
Runs wake-word inference/demo loop.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indentation, snake_case for functions/variables/files, PascalCase for classes.
- Keep modules focused (data pipeline in `utils`, model/runtime in `network`).
- Prefer explicit CLI args via `argparse` for reproducible scripts.

## Testing Guidelines
There is no formal automated test suite yet. For now:
- Validate training changes by running `training.py` on a small dataset split.
- Validate inference with `exec.py` using a known good optimized model.
- When adding features, include lightweight tests under a new `tests/` package using `pytest`.

## Commit & Pull Request Guidelines
Current history uses short, imperative commit messages (for example, `Added script to convert .mp3 files to .wav`). Keep that style:
- One logical change per commit.
- Start message with an action verb (`Add`, `Fix`, `Refactor`, `Update`).

PRs should include:
- Clear summary of behavior changes.
- Reproduction/validation steps (commands used).
- Any dataset/model artifact impacts and required environment variables.
