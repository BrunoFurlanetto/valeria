import argparse
import sys
from pathlib import Path

from app.assistant.config import AssistantConfig
from app.assistant.core import Assistant


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m app.assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    text_parser = subparsers.add_parser("text", help="Run Valeria without microphone access")
    text_parser.add_argument("--once", help="Process a single text command and exit")

    run_parser = subparsers.add_parser("run", help="Run Valeria with the wake-word engine")
    run_parser.add_argument("--model-file", help="Path to the optimized TorchScript wake-word model")
    run_parser.add_argument(
        "--no-wake-word",
        action="store_true",
        help="Skip wake-word startup and run the text loop",
    )

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    config = AssistantConfig.from_env()

    if args.command == "text":
        return run_text_mode(config, args.once)

    if args.command == "run":
        if args.no_wake_word:
            return run_text_mode(config, None)
        return run_wake_word_mode(args.model_file)

    parser.print_help()
    return 2


def run_text_mode(config, once):
    assistant = Assistant(config)

    if once is not None:
        print(assistant.respond(once))
        return 0

    print("Valeria text mode. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if user_input.lower() in {"exit", "quit", "sair"}:
            return 0

        if not user_input:
            continue

        print(assistant.respond(user_input))


def run_wake_word_mode(model_file):
    if not model_file:
        print("Configuration error: --model-file is required unless --no-wake-word is used.", file=sys.stderr)
        return 2

    model_path = Path(model_file)
    if not model_path.exists():
        print(f"Configuration error: model file not found: {model_path}", file=sys.stderr)
        return 2

    try:
        from app.network.exec import WakeWordEngine
    except ImportError as exc:
        print(f"Runtime error: wake-word dependencies are unavailable: {exc}", file=sys.stderr)
        return 1

    engine = WakeWordEngine(str(model_path))

    def report_prediction(prediction):
        if prediction == 1:
            print("Wake word detected.")

    print(f"Starting Valeria wake-word mode with model: {model_path}")
    engine.run(report_prediction)

    try:
        import threading

        threading.Event().wait()
    except KeyboardInterrupt:
        print()
        return 0

    return 0

