# Valeria - Virtual Assistant

## Description

Valeria is a virtual assistant developed to assist in day-to-day tasks. The current V1 work starts from a local wake-word engine and adds a CLI foundation for text and voice flows.

## Current V1 status

- **CLI foundation:** Run Valeria in text mode or start the wake-word engine from `python -m app.assistant`.
- **Wake-word runtime:** Load an optimized TorchScript model and listen for the local wake word when audio dependencies and a microphone are available.
- **Voice V1 flow:** After wake-word detection, Valeria records a short local command, sends that audio to Gemini for transcription, generates a local assistant response, sends the response text to Gemini TTS, and plays the returned audio locally.
- **Assistant core placeholder:** Text responses currently confirm receipt of the command. LLM-backed responses and tool integrations are planned for later V1 steps.

## Planned capabilities

- LLM-backed contextual responses.
- Service/tool integrations for useful day-to-day tasks.
- LLM-backed voice responses after the placeholder assistant core is replaced.

## Voice mode privacy

Voice mode uses the local microphone and speakers, but it is not fully local processing. Command audio recorded after the wake word is sent to Gemini for transcription, and assistant response text is sent to Gemini for text-to-speech. Temporary audio files are created under `tmp/valeria-audio` for each turn and removed after the turn finishes.

## Model artifacts

The repository intentionally keeps two small known-good wake-word artifacts under `app/network/save_model/`:

- `valeria_wake.pt`: baseline checkpoint for re-exporting the wake-word model.
- `valeria_optimized.zip`: baseline TorchScript model used by the local wake-word runtime.

New training outputs and temporary runtime files are ignored by default. In particular, `app/network/temporario_ww` is treated as runtime temporary data and is not versioned.

## Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/BrunoFurlanetto/valeria.git
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

```bash
venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r app/requirements.txt
```

5. Configure environment variables:

```bash
copy .env.example .env
```

6. Run text mode without microphone access:

```bash
python -m app.assistant text
```

7. Run wake-word mode with an optimized model:

```bash
python -m app.assistant run --model-file app/network/save_model/valeria_optimized.zip
```

## Development

Run the basic test suite:

```bash
python -m pytest -q
```

## License

This project is licensed under the [MIT License](LICENSE).
