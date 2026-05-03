# Valeria - Virtual Assistant

## Description

Valeria is a virtual assistant developed to assist in day-to-day tasks. The current V1 work starts from a local wake-word engine and adds a CLI foundation for text and voice flows.

## Features

- **Voice Recognition:** Valeria can understand and process voice commands to perform specific actions.
- **Smart Responses:** Uses LLM integration for contextual responses.
- **Integration with Services:** Connects to APIs to provide updated information, such as weather forecasts, news, among others.

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
