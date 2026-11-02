"""Gerador deterministico de arquivos de audio estaticos da Valeria.

Uso:
    python scripts/generate_static_audio.py

Gera sempre:
- earcon_positive.wav  — tom agudo curto (440 Hz, 0.3 s)
- earcon_error.wav     — dois pulsos graves (220 Hz, 2x 0.2 s com pausa)

Gera se ELEVENLABS_API_KEY estiver presente no .env:
- fallback_positive.wav — frase de sucesso via ElevenLabs TTS
- fallback_error.wav    — frase de erro via ElevenLabs TTS

Todos os WAVs: 24 kHz, mono, 16-bit PCM.
Saida: app/assistant/assets/sounds/
"""

import math
import os
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit
OUTPUT_DIR = Path(__file__).parent.parent / "app" / "assistant" / "assets" / "sounds"


def _sine_frames(frequency_hz: float, duration_s: float, amplitude: float = 0.6) -> bytes:
    """Gera frames PCM 16-bit de uma onda senoidal pura."""
    num_samples = int(SAMPLE_RATE * duration_s)
    frames = bytearray()
    for i in range(num_samples):
        sample = amplitude * math.sin(2 * math.pi * frequency_hz * i / SAMPLE_RATE)
        packed = struct.pack("<h", int(sample * 32767))
        frames.extend(packed)
    return bytes(frames)


def _silence_frames(duration_s: float) -> bytes:
    """Gera frames de silencio PCM 16-bit."""
    return b"\x00\x00" * int(SAMPLE_RATE * duration_s)


def _write_wav(path: Path, frames: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(frames)
    print(f"  escrito: {path}")


def generate_earcons(output_dir: Path) -> None:
    print("Gerando earcons...")

    # earcon positivo: tom agudo 880 Hz por 0.25 s com fade-in/fade-out curto
    positive_frames = _sine_frames(880.0, 0.08, amplitude=0.3)
    positive_frames += _sine_frames(1100.0, 0.20, amplitude=0.5)
    positive_frames += _silence_frames(0.05)
    _write_wav(output_dir / "earcon_positive.wav", positive_frames)

    # earcon de erro: dois pulsos graves 220 Hz, 0.15 s cada, com pausa de 0.1 s
    error_pulse = _sine_frames(220.0, 0.15, amplitude=0.6)
    error_frames = error_pulse + _silence_frames(0.1) + error_pulse + _silence_frames(0.1)
    _write_wav(output_dir / "earcon_error.wav", error_frames)


def generate_phrases_elevenlabs(output_dir: Path) -> None:
    """Gera WAVs de frases de fallback via ElevenLabs (one-shot)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()

    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")

    if not api_key:
        print("ELEVENLABS_API_KEY ausente — frases estaticas nao geradas via API.")
        print("  Para gerar: defina ELEVENLABS_API_KEY e ELEVENLABS_VOICE_ID no .env e re-execute.")
        return

    if not voice_id:
        print("ELEVENLABS_VOICE_ID ausente — frases estaticas nao geradas via API.")
        return

    try:
        from elevenlabs import ElevenLabs
    except ImportError:
        print("SDK elevenlabs nao instalado — frases estaticas nao geradas.")
        return

    # Importar textos do modulo de mensagens
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.assistant.messages import FALLBACK_POSITIVE, FALLBACK_ERROR

    client = ElevenLabs(api_key=api_key)
    phrases = [
        ("fallback_positive.wav", FALLBACK_POSITIVE),
        ("fallback_error.wav", FALLBACK_ERROR),
    ]

    print("Gerando frases de fallback via ElevenLabs...")
    for filename, text in phrases:
        try:
            audio_bytes = b"".join(
                client.text_to_speech.convert(
                    text=text,
                    voice_id=voice_id,
                    model_id="eleven_flash_v2_5",
                    output_format="wav_24000",
                )
            )
            path = output_dir / filename
            path.write_bytes(audio_bytes)
            print(f"  escrito: {path} ({len(audio_bytes)} bytes)")
        except Exception as exc:
            print(f"  falha ao gerar {filename}: {exc}")


def main() -> None:
    print(f"Diretorio de saida: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_earcons(OUTPUT_DIR)
    generate_phrases_elevenlabs(OUTPUT_DIR)
    print("Concluido.")


if __name__ == "__main__":
    main()
