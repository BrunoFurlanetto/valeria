import argparse
import threading
import time
import wave
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torchaudio

try:
    from app.network.dataset import get_featurizer
except ModuleNotFoundError:
    from dataset import get_featurizer


WAKE_WORD_TMP_DIR = Path("tmp") / "valeria-audio" / "wake-word"


class Listener:
    def __init__(self, sample_rate=8000, record_seconds=2):
        import pyaudio

        self.pyaudio = pyaudio
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            output=True,
            frames_per_buffer=self.chunk,
        )

    def listen(self, queue):
        while True:
            data = self.stream.read(self.chunk)
            queue.append(data)
            time.sleep(0.01)

    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("\nWake Word Engine agora esta ouvindo... \n")


class WakeWordEngine:
    def __init__(self, model_file):
        self.listener = Listener(sample_rate=8000, record_seconds=2)
        self.model = torch.jit.load(model_file)
        self.model.eval().to("cpu")  # run on cpu
        self.featurizer = get_featurizer(sample_rate=8000)
        self.audio_q = list()
        self.b = 0

    def save(self, waveforms, output_dir=WAKE_WORD_TMP_DIR):
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        fname = directory / f"wake-word-{time.time_ns()}-{uuid4().hex}.wav"

        with wave.open(str(fname), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.listener.p.get_sample_size(self.listener.pyaudio.paInt16))
            wf.setframerate(8000)
            wf.writeframes(b"".join(waveforms))

        return fname

    def prever(self, audio):
        with torch.no_grad():
            fname = None
            try:
                fname = self.save(audio)
                waveform, _ = torchaudio.backend.sox_io_backend.load(str(fname), normalize=False)

                mfcc = self.featurizer(waveform).transpose(1, 2).transpose(0, 1)

                out = self.model(mfcc)
                pred = torch.round(torch.sigmoid(out))
                a = int(np.array(pred))

                return pred.item()
            finally:
                if fname is not None:
                    try:
                        fname.unlink(missing_ok=True)
                    except OSError:
                        pass

    def inference_loop(self, action):
        while True:
            if len(self.audio_q) > 15:  # remover parte do stream
                diff = len(self.audio_q) - 15
                for _ in range(diff):
                    self.audio_q.pop(0)
                action(self.prever(self.audio_q))
            elif len(self.audio_q) == 15:
                action(self.prever(self.audio_q))
            time.sleep(0.05)

    def run(self, action):
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop, args=(action,), daemon=True)
        thread.start()


class Demonstrar:
    """Acao de demonstracao para tocar audios aleatorios ao detectar wake word.

    Args:
        sensibilidade: quanto menor o numero, mais sensivel sera a ativacao.
    """

    def __init__(self, sensibilidade=10):
        import os
        import random
        import subprocess
        from os.path import join, realpath

        self.random = random
        self.subprocess = subprocess
        self.detect_in_row = 0

        self.sensibilidade = sensibilidade
        folder = realpath(join(realpath(__file__), "..", "..", "..", "respostas", "audios"))
        self.arnold_mp3 = [
            os.path.join(folder, x)
            for x in os.listdir(folder)
            if ".wav" in x
        ]

    def __call__(self, prediction):
        if prediction == 1:
            self.detect_in_row += 1
            if self.detect_in_row == self.sensibilidade:
                self.play()
                self.detect_in_row = 0
        else:
            self.detect_in_row = 0

    def play(self):
        filename = self.random.choice(self.arnold_mp3)
        try:
            print("playing", filename)
            self.subprocess.check_output(["play", "-v", ".1", filename])
        except Exception as e:
            print(str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demonstrando o wakeword")
    parser.add_argument(
        "--model_file",
        type=str,
        default=None,
        required=True,
        help="arquivo otimizado para carregar. use modelo_optimizado.py",
    )
    parser.add_argument(
        "--sensibilidade",
        type=int,
        default=10,
        required=False,
        help="menor valor e mais sensivel a ativacoes",
    )

    args = parser.parse_args()
    wakeword_engine = WakeWordEngine(args.model_file)
    action = Demonstrar(args.sensibilidade)

    print("""\n*** Certifique-se de ter o sox instalado em seu sistema para que o demo funcione !!!
    Se voce nao quiser usar sox, altere a funcao play na classe Demonstrar
    no modulo engine.py para algo que funcione com o seu sistema.\n
    """)
    # action = lambda x: print(x)
    wakeword_engine.run(action)
    threading.Event().wait()
