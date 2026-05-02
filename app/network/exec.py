import pyaudio
import threading
import time
import argparse
import wave
from torch.nn.functional import normalize
import torchaudio
import torch
import numpy as np
from dataset import get_featurizer
from threading import Event
import os


class Listener:
    def __init__(self, sample_rate=8000, record_seconds=2):
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
            frames_per_buffer=self.chunk
        )

    def listen(self, queue):
        while True:
            data = self.stream.read(self.chunk)
            queue.append(data)
            time.sleep(0.01)

    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("\nWake Word Engine agora está ouvindo... \n")


class WakeWordEngine:
    def __init__(self, model_file):
        self.listener = Listener(sample_rate=8000, record_seconds=2)
        self.model = torch.jit.load(model_file)
        self.model.eval().to('cpu')  # run on cpu
        self.featurizer = get_featurizer(sample_rate=8000)
        self.audio_q = list()
        self.b = 0

    def save(self, waveforms, fname="temporario_ww"):
        wf = wave.open(fname, "wb")
        # set the channels
        wf.setnchannels(1)
        # set the sample format
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        # set the sample rate
        wf.setframerate(8000)
        # write the frames as bytes
        wf.writeframes(b"".join(waveforms))
        # close the file
        wf.close()
        return fname

    def prever(self, audio):
        with torch.no_grad():
            fname = self.save(audio)

            waveform, _ = torchaudio.backend.sox_io_backend.load(fname, normalize=False)  # não normalize

            mfcc = self.featurizer(waveform).transpose(1, 2).transpose(0, 1)

            out = self.model(mfcc)
            pred = torch.round(torch.sigmoid(out))
            a = int(np.array(pred))

            return pred.item()

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
        thread = threading.Thread(target=self.inference_loop,
                                  args=(action,), daemon=True)
        thread.start()


class Demonstrar:
    """Esta ação de demonstração dirá apenas citações de XXXX aleatoriamente
        args: sensibilidade. quanto menor o número, mais sensível será
        wakeword é para ativação."""

    def __init__(self, sensibilidade=10):
        # importar coisas aqui para evitar que executar.py
        # importar módulos desnecessários durante o uso de produção
        import os
        import subprocess
        import random
        from os.path import join, realpath

        self.random = random
        self.subprocess = subprocess
        self.detect_in_row = 0

        self.sensibilidade = sensibilidade
        folder = realpath(join(realpath(__file__), '..', '..', '..', 'respostas', 'audios'))
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
            self.subprocess.check_output(['play', '-v', '.1', filename])
        except Exception as e:
            print(str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demonstrando o wakeword")
    parser.add_argument('--model_file', type=str, default=None, required=True,
                        help='arquivo otimizado para carregar. use modelo_optimizado.py')
    parser.add_argument('--sensibilidade', type=int, default=10, required=False,
                        help='menor valor é mais sensível a ativações')

    args = parser.parse_args()
    wakeword_engine = WakeWordEngine(args.model_file)
    action = Demonstrar(args.sensibilidade)

    print("""\n*** Certifique-se de ter o sox instalado em seu sistema para que o demo funcione !!!
    Se você não quiser usar sox, altere a função play na classe Demonstrar
    no módulo engine.py para algo que funcione com o seu sistema.\n
    """)
    # action = lambda x: print(x)
    wakeword_engine.run(action)
    threading.Event().wait()
