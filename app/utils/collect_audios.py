import argparse
import os
import wave
from time import sleep

import pyaudio


class Listen:
    def __init__(self, args):
        self.chunk = 1024
        self.FORMAT = pyaudio.paInt16
        self.channels = 1
        self.rate = args.rate
        self.time_recording = args.seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.channels,
            rate=self.rate,
            input=True,
            output=True,
            frames_per_buffer=self.chunk
        )

    def save_audio(self, file_name, frames):
        print(f'Saving files to {file_name}')
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        wf = wave.open(file_name, "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.rate)
        wf.writeframes(b"".join(frames))

        wf.close()


def interactive(args):
    index = 0

    try:
        while True:
            listening = Listen(args)
            frames = []
            print('Start recording....')
            input(f'Press enter to continue. The recording will be of {args.seconds} seconds, or CTRL + C to stop ')
            sleep(0.2)

            for i in range(int((listening.rate / listening.chunk) * listening.time_recording)):
                data = listening.stream.read(listening.chunk, exception_on_overflow=False)
                frames.append(data)

            save_path = os.path.join(args.interactive_save, f'{index}.wav')
            listening.save_audio(save_path, frames)
            index += 1
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
    except Exception as e:
        print(str(e))


def main(args):
    listening = Listen(args)
    frames = []
    print('Recording...')

    try:
        while True:
            if listening.time_recording is None:
                print('Recording indefinitely... CTRL + C to stop', end="\r")
                data = listening.stream.read(listening.chunk)
                frames.append(data)
            else:
                for i in range(int((listening.rate / listening.chunk) * listening.time_recording)):
                    data = listening.stream.read(listening.chunk)
                    frames.append(data)
                raise Exception('End of recording')
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
    except Exception as e:
        print(str(e))

    print('Stopping recording..')
    listening.save_audio(args.save_path, frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    Script to collect data for wake word training.

     To record the ambiance, play the sound set seconds to None. This goes
     record indefinitely until ctrl+c.

     To record for a certain period of time, set seconds to whatever you want.

     To record interactively (usually for recording your own wake-up words N times)
     use --interactive mode.
    ''')
    parser.add_argument(
        '--rate',
        type=int,
        default=8000,
        help='The recording frequency in Hz'
    )
    parser.add_argument(
        '--seconds',
        type=int,
        default=None,
        help='If set to Name then it will record forever until keyboard interrupt'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        required=False,
        help='Full path to save the file. ex: /to/path/sound.wav'
    )
    parser.add_argument(
        '--interactive_save',
        type=str,
        default=None,
        required=False,
        help='Directory to save all 2 second interactive samples. ex: /to/path/'
    )
    parser.add_argument(
        '--interactive',
        default=False,
        action='store_true',
        required=False,
        help='Interactive mode'
    )

    args = parser.parse_args()

    if args.interactive_save:
        if args.interactive_save is None:
            raise Exception('Need to define --interactive_save')

        interactive(args)
    else:
        if args.save_path is None:
            raise Exception('Need to define --save_path')

        main(args)
