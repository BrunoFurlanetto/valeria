import argparse
import os
from time import sleep

from pydub import AudioSegment
from tqdm import tqdm


def main(args):
    origin = args.path_to_files
    number_of_files = len(os.listdir(origin))
    files_lost = 0

    for n_file, file in enumerate(tqdm(os.listdir(origin), desc="Converting files", unit="file"), start=1):
        if file.endswith('.mp3'):
            try:
                sound = AudioSegment.from_mp3(os.path.join(origin, file))
            except Exception as e:
                files_lost += 1
            else:
                if args.path_destination:
                    dest = os.path.join(args.path_destination, file.split('.')[0] + '.wav')
                else:
                    dest = os.path.join(origin, file.split('.')[0] + '.wav')

                sound.export(dest, format="wav")
            finally:
                if args.delete_original:
                    os.remove(os.path.join(origin, file))


    print('=' * 30)
    print('File conversion completed.')
    print(f'In the process {files_lost} out of {number_of_files} files were lost.')
    print(f'There are total {number_of_files - files_lost} files converted.')
    print('=' * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to convert .mp3 file in .wav')
    parser.add_argument(
        '--path_to_files',
        type=str,
        default=None,
        required=False,
        help='Path to .mp3 files'
    )
    parser.add_argument(
        '--path_destination',
        type=str,
        default=None,
        help='Path to .wav'
    )
    parser.add_argument(
        '--delete_original',
        type=bool,
        default=False
    )
    args = parser.parse_args()

    main(args)
