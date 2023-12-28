import os
import argparse
import shutil


def main(args):
    dest_dir = os.path.join(args.copy_destination)
    os.makedirs(dest_dir, exist_ok=True)

    for file in os.listdir(args.wake_words_dir):
        if file.endswith(".wav") or file.endswith(".mp3"):
            src_file = os.path.join(args.wake_words_dir, file)

            for i in range(args.number_copies):
                new_dst_file = os.path.join(dest_dir, f"{i}_{file}")
                shutil.copy2(src_file, new_dst_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
            Utility script to replicate wake word clips multiple times
        """
    )
    parser.add_argument(
        '--wake_words_dir',
        type=str,
        default=None,
        required=True,
        help='Clip directory with wake words'
    )
    parser.add_argument(
        '--copy_destination',
        type=str,
        default=None,
        required=True,
        help='Directory of wake words copy destinations'
    )
    parser.add_argument(
        '--number_copies',
        type=int,
        default=100,
        required=False,
        help='The number of copies you want'
    )

    args = parser.parse_args()

    main(args)
