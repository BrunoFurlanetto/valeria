import os
import argparse
import json
import random


def main(args):
    zeros = os.listdir(args.zero_label_dir)
    ones = os.listdir(args.one_label_dir)
    percent = args.percent / 100
    data = []

    for z in zeros:
        data.append({
            "key": os.path.join(args.zero_label_dir, z),
            "label": 0
        })

    for o in ones:
        data.append({
            "key": os.path.join(args.one_label_dir, o),
            "label": 1
        })

    random.shuffle(data)

    train_data = data[:int(len(data) * (1 - percent))]
    test_data = data[int(len(data) * (1 - percent)):]

    with open(os.path.join(args.save_json_path, 'train.json'), 'w') as f:
        for item in train_data:
            line = json.dumps(item)
            f.write(line + "\n")

    with open(os.path.join(args.save_json_path, 'test.json'), 'w') as f:
        for item in test_data:
            line = json.dumps(item)
            f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Utility script to create training JSON file for wake word recognition.
     There should be two directories, one with all 0 labels and one with all 1 labels.
    """
                                     )
    parser.add_argument(
        '--zero_label_dir',
        type=str,
        default=None,
        required=True,
        help='Directory with labels 0'
    )
    parser.add_argument(
        '--one_label_dir',
        type=str,
        default=None,
        required=True,
        help='Directory with label 1'
    )
    parser.add_argument(
        '--save_json_path',
        type=str,
        default=None,
        required=True,
        help='Path to save the JSON file'
    )
    parser.add_argument(
        '--percent',
        type=int,
        default=10,
        required=False,
        help='Percentage of clips placed in test.json instead of train.json'
    )
    args = parser.parse_args()

    main(args)
