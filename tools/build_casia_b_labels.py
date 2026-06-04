"""Build CASIA-B label pickle from CSV annotations."""

import argparse
import csv
import os
import pickle


def seq_name_from_image_name(image_name):
    seq_name = os.path.normpath(image_name).split(os.sep)[0]
    if seq_name == '.':
        seq_name = os.path.normpath(image_name).split(os.sep)[1]
    return seq_name


def build_labels(csv_path):
    labels_by_seq = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_name = seq_name_from_image_name(row['image_name'])
            if seq_name not in labels_by_seq:
                subject = seq_name.split('-')[0]
                labels_by_seq[seq_name] = int(subject) - 1
    return [labels_by_seq[seq_name] for seq_name in sorted(labels_by_seq)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--split',
        choices=['train', 'valid', 'test', 'train_valid'],
        default='valid',
        help='CASIA-B split to convert into labels')
    args = parser.parse_args()

    csv_path = f'data/casia-b/casia-b_pose_{args.split}.csv'
    pkl_path = f'data/casia-b/casia-b_labels_{args.split}.pkl'

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    labels = build_labels(csv_path)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, 'wb') as f:
        pickle.dump(labels, f)

    print(f'saved {pkl_path}')
    print(f'sequences: {len(labels)}')
    print(f'label range: {min(labels)}..{max(labels)}')


if __name__ == '__main__':
    main()
