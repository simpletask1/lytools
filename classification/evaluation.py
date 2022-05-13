import numpy as np
import cv2
import torch

import argparse


def accuracy(dets, labels):
    """

    :param dets: n_samples * n_classes (numpy arrays)
    :param labels: n_samples * n_classes (numpy arrays)
    :return: accuracy
    """
    dets_max_idx = np.argmax(dets, axis=1)
    labels_max_idx = np.argmax(labels, axis=1)
    rightness = (dets_max_idx == labels_max_idx)

    acc = list(rightness).count(1) / len(rightness)
    return np.round(acc, 5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute acc')
    parser.add_argument('--dets', help='classification results')
    parser.add_argument('--labels', help='labels')

    args = parser.parse_args()
    print(accuracy(args.dets, args.labels))
