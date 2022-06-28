#
# Copyright (c) 2016-2018 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

import argparse
import heapq
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser(description='Display inception v3 classification results.')
    parser.add_argument('-i', '--label_list', default='labels.txt',
                        help='File containing input list used to generate output_dir.')
    parser.add_argument('-o', '--output_dir', default='output',
                        help='Output directory containing Result_X/prob.raw files matching label_list.')
    parser.add_argument('-c', '--classes', default='classes.txt',
                        help='Path to ilsvrc_2012_labels.txt')
    parser.add_argument('-v', '--verbose_results',
                        help='Display top 5 classifications', action='store_true')
    args = parser.parse_args()

    label_list = os.path.abspath(args.label_list)
    output_dir = os.path.abspath(args.output_dir)
    classes = os.path.abspath(args.classes)
    display_top5 = args.verbose_results

    if not os.path.isfile(label_list):
        raise RuntimeError('label_list %s does not exist' % label_list)
    if not os.path.isdir(output_dir):
        raise RuntimeError('output_dir %s does not exist' % output_dir)
    if not os.path.isfile(classes):
        raise RuntimeError('classes %s does not exist' % classes)
    with open(classes, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    if len(class_names) != 2:
        raise RuntimeError('Invalid classes: need 2 categories')
    with open(label_list, 'r') as f:
        input_files = [line.strip() for line in f.readlines()]

    if len(input_files) <= 0:
        print('No files listed in input_files')
    else:
        print('Classification results')
        max_filename_len = max([len(file) for file in input_files])

        right_class = 0
        for idx, val in enumerate(input_files):
            img_name, label = val.split(' ')
            cur_results_dir = 'Result_' + str(idx)
            cur_results_file = os.path.join(output_dir, cur_results_dir, 'probs.raw')
            if not os.path.isfile(cur_results_file):
                raise RuntimeError('missing results file: ' + cur_results_file)

            float_array = np.fromfile(cur_results_file, dtype=np.float32)
            print(float_array)
            if len(float_array) != 2:
                raise RuntimeError(str(len(float_array)) + ' outputs in ' + cur_results_file)

            if not display_top5:
                max_prob = max(float_array)
                max_prob_index = np.where(float_array == max_prob)[0][0]
                max_prob_category = class_names[max_prob_index]

                display_text = '%s %f %s %s' % (
                img_name.ljust(max_filename_len), max_prob, str(max_prob_index).rjust(3), max_prob_category)
                print(display_text)
            else:
                top5_prob = heapq.nlargest(5, range(len(float_array)), float_array.take)
                for i, idx in enumerate(top5_prob):
                    prob = float_array[idx]
                    prob_category = class_names[idx]
                    display_text = '%s %f %s %s' % (
                        img_name.ljust(max_filename_len), prob, str(idx).rjust(3), prob_category)
                    print(display_text)

            # compute accuracy
            if np.argmax(float_array) == int(label):
                right_class += 1

        acc = np.round((right_class / len(input_files)), 5)
        print('accuracy : {}'.format(acc))


def show_res():
    for i in range(4):
        cur_results_dir = 'Result_' + str(i)
        cur_results_file = os.path.join('output', cur_results_dir, 'probs.raw')
        if not os.path.isfile(cur_results_file):
            raise RuntimeError('missing results file: ' + cur_results_file)

        float_array = np.fromfile(cur_results_file, dtype=np.float32)
        print(float_array)


if __name__ == '__main__':
    show_res()
