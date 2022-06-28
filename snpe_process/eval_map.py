import os
import numpy as np


def main():
    # with open('label_list.txt', 'r') as f:
    #     input_files = [line.strip() for line in f.readlines()]
    result_dir = 'output3'
    dirs = os.listdir(result_dir)
    for val in dirs:
        # img_name, label = val.split(' ')
        cur_results_file = os.path.join(result_dir, val, '621.raw')
        if not os.path.isfile(cur_results_file):
            raise RuntimeError('missing results file: ' + cur_results_file)

        float_array = np.fromfile(cur_results_file, dtype=np.float32)
        print(float_array)


if __name__ == '__main__':
    main()
