import os
import cv2


def create_labels(src):
    with open('labels.txt', 'w') as f:
        for img in os.listdir(src):
            name = os.path.join(src, img)
            class_id = '0' if img[0] == 'a' else '1'
            f.write(name+' ' + class_id + '\n')


if __name__ == '__main__':
    src = 'test4'
    dst = 'resized'
    for img_name in os.listdir(src):
        name = os.path.join(src, img_name)
        img = cv2.imread(name)
        img_299 = cv2.resize(img, (299, 299), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(dst, img_name), img_299)

