import onnx
import onnxruntime as ort
import numpy as np
import cv2
import os


def preprocess(img_file):
    img = cv2.imread(img_file)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_data = np.transpose(img, (2, 0, 1))

    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(input_data.shape).astype('float32')
    for i in range(input_data.shape[0]):
        # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i, :, :] = (input_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]

    norm_img_data = norm_img_data.reshape([1, 3, 224, 224])
    return norm_img_data


if __name__ == '__main__':
    img_dir = 'D:/program file/projects2/mmclassification/data/2_class/train/airplane/'
    model_path = 'D:/program file/projects2/mmclassification/my_work/mbnet.onnx'
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        input_data = preprocess(img_path)
        sess = ort.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name
        result = sess.run([], {input_name: input_data})
        result = np.reshape(result, [1, -1])
        print(result)
        index = np.argmax(result)
        print("max index:", index)
