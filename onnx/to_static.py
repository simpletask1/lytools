import cv2
import numpy as np
import torch
import onnx
# import onnxmltools
import onnxruntime as ort


def to_static(model_path='xxx.onnx', output_path='./staticmodel.onnx', do_simplify=False):
    model_onnx = onnx.load(model_path)
    print(onnx.helper.printable_graph(model_onnx.graph))

    # 将动态输入的shape修改为静态
    d = model_onnx.graph.input[0].type.tensor_type.shape.dim
    d[0].dim_value = 1
    d[1].dim_value = 3
    d[2].dim_value = 224
    d[3].dim_value = 224
    print(d)

    onnx.save(model_onnx, output_path)

    # 重新加载
    model_static = onnx.load(output_path)
    session = ort.InferenceSession(output_path)
    print('input info'.center(100, '='))
    for ii in session.get_inputs():
        print('Input: ', ii)

    session = ort.InferenceSession(output_path)
    print('output info'.center(100, '='))
    for oo in session.get_outputs():
        print('Input: ', oo)

    if do_simplify:
        from onnxsim import simplify
        sim_model, check = simplify(model_static, check_n=3)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(model_static, output_path)


if __name__ == '__main__':
    src = 'D:/program file/projects2/mmclassification/my_work/retinanet-9.onnx'
    # to_static(model_path=src)
    model_onnx = onnx.load(src)
    print(onnx.helper.printable_graph(model_onnx.graph))

    # model = torch.load(src)
    # for i, key_name in enumerate(model['state_dict'].keys()):
    #     print('【{}】: {}'.format(i, key_name))




