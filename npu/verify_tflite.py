"""NPU용 INT8 TFLite 모델의 정확도를 검증하는 스크립트

컴파일이 됐다고 끝이 아니라, 양자화 과정에서 정확도가 유지되는지 확인해야 한다.
ONNX Runtime INT8 모델과 같은 테스트셋으로 평가해서 두 경로의 결과를 비교한다.

TFLite INT8 모델은 입력·출력이 int8이라, 스케일과 제로포인트를 이용해
실수값 <-> int8 변환을 직접 해줘야 한다. (NPU에 올릴 때도 같은 처리가 필요하다)

사용 예시:
    python npu/verify_tflite.py --model simple_cnn
"""
import os
import glob
import argparse

import numpy as np
from PIL import Image

CLASS_NAMES = ['0_normal', '1_open', '2_short', '3_mousebite', '4_spur', '5_copper', '6_pinhole']


def load_test_set(data_dir, channels):
    xs, ys = [], []
    for label, cls in enumerate(CLASS_NAMES):
        for path in sorted(glob.glob(os.path.join(data_dir, 'test', cls, '*.png'))):
            arr = np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            arr = arr[..., None]  # NHWC
            if channels == 3:
                arr = np.repeat(arr, 3, axis=2)
            xs.append(arr)
            ys.append(label)
    return np.stack(xs).astype(np.float32), np.array(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='simple_cnn', choices=['simple_cnn', 'mobilenet_v2'])
    parser.add_argument('--tflite', default=None)
    parser.add_argument('--data-dir', default='data/patches')
    args = parser.parse_args()

    tflite_path = args.tflite or f'npu/models/{args.model}_int8.tflite'
    channels = 1 if args.model == 'simple_cnn' else 3

    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    # int8 양자화 파라미터 (실수 = (int8 - zero_point) * scale)
    in_scale, in_zp = inp['quantization']
    print(f'입력 양자화: scale={in_scale:.6f}, zero_point={in_zp}')

    x, y = load_test_set(args.data_dir, channels)
    print(f'테스트 패치 {len(x)}장으로 검증 중...')

    correct = 0
    for i in range(len(x)):
        q = np.round(x[i] / in_scale + in_zp).clip(-128, 127).astype(np.int8)
        interpreter.set_tensor(inp['index'], q[None, ...])
        interpreter.invoke()
        pred = int(interpreter.get_tensor(out['index'])[0].argmax())
        correct += int(pred == y[i])

    acc = correct / len(y)
    print(f'TFLite INT8 정확도: {acc:.4f} ({correct}/{len(y)})')


if __name__ == '__main__':
    main()
