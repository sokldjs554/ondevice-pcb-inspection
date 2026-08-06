"""ONNX 모델을 INT8로 정적 양자화하는 스크립트

NPU 같은 엣지 가속기는 대부분 INT8 연산 기반이라, 배포 전에 양자화가 필요하다.
정적 양자화는 캘리브레이션 데이터로 각 레이어의 활성값 범위를 미리 측정해서
스케일을 정하는 방식이라, 실제 NPU 컴파일 과정과 흐름이 비슷하다.

사용 예시:
    python src/quantize_onnx.py --model simple_cnn
"""
import os
import glob
import random
import argparse

import numpy as np
from PIL import Image
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

from models import input_channels


class PatchCalibrationReader(CalibrationDataReader):
    """학습 패치 일부를 캘리브레이션 데이터로 공급하는 리더"""

    def __init__(self, data_dir, channels, num_samples=300, seed=42):
        files = glob.glob(os.path.join(data_dir, 'train', '*', '*.png'))
        random.Random(seed).shuffle(files)
        self.files = files[:num_samples]
        self.channels = channels
        self.idx = 0

    def preprocess(self, path):
        img = Image.open(path).convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # 학습 때와 같은 정규화
        arr = arr[None, :, :]  # (1, H, W)
        if self.channels == 3:
            arr = np.repeat(arr, 3, axis=0)
        return arr[None, :, :, :]  # (1, C, H, W)

    def get_next(self):
        if self.idx >= len(self.files):
            return None
        data = {'input': self.preprocess(self.files[self.idx])}
        self.idx += 1
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='simple_cnn', choices=['simple_cnn', 'mobilenet_v2'])
    parser.add_argument('--onnx', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--data-dir', default='data/patches')
    parser.add_argument('--num-samples', type=int, default=300)
    args = parser.parse_args()

    onnx_path = args.onnx or f'models/{args.model}_fp32.onnx'
    out_path = args.out or f'models/{args.model}_int8.onnx'

    reader = PatchCalibrationReader(args.data_dir, input_channels(args.model), args.num_samples)
    print(f'캘리브레이션 샘플 {len(reader.files)}장으로 양자화 중...')

    quantize_static(
        onnx_path, out_path, reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    before = os.path.getsize(onnx_path) / 1024 / 1024
    after = os.path.getsize(out_path) / 1024 / 1024
    print(f'FP32: {before:.2f} MB -> INT8: {after:.2f} MB ({before / after:.1f}배 감소)')
    print(f'저장: {out_path}')


if __name__ == '__main__':
    main()
