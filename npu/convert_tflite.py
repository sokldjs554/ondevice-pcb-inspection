"""ONNX 모델을 NPU 컴파일용 INT8 TFLite로 변환하는 스크립트

NPU 컴파일러(ARM Ethos-U Vela)는 INT8로 완전 양자화된 TFLite를 입력으로 받는다.
그래서 배포용으로 만들어둔 ONNX를 TFLite로 옮기고, 학습 패치로 캘리브레이션해서
입력·출력까지 전부 INT8인 모델을 만든다. (ONNX Runtime 양자화 때와 같은 데이터 사용)

주의: ONNX는 NCHW, TFLite는 NHWC라 채널 순서가 바뀐다. onnx2tf가 이 변환을 처리한다.

사용 예시:
    python npu/convert_tflite.py --model simple_cnn
    python npu/convert_tflite.py --model mobilenet_v2
"""
import os
import glob
import random
import shutil
import tempfile
import argparse
import subprocess

import numpy as np
from PIL import Image


def load_calibration(data_dir, channels, num_samples, seed=42):
    """학습 패치를 NHWC 형식으로 읽어서 캘리브레이션 데이터로 만든다"""
    files = glob.glob(os.path.join(data_dir, 'train', '*', '*.png'))
    if not files:
        raise SystemExit(f'캘리브레이션 패치를 찾을 수 없습니다: {data_dir}\n'
                         'data/prepare_patches.py를 먼저 실행하세요.')
    random.Random(seed).shuffle(files)

    samples = []
    for path in files[:num_samples]:
        arr = np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # 학습 때와 같은 정규화
        arr = arr[..., None]  # (H, W, 1)
        if channels == 3:
            arr = np.repeat(arr, 3, axis=2)
        samples.append(arr)
    return np.stack(samples).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='simple_cnn', choices=['simple_cnn', 'mobilenet_v2'])
    parser.add_argument('--onnx', default=None)
    parser.add_argument('--data-dir', default='data/patches')
    parser.add_argument('--out-dir', default='npu/models')
    parser.add_argument('--num-samples', type=int, default=200)
    args = parser.parse_args()

    onnx_path = args.onnx or f'models/{args.model}_fp32.onnx'
    channels = 1 if args.model == 'simple_cnn' else 3
    os.makedirs(args.out_dir, exist_ok=True)
    saved_model_dir = os.path.join(args.out_dir, f'{args.model}_saved_model')

    # 1. ONNX -> TensorFlow SavedModel (onnx2tf가 NCHW -> NHWC 변환까지 처리)
    # onnx2tf는 입력 ONNX 파일을 단순화하면서 덮어쓰기 때문에, 복사본을 만들어서 변환한다.
    print(f'[1/2] ONNX -> SavedModel 변환: {onnx_path}')
    with tempfile.TemporaryDirectory() as tmp:
        tmp_onnx = os.path.join(tmp, os.path.basename(onnx_path))
        shutil.copy2(onnx_path, tmp_onnx)
        subprocess.run(['onnx2tf', '-i', tmp_onnx, '-o', saved_model_dir, '-osd'],
                       check=True, stdout=subprocess.DEVNULL)

    # 2. SavedModel -> INT8 TFLite (캘리브레이션 기반 완전 양자화)
    import tensorflow as tf  # onnx2tf 실행 후에 import (로그 정리 목적)

    cal = load_calibration(args.data_dir, channels, args.num_samples)
    print(f'[2/2] INT8 양자화 (캘리브레이션 {len(cal)}장, 입력 {cal.shape[1:]})')

    def representative_dataset():
        for i in range(len(cal)):
            yield [cal[i:i + 1]]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    # NPU는 정수 연산만 지원하므로 입력·출력까지 INT8로 강제한다
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out_path = os.path.join(args.out_dir, f'{args.model}_int8.tflite')
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    print(f'저장: {out_path} ({len(tflite_model) / 1024:.1f} KB)')


if __name__ == '__main__':
    main()
