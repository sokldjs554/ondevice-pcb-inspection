"""ONNX 모델 비교 벤치마크 (정확도 / 지연시간 / 모델 크기)

FP32 모델과 INT8 양자화 모델을 같은 조건에서 비교한다.
라즈베리파이에서도 같은 스크립트를 그대로 실행할 수 있다.

사용 예시:
    python src/benchmark.py --models models/simple_cnn_fp32.onnx models/simple_cnn_int8.onnx
"""
import os
import glob
import json
import time
import argparse

import numpy as np
from PIL import Image
import onnxruntime as ort

CLASS_NAMES = ['0_normal', '1_open', '2_short', '3_mousebite', '4_spur', '5_copper', '6_pinhole']


def load_test_set(data_dir, channels):
    """테스트 패치 전체를 numpy 배열로 로드"""
    xs, ys = [], []
    for label, cls in enumerate(CLASS_NAMES):
        for path in sorted(glob.glob(os.path.join(data_dir, 'test', cls, '*.png'))):
            img = Image.open(path).convert('L')
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            xs.append(arr[None])
            ys.append(label)
    x = np.stack(xs)  # (N, 1, 64, 64)
    if channels == 3:
        x = np.repeat(x, 3, axis=1)
    return x, np.array(ys)


def measure_accuracy(sess, x, y, batch_size=256):
    correct = 0
    for i in range(0, len(x), batch_size):
        out = sess.run(None, {'input': x[i:i + batch_size]})[0]
        correct += (out.argmax(1) == y[i:i + batch_size]).sum()
    return correct / len(y)


def measure_latency(sess, channels, warmup=20, runs=200):
    """이미지 1장 기준 추론 지연시간 측정 (ms)"""
    x = np.random.randn(1, channels, 64, 64).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {'input': x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {'input': x})
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return times.mean(), np.percentile(times, 95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True, help='비교할 onnx 파일들')
    parser.add_argument('--data-dir', default='data/patches')
    parser.add_argument('--threads', type=int, default=None, help='추론 스레드 수 (기본: onnxruntime 기본값)')
    args = parser.parse_args()

    results = []
    cache = {}
    for path in args.models:
        so = ort.SessionOptions()
        if args.threads:
            so.intra_op_num_threads = args.threads
        sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
        channels = sess.get_inputs()[0].shape[1]

        if channels not in cache:
            print(f'테스트 패치 로드 중 (채널 {channels})...')
            cache[channels] = load_test_set(args.data_dir, channels)
        x, y = cache[channels]

        acc = measure_accuracy(sess, x, y)
        mean_ms, p95_ms = measure_latency(sess, channels)
        size_mb = os.path.getsize(path) / 1024 / 1024
        results.append({
            'model': os.path.basename(path),
            'size_mb': round(size_mb, 2),
            'accuracy': round(float(acc), 4),
            'latency_ms': round(float(mean_ms), 2),
            'latency_p95_ms': round(float(p95_ms), 2),
            'fps': round(1000 / mean_ms, 1),
        })
        print(f'{os.path.basename(path)}: acc {acc:.4f}, {mean_ms:.2f} ms/장 (p95 {p95_ms:.2f}), {size_mb:.2f} MB')

    print()
    print(f'{"모델":<28} {"크기(MB)":>8} {"정확도":>8} {"지연(ms)":>9} {"FPS":>6}')
    for r in results:
        print(f'{r["model"]:<28} {r["size_mb"]:>8} {r["accuracy"]:>8} {r["latency_ms"]:>9} {r["fps"]:>6}')

    os.makedirs('results', exist_ok=True)
    with open('results/benchmark.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('결과 저장: results/benchmark.json')


if __name__ == '__main__':
    main()
