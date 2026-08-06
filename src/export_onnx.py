"""학습한 PyTorch 모델을 ONNX로 변환하는 스크립트

NPU나 다른 추론 엔진에 모델을 올리려면 프레임워크 중립 포맷이 필요해서
ONNX로 변환한 뒤, PyTorch 출력과 ONNX Runtime 출력이 일치하는지 검증한다.

사용 예시:
    python src/export_onnx.py --model simple_cnn
"""
import os
import argparse

import numpy as np
import torch
import onnxruntime as ort

from models import build_model, input_channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='simple_cnn', choices=['simple_cnn', 'mobilenet_v2'])
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    ckpt = args.ckpt or f'models/{args.model}_best.pt'
    out_path = args.out or f'models/{args.model}_fp32.onnx'

    model = build_model(args.model)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()

    channels = input_channels(args.model)
    dummy = torch.randn(1, channels, 64, 64)

    torch.onnx.export(
        model, dummy, out_path,
        input_names=['input'], output_names=['logits'],
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=13,
        external_data=False,  # 가중치를 .data 파일로 분리하지 않고 onnx 파일 하나로 저장
    )
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f'ONNX 저장: {out_path} ({size_mb:.2f} MB)')

    # PyTorch와 ONNX Runtime 출력 비교 검증
    x = torch.randn(8, channels, 64, 64)
    with torch.no_grad():
        torch_out = model(x).numpy()
    sess = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    onnx_out = sess.run(None, {'input': x.numpy()})[0]
    max_diff = np.abs(torch_out - onnx_out).max()
    print(f'PyTorch vs ONNX 최대 오차: {max_diff:.6f}')
    if max_diff < 1e-4:
        print('검증 통과')
    else:
        print('경고: 오차가 큽니다. 변환 과정을 확인하세요.')


if __name__ == '__main__':
    main()
