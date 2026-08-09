"""QAT(양자화 인식 학습)로 SimpleCNN을 재학습하는 스크립트

지금까지 쓴 방식은 PTQ(학습 후 양자화)입니다. 이미 학습된 모델을 나중에 INT8로 바꾸는
방식이라 간단하지만, 양자화 오차를 모델이 모른 채 학습됩니다.

QAT는 학습 중에 "양자화된 것처럼" 값을 반올림하는 가짜 양자화(fake quantize) 노드를
넣어두고 학습해서, 모델이 양자화 오차에 적응하도록 만듭니다. 보통 PTQ에서 정확도가
많이 떨어질 때 쓰는 방법입니다.

이 프로젝트는 PTQ로도 손실이 거의 없었기 때문에 QAT가 얼마나 더 도움이 되는지
직접 확인해보려고 만들었습니다.

사용 예시:
    python src/train_qat.py --epochs 5
"""
import os
import copy
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from models import build_model, input_channels
from train import make_transforms, run_epoch


def evaluate(model, loader, device='cpu'):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            correct += (out.argmax(1).cpu() == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/patches')
    parser.add_argument('--ckpt', default='models/simple_cnn_best.pt')
    parser.add_argument('--epochs', type=int, default=5, help='QAT 파인튜닝 epoch 수')
    parser.add_argument('--lr', type=float, default=1e-4, help='QAT는 낮은 학습률로 미세조정')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cpu')  # QAT 관측/변환은 CPU에서 수행

    channels = input_channels('simple_cnn')
    train_full = datasets.ImageFolder(os.path.join(args.data_dir, 'train'),
                                      transform=make_transforms(channels, train=True))
    n_val = int(len(train_full) * 0.1)
    train_set, _ = random_split(train_full, [len(train_full) - n_val, n_val],
                                generator=torch.Generator().manual_seed(args.seed))
    test_set = datasets.ImageFolder(os.path.join(args.data_dir, 'test'),
                                    transform=make_transforms(channels, train=False))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=128, num_workers=2)

    # 1. 학습된 FP32 모델 불러오기 (QAT는 보통 학습된 모델에서 파인튜닝한다)
    model = build_model('simple_cnn')
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    model.eval()
    fp32_acc = evaluate(model, test_loader)
    print(f'FP32 기준 정확도: {fp32_acc:.4f}')

    # 2. Conv-BN-ReLU를 하나로 합친다 (양자화 시 정확도·속도에 유리)
    # QuantWrapper로 감싸서 모델 입출력에 양자화/역양자화 경계를 만든다.
    # 이게 없으면 변환 후 추론에서 float 텐서가 양자화 연산에 들어가 에러가 난다.
    qat_model = torch.ao.quantization.QuantWrapper(copy.deepcopy(model))
    qat_model.train()
    fuse_list = [['module.features.0', 'module.features.1', 'module.features.2'],
                 ['module.features.4', 'module.features.5', 'module.features.6'],
                 ['module.features.8', 'module.features.9', 'module.features.10']]
    qat_model = torch.ao.quantization.fuse_modules_qat(qat_model, fuse_list)

    # 3. 가짜 양자화 노드 삽입
    qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
    torch.ao.quantization.prepare_qat(qat_model, inplace=True)

    # 4. 낮은 학습률로 파인튜닝
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, acc = run_epoch(qat_model, train_loader, criterion, optimizer, device, train=True)
        print(f'[QAT {epoch}/{args.epochs}] train loss {loss:.4f} acc {acc:.4f} | {time.time() - t0:.1f}s')

    # 5. 진짜 INT8 모델로 변환 후 평가
    qat_model.eval()
    int8_model = torch.ao.quantization.convert(qat_model.cpu(), inplace=False)
    qat_acc = evaluate(int8_model, test_loader)

    os.makedirs('models', exist_ok=True)
    out_path = 'models/simple_cnn_qat_int8.pt'
    torch.save(int8_model.state_dict(), out_path)

    print()
    print(f'FP32           : {fp32_acc:.4f}')
    print(f'QAT INT8       : {qat_acc:.4f}  ({(qat_acc - fp32_acc) * 100:+.2f}%p)')
    print(f'저장: {out_path}')
    print('\n(PTQ INT8 결과는 src/benchmark.py 출력과 비교하세요)')


if __name__ == '__main__':
    main()
