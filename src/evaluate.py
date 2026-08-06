"""테스트셋 평가 스크립트 (정확도, 클래스별 리포트, confusion matrix)

사용 예시:
    python src/evaluate.py --model simple_cnn --ckpt models/simple_cnn_best.pt
"""
import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import build_model, input_channels
from train import make_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='simple_cnn', choices=['simple_cnn', 'mobilenet_v2'])
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--data-dir', default='data/patches')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    ckpt = args.ckpt or f'models/{args.model}_best.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    channels = input_channels(args.model)
    test_set = datasets.ImageFolder(os.path.join(args.data_dir, 'test'),
                                    transform=make_transforms(channels, train=False))
    loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=2)

    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            all_pred.append(out.argmax(1).cpu().numpy())
            all_true.append(y.numpy())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)

    acc = (y_pred == y_true).mean()
    print(f'test accuracy: {acc:.4f} ({(y_pred == y_true).sum()}/{len(y_true)})')
    print(classification_report(y_true, y_pred, target_names=test_set.classes, digits=4))

    # confusion matrix 그림 저장
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(test_set.classes)))
    ax.set_yticks(range(len(test_set.classes)))
    ax.set_xticklabels(test_set.classes, rotation=45, ha='right')
    ax.set_yticklabels(test_set.classes)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(f'{args.model} confusion matrix (test)')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, cm[i, j], ha='center', va='center', color=color, fontsize=8)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    out_png = f'results/confusion_matrix_{args.model}.png'
    fig.savefig(out_png, dpi=120)
    print(f'confusion matrix 저장: {out_png}')

    # README 표에 쓰기 위한 지표 저장
    report = classification_report(y_true, y_pred, target_names=test_set.classes, output_dict=True)
    with open(f'results/metrics_{args.model}.json', 'w') as f:
        json.dump({'accuracy': float(acc), 'report': report}, f, indent=2)


if __name__ == '__main__':
    main()
