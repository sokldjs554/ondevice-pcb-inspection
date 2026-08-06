"""패치 분류 모델 학습 스크립트

사용 예시:
    python src/train.py --model simple_cnn --epochs 10
    python src/train.py --model mobilenet_v2 --epochs 10
"""
import os
import json
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import build_model, input_channels


def make_transforms(channels, train=True):
    tf = [transforms.Grayscale(num_output_channels=channels)]
    if train:
        # PCB 패치는 상하/좌우 뒤집어도 결함 종류가 변하지 않아서 flip 증강을 사용
        tf += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    tf += [transforms.ToTensor(), transforms.Normalize([0.5] * channels, [0.5] * channels)]
    return transforms.Compose(tf)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='simple_cnn', choices=['simple_cnn', 'mobilenet_v2'])
    parser.add_argument('--data-dir', default='data/patches')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', default='models')
    parser.add_argument('--resume', action='store_true',
                        help='마지막 체크포인트에서 이어서 학습 (학습이 중간에 끊겼을 때)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    channels = input_channels(args.model)
    train_full = datasets.ImageFolder(os.path.join(args.data_dir, 'train'),
                                      transform=make_transforms(channels, train=True))
    # 검증용 데이터는 학습 데이터에서 10%를 떼어서 사용 (공식 분할에는 val이 따로 없음)
    n_val = int(len(train_full) * args.val_ratio)
    train_set, val_set = random_split(train_full, [len(train_full) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(args.seed))
    print(f'train {len(train_set)} / val {len(val_set)} / classes {train_full.classes}')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=2)

    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{args.model} 파라미터 수: {n_params:,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 후반부에 학습률을 낮춰서 수렴을 안정시킨다
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f'{args.model}_best.pt')
    last_path = os.path.join(args.out_dir, f'{args.model}_last.pt')
    history_path = f'results/history_{args.model}.json'

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    start_epoch = 1
    if args.resume and os.path.exists(last_path):
        model.load_state_dict(torch.load(last_path, map_location=device))
        with open(history_path) as f:
            history = json.load(f)
        best_acc = max(history['val_acc'])
        start_epoch = len(history['val_acc']) + 1
        # 스케줄러를 이어받은 epoch 위치까지 진행시킨다
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f'{last_path}에서 이어서 학습 (epoch {start_epoch}부터, best val acc {best_acc:.4f})')

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)
        scheduler.step()

        mark = ''
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), ckpt_path)
            mark = ' *'
        print(f'[{epoch:2d}/{args.epochs}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | '
              f'val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - t0:.1f}s{mark}')

        # 중간에 끊겨도 이어서 학습할 수 있게 매 epoch마다 저장
        torch.save(model.state_dict(), last_path)
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    print(f'best val acc: {best_acc:.4f} -> {ckpt_path}')

    # 학습 곡선 저장
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history['train_loss'], label='train')
    axes[0].plot(epochs, history['val_loss'], label='val')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend()
    axes[1].plot(epochs, history['train_acc'], label='train')
    axes[1].plot(epochs, history['val_acc'], label='val')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('accuracy')
    axes[1].legend()
    fig.suptitle(f'{args.model} training curves')
    fig.tight_layout()
    fig.savefig(f'results/curves_{args.model}.png', dpi=120)
    print(f'학습 곡선 저장: results/curves_{args.model}.png')


if __name__ == '__main__':
    main()
