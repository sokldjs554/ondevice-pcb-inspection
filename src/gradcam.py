"""Grad-CAM 시각화 — 모델이 패치의 어디를 보고 판단했는지 확인

정확도 숫자만으로는 모델이 "결함을 보고" 맞힌 건지, 주변 패턴을 외운 건지 알 수 없어서
마지막 conv 특징맵의 기울기로 판단 근거 영역을 히트맵으로 그려봤습니다.

사용 예시:
    python src/gradcam.py --model mobilenet_v2
출력:
    results/gradcam_mobilenet_v2.png         (결함 6종, 정분류 사례)
    results/gradcam_errors_mobilenet_v2.png  (오분류 사례 — 정상→short, spur→정상 등)
"""
import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import build_model, input_channels
from train import make_transforms


class GradCAM:
    """마지막 conv 블록의 특징맵과 기울기를 후킹해서 CAM을 계산"""

    def __init__(self, model, target_layer):
        self.model = model
        self.feat = None
        self.grad = None
        target_layer.register_forward_hook(self._save_feat)

    def _save_feat(self, module, inp, out):
        self.feat = out
        out.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.grad = grad

    def __call__(self, x, class_idx=None):
        """x: (1,C,64,64) → (cam 64x64 [0~1], 예측 클래스, softmax 확신도)"""
        logits = self.model(x)
        prob = F.softmax(logits, dim=1)
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # 채널별 기울기 평균을 가중치로 특징맵을 합산 (Grad-CAM 원논문 방식)
        weights = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.feat).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx, float(prob[0, class_idx])


def draw_grid(samples, classes, out_png, title):
    """samples: [(원본 2D, cam 2D, 제목)] → 위 원본 / 아래 Grad-CAM 오버레이"""
    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 4.2))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i, (img, cam, label) in enumerate(samples):
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(label, fontsize=8)
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].imshow(cam, cmap='jet', alpha=0.45)
        for ax in (axes[0, i], axes[1, i]):
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel('patch', fontsize=9)
    axes[1, 0].set_ylabel('Grad-CAM', fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'저장: {out_png}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mobilenet_v2', choices=['simple_cnn', 'mobilenet_v2'])
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--data-dir', default='data/patches')
    parser.add_argument('--max-scan', type=int, default=4000,
                        help='사례를 찾기 위해 훑을 테스트 패치 수')
    parser.add_argument('--layer', type=int, default=None,
                        help='CAM을 계산할 features 블록 인덱스 (기본: 아래 참고)')
    args = parser.parse_args()

    ckpt = args.ckpt or f'models/{args.model}_best.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    channels = input_channels(args.model)

    test_set = datasets.ImageFolder(os.path.join(args.data_dir, 'test'),
                                    transform=make_transforms(channels, train=False))
    classes = test_set.classes  # 0_normal, 1_open, ...

    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # CAM 층 선택: 마지막 conv가 정석이지만 64x64 입력에서 MobileNetV2의 마지막
    # 특징맵은 2x2라 위치 정보가 거의 없음 → stride 8 구간(features[6], 8x8)을 기본으로 사용.
    # SimpleCNN은 마지막 features가 이미 8x8이라 그대로 사용.
    if args.layer is not None:
        cam_layer = model.features[args.layer]
    elif args.model == 'mobilenet_v2':
        cam_layer = model.features[6]   # 8x8 (c=32 블록 끝)
    else:
        cam_layer = model.features[-1]  # SimpleCNN: 8x8
    gradcam = GradCAM(model, cam_layer)

    correct = {}   # 결함 클래스별 정분류 대표 사례 (확신도 최고)
    errors = []    # 재미있는 오분류: 정상→short, spur/mousebite→정상
    idx_order = np.random.RandomState(0).permutation(len(test_set))[:args.max_scan]

    for i in idx_order:
        x, y = test_set[int(i)]
        x = x.unsqueeze(0).to(device)
        cam, pred, conf = gradcam(x)
        img = x[0, 0].cpu().numpy()  # 흑백 (3채널이어도 복사본이라 첫 채널 사용)

        if pred == y and y != 0:  # 결함 정분류
            name = classes[y]
            if name not in correct or conf > correct[name][3]:
                correct[name] = (img, cam, f'{name.split("_")[1]}\npred {conf:.2f}', conf)
        elif pred != y and len(errors) < 8:
            true_n, pred_n = classes[y].split('_')[1], classes[pred].split('_')[1]
            # 분석 대상: 정상을 short로 본 경우 / 작은 결함(spur 등)을 정상으로 놓친 경우
            if (y == 0 and pred == 2) or (y != 0 and pred == 0):
                errors.append((img, cam, f'true {true_n}\npred {pred_n} ({conf:.2f})'))

    os.makedirs('results', exist_ok=True)
    order = [c for c in classes if c != '0_normal']
    samples = [correct[c][:3] for c in order if c in correct]
    draw_grid(samples, classes, f'results/gradcam_{args.model}.png',
              f'Grad-CAM: correctly classified defects ({args.model})')
    if errors:
        draw_grid(errors[:6], classes, f'results/gradcam_errors_{args.model}.png',
                  f'Grad-CAM: misclassified cases ({args.model})')


if __name__ == '__main__':
    main()
