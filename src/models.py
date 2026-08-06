"""모델 정의

- SimpleCNN: 엣지 디바이스를 생각해서 직접 설계한 작은 CNN (입력: 1x64x64 흑백)
- MobileNetV2: torchvision의 경량 아키텍처 (입력: 3x64x64, 흑백을 3채널로 복사해서 사용)
"""
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(name, num_classes=7):
    if name == 'simple_cnn':
        return SimpleCNN(num_classes)
    elif name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model
    else:
        raise ValueError(f'지원하지 않는 모델: {name}')


def input_channels(name):
    """모델별 입력 채널 수 (SimpleCNN은 흑백 1채널, MobileNetV2는 3채널)"""
    return 1 if name == 'simple_cnn' else 3
