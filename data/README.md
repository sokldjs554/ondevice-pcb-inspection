# 데이터셋 준비

이 프로젝트는 [DeepPCB](https://github.com/tangsanli5201/DeepPCB) 데이터셋을 사용합니다. (MIT License)

- 640x640 흑백 PCB 이미지 1,500쌍 (결함 없는 템플릿 + 결함이 있는 테스트 이미지)
- 결함 6종: open, short, mousebite, spur, copper, pin-hole
- 어노테이션 형식: `x1 y1 x2 y2 type` (바운딩 박스 + 결함 종류)
- 공식 분할: 학습 1,000장 / 테스트 500장 (`trainval.txt`, `test.txt`)

## 1. 데이터셋 다운로드

```bash
git clone https://github.com/tangsanli5201/DeepPCB.git data/DeepPCB
```

## 2. 패치 추출

보드 한 장을 그대로 분류할 수는 없기 때문에, 결함 위치를 중심으로 64x64 패치를 잘라서
분류용 데이터셋을 만듭니다. 이때 크롭 위치에 ±16px 랜덤 오프셋을 줍니다. (항상 결함이
정중앙에 오게 자르면, 슬라이딩 윈도우 추론에서 결함이 가장자리에 걸릴 때 잘 못 잡습니다.)
정상 패치는 결함 박스와 겹치지 않는 위치에서 랜덤하게 뽑습니다.

```bash
python data/prepare_patches.py
```

실행하면 `data/patches/` 아래에 이런 구조로 저장됩니다.

```
data/patches/
├── train/
│   ├── 0_normal/
│   ├── 1_open/
│   ├── 2_short/
│   ├── 3_mousebite/
│   ├── 4_spur/
│   ├── 5_copper/
│   └── 6_pinhole/
└── test/
    └── (동일 구조)
```

## 출처

> Tang, S., He, F., Huang, X., Yang, J. "Online PCB Defect Detector On A New PCB Defect Dataset." (2019)
> https://github.com/tangsanli5201/DeepPCB

데이터셋 라이선스는 원본 저장소(MIT)를 따릅니다.
