# NPU 컴파일 실험 (ARM Ethos-U / Vela)

ONNX Runtime으로 CPU 추론까지는 확인했는데, 정작 **"이 모델이 실제 NPU에 올라가긴 하나?"** 는
확인하지 못한 게 계속 걸렸습니다. NPU 개발보드가 없어서 포기하고 있다가, ARM Ethos-U 계열
NPU의 공식 컴파일러인 **Vela**가 하드웨어 없이도 컴파일과 성능 추정을 해준다는 걸 알게 되어
직접 돌려봤습니다.

```
PyTorch → ONNX → TFLite(INT8) → Vela 컴파일 → NPU 성능 리포트
                                    ↑ 여기가 이 폴더
```

## 왜 TFLite로 다시 변환했나

NPU 컴파일러는 대부분 정해진 입력 포맷을 요구합니다. Vela는 **INT8로 완전 양자화된 TFLite**만
받기 때문에, 배포용으로 만들어둔 ONNX를 TFLite로 옮기고 다시 양자화해야 했습니다.
이때 두 가지를 맞춰야 했습니다.

- **채널 순서**: ONNX는 NCHW, TFLite는 NHWC (onnx2tf가 변환 처리)
- **입출력 타입**: NPU는 정수 연산만 하므로 입력·출력까지 int8로 강제
  (`inference_input_type = tf.int8`)

캘리브레이션에는 ONNX 양자화 때와 같은 학습 패치 200장을 썼습니다.

## 결과 1 — 연산 배치 (가장 궁금했던 것)

NPU가 지원하지 않는 연산이 있으면 그 부분만 CPU로 떨어지면서(fallback) 성능이 급락합니다.
그래서 컴파일 후 제일 먼저 확인한 게 이 비율이었습니다.

| 모델 | NPU 연산 | CPU 폴백 |
|---|---|---|
| SimpleCNN | 10개 (100%) | 0개 |
| MobileNetV2 | 65개 (100%) | 0개 |

둘 다 폴백 없이 전부 NPU에 올라갔습니다. MobileNetV2의 depthwise convolution까지 지원되는 걸
확인했고, 처음부터 엣지 배포를 염두에 두고 단순한 연산만 쓴 게 결과적으로 맞았습니다.

## 결과 2 — NPU 구성별 성능 추정

Ethos-U55는 MAC 유닛 수(32/128/256)로 구성이 나뉩니다. 같은 모델이 하드웨어 사양에 따라
어떻게 달라지는지 비교해봤습니다.

| 모델 | NPU 구성 | 추론 시간 | FPS | SRAM |
|---|---|---|---|---|
| SimpleCNN | Ethos-U55-32 | 1.938 ms | 516 | 85.3 KB |
| SimpleCNN | Ethos-U55-128 | 1.122 ms | 892 | 85.3 KB |
| SimpleCNN | Ethos-U55-256 | **0.987 ms** | **1013** | 85.3 KB |
| MobileNetV2 | Ethos-U55-32 | 6.965 ms | 144 | 124.4 KB |
| MobileNetV2 | Ethos-U55-128 | 5.069 ms | 197 | 124.4 KB |
| MobileNetV2 | Ethos-U55-256 | **4.889 ms** | **205** | 124.4 KB |

여기서 배운 것:

- **MAC 유닛을 8배(32→256) 늘려도 속도는 2배도 안 올랐습니다.** 연산 자체보다 메모리
  대역폭이 병목이라는 뜻으로 보입니다. Vela 리포트의 Flash 대역폭 항목이 이를 뒷받침합니다.
  하드웨어 사양을 올리는 것만으로는 한계가 있고 모델 쪽 최적화가 같이 가야 한다는 걸
  숫자로 보게 됐습니다.
- **SRAM 사용량이 85~124KB**로 나와서, 메모리가 작은 MCU급 장비에도 올릴 수 있는
  수준이라는 걸 확인했습니다.

## 결과 3 — 양자화 후 정확도 검증 (동작 검증)

컴파일이 됐다고 끝이 아니라 정확도가 유지되는지 확인해야 합니다. ONNX Runtime INT8 경로와
TFLite INT8 경로를 같은 테스트셋(패치 6,140장)으로 비교했습니다.

| 모델 | ONNX Runtime INT8 | TFLite INT8 (NPU용) | 차이 |
|---|---|---|---|
| SimpleCNN | 93.05% | 92.96% | -0.09%p |
| MobileNetV2 | 96.01% | 95.98% | -0.03%p |

양자화 툴체인이 달라도 정확도는 거의 동일했습니다. 두 경로 모두 캘리브레이션 방식이
비슷해서인 것으로 보입니다.

## 실행 방법

변환이 끝난 INT8 TFLite 모델은 `npu/models/`에 포함해 뒀습니다. 컴파일과 검증만 하려면
변환 단계(=tensorflow 설치)는 건너뛰어도 됩니다.

```bash
pip install -r npu/requirements.txt

python npu/compile_npu.py                            # Vela로 NPU 컴파일 + 성능 비교
python npu/verify_tflite.py --model mobilenet_v2     # 양자화 후 정확도 검증
```

ONNX에서 TFLite 변환까지 직접 다시 하려면 아래를 추가로 설치하고 실행합니다.

```bash
pip install -r npu/requirements-convert.txt

python npu/convert_tflite.py --model simple_cnn      # ONNX -> INT8 TFLite
python npu/convert_tflite.py --model mobilenet_v2
```

### 인텔 맥(x86_64)에서 설치할 때

`ethos-u-vela`는 macOS x86_64 휠이 없어서 소스로 빌드해야 합니다(Xcode Command Line
Tools 필요). `--only-binary :all:` 옵션을 쓰면 설치가 실패합니다.

```bash
pip install ethos-u-vela
```

변환까지 직접 하려면 버전을 맞춰야 합니다. 최신 `onnx2tf`가 쓰는 `ai_edge_litert`
바이너리가 macOS 14.2용으로 빌드돼 있어서 그보다 낮은 macOS에서는 로드되지 않습니다.
`ai_edge_litert`를 쓰지 않는 마지막 버전(1.26.9)과, 인텔 맥 휠이 있는 마지막
tensorflow(2.16.2)를 묶어둔 파일을 따로 만들어 뒀습니다.

```bash
pip install -r npu/requirements-convert-intel-mac.txt
```

버전을 내리고 나면 이번엔 변환이 `ValueError: axes don't match array`로 실패합니다.
`models/*_fp32.onnx`는 최신 PyTorch의 새 exporter로 내보낸 opset 18 그래프인데,
onnx2tf 1.26.x가 이걸 처리하지 못하기 때문입니다. 그래서 같은 가중치를 opset 13으로
내보낸 사본(`npu/models/*_fp32_opset13.onnx`)을 함께 넣어뒀고, `convert_tflite.py`가
이 파일이 있으면 자동으로 먼저 사용합니다.

컴파일 결과는 `npu/vela_out/<구성>/`에, 성능 요약은 `results/npu_compile.json`에 저장됩니다.

## 한계

- **추정치입니다.** Vela가 내놓는 건 실제 측정값이 아니라 사이클 기반 추정이라,
  실제 보드에서는 드라이버·메모리 구성에 따라 달라질 수 있습니다.
- **ARM Ethos-U 기준입니다.** NPU마다 지원 연산과 컴파일러가 달라서, 다른 NPU에서는
  폴백이 생기거나 모델 수정이 필요할 수 있습니다. 다만 "컴파일 → 연산 배치 확인 →
  성능·메모리 추정 → 정확도 검증"이라는 흐름 자체는 공통일 것이라고 생각합니다.
- Ethos-U65 구성도 함께 측정했는데 메모리 모드가 달라(Dedicated SRAM) 다른 구성과 직접
  비교하기 어려워 표에서는 제외했습니다. (`results/npu_compile.json`에는 남아 있습니다)
