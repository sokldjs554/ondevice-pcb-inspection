# 실행 화면 캡처

저장소를 클론해서 직접 실행한 화면입니다. 학습 없이 저장소에 포함된 ONNX/TFLite 모델만으로
대부분 재현할 수 있습니다. (환경: Intel 2코어 노트북, macOS)

## 1. 보드 결함 검출 — Python

```bash
python deploy/detect_board.py --image data/DeepPCB/PCBData/group00041/00041/00041000_test.jpg
```

![](01_detect_python.png)

## 2. 같은 보드 — C++ 포팅 버전

```bash
./deploy/cpp/build/detect_board --image <같은 보드>
```

![](02_detect_cpp.png)

## 3. 모델 벤치마크 (FP32 vs INT8, 테스트 패치 6,140장)

```bash
python src/benchmark.py --models models/simple_cnn_fp32.onnx models/simple_cnn_int8.onnx
```

![](03_benchmark.png)

## 4. 보드 단위 검출 성능 (테스트 보드 100장)

```bash
python deploy/eval_detection.py --num-boards 100
```

![](04_eval_detection.png)

## 5. 검사 결과 TCP 전송 → 모니터링 서버 수신

```bash
python deploy/monitor_server.py --port 5000        # 터미널 1
python deploy/detect_board.py --image <보드> --send 127.0.0.1:5000   # 터미널 2
```

![](05_tcp_monitor.png)

## 6~7. ONNX → INT8 TFLite 변환 (NPU 입력 포맷)

```bash
python npu/convert_tflite.py --model simple_cnn
python npu/convert_tflite.py --model mobilenet_v2
```

![](06_npu_simple_cnn.png)
![](07_npu_mobilenet_v2.png)

## 8. ARM Ethos-U(Vela) NPU 컴파일 — 연산 배치와 구성별 성능 추정

```bash
python npu/compile_npu.py
```

![](08_npu_compile.png)

두 모델 모두 CPU 폴백 0. MAC 유닛을 8배(32→256) 늘려도 속도는 2배 미만 → 메모리 대역폭이 병목.

## 9. NPU용 INT8 TFLite 정확도 검증

```bash
python npu/verify_tflite.py --model mobilenet_v2
```

![](09_npu_verify.png)

## 10. QAT vs 대조군(FP32 동일 조건 파인튜닝)

```bash
python src/train_qat.py --epochs 3
```

![](10_qat.png)

QAT의 순수 이득은 +0.16%p. 이 모델에는 QAT의 추가 복잡도가 정당화되지 않는다는 결론.

## 11. C++ 추론 스레드 수별 벤치마크

```bash
./deploy/cpp/build/detect_board --image <보드> --bench 10
```

![](11_cpp_bench.png)

물리 코어가 2개인 장비라 4스레드가 2스레드보다 오히려 느립니다.

## 12. C++ 결과와 파이썬 결과 자동 대조

```bash
python deploy/cpp/compare_with_python.py --num-boards 10 --onnx models/mobilenet_v2_fp32.onnx
```

![](12_cpp_compare.png)

## 13. C++ 버전 템플릿 비교 + 박스 정밀화

```bash
./deploy/cpp/build/detect_board --image <보드> --template auto --refine
```

![](13_cpp_template.png)

파이썬과 동일한 후처리를 OpenCV 없이 구현했고, 정밀화된 박스 좌표까지 같습니다.

## 14. 논문 기준(IoU≥0.33) 검출 성능

```bash
python deploy/eval_detection.py --use-template --refine --iou 0.33 --num-boards 100
```

![](14_eval_iou.png)
