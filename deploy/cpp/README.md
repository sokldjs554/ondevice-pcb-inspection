# C++ 추론 데모 (ONNX Runtime C++ API)

파이썬을 올리기 어려운 장비를 대비해, `deploy/detect_board.py`의 검출 파이프라인 전체를
C++로 포팅한 데모입니다. 의존성은 onnxruntime 하나뿐이고, 이미지 입출력은 저장소에 포함된
stb 싱글헤더(`third_party/`)를 사용합니다. OpenCV가 하던 일(리사이즈, 모폴로지,
connected components)은 직접 구현했습니다.

파이썬 버전과 같은 기능을 지원합니다.

| 기능 | 파이썬 | C++ | C++ 구현 방식 |
|---|---|---|---|
| 슬라이딩 윈도우 배치 추론 | O | O | ONNX Runtime C++ API, 256개씩 나눠서 |
| 인접 윈도우 묶기 | O | O | `cv2.connectedComponents` → BFS 직접 구현 |
| 템플릿 차분 오검출 제거 | O | O | 적분 영상으로 블록 평균 계산 |
| 박스 정밀화 | O | O | 7x7 닫힘 연산 + BFS 라벨링 직접 구현 |
| 멀티스케일 검출 | O | O | 면적 평균 축소(INTER_AREA 방식) 직접 구현 |
| 결과 TCP(JSON) 전송 | O | O | POSIX 소켓(`getaddrinfo`/`connect`/`send`) |
| 스레드 수별 벤치마크 | X | O | `--bench` |
| 폴더 연속 검사 (배치) | X | O | `--dir`, 결과 JSONL 저장 |
| 읽기·전처리 / 추론 파이프라인 | X | O | `--pipeline` (생산자-소비자 스레드) |

## 파일 구성

```
main.cpp              커맨드라인 파싱, 전체 흐름, 벤치마크 모드
src/detector.h/.cpp   ONNX 세션, 슬라이딩 윈도우, 인접 윈도우 묶기, 멀티스케일, NMS
src/postprocess.*     템플릿 차분 필터, 박스 정밀화 (모폴로지·라벨링 직접 구현)
src/image_io.*        stb 래퍼: 흑백 로드, 면적 평균 축소, 박스 그려서 PNG 저장
src/tcp_sender.*      JSON 문자열 생성, TCP 전송
src/batch.*           폴더 연속 검사, 생산자-소비자 파이프라인 (스레드·큐)
compare_with_python.py  파이썬 결과와 C++ 결과가 같은지 자동 비교
```

## 빌드

1. onnxruntime C++ 라이브러리 다운로드: https://github.com/microsoft/onnxruntime/releases
   - x86 리눅스: `onnxruntime-linux-x64-<버전>.tgz`
   - 라즈베리파이(64bit OS): `onnxruntime-linux-aarch64-<버전>.tgz`
   - macOS: `onnxruntime-osx-x86_64-<버전>.tgz` (Apple Silicon은 arm64)
   - 압축만 풀면 됩니다 (1.19.2 / 1.20.1로 확인함)

2. 빌드 (이 폴더에서)

```bash
cmake -B build -DORT_HOME=/path/to/onnxruntime-linux-x64-1.20.1
cmake --build build -j4
```

## 실행 (프로젝트 루트에서)

```bash
# 기본 검출
./deploy/cpp/build/detect_board --model models/mobilenet_v2_int8.onnx \
    --image data/DeepPCB/PCBData/group00041/00041/00041000_test.jpg

# 템플릿 비교로 오검출 제거 + 박스를 실제 결함 크기로 정밀화
./deploy/cpp/build/detect_board --image <보드> --template auto --refine

# 큰 결함까지 잡기 (이미지 피라미드)
./deploy/cpp/build/detect_board --image <보드> --multiscale

# 결과를 모니터링 서버로 전송 (다른 터미널에서 python deploy/monitor_server.py --port 5000)
./deploy/cpp/build/detect_board --image <보드> --send 127.0.0.1:5000

# 스레드 수별 속도 비교
./deploy/cpp/build/detect_board --image <보드> --bench 10

# 폴더 안의 보드를 연속으로 검사 (처리량 측정, 결과 JSONL 저장)
./deploy/cpp/build/detect_board --dir <폴더> --limit 20 --jsonl results.jsonl
```

`--help`로 전체 옵션을 볼 수 있습니다. 예전 문서에 적어둔 위치 인자 방식
(`detect_board <모델> <이미지> [thresh]`)도 그대로 동작합니다.

출력 예시:

```
템플릿 비교: 13건 -> 12건 (1건 제거, 박스 정밀화 적용)
검출 완료: 12건, 0.13초 (윈도우 361개, thresh 0.80)
  단계별: 전처리 10.6 ms / 추론 116.4 ms / 후처리 0.0 ms
  pinhole    box=[426, 26, 460, 52] score=1.000 diff=0.238
  short      box=[494, 30, 520, 54] score=0.963 diff=0.254
  ...
결과 이미지 저장: result_cpp.png
```

## 파이썬 결과와 같은지 확인

`compare_with_python.py`가 같은 보드를 파이썬과 C++로 각각 돌려서 검출 개수, 클래스,
박스 좌표까지 전부 비교합니다.

```bash
python deploy/cpp/compare_with_python.py --num-boards 30 --onnx models/mobilenet_v2_fp32.onnx
python deploy/cpp/compare_with_python.py --num-boards 30 --onnx models/mobilenet_v2_int8.onnx
```

결과 (테스트 보드 30장):

| 모델 | 결과가 완전히 같은 보드 | 총 검출 수 (파이썬 / C++) |
|---|---|---|
| FP32 | 30 / 30 | 270 / 270 |
| INT8 | 16 / 30 | 269 / 270 |

### INT8에서 왜 결과가 갈렸나

처음엔 C++ 코드를 잘못 옮긴 줄 알고 전처리 값을 하나씩 찍어봤는데, 원인은 모델이 아니라
**JPEG 디코더**였습니다. 파이썬은 OpenCV(libjpeg-turbo), C++은 stb_image로 이미지를 읽는데
두 디코더의 결과가 완전히 같지 않습니다.

- 640x640 보드 한 장에서 **316픽셀(0.08%)**이 값 1만큼 달랐습니다.
- FP32에서는 이 정도 차이가 결과를 바꾸지 않았습니다 (30장 전부 동일).
- INT8은 값을 256단계로 뭉개기 때문에, 확신도가 임계값 0.8 근처인 윈도우에서
  0.79 ↔ 0.83처럼 판정이 뒤집혔습니다. 달라진 곳은 전부 이런 경계 윈도우 하나씩이었고,
  결함을 통째로 놓치거나 없는 걸 만들어낸 경우는 없었습니다.

확인 방법: 같은 JPEG을 OpenCV로 디코딩해 PNG로 저장한 뒤 그 PNG를 양쪽에 넣으니,
결과가 달랐던 보드 7장이 **전부 동일**해졌습니다. 디코더가 원인이라는 게 확실해졌습니다.

실제 장비에서는 카메라 프레임을 메모리로 바로 받아 쓰기 때문에 JPEG 디코딩 자체가
빠지지만, "PC에서 검증한 결과와 장비에서 나온 결과가 왜 다른가"를 따질 때 모델뿐 아니라
입력 경로까지 봐야 한다는 걸 이 과정에서 배웠습니다.

## 스레드 수별 속도 (`--bench`)

4코어 x86 리눅스, 보드 1장(윈도우 361개), 10회 평균:

| 모델 | 1스레드 | 2스레드 | 4스레드 | 4스레드 배속 |
|---|---|---|---|---|
| SimpleCNN INT8 | 87.2 ms | 49.6 ms | 28.7 ms | 3.04x |
| MobileNetV2 INT8 | 209.5 ms | 125.8 ms | 83.1 ms | 2.52x |

코어를 4배 늘려도 4배가 되지는 않습니다. MobileNetV2 쪽 이득이 더 작은데, depthwise
convolution은 연산량 대비 메모리 접근이 많아서 스레드를 늘려도 메모리 대역폭에서
막히기 때문으로 보입니다. (NPU 실험에서 MAC 유닛을 32→256으로 늘려도 2배가 안 됐던
것과 같은 이유입니다.)

전처리(윈도우 자르기 + 정규화)는 3~5 ms, 후처리(BFS 묶기)는 0.1 ms 미만이라
전체의 대부분은 추론 시간입니다. 그래서 최적화가 필요하다면 전처리보다 모델 쪽을
건드리는 게 맞다고 판단했습니다.

## 배치 검사와 파이프라인 (`--dir`, `--pipeline`)

보드 한 장이 아니라 라인 위로 계속 흘러오는 상황을 가정해서, 폴더를 통째로 검사하고
처리량(장/초)을 재는 모드를 넣었습니다.

```bash
# 폴더 안의 보드를 순서대로 검사하고 결과를 JSONL로 저장
./deploy/cpp/build/detect_board --dir data/DeepPCB/PCBData/group00041/00041 \
    --limit 20 --jsonl results.jsonl

# 읽기·전처리를 별도 스레드로 돌려 추론과 겹치기
./deploy/cpp/build/detect_board --dir <폴더> --limit 20 --pipeline
```

한 장을 처리하는 과정은 `읽기 → 전처리 → 추론` 세 단계인데, 순차 방식에서는 추론이 도는
동안 다음 장을 읽는 시간이 그냥 낭비됩니다. 그래서 **생산자-소비자 구조**로 나눴습니다.
생산자 스레드가 읽기+전처리를 맡아 큐에 넣고, 메인 스레드가 큐에서 꺼내 추론합니다.
큐 크기는 2로 제한했는데, 보드 한 장의 입력 텐서가 361 × 3 × 64 × 64 × 4바이트 = 약 17MB라
제한이 없으면 메모리가 계속 늘어나기 때문입니다. (`std::mutex` + `std::condition_variable`)

### 측정 결과 — 이득이 나는 조건이 따로 있었습니다

4코어 리눅스, 보드 20장:

| 추론 스레드 | 순차 | 파이프라인 | 결과 |
|---|---|---|---|
| 4 (코어 전부) | 3.62초 | 3.90초 | **오히려 8% 느림** |
| 1 | 7.57초 | 6.83초 | **10% 빠름** |

추론이 이미 코어를 전부 쓰고 있으면, 생산자 스레드가 끼어들면서 서로 CPU를 뺏습니다.
반대로 추론 스레드를 1개로 묶어 코어를 남겨두면 읽기·전처리가 그 여유 코어에서 돌아
추론과 진짜로 겹칩니다.

**파이프라인은 "남는 코어가 있을 때만" 이득**이라는 걸 측정으로 확인했습니다. 코어가 적은
임베디드 장비라면 추론에 코어를 몰아주고 순차로 도는 편이 낫고, 코어가 넉넉하거나 읽기가
느린 장치(네트워크 스토리지, 느린 카메라)라면 파이프라인이 유리합니다. 그래서 기본은
순차로 두고 옵션으로 뒀습니다.

## 문제 해결

**`Library not loaded ... Reason: image not found` (macOS) 또는 `error while loading shared libraries` (Linux)**

실행 파일이 onnxruntime 라이브러리 위치를 못 찾는 경우입니다. 라이브러리 경로를
직접 지정해서 실행하면 바로 확인할 수 있습니다.

```bash
# macOS
DYLD_LIBRARY_PATH=/path/to/onnxruntime/lib ./deploy/cpp/build/detect_board --image <보드>

# Linux
LD_LIBRARY_PATH=/path/to/onnxruntime/lib ./deploy/cpp/build/detect_board --image <보드>
```

빌드할 때 경로가 실행 파일에 박히도록 CMakeLists에 rpath 설정을 넣어뒀는데,
예전 버전으로 빌드한 경우 `build` 폴더를 지우고 다시 빌드하면 해결됩니다.

**macOS에서 "개발자를 확인할 수 없어 열 수 없습니다"**

인터넷에서 받은 dylib에 격리 속성이 붙어서 나는 경고입니다. 서명되지 않은 파일이라는
뜻이고, 아래 명령으로 속성을 지우면 실행됩니다.

```bash
xattr -dr com.apple.quarantine /path/to/onnxruntime
```
