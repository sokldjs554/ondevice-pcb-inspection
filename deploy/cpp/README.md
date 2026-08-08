# C++ 추론 데모 (ONNX Runtime C++ API)

파이썬을 설치할 수 없는 장비를 대비해, `detect_board.py`의 추론과 후처리를
C++로 포팅한 데모입니다. 의존성은 onnxruntime 하나뿐이고, 이미지 입출력은
저장소에 포함된 stb 싱글헤더(`third_party/`)를 사용합니다.

같은 보드 이미지에서 파이썬 버전과 **동일한 검출 결과**(클래스, 박스 좌표까지)가
나오는 것을 확인했습니다. 인접 윈도우 묶기는 OpenCV connectedComponents 대신
BFS로 직접 구현했습니다. 박스 라벨 텍스트는 그리지 않고 콘솔 출력으로 대체했습니다.

## 빌드

1. onnxruntime C++ 라이브러리 다운로드: https://github.com/microsoft/onnxruntime/releases
   - x86 리눅스: `onnxruntime-linux-x64-<버전>.tgz`
   - 라즈베리파이(64bit OS): `onnxruntime-linux-aarch64-<버전>.tgz`
   - 압축만 풀면 됩니다 (1.20.1로 확인함)

2. 빌드 (이 폴더에서)

```bash
cmake -B build -DORT_HOME=/path/to/onnxruntime-linux-x64-1.20.1
cmake --build build
```

## 실행 (프로젝트 루트에서)

```bash
./deploy/cpp/build/detect_board models/mobilenet_v2_int8.onnx \
    data/DeepPCB/PCBData/group90100/90100/90100000_test.jpg 0.8
```

출력 예시:

```
검출 완료: 6건, 추론 218.4 ms (윈도우 361개, thresh 0.80)
  mousebite  box=[32, 0, 128, 64] score=0.999
  copper     box=[160, 128, 224, 224] score=1.000
  ...
결과 이미지 저장: result_cpp.png
```
