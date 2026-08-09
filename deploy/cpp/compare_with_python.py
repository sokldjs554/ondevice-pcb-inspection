"""C++ 포팅이 파이썬 버전과 같은 결과를 내는지 자동으로 확인하는 스크립트

C++로 옮긴 뒤 "같은 결과가 나온다"고 말하려면 보드 한 장만 봐서는 부족해서,
여러 장을 돌려서 검출 개수·클래스·박스 좌표가 전부 같은지 비교하도록 만들었다.

사용 예시:
    python deploy/cpp/compare_with_python.py --num-boards 30
    python deploy/cpp/compare_with_python.py --onnx models/mobilenet_v2_fp32.onnx
"""
import os
import re
import argparse
import subprocess

import cv2

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from detect_board import SlidingWindowDetector  # noqa: E402

# C++ 출력 한 줄 예시: "  pinhole    box=[416, 0, 480, 96] score=1.000"
LINE_RE = re.compile(r'^\s+(\w+)\s+box=\[(\d+), (\d+), (\d+), (\d+)\]\s+score=([\d.]+)')


def parse_cpp_output(text):
    out = []
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if m:
            out.append((m.group(1), tuple(int(m.group(i)) for i in range(2, 6))))
    return sorted(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', default='deploy/cpp/build/detect_board')
    parser.add_argument('--onnx', default='models/mobilenet_v2_int8.onnx')
    parser.add_argument('--pcb-root', default='data/DeepPCB/PCBData')
    parser.add_argument('--num-boards', type=int, default=30)
    parser.add_argument('--thresh', type=float, default=0.8)
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        raise SystemExit(f'C++ 실행 파일이 없습니다: {args.binary} (먼저 빌드하세요)')

    items = []
    with open(os.path.join(args.pcb_root, 'test.txt')) as f:
        for line in f:
            parts = line.split()
            if parts:
                items.append(parts[0].replace('.jpg', '_test.jpg'))
    items = items[:args.num_boards]

    detector = SlidingWindowDetector(args.onnx, thresh=args.thresh)

    same, diff = 0, 0
    total_py, total_cpp = 0, 0
    for rel in items:
        path = os.path.join(args.pcb_root, rel)
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        py = sorted((d['type'], tuple(d['box'])) for d in detector.detect(gray))

        proc = subprocess.run(
            [args.binary, '--model', args.onnx, '--image', path,
             '--thresh', str(args.thresh), '--out', '/dev/null'],
            capture_output=True, text=True)
        cpp = parse_cpp_output(proc.stdout)

        total_py += len(py)
        total_cpp += len(cpp)
        if py == cpp:
            same += 1
        else:
            diff += 1
            only_py = [d for d in py if d not in cpp]
            only_cpp = [d for d in cpp if d not in py]
            print(f'[다름] {os.path.basename(path)}  파이썬 {len(py)}건 / C++ {len(cpp)}건')
            for d in only_py:
                print(f'    파이썬에만: {d[0]} {list(d[1])}')
            for d in only_cpp:
                print(f'    C++에만  : {d[0]} {list(d[1])}')

    print()
    print(f'모델: {args.onnx}')
    print(f'보드 {len(items)}장 중 결과가 완전히 같은 보드: {same}장 (다른 보드 {diff}장)')
    print(f'총 검출 수: 파이썬 {total_py}건 / C++ {total_cpp}건')


if __name__ == '__main__':
    main()
