"""결함 크기별 검출률을 재는 스크립트

전체 recall만 보면 "어떤 결함을 놓치는지"를 알 수 없어서, 정답 박스의 긴 변 길이로
구간을 나눠 따로 측정한다. 64x64 윈도우로 훑는 방식이라 윈도우보다 큰 결함이
불리할 것이라는 가설을 확인하려고 만들었다.

사용 예시:
    python deploy/eval_by_size.py --num-boards 100
    python deploy/eval_by_size.py --num-boards 100 --multiscale
"""
import os
import argparse

import cv2

from detect_board import SlidingWindowDetector, load_gt_boxes, detect_multiscale

BINS = [(0, 32, '~32px'), (32, 64, '32~64px'), (64, 96, '64~96px'), (96, 10000, '96px 초과')]


def coverage(gt, det):
    gx1, gy1, gx2, gy2 = gt
    dx1, dy1, dx2, dy2 = det
    ix1, iy1 = max(gx1, dx1), max(gy1, dy1)
    ix2, iy2 = min(gx2, dx2), min(gy2, dy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = (gx2 - gx1) * (gy2 - gy1)
    return inter / area if area > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', default='models/mobilenet_v2_int8.onnx')
    parser.add_argument('--pcb-root', default='data/DeepPCB/PCBData')
    parser.add_argument('--num-boards', type=int, default=100)
    parser.add_argument('--thresh', type=float, default=0.8)
    parser.add_argument('--cover', type=float, default=0.5)
    parser.add_argument('--multiscale', action='store_true',
                        help='이미지 피라미드로 큰 결함까지 검출')
    args = parser.parse_args()

    items = []
    with open(os.path.join(args.pcb_root, 'test.txt')) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                items.append((parts[0].replace('.jpg', '_test.jpg'), parts[1]))
    items = items[:args.num_boards]

    detector = SlidingWindowDetector(args.onnx, thresh=args.thresh)
    found = {b[2]: 0 for b in BINS}
    total = {b[2]: 0 for b in BINS}

    for img_rel, ann_rel in items:
        gray = cv2.imread(os.path.join(args.pcb_root, img_rel), cv2.IMREAD_GRAYSCALE)
        gt = load_gt_boxes(os.path.join(args.pcb_root, ann_rel))
        dets = detect_multiscale(detector, gray) if args.multiscale else detector.detect(gray)
        for (x1, y1, x2, y2, _) in gt:
            size = max(x2 - x1, y2 - y1)  # 긴 변 기준
            name = next(b[2] for b in BINS if b[0] <= size < b[1])
            total[name] += 1
            if any(coverage((x1, y1, x2, y2), d['box']) >= args.cover for d in dets):
                found[name] += 1

    mode = '멀티스케일' if args.multiscale else '기본'
    print(f'보드 {len(items)}장, {mode} (thresh={args.thresh}, cover>={args.cover})')
    print(f"{'결함 크기':<12}{'개수':>6}{'검출':>6}{'recall':>10}")
    for _, _, name in BINS:
        n = total[name]
        r = found[name] / n if n else 0
        print(f'{name:<12}{n:>6}{found[name]:>6}{r:>9.1%}')
    n_all = sum(total.values())
    print(f"{'전체':<12}{n_all:>6}{sum(found.values()):>6}"
          f"{sum(found.values()) / n_all if n_all else 0:>9.1%}")


if __name__ == '__main__':
    main()
