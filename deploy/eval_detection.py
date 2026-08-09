"""보드 단위 검출 성능 평가 스크립트

패치 분류 정확도와 별개로 "실제 보드에서 결함을 얼마나 찾아내는지"를 확인하기 위한 평가.
테스트 보드들에 대해 슬라이딩 윈도우 검출을 돌리고, 정답 박스와 비교해서
recall(정답 결함 중 찾은 비율)과 precision(검출 중 실제 결함 비율)을 계산한다.

매칭 기준: 정답 박스 면적의 50% 이상이 검출 박스에 덮이면 "찾았다"로 본다.
(검출 박스가 윈도우 단위(64px 배수)로 나와서 작은 정답 박스와의 IoU는 구조적으로
낮게 나온다. 처음엔 IoU 기준으로 쟀다가, 맞게 찾은 것도 미스로 세길래 기준을 바꿨다.)

사용 예시:
    python deploy/eval_detection.py --onnx models/simple_cnn_int8.onnx --num-boards 100
"""
import os
import argparse

import cv2
import numpy as np

from detect_board import SlidingWindowDetector, load_gt_boxes, filter_by_template, CLASS_NAMES


def coverage(gt, det):
    """정답 박스(gt) 면적 중 검출 박스(det)에 덮인 비율"""
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
    parser.add_argument('--num-boards', type=int, default=100, help='평가할 테스트 보드 수 (전체는 500)')
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--thresh', type=float, default=0.8)
    parser.add_argument('--cover', type=float, default=0.5, help='정답 박스가 이 비율 이상 덮이면 검출로 인정')
    parser.add_argument('--use-template', action='store_true',
                        help='결함 없는 템플릿 이미지와 비교해 정상 구조물 오검출을 제거')
    args = parser.parse_args()

    items = []
    with open(os.path.join(args.pcb_root, 'test.txt')) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                items.append((parts[0].replace('.jpg', '_test.jpg'), parts[1]))
    items = items[:args.num_boards]

    detector = SlidingWindowDetector(args.onnx, stride=args.stride, thresh=args.thresh)

    n_gt, n_det = 0, 0
    n_found, n_found_type = 0, 0
    n_correct_det = 0
    for img_rel, ann_rel in items:
        gray = cv2.imread(os.path.join(args.pcb_root, img_rel), cv2.IMREAD_GRAYSCALE)
        gt = load_gt_boxes(os.path.join(args.pcb_root, ann_rel))
        dets = detector.detect(gray)

        if args.use_template:
            tpl_path = os.path.join(args.pcb_root, img_rel.replace('_test.jpg', '_temp.jpg'))
            template = cv2.imread(tpl_path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                dets = filter_by_template(dets, gray, template)

        n_gt += len(gt)
        n_det += len(dets)
        # recall: 정답 박스 기준으로 매칭되는 검출이 있는지
        for (x1, y1, x2, y2, tname) in gt:
            matched = [d for d in dets if coverage((x1, y1, x2, y2), d['box']) >= args.cover]
            if matched:
                n_found += 1
                if any(d['type'] == tname for d in matched):
                    n_found_type += 1
        # precision: 검출 박스 기준으로 매칭되는 정답이 있는지
        for d in dets:
            if any(coverage((x1, y1, x2, y2), d['box']) >= args.cover for (x1, y1, x2, y2, _) in gt):
                n_correct_det += 1

    recall = n_found / n_gt if n_gt else 0
    recall_type = n_found_type / n_gt if n_gt else 0
    precision = n_correct_det / n_det if n_det else 0
    tpl_note = ', 템플릿 비교 ON' if args.use_template else ''
    print(f'보드 {len(items)}장 / 정답 결함 {n_gt}개 / 검출 {n_det}개  '
          f'(thresh={args.thresh}, stride={args.stride}, cover={args.cover}{tpl_note})')
    print(f'recall (위치):        {recall:.3f}  ({n_found}/{n_gt})')
    print(f'recall (위치+종류):   {recall_type:.3f}  ({n_found_type}/{n_gt})')
    print(f'precision:            {precision:.3f}  ({n_correct_det}/{n_det})')


if __name__ == '__main__':
    main()
