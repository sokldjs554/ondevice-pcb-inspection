"""보드 이미지 한 장에서 결함 위치를 찾는 스크립트 (슬라이딩 윈도우 방식)

640x640 보드 이미지를 64x64 윈도우로 훑으면서 패치 분류기를 돌리고,
결함으로 판정된 윈도우들을 묶어서 결함 위치(박스)와 종류를 출력한다.
--send 옵션을 주면 검출 결과를 TCP(JSON)로 모니터링 서버에 전송한다.

사용 예시:
    python deploy/detect_board.py --image board.jpg
    python deploy/detect_board.py --image board.jpg --annot board.txt --send 192.168.0.10:5000
"""
import os
import json
import time
import socket
import argparse

import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES = ['normal', 'open', 'short', 'mousebite', 'spur', 'copper', 'pinhole']


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class SlidingWindowDetector:
    def __init__(self, onnx_path, patch_size=64, stride=32, thresh=0.8, threads=None):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])
        self.channels = self.sess.get_inputs()[0].shape[1]
        self.patch_size = patch_size
        self.stride = stride
        self.thresh = thresh

    def detect(self, gray):
        """흑백 이미지(np.uint8, HxW)에서 결함 목록을 반환"""
        h, w = gray.shape
        p, s = self.patch_size, self.stride
        xs = list(range(0, w - p + 1, s))
        ys = list(range(0, h - p + 1, s))

        # 윈도우를 전부 잘라서 한 번에 배치 추론
        patches = []
        for y in ys:
            for x in xs:
                patches.append(gray[y:y + p, x:x + p])
        batch = np.stack(patches).astype(np.float32) / 255.0
        batch = (batch - 0.5) / 0.5
        batch = batch[:, None, :, :]
        if self.channels == 3:
            batch = np.repeat(batch, 3, axis=1)

        probs = []
        for i in range(0, len(batch), 256):
            out = self.sess.run(None, {'input': batch[i:i + 256]})[0]
            probs.append(softmax(out))
        probs = np.concatenate(probs).reshape(len(ys), len(xs), -1)

        # 결함 확률이 높은 윈도우만 골라서 클래스별로 인접 윈도우를 묶는다
        cls_map = probs.argmax(axis=2)
        conf_map = probs.max(axis=2)
        defects = []
        for c in range(1, len(CLASS_NAMES)):
            mask = ((cls_map == c) & (conf_map >= self.thresh)).astype(np.uint8)
            if mask.sum() == 0:
                continue
            n_comp, comp = cv2.connectedComponents(mask, connectivity=8)
            for k in range(1, n_comp):
                iy, ix = np.where(comp == k)
                x1 = int(min(ix) * s)
                y1 = int(min(iy) * s)
                x2 = int(max(ix) * s + p)
                y2 = int(max(iy) * s + p)
                score = float(conf_map[iy, ix].max())
                defects.append({'type': CLASS_NAMES[c], 'box': [x1, y1, x2, y2], 'score': round(score, 3)})
        return defects


def load_gt_boxes(ann_path):
    boxes = []
    with open(ann_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                x1, y1, x2, y2, t = map(int, parts[:5])
                boxes.append((x1, y1, x2, y2, CLASS_NAMES[t]))
    return boxes


def draw_result(gray, defects, gt_boxes=None):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if gt_boxes:
        for (x1, y1, x2, y2, name) in gt_boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)  # 빨강: 정답
    for d in defects:
        x1, y1, x2, y2 = d['box']
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록: 검출 결과
        label = f"{d['type']} {d['score']:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return vis


def send_json(host, port, payload):
    """검출 결과를 모니터링 서버로 전송 (한 줄 JSON)"""
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall((json.dumps(payload, ensure_ascii=False) + '\n').encode('utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--onnx', default='models/mobilenet_v2_int8.onnx')
    parser.add_argument('--annot', default=None, help='정답 어노테이션 파일 (비교 표시용)')
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--thresh', type=float, default=0.8)
    parser.add_argument('--out-dir', default='results/detections')
    parser.add_argument('--send', default=None, help='host:port 형식. 결과를 TCP로 전송')
    args = parser.parse_args()

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f'이미지를 열 수 없습니다: {args.image}')

    detector = SlidingWindowDetector(args.onnx, stride=args.stride, thresh=args.thresh)
    t0 = time.time()
    defects = detector.detect(gray)
    elapsed = time.time() - t0

    print(f'검출 완료: {len(defects)}건, {elapsed:.2f}초')
    for d in defects:
        print(f"  {d['type']:<10} box={d['box']} score={d['score']}")

    gt_boxes = load_gt_boxes(args.annot) if args.annot else None
    vis = draw_result(gray, defects, gt_boxes)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, os.path.basename(args.image).replace('.jpg', '_result.png'))
    cv2.imwrite(out_path, vis)
    print(f'결과 이미지 저장: {out_path}')

    if args.send:
        host, port = args.send.split(':')
        payload = {
            'image': os.path.basename(args.image),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_sec': round(elapsed, 2),
            'defects': defects,
        }
        send_json(host, int(port), payload)
        print(f'검출 결과 전송 완료 -> {args.send}')


if __name__ == '__main__':
    main()
