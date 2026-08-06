"""라즈베리파이 + USB 카메라 실시간 검사 데모

카메라 프레임을 640x640으로 잘라서 슬라이딩 윈도우 검출을 돌리고,
결함 박스와 FPS를 화면에 표시한다. 모니터가 없는 환경에서는 --save 옵션으로
결과 프레임을 파일로 저장할 수 있다.

사용 예시:
    python deploy/camera_demo.py
    python deploy/camera_demo.py --send 192.168.0.10:5000 --save
"""
import os
import time
import argparse

import cv2

from detect_board import SlidingWindowDetector, draw_result, send_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', default='models/simple_cnn_int8.onnx')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--thresh', type=float, default=0.8)
    parser.add_argument('--send', default=None, help='host:port 형식. 결과를 TCP로 전송')
    parser.add_argument('--save', action='store_true', help='결과 프레임을 파일로 저장 (모니터 없는 환경용)')
    args = parser.parse_args()

    detector = SlidingWindowDetector(args.onnx, stride=args.stride, thresh=args.thresh)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit('카메라를 열 수 없습니다')
    print('실시간 검사 시작 (q 키로 종료)')

    if args.save:
        os.makedirs('capture', exist_ok=True)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 프레임 가운데를 640x640으로 잘라서 사용
        h, w = frame.shape[:2]
        size = min(h, w, 640)
        y0, x0 = (h - size) // 2, (w - size) // 2
        crop = frame[y0:y0 + size, x0:x0 + size]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        t0 = time.time()
        defects = detector.detect(gray)
        elapsed = time.time() - t0

        vis = draw_result(gray, defects)
        fps_text = f'{1 / elapsed:.1f} FPS ({elapsed * 1000:.0f} ms)'
        cv2.putText(vis, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if args.send and defects:
            payload = {
                'image': f'frame_{frame_idx}',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed_sec': round(elapsed, 2),
                'defects': defects,
            }
            try:
                host, port = args.send.split(':')
                send_json(host, int(port), payload)
            except OSError as e:
                print(f'전송 실패: {e}')

        if args.save:
            cv2.imwrite(f'capture/frame_{frame_idx:04d}.png', vis)
        else:
            cv2.imshow('PCB defect detection', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
