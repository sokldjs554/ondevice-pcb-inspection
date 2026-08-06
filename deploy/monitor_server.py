"""검사 결과 수신용 모니터링 서버 (PC에서 실행)

라즈베리파이(검사 장비 쪽)에서 보낸 검출 결과(JSON)를 받아서 콘솔에 출력하고
로그 파일로도 남긴다. 공장에서 검사 장비 여러 대를 한 곳에서 모니터링하는
상황을 가정한 간단한 데모.

사용 예시:
    python deploy/monitor_server.py --port 5000
"""
import json
import socket
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--log', default='monitor_log.jsonl')
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', args.port))
    server.listen(5)
    print(f'모니터링 서버 시작 (포트 {args.port}), Ctrl+C로 종료')

    try:
        while True:
            conn, addr = server.accept()
            with conn:
                data = b''
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
            for line in data.decode('utf-8').strip().splitlines():
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    print(f'[{addr[0]}] JSON 파싱 실패: {line[:80]}')
                    continue
                n = len(msg.get('defects', []))
                status = '불량' if n > 0 else '정상'
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {addr[0]} | {msg.get('image')} | "
                      f"{status} (결함 {n}건, {msg.get('elapsed_sec')}초)")
                for d in msg.get('defects', []):
                    print(f"    - {d['type']} box={d['box']} score={d['score']}")
                with open(args.log, 'a') as f:
                    f.write(line + '\n')
    except KeyboardInterrupt:
        print('\n서버 종료')
    finally:
        server.close()


if __name__ == '__main__':
    main()
