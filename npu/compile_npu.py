"""INT8 TFLite 모델을 NPU용으로 컴파일하고 성능을 비교하는 스크립트

ARM Ethos-U 계열 NPU의 공식 컴파일러인 Vela를 사용한다. 실제 NPU 보드가 없어도
컴파일과 성능 추정이 가능해서, 모델이 NPU에 얼마나 잘 맞는지 미리 확인할 수 있다.

Vela가 알려주는 것:
  - NPU에서 실행되는 연산 비율 (CPU로 폴백되는 연산이 있는지)
  - 추정 추론 시간과 초당 추론 횟수
  - SRAM / Flash 사용량 (메모리가 작은 MCU에 올릴 수 있는지 판단)

사용 예시:
    python npu/compile_npu.py
    python npu/compile_npu.py --configs ethos-u55-128 ethos-u65-256
"""
import os
import csv
import glob
import json
import argparse
import subprocess


def run_vela(tflite_path, config, out_dir):
    """Vela 컴파일 실행 후 요약 CSV를 파싱해서 결과 반환"""
    subprocess.run(
        ['vela', tflite_path, '--accelerator-config', config, '--output-dir', out_dir],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    name = os.path.basename(tflite_path).replace('.tflite', '')
    csv_files = glob.glob(os.path.join(out_dir, f'{name}_summary_*.csv'))
    if not csv_files:
        raise RuntimeError(f'요약 파일을 찾을 수 없습니다: {out_dir}')
    with open(sorted(csv_files)[-1]) as f:
        row = list(csv.DictReader(f))[0]

    vela_model = os.path.join(out_dir, f'{name}_vela.tflite')
    return {
        'model': name,
        'accelerator': config,
        'nn_macs': int(row['nn_macs']),
        'inference_time_ms': round(float(row['inference_time']) * 1000, 3),
        'inferences_per_second': round(float(row['inferences_per_second']), 1),
        'sram_kb': round(float(row['sram_memory_used']), 1),
        'flash_kb': round(float(row['off_chip_flash_memory_used']), 1),
        'vela_model_kb': round(os.path.getsize(vela_model) / 1024, 1) if os.path.exists(vela_model) else None,
    }


def count_operator_placement(tflite_path, config, out_dir):
    """CPU로 폴백된 연산이 있는지 Vela 출력 로그에서 확인"""
    proc = subprocess.run(
        ['vela', tflite_path, '--accelerator-config', config, '--output-dir', out_dir],
        capture_output=True, text=True)
    cpu_ops = npu_ops = None
    for line in proc.stdout.splitlines():
        if line.startswith('CPU operators'):
            cpu_ops = line.split('=')[1].strip()
        elif line.startswith('NPU operators'):
            npu_ops = line.split('=')[1].strip()
    return cpu_ops, npu_ops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', default='npu/models')
    parser.add_argument('--out-dir', default='npu/vela_out')
    parser.add_argument('--configs', nargs='+',
                        default=['ethos-u55-32', 'ethos-u55-128', 'ethos-u55-256', 'ethos-u65-256'],
                        help='비교할 NPU 구성 (숫자는 MAC 유닛 수)')
    args = parser.parse_args()

    tflite_files = sorted(glob.glob(os.path.join(args.models_dir, '*_int8.tflite')))
    if not tflite_files:
        raise SystemExit(f'INT8 TFLite 모델이 없습니다: {args.models_dir}\n'
                         'npu/convert_tflite.py를 먼저 실행하세요.')

    results = []
    for tflite in tflite_files:
        name = os.path.basename(tflite).replace('_int8.tflite', '')
        print(f'\n=== {name} ===')
        cpu_ops, npu_ops = count_operator_placement(tflite, args.configs[0], args.out_dir)
        print(f'연산 배치: NPU {npu_ops} / CPU {cpu_ops}')

        for config in args.configs:
            out_dir = os.path.join(args.out_dir, config)
            os.makedirs(out_dir, exist_ok=True)
            r = run_vela(tflite, config, out_dir)
            r['cpu_operators'] = cpu_ops
            r['npu_operators'] = npu_ops
            results.append(r)
            print(f'  {config:16} {r["inference_time_ms"]:>8.3f} ms  '
                  f'{r["inferences_per_second"]:>8.1f} FPS  '
                  f'SRAM {r["sram_kb"]:>7.1f} KB  Flash {r["flash_kb"]:>7.1f} KB')

    os.makedirs('results', exist_ok=True)
    with open('results/npu_compile.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('\n결과 저장: results/npu_compile.json')


if __name__ == '__main__':
    main()
