"""DeepPCB 데이터셋에서 분류용 64x64 패치를 추출하는 스크립트

- 결함 패치: 어노테이션 박스 중심으로 64x64 크롭 (라벨 = 결함 종류)
- 정상 패치: 결함 박스와 겹치지 않는 위치에서 랜덤 크롭 (라벨 = 0_normal)
- 공식 분할(trainval.txt / test.txt)을 그대로 따라간다
"""
import os
import random
import argparse
from PIL import Image

CLASS_NAMES = ['0_normal', '1_open', '2_short', '3_mousebite', '4_spur', '5_copper', '6_pinhole']


def load_split_list(pcb_root, split_file):
    """trainval.txt / test.txt 를 읽어서 (이미지 경로, 어노테이션 경로) 목록으로 반환"""
    items = []
    with open(os.path.join(pcb_root, split_file)) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            img_rel, ann_rel = parts[0], parts[1]
            # 목록에는 00041000.jpg 로 적혀 있지만 실제 파일명은 00041000_test.jpg
            img_path = os.path.join(pcb_root, img_rel.replace('.jpg', '_test.jpg'))
            ann_path = os.path.join(pcb_root, ann_rel)
            items.append((img_path, ann_path))
    return items


def load_boxes(ann_path):
    """어노테이션 파일에서 (x1, y1, x2, y2, type) 목록 읽기"""
    boxes = []
    with open(ann_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            x1, y1, x2, y2, t = map(int, parts[:5])
            boxes.append((x1, y1, x2, y2, t))
    return boxes


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def overlaps(x, y, size, boxes, margin=8):
    """(x, y)에서 자른 패치가 결함 박스와 겹치는지 확인 (margin 만큼 여유를 둠)"""
    for (x1, y1, x2, y2, _) in boxes:
        if x < x2 + margin and x + size > x1 - margin and y < y2 + margin and y + size > y1 - margin:
            return True
    return False


def extract_patches(items, out_dir, patch_size, neg_per_image, rng, jitter):
    for name in CLASS_NAMES:
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)

    counts = {name: 0 for name in CLASS_NAMES}
    for img_path, ann_path in items:
        img = Image.open(img_path).convert('L')
        w, h = img.size
        boxes = load_boxes(ann_path)
        img_id = os.path.basename(img_path).replace('_test.jpg', '')

        # 결함 패치: 박스 중심 기준으로 크롭하되, 위치를 랜덤하게 흔들어준다.
        # (항상 중앙에 결함이 오게 자르면, 슬라이딩 윈도우에서 결함이 가장자리에
        #  걸리는 경우를 잘 못 잡는 문제가 있었음)
        for i, (x1, y1, x2, y2, t) in enumerate(boxes):
            cx = (x1 + x2) // 2 + rng.randint(-jitter, jitter)
            cy = (y1 + y2) // 2 + rng.randint(-jitter, jitter)
            px = clamp(cx - patch_size // 2, 0, w - patch_size)
            py = clamp(cy - patch_size // 2, 0, h - patch_size)
            patch = img.crop((px, py, px + patch_size, py + patch_size))
            cls = CLASS_NAMES[t]
            patch.save(os.path.join(out_dir, cls, f'{img_id}_{i}.png'))
            counts[cls] += 1

        # 정상 패치: 결함과 겹치지 않는 랜덤 위치에서 크롭
        made, tries = 0, 0
        while made < neg_per_image and tries < 100:
            tries += 1
            px = rng.randint(0, w - patch_size)
            py = rng.randint(0, h - patch_size)
            if overlaps(px, py, patch_size, boxes):
                continue
            patch = img.crop((px, py, px + patch_size, py + patch_size))
            patch.save(os.path.join(out_dir, '0_normal', f'{img_id}_n{made}.png'))
            counts['0_normal'] += 1
            made += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcb-root', default='data/DeepPCB/PCBData', help='DeepPCB의 PCBData 폴더 경로')
    parser.add_argument('--out', default='data/patches')
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--neg-per-image', type=int, default=6, help='이미지당 정상 패치 개수')
    parser.add_argument('--jitter', type=int, default=16, help='결함 패치 크롭 위치를 흔드는 범위 (px)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    for split, split_file in [('train', 'trainval.txt'), ('test', 'test.txt')]:
        items = load_split_list(args.pcb_root, split_file)
        print(f'[{split}] 보드 이미지 {len(items)}장에서 패치 추출 중...')
        counts = extract_patches(items, os.path.join(args.out, split), args.patch_size, args.neg_per_image, rng, args.jitter)
        total = sum(counts.values())
        print(f'[{split}] 총 {total}장 저장')
        for name, c in counts.items():
            print(f'    {name}: {c}')


if __name__ == '__main__':
    main()
