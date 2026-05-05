import re
import numpy as np
from sklearn.cluster import DBSCAN


def parse_spotting_output(raw: str, img_w: int, img_h: int) -> list[dict]:
    pattern = r'(.+?)((?:<\|LOC_\d+\|>){8})'
    boxes = []
    for line in raw.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        match = re.match(pattern, line)
        if not match:
            continue
        text = match.group(1).strip()
        loc_tokens = re.findall(r'<\|LOC_(\d+)\|>', match.group(2))
        if len(loc_tokens) != 8:
            continue
        coords = list(map(int, loc_tokens))
        xs = coords[0::2]
        ys = coords[1::2]
        boxes.append({
            'text': text,
            'x_min': int(min(xs) / 1000 * img_w),
            'y_min': int(min(ys) / 1000 * img_h),
            'x_max': int(max(xs) / 1000 * img_w),
            'y_max': int(max(ys) / 1000 * img_h),
        })
    return boxes


def cluster_into_bubbles(boxes: list[dict], eps: float) -> list[list[dict]]:
    if not boxes:
        return []
    centers = np.array([
        [(b['x_min'] + b['x_max']) / 2, (b['y_min'] + b['y_max']) / 2]
        for b in boxes
    ])
    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(centers)
    groups: dict[int, list[dict]] = {}
    for label, box in zip(labels, boxes):
        groups.setdefault(int(label), []).append(box)
    return list(groups.values())


def boxes_from_clusters(groups: list[list[dict]]) -> list[dict]:
    result = []
    for group in groups:
        text = ' '.join(b['text'] for b in group)
        x_min = min(b['x_min'] for b in group)
        y_min = min(b['y_min'] for b in group)
        x_max = max(b['x_max'] for b in group)
        y_max = max(b['y_max'] for b in group)
        result.append({
            'text': text,
            'insertion_polygon': [x_min, y_min, x_max, y_max],
        })
    return result
