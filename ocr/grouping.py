GROUPING_METHODS = ("none", "dbscan", "overlap")
_METHOD_PARAMS = {"none": (), "dbscan": ("eps",), "overlap": ("threshold",)}


def union_box(group: list[dict]) -> tuple[int, int, int, int]:
    return (
        min(b["x_min"] for b in group),
        min(b["y_min"] for b in group),
        max(b["x_max"] for b in group),
        max(b["y_max"] for b in group),
    )


def group_none(boxes: list[dict], img_w: int, img_h: int) -> list[list[dict]]:
    return [[b] for b in boxes]


def group_dbscan(boxes: list[dict], eps_pixels: float) -> list[list[dict]]:
    if not boxes:
        return []
    import numpy as np
    from sklearn.cluster import DBSCAN

    centers = np.array([
        [(b["x_min"] + b["x_max"]) / 2, (b["y_min"] + b["y_max"]) / 2]
        for b in boxes
    ])
    labels = DBSCAN(eps=eps_pixels, min_samples=1).fit_predict(centers)
    groups: dict[int, list[dict]] = {}
    for label, box in zip(labels, boxes):
        groups.setdefault(int(label), []).append(box)
    return list(groups.values())


def _overlap_ratio(a: dict, b: dict) -> float:
    """Intersection area as a fraction of the smaller box's area (1.0 when one box is inside the other)."""
    ix = min(a["x_max"], b["x_max"]) - max(a["x_min"], b["x_min"])
    iy = min(a["y_max"], b["y_max"]) - max(a["y_min"], b["y_min"])
    if ix <= 0 or iy <= 0:
        return 0.0
    smaller = min(
        (a["x_max"] - a["x_min"]) * (a["y_max"] - a["y_min"]),
        (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"]),
    )
    return ix * iy / smaller if smaller > 0 else 0.0


def group_overlap(boxes: list[dict], threshold: float) -> list[list[dict]]:
    """Merge boxes that overlap by at least `threshold` of the smaller box, transitively."""
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _overlap_ratio(boxes[i], boxes[j]) >= threshold:
                parent[find(j)] = find(i)

    groups: dict[int, list[dict]] = {}
    for i, box in enumerate(boxes):
        groups.setdefault(find(i), []).append(box)
    return list(groups.values())


def make_grouper(config: dict | None):
    """Return grouper(boxes, img_w, img_h) -> list[list[dict]]. Validates config eagerly."""
    params = dict(config) if config else {}
    method = params.pop("method", "none")
    if method not in GROUPING_METHODS:
        raise ValueError(
            f"Unknown grouping method '{method}'. Valid options: {list(GROUPING_METHODS)}"
        )
    allowed = _METHOD_PARAMS[method]
    unknown = set(params) - set(allowed)
    if unknown:
        raise ValueError(
            f"Unknown grouping params {sorted(unknown)} for method '{method}'. Valid: {list(allowed)}"
        )
    if method == "none":
        return group_none

    if method == "overlap":
        threshold = params.get("threshold", 0.5)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
            raise ValueError(f"Grouping 'threshold' must be a number in (0, 1], got {threshold!r}")

        def overlap_grouper(boxes: list[dict], img_w: int, img_h: int) -> list[list[dict]]:
            return group_overlap(boxes, threshold)

        return overlap_grouper

    eps = params.get("eps", 80)

    def grouper(boxes: list[dict], img_w: int, img_h: int) -> list[list[dict]]:
        eps_pixels = max(1, int(eps / 1000 * max(img_w, img_h)))
        return group_dbscan(boxes, eps_pixels)

    return grouper
