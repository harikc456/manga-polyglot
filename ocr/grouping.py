GROUPING_METHODS = ("none", "dbscan")


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


def make_grouper(config: dict | None):
    """Return grouper(boxes, img_w, img_h) -> list[list[dict]]. Validates config eagerly."""
    params = dict(config) if config else {}
    method = params.pop("method", "none")
    if method not in GROUPING_METHODS:
        raise ValueError(
            f"Unknown grouping method '{method}'. Valid options: {list(GROUPING_METHODS)}"
        )
    allowed = ("eps",) if method == "dbscan" else ()
    unknown = set(params) - set(allowed)
    if unknown:
        raise ValueError(
            f"Unknown grouping params {sorted(unknown)} for method '{method}'. Valid: {list(allowed)}"
        )
    if method == "none":
        return group_none

    eps = params.get("eps", 80)

    def grouper(boxes: list[dict], img_w: int, img_h: int) -> list[list[dict]]:
        eps_pixels = max(1, int(eps / 1000 * max(img_w, img_h)))
        return group_dbscan(boxes, eps_pixels)

    return grouper
