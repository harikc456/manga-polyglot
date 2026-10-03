import pytest

from ocr.grouping import make_grouper, union_box


def _box(x0, y0, x1, y1, text="t"):
    return {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1, "text": text}


def test_union_box():
    group = [_box(10, 20, 50, 60), _box(5, 30, 40, 90)]
    assert union_box(group) == (5, 20, 50, 90)


def test_none_puts_each_box_in_its_own_group():
    boxes = [_box(0, 0, 10, 10), _box(0, 0, 10, 10)]
    groups = make_grouper({"method": "none"})(boxes, 100, 100)
    assert groups == [[boxes[0]], [boxes[1]]]


def test_default_grouper_is_none():
    boxes = [_box(0, 0, 10, 10)]
    assert make_grouper(None)(boxes, 100, 100) == [[boxes[0]]]


def test_none_with_no_boxes_returns_empty():
    assert make_grouper({"method": "none"})([], 100, 100) == []


def test_dbscan_merges_nearby_and_splits_far():
    grouper = make_grouper({"method": "dbscan", "eps": 80})
    near_a = _box(10, 10, 50, 30)
    near_b = _box(10, 35, 50, 55)
    far = _box(400, 400, 490, 490)
    groups = grouper([near_a, near_b, far], 500, 500)  # eps = 80/1000*500 = 40px
    assert sorted(len(g) for g in groups) == [1, 2]


def test_dbscan_with_no_boxes_returns_empty():
    assert make_grouper({"method": "dbscan"})([], 500, 500) == []


def test_dbscan_tiny_image_does_not_crash():
    # eps in pixels would round to 0, which DBSCAN rejects
    groups = make_grouper({"method": "dbscan", "eps": 80})([_box(0, 0, 2, 2)], 5, 5)
    assert len(groups) == 1


def test_unknown_method_raises():
    with pytest.raises(ValueError, match=r"Unknown grouping method 'kmeans'.*none.*dbscan"):
        make_grouper({"method": "kmeans"})


def test_unknown_param_raises():
    with pytest.raises(ValueError, match=r"Unknown grouping params \['eps'\] for method 'none'"):
        make_grouper({"method": "none", "eps": 3})
    with pytest.raises(ValueError, match=r"Unknown grouping params \['epz'\] for method 'dbscan'"):
        make_grouper({"method": "dbscan", "epz": 3})
