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


# --- overlap grouping -----------------------------------------------------

def _overlap(boxes, threshold=None):
    config = {"method": "overlap"}
    if threshold is not None:
        config["threshold"] = threshold
    return make_grouper(config)(boxes, 1000, 1000)


def test_overlap_merges_a_box_nested_inside_another():
    big, small = _box(0, 0, 100, 100), _box(10, 10, 30, 30)
    assert _overlap([big, small]) == [[big, small]]


def test_overlap_merges_at_the_threshold_and_splits_below_it():
    a = _box(0, 0, 100, 100)
    at = _box(50, 0, 150, 100)       # intersection 5000 / smaller area 10000 = 0.5
    below = _box(60, 0, 160, 100)    # 0.4
    assert _overlap([a, at], 0.5) == [[a, at]]
    assert _overlap([a, below], 0.5) == [[a], [below]]


def test_overlap_default_threshold_is_half():
    a = _box(0, 0, 100, 100)
    assert _overlap([a, _box(50, 0, 150, 100)]) == [[a, _box(50, 0, 150, 100)]]
    assert len(_overlap([a, _box(60, 0, 160, 100)])) == 2


def test_overlap_merges_chains_transitively():
    a, b, c = _box(0, 0, 100, 100), _box(50, 0, 150, 100), _box(100, 0, 200, 100)
    assert _overlap([a, b, c]) == [[a, b, c]]    # a and c do not overlap each other


def test_overlap_keeps_disjoint_and_edge_touching_boxes_apart():
    a, b, c = _box(0, 0, 10, 10), _box(10, 0, 20, 10), _box(500, 500, 600, 600)
    assert _overlap([a, b, c]) == [[a], [b], [c]]


def test_overlap_groups_are_ordered_by_first_member():
    a, far, a2 = _box(0, 0, 100, 100), _box(500, 500, 600, 600), _box(10, 10, 20, 20)
    assert _overlap([a, far, a2]) == [[a, a2], [far]]


def test_overlap_zero_area_box_does_not_divide_by_zero():
    flat, big = _box(10, 10, 10, 50), _box(0, 0, 100, 100)
    assert _overlap([flat, big]) == [[flat], [big]]


def test_overlap_with_no_boxes_returns_empty():
    assert _overlap([]) == []


@pytest.mark.parametrize("bad", [0, -0.2, 1.5, "0.5", None, True])
def test_overlap_rejects_a_bad_threshold(bad):
    with pytest.raises(ValueError, match="threshold"):
        make_grouper({"method": "overlap", "threshold": bad})


def test_overlap_rejects_params_of_other_methods():
    with pytest.raises(ValueError, match=r"Unknown grouping params \['eps'\] for method 'overlap'.*\['threshold'\]"):
        make_grouper({"method": "overlap", "eps": 80})


def test_unknown_method_message_lists_overlap():
    with pytest.raises(ValueError, match="overlap"):
        make_grouper({"method": "kmeans"})
