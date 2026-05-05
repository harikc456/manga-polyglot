import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_utils import parse_spotting_output, cluster_into_bubbles, boxes_from_clusters

# Two valid lines + one malformed line (no LOC tokens)
SAMPLE_RAW = (
    "FROM MY<|LOC_498|><|LOC_80|><|LOC_580|><|LOC_80|><|LOC_580|><|LOC_93|><|LOC_498|><|LOC_93|>\n"
    "TEACHER<|LOC_495|><|LOC_98|><|LOC_583|><|LOC_98|><|LOC_583|><|LOC_111|><|LOC_495|><|LOC_111|>\n"
    "BADLINE\n"
)


def test_parse_returns_one_box_per_valid_line():
    boxes = parse_spotting_output(SAMPLE_RAW, img_w=1000, img_h=1000)
    assert len(boxes) == 2


def test_parse_extracts_text():
    boxes = parse_spotting_output(SAMPLE_RAW, img_w=1000, img_h=1000)
    assert boxes[0]['text'] == 'FROM MY'
    assert boxes[1]['text'] == 'TEACHER'


def test_parse_denormalizes_coordinates():
    boxes = parse_spotting_output(SAMPLE_RAW, img_w=2000, img_h=500)
    # x_min for "FROM MY": min(498,580,580,498)=498 → 498/1000*2000=996
    assert boxes[0]['x_min'] == 996
    # y_min: min(80,80,93,93)=80 → 80/1000*500=40
    assert boxes[0]['y_min'] == 40


def test_parse_skips_malformed_lines():
    boxes = parse_spotting_output(SAMPLE_RAW, img_w=1000, img_h=1000)
    texts = [b['text'] for b in boxes]
    assert 'BADLINE' not in texts


def test_parse_empty_string():
    assert parse_spotting_output('', 1000, 1000) == []


def test_cluster_groups_nearby_boxes():
    boxes = [
        {'text': 'A', 'x_min': 100, 'y_min': 100, 'x_max': 200, 'y_max': 120},
        {'text': 'B', 'x_min': 105, 'y_min': 130, 'x_max': 205, 'y_max': 150},
        {'text': 'C', 'x_min': 800, 'y_min': 800, 'x_max': 900, 'y_max': 820},
    ]
    groups = cluster_into_bubbles(boxes, eps=100)
    assert len(groups) == 2


def test_cluster_isolated_line_is_own_group():
    boxes = [{'text': 'ALONE', 'x_min': 500, 'y_min': 500, 'x_max': 600, 'y_max': 520}]
    groups = cluster_into_bubbles(boxes, eps=50)
    assert len(groups) == 1
    assert groups[0][0]['text'] == 'ALONE'


def test_cluster_empty_input():
    assert cluster_into_bubbles([], eps=80) == []


def test_boxes_from_clusters_joins_text():
    groups = [
        [
            {'text': 'HELLO', 'x_min': 10, 'y_min': 10, 'x_max': 80, 'y_max': 30},
            {'text': 'WORLD', 'x_min': 12, 'y_min': 35, 'x_max': 78, 'y_max': 55},
        ]
    ]
    result = boxes_from_clusters(groups)
    assert len(result) == 1
    assert result[0]['text'] == 'HELLO WORLD'


def test_boxes_from_clusters_computes_bounding_box():
    groups = [
        [
            {'text': 'A', 'x_min': 10, 'y_min': 20, 'x_max': 80, 'y_max': 40},
            {'text': 'B', 'x_min': 5,  'y_min': 45, 'x_max': 90, 'y_max': 65},
        ]
    ]
    result = boxes_from_clusters(groups)
    assert result[0]['insertion_polygon'] == [5, 20, 90, 65]
