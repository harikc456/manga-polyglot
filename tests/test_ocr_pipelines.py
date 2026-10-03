import pytest
from PIL import Image

from ocr import build_pipeline, registry
from ocr.base import Detector, Recognizer, Spotter
from ocr.grouping import make_grouper
from ocr.pipelines import DetectRecognize, Spot

_BUILT = []
_DETECTORS = []


class FakeDetector(Detector):
    def __init__(self, boxes=None):
        _BUILT.append("detector")
        _DETECTORS.append(self)
        self.boxes = boxes or []
        self.modes = []
        self.closed = False

    def detect(self, pil_image):
        self.modes.append(pil_image.mode)
        return self.boxes

    def close(self):
        self.closed = True


class FakeRecognizer(Recognizer):
    def __init__(self, text="txt"):
        _BUILT.append("recognizer")
        self.text = text
        self.crops = []
        self.closed = False

    def read(self, crop):
        self.crops.append(crop.size)
        return self.text

    def close(self):
        self.closed = True


class FakeSpotter(Spotter):
    def __init__(self, boxes=None):
        _BUILT.append("spotter")
        self.boxes = boxes or []
        self.closed = False

    def spot(self, pil_image):
        return self.boxes

    def close(self):
        self.closed = True


def _page(tmp_path, size=(100, 100), mode="RGB"):
    path = tmp_path / "page.png"
    Image.new(mode, size).save(path)
    return str(path)


def _box(x0, y0, x1, y1, text="t"):
    return {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1, "text": text}


NONE = make_grouper({"method": "none"})


# --- DetectRecognize ------------------------------------------------------

def test_detect_recognize_empty_detection_returns_empty_and_skips_recognizer(tmp_path):
    rec = FakeRecognizer()
    pipe = DetectRecognize(FakeDetector([]), rec, NONE)
    assert pipe.run(_page(tmp_path)) == []
    assert rec.crops == []


def test_detect_recognize_returns_text_and_polygon_per_box(tmp_path):
    det = FakeDetector([_box(20, 20, 40, 40), _box(60, 60, 80, 80)])
    pipe = DetectRecognize(det, FakeRecognizer(text="hello"), NONE, crop_padding=0)
    assert pipe.run(_page(tmp_path)) == [
        {"text": "hello", "insertion_polygon": [20, 20, 40, 40]},
        {"text": "hello", "insertion_polygon": [60, 60, 80, 80]},
    ]


def test_detect_recognize_applies_and_clamps_crop_padding(tmp_path):
    rec = FakeRecognizer()
    pipe = DetectRecognize(FakeDetector([_box(5, 5, 20, 20)]), rec, NONE, crop_padding=10)
    result = pipe.run(_page(tmp_path, size=(100, 100)))
    assert result[0]["insertion_polygon"] == [0, 0, 30, 30]
    assert rec.crops == [(30, 30)]


def test_detect_recognize_groups_nearby_boxes_with_dbscan(tmp_path):
    boxes = [_box(10, 10, 50, 30), _box(10, 35, 50, 55), _box(400, 400, 490, 490)]
    pipe = DetectRecognize(
        FakeDetector(boxes), FakeRecognizer(), make_grouper({"method": "dbscan", "eps": 80}), crop_padding=0
    )
    result = pipe.run(_page(tmp_path, size=(500, 500)))
    assert sorted(r["insertion_polygon"] for r in result) == [[10, 10, 50, 55], [400, 400, 490, 490]]


def test_detect_recognize_skips_zero_area_boxes(tmp_path):
    rec = FakeRecognizer()
    pipe = DetectRecognize(FakeDetector([_box(10, 10, 10, 40)]), rec, NONE, crop_padding=0)
    assert pipe.run(_page(tmp_path)) == []
    assert rec.crops == []


def test_detect_recognize_converts_page_to_rgb(tmp_path):
    det = FakeDetector([])
    DetectRecognize(det, FakeRecognizer(), NONE).run(_page(tmp_path, mode="L"))
    assert det.modes == ["RGB"]


def test_detect_recognize_close_closes_components():
    det, rec = FakeDetector(), FakeRecognizer()
    DetectRecognize(det, rec, NONE).close()
    assert det.closed and rec.closed


# --- Spot -----------------------------------------------------------------

def test_spot_empty_returns_empty(tmp_path):
    assert Spot(FakeSpotter([]), NONE).run(_page(tmp_path)) == []


def test_spot_passes_through_boxes_with_none_grouping(tmp_path):
    pipe = Spot(FakeSpotter([_box(1, 2, 3, 4, "A")]), NONE)
    assert pipe.run(_page(tmp_path)) == [{"text": "A", "insertion_polygon": [1, 2, 3, 4]}]


def test_spot_merges_text_and_box_of_grouped_lines(tmp_path):
    lines = [_box(10, 10, 50, 30, "A"), _box(10, 35, 50, 55, "B")]
    pipe = Spot(FakeSpotter(lines), make_grouper({"method": "dbscan", "eps": 80}))
    assert pipe.run(_page(tmp_path, size=(500, 500))) == [
        {"text": "A B", "insertion_polygon": [10, 10, 50, 55]}
    ]


def test_spot_converts_page_to_rgb(tmp_path):
    seen = []

    class Recording(FakeSpotter):
        def spot(self, pil_image):
            seen.append(pil_image.mode)
            return []

    Spot(Recording(), NONE).run(_page(tmp_path, mode="L"))
    assert seen == ["RGB"]


def test_spot_close_closes_spotter():
    spotter = FakeSpotter()
    Spot(spotter, NONE).close()
    assert spotter.closed


# --- build_pipeline -------------------------------------------------------

@pytest.fixture
def fakes(isolated_registry):
    _BUILT.clear()
    _DETECTORS.clear()
    registry.register("detector", "fakedet")(FakeDetector)
    registry.register("recognizer", "fakerec")(FakeRecognizer)
    registry.register("spotter", "fakespot")(FakeSpotter)


def test_build_detect_recognize(fakes):
    pipe = build_pipeline({
        "pipeline": "detect_recognize",
        "detector": {"name": "fakedet"},
        "recognizer": {"name": "fakerec", "text": "hi", "crop_padding": 7},
        "grouping": {"method": "none"},
    })
    assert isinstance(pipe, DetectRecognize)
    assert pipe._crop_padding == 7          # removed from the recognizer params
    assert pipe._recognizer.text == "hi"


def test_build_detect_recognize_default_crop_padding(fakes):
    pipe = build_pipeline({
        "pipeline": "detect_recognize",
        "detector": {"name": "fakedet"},
        "recognizer": {"name": "fakerec"},
    })
    assert pipe._crop_padding == 10


def test_build_spot_with_dbscan(fakes):
    pipe = build_pipeline({
        "pipeline": "spot",
        "spotter": {"name": "fakespot"},
        "grouping": {"method": "dbscan", "eps": 80},
    })
    assert isinstance(pipe, Spot)


@pytest.mark.parametrize("config, match", [
    (None, "must contain an 'ocr' block"),
    ("yolo", "must contain an 'ocr' block"),
    ({}, "Invalid ocr.pipeline"),
    ({"pipeline": "nope"}, "Invalid ocr.pipeline 'nope'"),
    ({"pipeline": "spot"}, "requires blocks"),
    ({"pipeline": "detect_recognize", "detector": {"name": "fakedet"}}, "requires blocks"),
    ({"pipeline": "spot", "spotter": {"name": "fakespot"}, "detector": {"name": "fakedet"}},
     "Unexpected ocr blocks"),
    ({"pipeline": "spot", "spotter": "fakespot"}, "must be an object"),
    ({"pipeline": "spot", "spotter": {}}, "needs a 'name'"),
    ({"pipeline": "spot", "spotter": {"name": "missing"}}, "Unknown spotter 'missing'"),
    ({"pipeline": "spot", "spotter": {"name": "fakespot", "bogus": 1}}, "Unknown params"),
    ({"pipeline": "spot", "spotter": {"name": "fakespot"}, "grouping": {"method": "kmeans"}},
     "Unknown grouping method"),
    ({"pipeline": "spot", "spotter": {"name": "fakespot"}, "grouping": {"method": "dbscan", "epz": 1}},
     "Unknown grouping params"),
])
def test_build_pipeline_config_errors(fakes, config, match):
    with pytest.raises(ValueError, match=match):
        build_pipeline(config)


def test_invalid_grouping_is_rejected_before_any_model_is_built(fakes):
    with pytest.raises(ValueError):
        build_pipeline({
            "pipeline": "spot",
            "spotter": {"name": "fakespot"},
            "grouping": {"method": "kmeans"},
        })
    assert _BUILT == []


def test_build_pipeline_does_not_mutate_the_config(fakes):
    config = {
        "pipeline": "detect_recognize",
        "detector": {"name": "fakedet"},
        "recognizer": {"name": "fakerec", "crop_padding": 7},
    }
    build_pipeline(config)
    assert config["recognizer"] == {"name": "fakerec", "crop_padding": 7}
    assert config["detector"] == {"name": "fakedet"}


@pytest.mark.parametrize("recognizer, match", [
    ({"name": "missing"}, "Unknown recognizer 'missing'"),
    ({"name": "fakerec", "bogus": 1}, "Unknown params"),
])
def test_bad_recognizer_is_rejected_before_detector_is_built(fakes, recognizer, match):
    with pytest.raises(ValueError, match=match):
        build_pipeline({
            "pipeline": "detect_recognize",
            "detector": {"name": "fakedet"},
            "recognizer": recognizer,
        })
    assert _BUILT == []


def test_detector_is_closed_if_recognizer_construction_fails(fakes):
    class Boom(Recognizer):
        def __init__(self):
            raise RuntimeError("boom")

        def read(self, crop):
            return ""

    registry.register("recognizer", "boom")(Boom)
    with pytest.raises(RuntimeError, match="boom"):
        build_pipeline({
            "pipeline": "detect_recognize",
            "detector": {"name": "fakedet"},
            "recognizer": {"name": "boom"},
        })
    assert len(_DETECTORS) == 1 and _DETECTORS[0].closed


def test_non_dict_recognizer_block_rejected(fakes):
    with pytest.raises(ValueError, match="must be an object"):
        build_pipeline({
            "pipeline": "detect_recognize",
            "detector": {"name": "fakedet"},
            "recognizer": "fakerec",
        })
    assert _BUILT == []


# --- overlap grouping through the pipeline --------------------------------

def test_detect_recognize_reads_overlapping_boxes_once(tmp_path):
    rec = FakeRecognizer(text="hello")
    nested = [_box(20, 20, 80, 80), _box(30, 30, 50, 50), _box(25, 25, 85, 70)]
    pipe = DetectRecognize(
        FakeDetector(nested), rec, make_grouper({"method": "overlap"}), crop_padding=0
    )
    assert pipe.run(_page(tmp_path)) == [{"text": "hello", "insertion_polygon": [20, 20, 85, 80]}]
    assert rec.crops == [(65, 60)]
