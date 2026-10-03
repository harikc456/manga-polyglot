import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from ocr import registry
from ocr.detectors.yolo import YoloTextDetector


class _Xyxy:
    def __init__(self, coords):
        self._coords = coords

    def tolist(self):
        return self._coords


def _box(cls, coords):
    return SimpleNamespace(cls=cls, conf=0.9, xyxy=[_Xyxy(coords)])


def _make(boxes, **kwargs):
    model = MagicMock()
    model.predict.return_value = [SimpleNamespace(boxes=boxes)]
    ultralytics = MagicMock()
    ultralytics.YOLO = MagicMock(return_value=model)
    hub = MagicMock()
    hub.hf_hub_download = MagicMock(return_value="/w/best.pt")
    with patch.dict(sys.modules, {"ultralytics": ultralytics, "huggingface_hub": hub}):
        detector = YoloTextDetector(**kwargs)
    return detector, model, ultralytics, hub


def test_registered_as_builtin_yolo():
    assert "yolo" in registry.available("detector")
    assert registry._REGISTRY["detector"]["yolo"] is YoloTextDetector


def test_downloads_default_weights_and_loads_them():
    _, _, ultralytics, hub = _make([])
    hub.hf_hub_download.assert_called_once_with("lordtrilink/manga-text-detector-v0", "best.pt")
    ultralytics.YOLO.assert_called_once_with("/w/best.pt")


def test_predict_uses_model_card_settings_on_the_unmodified_image():
    detector, model, _, _ = _make([])
    image = Image.new("RGB", (200, 300))
    detector.detect(image)
    model.predict.assert_called_once_with(image, imgsz=1024, conf=0.05, iou=0.7, verbose=False)


def test_custom_settings_are_forwarded():
    detector, model, _, _ = _make([], conf=0.3, iou=0.5, imgsz=1536)
    detector.detect(Image.new("RGB", (200, 300)))
    kwargs = model.predict.call_args.kwargs
    assert (kwargs["conf"], kwargs["iou"], kwargs["imgsz"]) == (0.3, 0.5, 1536)


def test_returns_only_text_region_boxes_by_default():
    boxes = [_box(0, [10.4, 20.9, 50.2, 60.7]), _box(1, [70, 70, 90, 90])]
    detector, _, _, _ = _make(boxes)
    assert detector.detect(Image.new("RGB", (100, 100))) == [
        {"x_min": 10, "y_min": 20, "x_max": 50, "y_max": 60}
    ]


def test_include_sfx_keeps_both_classes():
    boxes = [_box(0, [10, 10, 20, 20]), _box(1, [30, 30, 40, 40])]
    detector, _, _, _ = _make(boxes, include_sfx=True)
    assert len(detector.detect(Image.new("RGB", (100, 100)))) == 2


def test_boxes_are_clamped_to_the_image():
    detector, _, _, _ = _make([_box(0, [-5, -8, 150, 120])])
    assert detector.detect(Image.new("RGB", (100, 100))) == [
        {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100}
    ]


def test_zero_area_and_fully_outside_boxes_are_dropped():
    boxes = [
        _box(0, [10, 10, 10, 50]),     # zero width
        _box(0, [10, 10, 50, 10]),     # zero height
        _box(0, [120, 120, 150, 150]), # fully outside a 100x100 image
    ]
    detector, _, _, _ = _make(boxes)
    assert detector.detect(Image.new("RGB", (100, 100))) == []


def test_no_detections_returns_empty_list():
    detector, _, _, _ = _make([])
    assert detector.detect(Image.new("RGB", (100, 100))) == []


def test_close_drops_model_reference():
    detector, _, _, _ = _make([])
    detector.close()
    assert detector._model is None
