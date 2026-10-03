from PIL import Image

from ocr.base import Pipeline
from ocr.grouping import union_box


def _load_rgb(img_path: str):
    with Image.open(img_path) as im:
        return im.convert("RGB")


class DetectRecognize(Pipeline):
    """Detector -> grouping -> crop (with padding) -> Recognizer."""

    def __init__(self, detector, recognizer, grouper, crop_padding: int = 10):
        self._detector = detector
        self._recognizer = recognizer
        self._grouper = grouper
        self._crop_padding = crop_padding

    def run(self, img_path: str) -> list[dict]:
        image = _load_rgb(img_path)
        img_w, img_h = image.size
        boxes = self._detector.detect(image)
        result = []
        for group in self._grouper(boxes, img_w, img_h):
            x0, y0, x1, y1 = union_box(group)
            x0 = max(0, x0 - self._crop_padding)
            y0 = max(0, y0 - self._crop_padding)
            x1 = min(img_w, x1 + self._crop_padding)
            y1 = min(img_h, y1 + self._crop_padding)
            if x1 <= x0 or y1 <= y0:
                continue
            text = self._recognizer.read(image.crop((x0, y0, x1, y1)))
            result.append({"text": text, "insertion_polygon": [x0, y0, x1, y1]})
        return result

    def close(self) -> None:
        self._detector.close()
        self._recognizer.close()


class Spot(Pipeline):
    """Spotter (detect + read in one pass) -> grouping."""

    def __init__(self, spotter, grouper):
        self._spotter = spotter
        self._grouper = grouper

    def run(self, img_path: str) -> list[dict]:
        image = _load_rgb(img_path)
        img_w, img_h = image.size
        boxes = self._spotter.spot(image)
        result = []
        for group in self._grouper(boxes, img_w, img_h):
            x0, y0, x1, y1 = union_box(group)
            text = " ".join(b["text"] for b in group)
            result.append({"text": text, "insertion_polygon": [x0, y0, x1, y1]})
        return result

    def close(self) -> None:
        self._spotter.close()
