"""Interfaces for pluggable OCR components.

Box dicts use pixel coordinates in the page's own frame:
  detector boxes: {x_min, y_min, x_max, y_max}
  spotter boxes:  {text, x_min, y_min, x_max, y_max}
  pipeline output: {text, insertion_polygon: [x_min, y_min, x_max, y_max]}
"""


class Detector:
    def detect(self, pil_image) -> list[dict]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class Recognizer:
    def read(self, crop) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass


class Spotter:
    def spot(self, pil_image) -> list[dict]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class Pipeline:
    def run(self, img_path: str) -> list[dict]:
        raise NotImplementedError

    def close(self) -> None:
        pass
