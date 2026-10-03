"""YOLO11s manga text detector (lordtrilink/manga-text-detector-v0).

Classes: 0 = text-region, 1 = sfx. One box per text region, so it normally
pairs with grouping method "none".

License: the model weights are CC BY-NC-SA 4.0 (non-commercial use only).

The image is passed to the model as-is. Do not apply EXIF transposition here:
the rest of the pipeline (cleaning, drawing) opens pages without transposing,
so boxes must stay in that coordinate frame.
"""
from ocr.base import Detector
from ocr.registry import register

DEFAULT_REPO = "lordtrilink/manga-text-detector-v0"
TEXT_REGION = 0
SFX = 1


@register("detector", "yolo")
class YoloTextDetector(Detector):
    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        filename: str = "best.pt",
        conf: float = 0.05,
        iou: float = 0.7,
        imgsz: int = 1024,
        include_sfx: bool = False,
    ):
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        self._model = YOLO(hf_hub_download(repo, filename))
        self._predict_args = {"imgsz": imgsz, "conf": conf, "iou": iou, "verbose": False}
        self._keep = {TEXT_REGION, SFX} if include_sfx else {TEXT_REGION}

    def detect(self, pil_image) -> list[dict]:
        img_w, img_h = pil_image.size
        result = self._model.predict(pil_image, **self._predict_args)[0]
        boxes = []
        for box in result.boxes:
            if int(box.cls) not in self._keep:
                continue
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(img_w, int(x1)), min(img_h, int(y1))
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append({"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1})
        return boxes

    def close(self) -> None:
        self._model = None
