from pathlib import Path
import urllib.request
from tqdm import tqdm
import cv2
import numpy as np
import onnxruntime as ort

MODEL_URL = "https://huggingface.co/mayocream/lama-manga-onnx/resolve/main/lama-manga.onnx"
MODEL_PATH = Path.home() / ".manga-polyglot" / "models" / "lama-manga.onnx"


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tqdm(unit="B", unit_scale=True, miniters=1, desc="Downloading lama-manga.onnx") as t:
            def _reporthook(count, block_size, total_size):
                if t.total is None and total_size > 0:
                    t.total = total_size
                t.update(block_size)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, _reporthook)
    return MODEL_PATH


def build_text_mask(boxes: list[dict], img_w: int, img_h: int) -> np.ndarray:
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box["insertion_polygon"])
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    kernel_size = max(11, int(0.04 * max(img_w, img_h)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


class LamaInpainter:
    def __init__(self, model_path: Path):
        providers = ["CPUExecutionProvider"]
        try:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
        self._session = ort.InferenceSession(str(model_path), providers=providers)

    def infer(self, img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        img = img_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]       # [1, 3, H, W]
        msk = (mask.astype(np.float32) / 255.0)[np.newaxis, np.newaxis]  # [1, 1, H, W]

        inputs = self._session.get_inputs()
        feeds = {inputs[0].name: img, inputs[1].name: msk}
        output = self._session.run(None, feeds)[0]            # [1, 3, H, W]
        result = np.clip(output[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        return result
