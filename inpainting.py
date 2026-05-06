from pathlib import Path
import urllib.request
from tqdm import tqdm
from PIL import Image
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
        orig_h, orig_w = img_rgb.shape[:2]

        img_512 = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        msk_512 = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        img = img_512.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]          # [1, 3, 512, 512]
        msk = (msk_512.astype(np.float32) / 255.0)[np.newaxis, np.newaxis]  # [1, 1, 512, 512]

        inputs = self._session.get_inputs()
        feeds = {inputs[0].name: img, inputs[1].name: msk}
        output = self._session.run(None, feeds)[0]               # [1, 3, 512, 512]
        result_512 = np.clip(output[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)

        return cv2.resize(result_512, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def inpaint_page(pil_image: Image.Image, boxes: list[dict]) -> Image.Image:
    model_path = ensure_model()
    img_w, img_h = pil_image.size

    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    output = img_bgr.copy()

    mask = build_text_mask(boxes, img_w, img_h)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    blobs = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= 5:
            blobs.append((area, lbl, stats[lbl], centroids[lbl]))
    blobs.sort(key=lambda x: x[0], reverse=True)

    if not blobs:
        return pil_image

    inpainter = LamaInpainter(model_path)
    processed_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for area, lbl, stat, centroid in blobs:
        cx, cy = int(centroid[0]), int(centroid[1])
        if processed_mask[cy, cx] > 0:
            continue

        pad = 32
        x1 = max(0, stat[cv2.CC_STAT_LEFT] - pad)
        y1 = max(0, stat[cv2.CC_STAT_TOP] - pad)
        x2 = min(img_w, stat[cv2.CC_STAT_LEFT] + stat[cv2.CC_STAT_WIDTH] + pad)
        y2 = min(img_h, stat[cv2.CC_STAT_TOP] + stat[cv2.CC_STAT_HEIGHT] + pad)

        tile_bgr = img_bgr[y1:y2, x1:x2]
        mask_tile = mask[y1:y2, x1:x2]
        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)

        result_rgb = inpainter.infer(tile_rgb, mask_tile)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        region_mask = mask[y1:y2, x1:x2]
        output[y1:y2, x1:x2][region_mask > 0] = result_bgr[region_mask > 0]
        processed_mask[y1:y2, x1:x2][region_mask > 0] = 255

    result_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)
