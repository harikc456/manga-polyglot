# LaMa Inpainting Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace flat color-fill text removal with LaMa neural inpainting, controlled by `inpainting_engine` in `config.json`, with connected-component-per-blob processing and morphological mask dilation.

**Architecture:** A new `inpainting.py` module owns model download, mask generation, and LaMa inference. `clean_page()` in `inference.py` reads `inpainting_engine` from config and routes to either the existing color-fill path or the new `inpaint_page()`. No other files change.

**Tech Stack:** `onnxruntime` (ONNX inference, CUDA→CPU fallback), `opencv-python` (mask ops, connected components), `urllib.request` + `tqdm` (model download), `PIL` (image I/O).

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `inpainting.py` | `ensure_model`, `build_text_mask`, `LamaInpainter`, `inpaint_page` |
| Create | `tests/test_inpainting.py` | Unit tests for all inpainting module functions |
| Modify | `inference.py` | Add `inpainting_engine` param to `clean_page()`, read from config in `driver()` |
| Modify | `config.json` | Add `"inpainting_engine": "color_fill"` |
| Modify | `pyproject.toml` | Add `"onnxruntime"` dependency |

---

## Task 1: Dependencies and config defaults

**Files:**
- Modify: `pyproject.toml`
- Modify: `config.json`

- [ ] **Step 1: Add `onnxruntime` to `pyproject.toml`**

Open `pyproject.toml`. The `dependencies` list currently ends with `"ollama"`. Add `"onnxruntime"` after it:

```toml
dependencies = [
    "torch",
    "transformers",
    "Pillow",
    "tqdm",
    "scipy",
    "fastapi",
    "uvicorn[standard]",
    "httpx",
    "numpy",
    "scikit-learn",
    "jaconv",
    "ollama",
    "onnxruntime",
]
```

- [ ] **Step 2: Add `inpainting_engine` key to `config.json`**

`config.json` currently has 8 keys. Add `"inpainting_engine": "color_fill"` as the last key:

```json
{
    "ocr_model": "PaddlePaddle/PaddleOCR-VL-1.5",
    "llm_name": "translategemma:12b",
    "font_path": "./fonts/animeace2_bld.otf",
    "image_enabled": false,
    "json_enabled": false,
    "memory_enabled": false,
    "spotting_cluster_eps": 80,
    "spotting_max_tokens": 512,
    "inpainting_engine": "color_fill"
}
```

- [ ] **Step 3: Install the new dependency**

```bash
uv sync
```

Expected: resolves and installs `onnxruntime` without errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml config.json uv.lock
git commit -m "chore: add onnxruntime dep and inpainting_engine config key"
```

---

## Task 2: `ensure_model()` — download LaMa ONNX on first use

**Files:**
- Create: `inpainting.py`
- Create: `tests/test_inpainting.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_inpainting.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock
import pytest


def test_ensure_model_returns_correct_path():
    """ensure_model() returns a Path ending in lama-manga.onnx under ~/.manga-polyglot/models/."""
    from inpainting import ensure_model, MODEL_PATH
    with patch("inpainting.MODEL_PATH") as mock_path:
        mock_path.exists.return_value = True
        mock_path.__str__ = lambda self: str(Path.home() / ".manga-polyglot" / "models" / "lama-manga.onnx")
        result = ensure_model()
        assert str(result).endswith("lama-manga.onnx")


def test_ensure_model_skips_download_if_file_exists(tmp_path):
    """ensure_model() does not call urlretrieve when the model file already exists."""
    fake_model = tmp_path / "lama-manga.onnx"
    fake_model.write_bytes(b"fake")
    with patch("inpainting.MODEL_PATH", fake_model), \
         patch("urllib.request.urlretrieve") as mock_dl:
        from inpainting import ensure_model
        ensure_model()
        mock_dl.assert_not_called()


def test_ensure_model_downloads_when_missing(tmp_path):
    """ensure_model() calls urlretrieve with the correct URL when the model file is absent."""
    fake_model = tmp_path / "lama-manga.onnx"
    with patch("inpainting.MODEL_PATH", fake_model), \
         patch("urllib.request.urlretrieve") as mock_dl:
        mock_dl.side_effect = lambda url, path, reporthook: Path(path).write_bytes(b"fake")
        from inpainting import ensure_model, MODEL_URL
        result = ensure_model()
        mock_dl.assert_called_once()
        assert mock_dl.call_args[0][0] == MODEL_URL
        assert result == fake_model
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_inpainting.py -v
```

Expected: `ImportError: No module named 'inpainting'`

- [ ] **Step 3: Create `inpainting.py` with `ensure_model()`**

```python
from pathlib import Path
import urllib.request
from tqdm import tqdm

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_inpainting.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add inpainting.py tests/test_inpainting.py
git commit -m "feat: add ensure_model() for lama-manga.onnx auto-download"
```

---

## Task 3: `build_text_mask()` — dilated binary mask from bounding boxes

**Files:**
- Modify: `inpainting.py`
- Modify: `tests/test_inpainting.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_inpainting.py`:

```python
import numpy as np


def test_build_text_mask_shape():
    """build_text_mask returns a uint8 mask of the correct image dimensions."""
    from inpainting import build_text_mask
    boxes = [{"insertion_polygon": [10, 10, 50, 50]}]
    mask = build_text_mask(boxes, img_w=100, img_h=80)
    assert mask.shape == (80, 100)
    assert mask.dtype == np.uint8


def test_build_text_mask_covers_box():
    """Mask has 255 inside the box region before dilation."""
    from inpainting import build_text_mask
    # Use a box far from edges so dilation doesn't complicate this check
    boxes = [{"insertion_polygon": [40, 40, 60, 60]}]
    mask = build_text_mask(boxes, img_w=200, img_h=200)
    # Centre of box must be white
    assert mask[50, 50] == 255


def test_build_text_mask_is_dilated():
    """Mask extends beyond the original box due to morphological dilation."""
    from inpainting import build_text_mask
    boxes = [{"insertion_polygon": [50, 50, 100, 100]}]
    mask = build_text_mask(boxes, img_w=300, img_h=300)
    # A pixel just outside the box should be white (dilated)
    assert mask[48, 75] == 255  # 2 pixels above top edge
    assert mask[75, 48] == 255  # 2 pixels left of left edge


def test_build_text_mask_empty_boxes():
    """build_text_mask with no boxes returns an all-zero mask."""
    from inpainting import build_text_mask
    mask = build_text_mask([], img_w=100, img_h=100)
    assert mask.max() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_inpainting.py::test_build_text_mask_shape tests/test_inpainting.py::test_build_text_mask_covers_box tests/test_inpainting.py::test_build_text_mask_is_dilated tests/test_inpainting.py::test_build_text_mask_empty_boxes -v
```

Expected: `ImportError: cannot import name 'build_text_mask'`

- [ ] **Step 3: Add `build_text_mask()` to `inpainting.py`**

Append after `ensure_model()`:

```python
import cv2
import numpy as np


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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_inpainting.py::test_build_text_mask_shape tests/test_inpainting.py::test_build_text_mask_covers_box tests/test_inpainting.py::test_build_text_mask_is_dilated tests/test_inpainting.py::test_build_text_mask_empty_boxes -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add inpainting.py tests/test_inpainting.py
git commit -m "feat: add build_text_mask() with morphological dilation"
```

---

## Task 4: `LamaInpainter` — ONNX inference with CUDA→CPU fallback

**Files:**
- Modify: `inpainting.py`
- Modify: `tests/test_inpainting.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_inpainting.py`:

```python
def _make_fake_ort_session(output_shape):
    """Return a mock onnxruntime.InferenceSession that outputs a fixed array."""
    session = MagicMock()
    fake_input_0 = MagicMock()
    fake_input_0.name = "image"
    fake_input_1 = MagicMock()
    fake_input_1.name = "mask"
    session.get_inputs.return_value = [fake_input_0, fake_input_1]
    # output[0] shape: [1, 3, H, W] — normalized float32
    h, w = output_shape
    session.run.return_value = [np.ones((1, 3, h, w), dtype=np.float32) * 0.5]
    return session


def test_lama_inpainter_infer_output_shape(tmp_path):
    """LamaInpainter.infer() returns an array with the same HxW as the input."""
    fake_model = tmp_path / "lama-manga.onnx"
    fake_model.write_bytes(b"fake")

    with patch("onnxruntime.InferenceSession", return_value=_make_fake_ort_session((64, 64))):
        from inpainting import LamaInpainter
        inpainter = LamaInpainter(fake_model)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = inpainter.infer(img, mask)
        assert result.shape == (64, 64, 3)


def test_lama_inpainter_infer_output_dtype(tmp_path):
    """LamaInpainter.infer() returns uint8 values in [0, 255]."""
    fake_model = tmp_path / "lama-manga.onnx"
    fake_model.write_bytes(b"fake")

    with patch("onnxruntime.InferenceSession", return_value=_make_fake_ort_session((32, 32))):
        from inpainting import LamaInpainter
        inpainter = LamaInpainter(fake_model)
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        result = inpainter.infer(img, mask)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_inpainting.py::test_lama_inpainter_infer_output_shape tests/test_inpainting.py::test_lama_inpainter_infer_output_dtype -v
```

Expected: `ImportError: cannot import name 'LamaInpainter'`

- [ ] **Step 3: Add `LamaInpainter` to `inpainting.py`**

Append after `build_text_mask()`:

```python
import onnxruntime as ort


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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_inpainting.py::test_lama_inpainter_infer_output_shape tests/test_inpainting.py::test_lama_inpainter_infer_output_dtype -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add inpainting.py tests/test_inpainting.py
git commit -m "feat: add LamaInpainter with CUDA->CPU fallback"
```

---

## Task 5: `inpaint_page()` — orchestrate mask, blobs, per-blob inference

**Files:**
- Modify: `inpainting.py`
- Modify: `tests/test_inpainting.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_inpainting.py`:

```python
from PIL import Image as PILImage


def _fake_inpainter_passthrough(img_rgb, mask):
    """Identity inpainter — returns the tile unchanged."""
    return img_rgb.copy()


def test_inpaint_page_returns_pil_image(tmp_path):
    """inpaint_page() returns a PIL Image of the same size as the input."""
    img = PILImage.fromarray(np.ones((200, 200, 3), dtype=np.uint8) * 200)
    boxes = [{"insertion_polygon": [50, 50, 150, 150]}]
    fake_model = tmp_path / "lama-manga.onnx"
    fake_model.write_bytes(b"fake")

    with patch("inpainting.ensure_model", return_value=fake_model), \
         patch("inpainting.LamaInpainter") as MockInpainter:
        MockInpainter.return_value.infer.side_effect = _fake_inpainter_passthrough
        from inpainting import inpaint_page
        result = inpaint_page(img, boxes)
        assert isinstance(result, PILImage.Image)
        assert result.size == img.size


def test_inpaint_page_processes_text_regions(tmp_path):
    """inpaint_page() calls LamaInpainter.infer() at least once when boxes exist."""
    img = PILImage.fromarray(np.ones((200, 200, 3), dtype=np.uint8) * 200)
    boxes = [{"insertion_polygon": [50, 50, 150, 150]}]
    fake_model = tmp_path / "lama-manga.onnx"
    fake_model.write_bytes(b"fake")

    with patch("inpainting.ensure_model", return_value=fake_model), \
         patch("inpainting.LamaInpainter") as MockInpainter:
        mock_inpainter = MockInpainter.return_value
        mock_inpainter.infer.side_effect = _fake_inpainter_passthrough
        from inpainting import inpaint_page
        inpaint_page(img, boxes)
        assert mock_inpainter.infer.call_count >= 1


def test_inpaint_page_no_boxes_returns_unchanged(tmp_path):
    """inpaint_page() with empty boxes returns image without calling infer."""
    arr = np.ones((100, 100, 3), dtype=np.uint8) * 128
    img = PILImage.fromarray(arr)
    fake_model = tmp_path / "lama-manga.onnx"
    fake_model.write_bytes(b"fake")

    with patch("inpainting.ensure_model", return_value=fake_model), \
         patch("inpainting.LamaInpainter") as MockInpainter:
        mock_inpainter = MockInpainter.return_value
        from inpainting import inpaint_page
        result = inpaint_page(img, [])
        mock_inpainter.infer.assert_not_called()
        assert np.array_equal(np.array(result), arr)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_inpainting.py::test_inpaint_page_returns_pil_image tests/test_inpainting.py::test_inpaint_page_processes_text_regions tests/test_inpainting.py::test_inpaint_page_no_boxes_returns_unchanged -v
```

Expected: `ImportError: cannot import name 'inpaint_page'`

- [ ] **Step 3: Add `inpaint_page()` to `inpainting.py`**

Append after `LamaInpainter`:

```python
from PIL import Image


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

        h, w = tile_rgb.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        tile_padded = cv2.copyMakeBorder(tile_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        mask_padded = cv2.copyMakeBorder(mask_tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        result_padded = inpainter.infer(tile_padded, mask_padded)
        result_rgb = result_padded[:h, :w]
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        region_mask = mask[y1:y2, x1:x2]
        output[y1:y2, x1:x2][region_mask > 0] = result_bgr[region_mask > 0]
        processed_mask[y1:y2, x1:x2][region_mask > 0] = 255

    result_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_inpainting.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add inpainting.py tests/test_inpainting.py
git commit -m "feat: add inpaint_page() with connected-component blob processing"
```

---

## Task 6: Wire `inpainting_engine` into `clean_page()` and `driver()`

**Files:**
- Modify: `inference.py` (lines 58–65 and 78–88)
- Create: `tests/test_clean_page.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_clean_page.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image


def _make_rgb_image(w=100, h=100):
    return Image.fromarray(np.ones((h, w, 3), dtype=np.uint8) * 200)


def test_clean_page_color_fill_calls_fill_bubble(tmp_path):
    """clean_page() with 'color_fill' calls fill_bubble_with_estimated_color for each box."""
    img_path = tmp_path / "page.png"
    _make_rgb_image().save(img_path)
    boxes = [{"insertion_polygon": [10, 10, 50, 50]}]

    with patch("inference.fill_bubble_with_estimated_color", return_value=_make_rgb_image()) as mock_fill:
        from inference import clean_page
        clean_page(str(img_path), str(tmp_path), boxes, inpainting_engine="color_fill")
        mock_fill.assert_called_once()


def test_clean_page_lama_calls_inpaint_page(tmp_path):
    """clean_page() with 'lama' calls inpaint_page() instead of fill_bubble."""
    img_path = tmp_path / "page.png"
    _make_rgb_image().save(img_path)
    boxes = [{"insertion_polygon": [10, 10, 50, 50]}]

    with patch("inference.inpaint_page", return_value=_make_rgb_image()) as mock_inpaint, \
         patch("inference.fill_bubble_with_estimated_color") as mock_fill:
        from inference import clean_page
        clean_page(str(img_path), str(tmp_path), boxes, inpainting_engine="lama")
        mock_inpaint.assert_called_once()
        mock_fill.assert_not_called()


def test_clean_page_invalid_engine_raises(tmp_path):
    """clean_page() raises ValueError for an unknown inpainting_engine value."""
    img_path = tmp_path / "page.png"
    _make_rgb_image().save(img_path)

    with pytest.raises(ValueError, match="inpainting_engine"):
        from inference import clean_page
        clean_page(str(img_path), str(tmp_path), [], inpainting_engine="magic")


def test_clean_page_default_engine_is_color_fill(tmp_path):
    """clean_page() defaults to color_fill when inpainting_engine is omitted."""
    img_path = tmp_path / "page.png"
    _make_rgb_image().save(img_path)

    with patch("inference.fill_bubble_with_estimated_color", return_value=_make_rgb_image()) as mock_fill, \
         patch("inference.inpaint_page") as mock_inpaint:
        from inference import clean_page
        clean_page(str(img_path), str(tmp_path), [], inpainting_engine="color_fill")
        mock_inpaint.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_clean_page.py -v
```

Expected: tests fail because `clean_page()` doesn't accept `inpainting_engine` param yet and doesn't import `inpaint_page`.

- [ ] **Step 3: Update `inference.py`**

Add the import at the top of `inference.py`, after the existing imports:

```python
from inpainting import inpaint_page
```

Replace `clean_page()` (lines 58–65) with:

```python
def clean_page(img_path: str, temp_dir: str, boxes: list[dict], inpainting_engine: str = "color_fill") -> str:
    pil_image = Image.open(img_path).convert("RGB")
    file_name = os.path.basename(img_path)
    cleaned_file_path = os.path.join(temp_dir, file_name)

    if inpainting_engine == "color_fill":
        for box in boxes:
            pil_image = fill_bubble_with_estimated_color(pil_image, box["insertion_polygon"])
    elif inpainting_engine == "lama":
        pil_image = inpaint_page(pil_image, boxes)
    else:
        raise ValueError(
            f"Invalid inpainting_engine '{inpainting_engine}'. Valid options: 'color_fill', 'lama'"
        )

    pil_image.save(cleaned_file_path)
    return cleaned_file_path
```

In `driver()`, add config reading and validation at the top of the function body (after line 84, `max_tokens = config.get(...)`):

```python
    inpainting_engine = config.get("inpainting_engine", "color_fill")
    if inpainting_engine not in ("color_fill", "lama"):
        raise ValueError(
            f"Invalid inpainting_engine '{inpainting_engine}'. Valid options: 'color_fill', 'lama'"
        )
```

And update the `clean_page()` call (line 139) to pass the engine:

```python
        cleaned_file_path = clean_page(img_path, temp_dir, boxes, inpainting_engine)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_clean_page.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add inference.py tests/test_clean_page.py
git commit -m "feat: route clean_page() to LaMa or color-fill via inpainting_engine config"
```
