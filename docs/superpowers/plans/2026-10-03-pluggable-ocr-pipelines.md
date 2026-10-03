# Pluggable OCR Pipelines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hard-coded PaddleOCR-VL spotting flow with config-selected, registry-based OCR pipelines (`detect_recognize` and `spot`), and add a YOLO manga text detector as the first detector.

**Architecture:** A new `ocr/` package defines `Detector`, `Recognizer`, `Spotter` and `Pipeline` interfaces, a lazy-importing registry, a grouping step (`none` | `dbscan`), and two pipeline classes. `inference.py::driver()` builds one pipeline from `config["ocr"]`, calls `pipeline.run(img_path)` on a cache miss, and keeps everything downstream (sorting, color-fill cleaning, translation, drawing) unchanged. `ocr_utils.py` is removed.

**Tech Stack:** Python 3.13, PIL, scikit-learn (DBSCAN), transformers (PaddleOCR-VL), ultralytics + huggingface_hub (YOLO), pytest.

**Spec:** `docs/superpowers/specs/2026-10-03-pluggable-ocr-pipelines-design.md`

## Global Constraints

- Work only in the worktree `/home/harikrishnan-c/projects/manga-polyglot-ocr` (branch `feat/pluggable-ocr-pipelines`, based on `main`). Do NOT touch `/home/harikrishnan-c/projects/manga-polyglot` (it holds the user's uncommitted work on `feat/lama-inpainting`, PR #19).
- Nothing from PR #19 exists on this baseline: no Paddle `TextDetection` detector, no bubble expansion, no LaMa. `clean_page` stays color-fill and is not modified.
- Run tests with the main checkout's venv (the worktree has no `.venv`): `/home/harikrishnan-c/projects/manga-polyglot/.venv/bin/python -m pytest <args>` from the worktree root. Below this is written as `PYTEST`.
- `ultralytics` and `huggingface_hub` must only be imported inside `YoloTextDetector.__init__` (lazy), never at module top level, so other configs and the test suite don't need them.
- YOLO detector must NOT call `ImageOps.exif_transpose` (boxes must stay in the coordinate frame used by cleaning/drawing).
- Config shape (exact keys): `ocr.pipeline` is `"detect_recognize"` or `"spot"`; blocks are `detector`, `recognizer`, `spotter`, `grouping`; `crop_padding` is read from the `recognizer` block; grouping methods are `"none"` and `"dbscan"` (`eps` default 80, thousandths of the longer image side).
- Old flat keys `ocr_model`, `spotting_cluster_eps`, `spotting_max_tokens` are removed, not supported.
- Use `uv` for dependency changes. `main` does not track `uv.lock`; never `git add` a `uv.lock`.
- Stage files explicitly by path (never `git add -A`). Every commit message ends with the trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` (pass it as a second `-m`).
- The YOLO model is CC BY-NC-SA 4.0 (non-commercial); keep the note in the module docstring and README.

## Review Focus

Failure modes the spec implies but a casual implementation misses (each is pinned by a test in the named task):

1. A page with no detections must produce `[]` and the driver must still save the page without calling `translate` (Task 4 pipeline tests, Task 5 driver test).
2. YOLO returns a box partly outside the image, or with zero area after clamping: clamp it, or drop it (Task 3).
3. Grayscale/palette pages (non-RGB PNG/JPEG): the pipeline must convert to RGB before the detector sees the image (Task 4).
4. Config typos (unknown component name, unknown param, missing/extra block, bad grouping method or param) must raise `ValueError` naming valid options, before any model is loaded (Tasks 1 and 4).
5. A cache file written by the old code (has `spotting_raw`/`cluster_eps`, no `ocr` key) must be a cache miss, not a crash (Task 5).

---

## File Structure

Create:
- `ocr/__init__.py` — `build_pipeline(ocr_config)` (added in Task 4; empty until then)
- `ocr/base.py` — `Detector`, `Recognizer`, `Spotter`, `Pipeline` interfaces
- `ocr/registry.py` — `register`, `build`, `available`
- `ocr/grouping.py` — `make_grouper`, `union_box`
- `ocr/pipelines.py` — `DetectRecognize`, `Spot`
- `ocr/_paddleocr_vl.py` — `get_min_pixels`, `load_model_and_processor`
- `ocr/spotters/__init__.py`, `ocr/spotters/paddleocr_vl.py` — `parse_spotting_output`, `spot_text`, `PaddleOCRVLSpotter`
- `ocr/recognizers/__init__.py`, `ocr/recognizers/paddleocr_vl.py` — `read_crop_text`, `PaddleOCRVLRecognizer`
- `ocr/detectors/__init__.py`, `ocr/detectors/yolo.py` — `YoloTextDetector`
- `tests/test_ocr_registry.py`, `tests/test_ocr_grouping.py`, `tests/test_ocr_paddleocr_vl.py`, `tests/test_ocr_yolo.py`, `tests/test_ocr_pipelines.py`

Modify: `tests/conftest.py`, `inference.py`, `config.json`, `README.md`, `pyproject.toml`, `tests/test_inference_resume.py`, `tests/test_cache_extension.py`

Delete: `ocr_utils.py`

---

### Task 1: Core interfaces, registry, grouping

**Files:**
- Create: `ocr/__init__.py` (empty), `ocr/base.py`, `ocr/registry.py`, `ocr/grouping.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_ocr_registry.py`, `tests/test_ocr_grouping.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ocr.base.Detector.detect(pil_image) -> list[dict]` (`x_min,y_min,x_max,y_max`), `Recognizer.read(crop) -> str`, `Spotter.spot(pil_image) -> list[dict]` (`text,x_min,y_min,x_max,y_max`), `Pipeline.run(img_path) -> list[dict]` (`text, insertion_polygon`); every class has `close() -> None`.
  - `ocr.registry.register(kind, name)` decorator; `ocr.registry.build(kind, name, **params)`; `ocr.registry.available(kind) -> list[str]`; module attrs `_REGISTRY` and `_BUILTIN_MODULES` (dict keyed by `(kind, name)`).
  - `ocr.grouping.make_grouper(config: dict | None)` returning `grouper(boxes, img_w, img_h) -> list[list[dict]]`; `ocr.grouping.union_box(group) -> (x0, y0, x1, y1)`.
  - pytest fixture `isolated_registry` (in `tests/conftest.py`).

- [ ] **Step 1: Baseline and fixture**

Run: `PYTEST tests -q` from the worktree root.
Expected: all existing tests pass (this is the baseline on `main`; if anything fails, stop and report).

Append to `tests/conftest.py`:

```python
import pytest


@pytest.fixture
def isolated_registry():
    """Remove components registered during a test; keep built-ins (they are registered once at import)."""
    yield
    from ocr import registry

    for kind, names in registry._REGISTRY.items():
        for name in list(names):
            if (kind, name) not in registry._BUILTIN_MODULES:
                del names[name]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_ocr_registry.py`:

```python
import pytest

from ocr import registry
from ocr.base import Detector


def test_register_and_build(isolated_registry):
    @registry.register("detector", "fake")
    class Fake(Detector):
        def __init__(self, a=1):
            self.a = a

    inst = registry.build("detector", "fake", a=5)
    assert isinstance(inst, Fake)
    assert inst.a == 5


def test_unknown_name_lists_valid_options():
    with pytest.raises(ValueError, match=r"Unknown detector 'nope'.*yolo"):
        registry.build("detector", "nope")


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown kind 'widget'"):
        registry.build("widget", "x")


def test_register_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown kind 'widget'"):
        registry.register("widget", "x")


def test_unknown_param_rejected_and_lists_accepted(isolated_registry):
    @registry.register("detector", "fake2")
    class Fake(Detector):
        def __init__(self, a=1):
            self.a = a

    with pytest.raises(ValueError, match=r"Unknown params \['b'\] for detector 'fake2'.*\['a'\]"):
        registry.build("detector", "fake2", b=2)


def test_component_with_var_kwargs_skips_param_check(isolated_registry):
    @registry.register("detector", "fake3")
    class Fake(Detector):
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    assert registry.build("detector", "fake3", anything=1).kwargs == {"anything": 1}


def test_builtin_names_are_listed():
    assert registry.available("detector") == ["yolo"]
    assert registry.available("recognizer") == ["paddleocr_vl"]
    assert registry.available("spotter") == ["paddleocr_vl"]
```

Create `tests/test_ocr_grouping.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTEST tests/test_ocr_registry.py tests/test_ocr_grouping.py -q`
Expected: FAIL / collection errors with `ModuleNotFoundError: No module named 'ocr'`.

- [ ] **Step 4: Implement**

```bash
mkdir -p ocr && : > ocr/__init__.py
```

Create `ocr/base.py`:

```python
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
```

Create `ocr/registry.py`:

```python
import importlib
import inspect

KINDS = ("detector", "recognizer", "spotter")

_REGISTRY: dict[str, dict[str, type]] = {kind: {} for kind in KINDS}

# Built-in components: modules are imported lazily on first build, so a config
# that doesn't use a component never imports its (possibly heavy) dependencies.
_BUILTIN_MODULES = {
    ("detector", "yolo"): "ocr.detectors.yolo",
    ("recognizer", "paddleocr_vl"): "ocr.recognizers.paddleocr_vl",
    ("spotter", "paddleocr_vl"): "ocr.spotters.paddleocr_vl",
}


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"Unknown kind '{kind}'. Valid kinds: {list(KINDS)}")


def register(kind: str, name: str):
    _check_kind(kind)

    def decorator(cls):
        _REGISTRY[kind][name] = cls
        return cls

    return decorator


def available(kind: str) -> list[str]:
    _check_kind(kind)
    names = set(_REGISTRY[kind]) | {n for (k, n) in _BUILTIN_MODULES if k == kind}
    return sorted(names)


def _check_params(kind: str, name: str, cls: type, params: dict) -> None:
    signature = inspect.signature(cls.__init__)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return  # the component validates its own params
    accepted = {
        n
        for n, p in signature.parameters.items()
        if n != "self"
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    unknown = set(params) - accepted
    if unknown:
        raise ValueError(
            f"Unknown params {sorted(unknown)} for {kind} '{name}'. Valid: {sorted(accepted)}"
        )


def build(kind: str, name: str, **params):
    _check_kind(kind)
    if name not in _REGISTRY[kind]:
        module = _BUILTIN_MODULES.get((kind, name))
        if module is None:
            raise ValueError(f"Unknown {kind} '{name}'. Valid options: {available(kind)}")
        importlib.import_module(module)
        if name not in _REGISTRY[kind]:
            raise ValueError(f"Module '{module}' did not register {kind} '{name}'")
    cls = _REGISTRY[kind][name]
    _check_params(kind, name, cls, params)
    return cls(**params)
```

Create `ocr/grouping.py`:

```python
GROUPING_METHODS = ("none", "dbscan")


def union_box(group: list[dict]) -> tuple[int, int, int, int]:
    return (
        min(b["x_min"] for b in group),
        min(b["y_min"] for b in group),
        max(b["x_max"] for b in group),
        max(b["y_max"] for b in group),
    )


def group_none(boxes: list[dict], img_w: int, img_h: int) -> list[list[dict]]:
    return [[b] for b in boxes]


def group_dbscan(boxes: list[dict], eps_pixels: float) -> list[list[dict]]:
    if not boxes:
        return []
    import numpy as np
    from sklearn.cluster import DBSCAN

    centers = np.array([
        [(b["x_min"] + b["x_max"]) / 2, (b["y_min"] + b["y_max"]) / 2]
        for b in boxes
    ])
    labels = DBSCAN(eps=eps_pixels, min_samples=1).fit_predict(centers)
    groups: dict[int, list[dict]] = {}
    for label, box in zip(labels, boxes):
        groups.setdefault(int(label), []).append(box)
    return list(groups.values())


def make_grouper(config: dict | None):
    """Return grouper(boxes, img_w, img_h) -> list[list[dict]]. Validates config eagerly."""
    params = dict(config) if config else {}
    method = params.pop("method", "none")
    if method not in GROUPING_METHODS:
        raise ValueError(
            f"Unknown grouping method '{method}'. Valid options: {list(GROUPING_METHODS)}"
        )
    allowed = ("eps",) if method == "dbscan" else ()
    unknown = set(params) - set(allowed)
    if unknown:
        raise ValueError(
            f"Unknown grouping params {sorted(unknown)} for method '{method}'. Valid: {list(allowed)}"
        )
    if method == "none":
        return group_none

    eps = params.get("eps", 80)

    def grouper(boxes: list[dict], img_w: int, img_h: int) -> list[list[dict]]:
        eps_pixels = max(1, int(eps / 1000 * max(img_w, img_h)))
        return group_dbscan(boxes, eps_pixels)

    return grouper
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTEST tests/test_ocr_registry.py tests/test_ocr_grouping.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add ocr/__init__.py ocr/base.py ocr/registry.py ocr/grouping.py tests/conftest.py tests/test_ocr_registry.py tests/test_ocr_grouping.py
git commit -m "feat: add OCR component interfaces, registry and grouping" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: PaddleOCR-VL spotter and recognizer

**Files:**
- Create: `ocr/_paddleocr_vl.py`, `ocr/spotters/__init__.py` (empty), `ocr/spotters/paddleocr_vl.py`, `ocr/recognizers/__init__.py` (empty), `ocr/recognizers/paddleocr_vl.py`
- Test: `tests/test_ocr_paddleocr_vl.py`

**Interfaces:**
- Consumes: `ocr.base.Spotter`, `ocr.base.Recognizer`, `ocr.registry.register`.
- Produces:
  - `ocr._paddleocr_vl.get_min_pixels(processor) -> int`; `load_model_and_processor(model_id) -> (model, processor)`.
  - `ocr.spotters.paddleocr_vl.parse_spotting_output(raw, img_w, img_h) -> list[dict]`; `spot_text(image, model, processor, max_tokens=512) -> str` (takes a PIL image, not a path); `PaddleOCRVLSpotter(model="PaddlePaddle/PaddleOCR-VL-1.5", max_tokens=512)` registered as `("spotter", "paddleocr_vl")`.
  - `ocr.recognizers.paddleocr_vl.read_crop_text(crop, model, processor, max_tokens=128) -> str`; `PaddleOCRVLRecognizer(model="PaddlePaddle/PaddleOCR-VL-1.5", max_tokens=128)` registered as `("recognizer", "paddleocr_vl")`.

Note: this code deliberately uses `get_min_pixels` and `processor_kwargs={"images_kwargs": ...}` (the installed transformers is 5.x, where `image_processor.min_pixels` is gone and chat-template image kwargs go through `processor_kwargs`). The old `inference.spot_text` on `main` used the pre-5.x form.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ocr_paddleocr_vl.py`:

```python
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from ocr._paddleocr_vl import get_min_pixels
from ocr.recognizers import paddleocr_vl as recognizer_mod
from ocr.spotters import paddleocr_vl as spotter_mod

_RAW = "HELLO<|LOC_100|><|LOC_100|><|LOC_200|><|LOC_100|><|LOC_200|><|LOC_200|><|LOC_100|><|LOC_200|>"


def _model_processor(decoded="hello"):
    processor = MagicMock()
    processor.image_processor.min_pixels = 256
    inputs = MagicMock()
    inputs.__getitem__ = lambda self, key: MagicMock(shape=[-1, 4])
    inputs.to.return_value = inputs
    processor.apply_chat_template.return_value = inputs
    processor.decode.return_value = decoded
    model = MagicMock()
    model.device = "cpu"
    model.generate.return_value = [list(range(10))]
    return model, processor


def _prompt_texts(processor):
    messages = processor.apply_chat_template.call_args[0][0]
    return [c["text"] for c in messages[0]["content"] if c.get("type") == "text"]


# --- shared helpers -------------------------------------------------------

def test_get_min_pixels_prefers_min_pixels_attribute():
    processor = SimpleNamespace(image_processor=SimpleNamespace(min_pixels=123, size={"shortest_edge": 99}))
    assert get_min_pixels(processor) == 123


def test_get_min_pixels_falls_back_to_shortest_edge():
    processor = SimpleNamespace(image_processor=SimpleNamespace(min_pixels=None, size={"shortest_edge": 99}))
    assert get_min_pixels(processor) == 99


# --- spotter --------------------------------------------------------------

def test_parse_spotting_output_scales_to_pixels():
    boxes = spotter_mod.parse_spotting_output(_RAW, 2000, 1000)
    assert boxes == [{"text": "HELLO", "x_min": 200, "y_min": 100, "x_max": 400, "y_max": 200}]


def test_parse_spotting_output_skips_malformed_and_blank_lines():
    raw = "no coordinates here\n\n" + _RAW + "\nTOO FEW<|LOC_1|><|LOC_2|>"
    boxes = spotter_mod.parse_spotting_output(raw, 1000, 1000)
    assert [b["text"] for b in boxes] == ["HELLO"]


def test_spot_text_uses_spotting_prompt_and_does_not_resize():
    model, processor = _model_processor(decoded="raw")
    image = Image.new("RGB", (800, 1200))
    result = spotter_mod.spot_text(image, model, processor)
    assert "Spotting:" in _prompt_texts(processor)
    messages = processor.apply_chat_template.call_args[0][0]
    assert messages[0]["content"][0]["image"] is image
    assert result == "raw"


def test_spotter_spot_returns_pixel_boxes():
    model, processor = _model_processor()
    with patch.object(spotter_mod, "load_model_and_processor", return_value=(model, processor)), \
         patch.object(spotter_mod, "spot_text", return_value=_RAW) as spot:
        spotter = spotter_mod.PaddleOCRVLSpotter(model="m", max_tokens=77)
        boxes = spotter.spot(Image.new("RGB", (1000, 1000)))
    assert boxes == [{"text": "HELLO", "x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200}]
    assert spot.call_args[0][3] == 77


def test_spotter_close_drops_model_references():
    model, processor = _model_processor()
    with patch.object(spotter_mod, "load_model_and_processor", return_value=(model, processor)):
        spotter = spotter_mod.PaddleOCRVLSpotter()
    spotter.close()
    assert spotter._model is None and spotter._processor is None


# --- recognizer -----------------------------------------------------------

def test_read_crop_text_uses_ocr_prompt():
    model, processor = _model_processor()
    recognizer_mod.read_crop_text(Image.new("RGB", (64, 64)), model, processor)
    assert "OCR:" in _prompt_texts(processor)


def test_read_crop_text_returns_stripped_string():
    model, processor = _model_processor(decoded="  hello world  ")
    assert recognizer_mod.read_crop_text(Image.new("RGB", (64, 64)), model, processor) == "hello world"


def test_recognizer_read_passes_max_tokens():
    model, processor = _model_processor(decoded="hi")
    with patch.object(recognizer_mod, "load_model_and_processor", return_value=(model, processor)):
        recognizer = recognizer_mod.PaddleOCRVLRecognizer(model="m", max_tokens=33)
    assert recognizer.read(Image.new("RGB", (64, 64))) == "hi"
    assert model.generate.call_args.kwargs["max_new_tokens"] == 33


def test_recognizer_close_drops_model_references():
    model, processor = _model_processor()
    with patch.object(recognizer_mod, "load_model_and_processor", return_value=(model, processor)):
        recognizer = recognizer_mod.PaddleOCRVLRecognizer()
    recognizer.close()
    assert recognizer._model is None and recognizer._processor is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST tests/test_ocr_paddleocr_vl.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ocr._paddleocr_vl'`.

- [ ] **Step 3: Implement**

```bash
mkdir -p ocr/spotters ocr/recognizers ocr/detectors && : > ocr/spotters/__init__.py && : > ocr/recognizers/__init__.py && : > ocr/detectors/__init__.py
```

Create `ocr/_paddleocr_vl.py`:

```python
DEFAULT_MODEL = "PaddlePaddle/PaddleOCR-VL-1.5"


def get_min_pixels(processor) -> int:
    """min_pixels was replaced by size.shortest_edge in newer transformers releases."""
    ip = processor.image_processor
    min_pixels = getattr(ip, "min_pixels", None)
    if min_pixels is not None:
        return min_pixels
    return ip.size["shortest_edge"]


def load_model_and_processor(model_id: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (
        AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor
```

Create `ocr/spotters/paddleocr_vl.py`:

```python
import re

from ocr._paddleocr_vl import DEFAULT_MODEL, get_min_pixels, load_model_and_processor
from ocr.base import Spotter
from ocr.registry import register

# Dense pages (many sound effects/narration boxes) can approach max_tokens; raise it in config if spotting looks incomplete.
MAX_PIXELS = 2048 * 28 * 28


def parse_spotting_output(raw: str, img_w: int, img_h: int) -> list[dict]:
    pattern = r'(.+?)((?:<\|LOC_\d+\|>){8})'
    boxes = []
    for line in raw.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        match = re.match(pattern, line)
        if not match:
            continue
        text = match.group(1).strip()
        loc_tokens = re.findall(r'<\|LOC_(\d+)\|>', match.group(2))
        if len(loc_tokens) != 8:
            continue
        coords = list(map(int, loc_tokens))
        xs = coords[0::2]
        ys = coords[1::2]
        boxes.append({
            'text': text,
            'x_min': int(min(xs) / 1000 * img_w),
            'y_min': int(min(ys) / 1000 * img_h),
            'x_max': int(max(xs) / 1000 * img_w),
            'y_max': int(max(ys) / 1000 * img_h),
        })
    return boxes


def spot_text(image, model, processor, max_tokens: int = 512) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Spotting:"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "images_kwargs": {
                "size": {
                    "shortest_edge": get_min_pixels(processor),
                    "longest_edge": MAX_PIXELS,
                }
            },
        },
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])


@register("spotter", "paddleocr_vl")
class PaddleOCRVLSpotter(Spotter):
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 512):
        self._model, self._processor = load_model_and_processor(model)
        self._max_tokens = max_tokens

    def spot(self, pil_image) -> list[dict]:
        raw = spot_text(pil_image, self._model, self._processor, self._max_tokens)
        img_w, img_h = pil_image.size
        return parse_spotting_output(raw, img_w, img_h)

    def close(self) -> None:
        self._model = None
        self._processor = None
```

Create `ocr/recognizers/paddleocr_vl.py`:

```python
from ocr._paddleocr_vl import DEFAULT_MODEL, get_min_pixels, load_model_and_processor
from ocr.base import Recognizer
from ocr.registry import register

MAX_PIXELS = 512 * 28 * 28


def read_crop_text(crop, model, processor, max_tokens: int = 128) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop},
                {"type": "text", "text": "OCR:"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "images_kwargs": {
                "size": {
                    "shortest_edge": get_min_pixels(processor),
                    "longest_edge": MAX_PIXELS,
                }
            },
        },
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1]).strip()


@register("recognizer", "paddleocr_vl")
class PaddleOCRVLRecognizer(Recognizer):
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 128):
        self._model, self._processor = load_model_and_processor(model)
        self._max_tokens = max_tokens

    def read(self, crop) -> str:
        return read_crop_text(crop, self._model, self._processor, self._max_tokens)

    def close(self) -> None:
        self._model = None
        self._processor = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST tests/test_ocr_paddleocr_vl.py tests/test_ocr_registry.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ocr/_paddleocr_vl.py ocr/spotters ocr/recognizers ocr/detectors/__init__.py tests/test_ocr_paddleocr_vl.py
git commit -m "feat: add PaddleOCR-VL spotter and recognizer components" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: YOLO text detector

**Files:**
- Create: `ocr/detectors/yolo.py`
- Modify: `pyproject.toml` (via `uv add`)
- Test: `tests/test_ocr_yolo.py`

**Interfaces:**
- Consumes: `ocr.base.Detector`, `ocr.registry.register`.
- Produces: `YoloTextDetector(repo="lordtrilink/manga-text-detector-v0", filename="best.pt", conf=0.05, iou=0.7, imgsz=1024, include_sfx=False)` registered as `("detector", "yolo")`; `detect(pil_image) -> [{x_min, y_min, x_max, y_max}]` (int pixels, clamped to the image, zero-area boxes dropped).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ocr_yolo.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST tests/test_ocr_yolo.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ocr.detectors.yolo'`.

- [ ] **Step 3: Implement**

Create `ocr/detectors/yolo.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST tests/test_ocr_yolo.py -v`
Expected: all PASS.

- [ ] **Step 5: Add the dependencies**

Run: `uv add --no-sync ultralytics huggingface_hub`
Then: `git status --short`
Expected: `pyproject.toml` modified; a `uv.lock` may appear as untracked. Leave it untracked (do not stage it).
Check: `git diff pyproject.toml` shows exactly the two added dependency lines.

- [ ] **Step 6: Commit**

```bash
git add ocr/detectors/yolo.py tests/test_ocr_yolo.py pyproject.toml
git commit -m "feat: add YOLO manga text detector component" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Pipelines and `build_pipeline`

**Files:**
- Create: `ocr/pipelines.py`
- Modify: `ocr/__init__.py` (currently empty)
- Test: `tests/test_ocr_pipelines.py`

**Interfaces:**
- Consumes: `ocr.base.*`, `ocr.registry.build/available`, `ocr.grouping.make_grouper/union_box`.
- Produces:
  - `ocr.pipelines.DetectRecognize(detector, recognizer, grouper, crop_padding=10)` and `ocr.pipelines.Spot(spotter, grouper)`; both `run(img_path) -> [{text, insertion_polygon}]` and `close()`. Attributes `_crop_padding`, `_recognizer` are used by tests.
  - `ocr.build_pipeline(ocr_config) -> Pipeline`. Raises `ValueError` for every config problem before any component is built.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ocr_pipelines.py`:

```python
import pytest
from PIL import Image

from ocr import build_pipeline, registry
from ocr.base import Detector, Recognizer, Spotter
from ocr.grouping import make_grouper
from ocr.pipelines import DetectRecognize, Spot

_BUILT = []


class FakeDetector(Detector):
    def __init__(self, boxes=None):
        _BUILT.append("detector")
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST tests/test_ocr_pipelines.py -q`
Expected: FAIL with `ImportError: cannot import name 'build_pipeline' from 'ocr'`.

- [ ] **Step 3: Implement**

Create `ocr/pipelines.py`:

```python
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
```

Write `ocr/__init__.py`:

```python
from ocr import registry
from ocr.grouping import make_grouper
from ocr.pipelines import DetectRecognize, Spot

PIPELINES = ("detect_recognize", "spot")
_BLOCKS = {
    "detect_recognize": {"detector", "recognizer"},
    "spot": {"spotter"},
}


def _build_component(kind: str, block):
    if not isinstance(block, dict):
        raise ValueError(f"ocr.{kind} must be an object with a 'name'")
    params = dict(block)
    name = params.pop("name", None)
    if name is None:
        raise ValueError(f"ocr.{kind} needs a 'name'. Valid options: {registry.available(kind)}")
    return registry.build(kind, name, **params)


def build_pipeline(ocr_config):
    """Build an OCR pipeline from the config's "ocr" block. All validation happens before any model loads."""
    if not isinstance(ocr_config, dict):
        raise ValueError("config.json must contain an 'ocr' block (see README, 'OCR pipelines')")
    config = dict(ocr_config)

    pipeline = config.pop("pipeline", None)
    if pipeline not in PIPELINES:
        raise ValueError(f"Invalid ocr.pipeline '{pipeline}'. Valid options: {list(PIPELINES)}")

    grouping_config = config.pop("grouping", None)
    required = _BLOCKS[pipeline]
    unexpected = set(config) - required
    if unexpected:
        raise ValueError(
            f"Unexpected ocr blocks {sorted(unexpected)} for pipeline '{pipeline}'. "
            f"Valid blocks: {sorted(required | {'grouping'})}"
        )
    missing = required - set(config)
    if missing:
        raise ValueError(f"ocr.pipeline '{pipeline}' requires blocks: {sorted(missing)}")

    grouper = make_grouper(grouping_config)

    if pipeline == "spot":
        return Spot(_build_component("spotter", config["spotter"]), grouper)

    recognizer_block = config["recognizer"]
    crop_padding = 10
    if isinstance(recognizer_block, dict):
        recognizer_block = dict(recognizer_block)
        crop_padding = recognizer_block.pop("crop_padding", 10)
    detector = _build_component("detector", config["detector"])
    recognizer = _build_component("recognizer", recognizer_block)
    return DetectRecognize(detector, recognizer, grouper, crop_padding=crop_padding)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST tests/test_ocr_pipelines.py tests/test_ocr_registry.py tests/test_ocr_grouping.py tests/test_ocr_paddleocr_vl.py tests/test_ocr_yolo.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ocr/__init__.py ocr/pipelines.py tests/test_ocr_pipelines.py
git commit -m "feat: add detect_recognize and spot pipelines with config validation" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Driver, config, README, cleanup

**Files:**
- Modify: `inference.py`, `config.json`, `README.md`, `tests/test_inference_resume.py`, `tests/test_cache_extension.py`
- Delete: `ocr_utils.py`

**Interfaces:**
- Consumes: `ocr.build_pipeline(ocr_config)`; pipeline `run(img_path)` and `close()`; `img_utils.sort_manga_reading_order` (already imported in `inference.py`).
- Produces: `.ocr.json` cache keys `hash`, `texts`, `text_boxes`, `page_context`, `ocr`, `ocr_boxes` (plus `translated`, `translations` set later, as today).

- [ ] **Step 1: Rewrite the driver tests (they fail until the driver is migrated)**

In `tests/test_inference_resume.py`:

1. In `_MOCKS`, replace the line `'ocr_utils': MagicMock(),` with `'ocr': MagicMock(),`.
2. Replace everything from `_FAKE_SPOTTING_RAW = ...` through the end of `_BASE_CONFIG = {...}` with:

```python
_FAKE_BOXES = [{"text": "Hello", "insertion_polygon": [0, 0, 10, 10]}]
_OCR_CONFIG = {
    "pipeline": "spot",
    "spotter": {"name": "paddleocr_vl"},
    "grouping": {"method": "dbscan", "eps": 80},
}


def _make_driver_deps():
    pipeline = MagicMock()
    pipeline.run.return_value = _FAKE_BOXES
    patches = {
        "inference.build_pipeline": MagicMock(return_value=pipeline),
        "inference.sort_manga_reading_order": MagicMock(side_effect=lambda boxes: boxes),
        "inference.clean_page": MagicMock(side_effect=lambda img_path, temp_dir, *a, **kw: img_path),
        "inference.translate": MagicMock(return_value="こんにちは"),
        "inference.update_session_memory": MagicMock(return_value=MagicMock()),
        "inference.replace_text_with_translation": MagicMock(return_value=MagicMock()),
        "inference.load_memory": MagicMock(return_value=MagicMock()),
        "inference.tqdm": MagicMock(side_effect=lambda x: x),
    }
    return patches


def _pipeline(patches):
    return patches["inference.build_pipeline"].return_value


def _cache(page_hash, **extra):
    """A valid, current-format OCR cache entry."""
    return {
        "hash": page_hash,
        "texts": ["Hello"],
        "text_boxes": [[0, 0, 10, 10]],
        "page_context": "Hello",
        "ocr": _OCR_CONFIG,
        "ocr_boxes": _FAKE_BOXES,
        **extra,
    }


_BASE_CONFIG = {
    "llm_name": "d",
    "font_path": "d",
    "image_enabled": False,
    "ocr": _OCR_CONFIG,
}
```

3. In `test_translate_not_called_for_translated_cache`, replace the whole `cache_001 = {...}` literal with:

```python
    cache_001 = _cache(page1_hash, translated=True)
```

4. In `test_translate_reruns_when_input_image_changes`, replace the whole `stale_cache = {...}` literal with:

```python
    stale_cache = _cache("a" * 64, translated=True)
```

5. In `test_ocr_cache_written_after_run`, replace the last two assertions (`assert "spotting_raw" in data` and `assert "cluster_eps" in data`) with:

```python
    assert data["ocr"] == _OCR_CONFIG
    assert data["ocr_boxes"] == _FAKE_BOXES
```

6. Replace `test_ocr_cache_hit_skips_spotting` entirely with:

```python
def test_ocr_cache_hit_skips_ocr(tmp_path):
    """The pipeline is not run when a valid OCR cache with a matching ocr block exists."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir(); output_dir.mkdir(); temp_dir.mkdir()

    img_bytes = b"fake image"
    (input_dir / "page_001.jpg").write_bytes(img_bytes)
    (temp_dir / "page_001.jpg").write_bytes(img_bytes)

    img_hash = hashlib.sha256(img_bytes).hexdigest()
    (temp_dir / "page_001.jpg.ocr.json").write_text(
        json.dumps(_cache(img_hash, texts=["cached text"], page_context="cached text"))
    )

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), _BASE_CONFIG, "Japanese", "English")

    _pipeline(patches).run.assert_not_called()
    patches["inference.clean_page"].assert_not_called()
    patches["inference.translate"].assert_called_once()
```

7. In `test_ocr_cache_miss_on_hash_mismatch`, replace the `stale_cache = {...}` literal with `stale_cache = _cache("a" * 64)` and replace the last line `patches["inference.spot_text"].assert_called_once()` with `_pipeline(patches).run.assert_called_once()`. Update the docstring's `spot_text runs` to `the pipeline runs`.

8. Append these new tests at the end of the file:

```python
def test_ocr_cache_miss_when_ocr_config_changes(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir(); output_dir.mkdir(); temp_dir.mkdir()

    img_bytes = b"fake image"
    (input_dir / "page_001.jpg").write_bytes(img_bytes)
    (temp_dir / "page_001.jpg").write_bytes(img_bytes)
    img_hash = hashlib.sha256(img_bytes).hexdigest()
    other_ocr = {**_OCR_CONFIG, "grouping": {"method": "dbscan", "eps": 40}}
    (temp_dir / "page_001.jpg.ocr.json").write_text(json.dumps(_cache(img_hash, ocr=other_ocr)))

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), _BASE_CONFIG, "Japanese", "English")

    _pipeline(patches).run.assert_called_once()
    patches["inference.clean_page"].assert_called_once()


def test_old_format_cache_is_a_miss_not_a_crash(tmp_path):
    """Caches written before the ocr block existed (spotting_raw/cluster_eps) are simply re-computed."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir(); output_dir.mkdir(); temp_dir.mkdir()

    img_bytes = b"fake image"
    (input_dir / "page_001.jpg").write_bytes(img_bytes)
    (temp_dir / "page_001.jpg").write_bytes(img_bytes)
    old_cache = {
        "hash": hashlib.sha256(img_bytes).hexdigest(),
        "texts": ["old"],
        "text_boxes": [[0, 0, 5, 5]],
        "page_context": "old",
        "spotting_raw": "HELLO<|LOC_1|>",
        "cluster_eps": 80,
    }
    (temp_dir / "page_001.jpg.ocr.json").write_text(json.dumps(old_cache))

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), _BASE_CONFIG, "Japanese", "English")

    _pipeline(patches).run.assert_called_once()


def test_cleaning_reruns_without_ocr_when_clean_image_missing(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir(); output_dir.mkdir(); temp_dir.mkdir()

    img_bytes = b"fake image"
    (input_dir / "page_001.jpg").write_bytes(img_bytes)   # no cleaned image in temp_dir
    img_hash = hashlib.sha256(img_bytes).hexdigest()
    (temp_dir / "page_001.jpg.ocr.json").write_text(json.dumps(_cache(img_hash)))

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), _BASE_CONFIG, "Japanese", "English")

    _pipeline(patches).run.assert_not_called()
    patches["inference.clean_page"].assert_called_once()


def test_pipeline_built_from_ocr_block_and_closed(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir(); output_dir.mkdir(); temp_dir.mkdir()
    (input_dir / "page_001.jpg").write_bytes(b"fake image")

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), _BASE_CONFIG, "Japanese", "English")

    patches["inference.build_pipeline"].assert_called_once_with(_OCR_CONFIG)
    _pipeline(patches).close.assert_called_once()


def test_page_with_no_detected_text_is_saved_without_translating(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir(); output_dir.mkdir(); temp_dir.mkdir()
    (input_dir / "page_001.jpg").write_bytes(b"fake image")

    patches = _make_driver_deps()
    _pipeline(patches).run.return_value = []
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), _BASE_CONFIG, "Japanese", "English")

    patches["inference.translate"].assert_not_called()
    patches["inference.replace_text_with_translation"].assert_called_once()
    data = json.loads((tmp_path / "temp" / "page_001.jpg.ocr.json").read_text())
    assert data["texts"] == [] and data["ocr_boxes"] == []
```

In `tests/test_cache_extension.py`: replace `'ocr_utils': MagicMock(),` with `'ocr': MagicMock(),`, and replace the first test (`test_cache_write_includes_spotting_raw_and_cluster_eps`) with:

```python
def test_cache_write_includes_ocr_block_and_boxes():
    """First-pass cache must include the ocr config and the pipeline's boxes."""
    cache_data = {
        "hash": "abc123",
        "texts": ["FROM MY TEACHER"],
        "text_boxes": [[498, 80, 583, 111]],
        "page_context": "FROM MY TEACHER",
        "ocr": {"pipeline": "spot", "spotter": {"name": "paddleocr_vl"}},
        "ocr_boxes": [{"text": "FROM MY TEACHER", "insertion_polygon": [498, 80, 583, 111]}],
    }
    assert "ocr" in cache_data
    assert cache_data["ocr_boxes"][0]["insertion_polygon"] == cache_data["text_boxes"][0]
```

- [ ] **Step 2: Run the driver tests to verify they fail**

Run: `PYTEST tests/test_inference_resume.py -q`
Expected: FAIL (`AttributeError: <module 'inference'> does not have the attribute 'build_pipeline'`).

- [ ] **Step 3: Migrate `inference.py`**

3a. Replace the import block at the top (everything from `import os` through `from memory_utils import load_memory`) with:

```python
import os
import gc
import json
import torch
import argparse
import hashlib
from PIL import Image
from tqdm import tqdm
from img_utils import (
    replace_text_with_translation,
    fill_bubble_with_estimated_color,
    sort_manga_reading_order,
)
from ocr import build_pipeline
from text_utils import translate, update_session_memory
from data_model import SessionMemory
from memory_utils import load_memory
```

3b. Delete the whole `spot_text` function and the whole `_cluster_boxes` function. Leave `_file_hash` and `clean_page` untouched.

3c. Replace everything from `def driver(` down to and including the first CUDA block that ends with `        torch.cuda.empty_cache()` (the lines just before `lookback_pages = 2`) with:

```python
def driver(input_dir, temp_dir, output_dir, config, source_language, target_language):
    llm_name = config["llm_name"]
    font_path = config["font_path"]
    image_enabled = config.get("image_enabled", False)
    json_enabled = config.get("json_enabled", True)
    memory_enabled = config.get("memory_enabled", True)
    ocr_config = config.get("ocr")

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    memory_path = os.path.join(temp_dir, "memory.md")
    session_memory = load_memory(memory_path) if memory_enabled else None

    pipeline = build_pipeline(ocr_config)

    img_paths = sorted(os.listdir(input_dir))
    computed = {}

    for img_name in tqdm(img_paths):
        img_path = os.path.join(input_dir, img_name)
        cache_path = os.path.join(temp_dir, img_name + ".ocr.json")
        current_hash = _file_hash(img_path)
        clean_img_path = os.path.join(temp_dir, img_name)

        boxes_raw = None

        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            ocr_ok = (
                cached.get("hash") == current_hash
                and cached.get("ocr") == ocr_config
                and "ocr_boxes" in cached
            )
            if ocr_ok:
                if os.path.exists(clean_img_path):
                    computed[img_path] = {
                        "texts": cached["texts"],
                        "text_boxes": cached["text_boxes"],
                        "page_context": cached["page_context"],
                        "clean_img_path": clean_img_path,
                        "cache_path": cache_path,
                        "hash": current_hash,
                    }
                    continue
                boxes_raw = cached["ocr_boxes"]

        if boxes_raw is None:
            boxes_raw = pipeline.run(img_path)

        boxes = sort_manga_reading_order(boxes_raw)
        texts = [b["text"] for b in boxes]
        text_boxes = [b["insertion_polygon"] for b in boxes]
        page_context = "\n\n".join(texts)
        cleaned_file_path = clean_page(img_path, temp_dir, boxes)

        computed[img_path] = {
            "texts": texts,
            "text_boxes": text_boxes,
            "page_context": page_context,
            "clean_img_path": cleaned_file_path,
            "cache_path": cache_path,
            "hash": current_hash,
        }

        with open(cache_path, "w") as f:
            json.dump({
                "hash": current_hash,
                "texts": texts,
                "text_boxes": [list(b) for b in text_boxes],
                "page_context": page_context,
                "ocr": ocr_config,
                "ocr_boxes": boxes_raw,
            }, f)

    pipeline.close()
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
```

The rest of the file (from `lookback_pages = 2` onward, `main()` included) is unchanged.

- [ ] **Step 4: Run the full suite**

Run: `PYTEST tests -v`
Expected: all PASS (the `ocr_utils` mock entry is gone from both test files; nothing imports `ocr_utils` any more).

- [ ] **Step 5: Remove `ocr_utils.py`, update config and README**

```bash
git rm ocr_utils.py
grep -rn "ocr_utils" --include=*.py . | grep -v "^./docs/"
```
Expected: the grep prints nothing.

Replace the contents of `config.json` with (same models and settings as before, now in the `ocr` block):

```json
{
    "llm_name": "translategemma:12b",
    "font_path": "./fonts/animeace2_bld.otf",
    "image_enabled": false,
    "json_enabled": false,
    "memory_enabled": false,
    "ocr": {
        "pipeline": "spot",
        "spotter": {
            "name": "paddleocr_vl",
            "model": "PaddlePaddle/PaddleOCR-VL-1.5",
            "max_tokens": 512
        },
        "grouping": {"method": "dbscan", "eps": 80}
    }
}
```

In `README.md`, replace the line

```
*   `ocr_model`: The name of the OCR model to use from the Hugging Face Hub.
```

with

```
*   `ocr`: OCR pipeline settings (see "OCR pipelines" below).
```

and insert this section immediately before the `## Usage` heading:

````markdown
### OCR pipelines

Text detection and recognition are pluggable. The `ocr` block of `config.json` selects a pipeline and its components:

*   `"pipeline": "spot"` — one model finds and reads text (`spotter` block). Available spotters: `paddleocr_vl`.
*   `"pipeline": "detect_recognize"` — a `detector` finds text regions and a `recognizer` reads each crop. Available detectors: `yolo`. Available recognizers: `paddleocr_vl`. `crop_padding` (default 10) is set in the `recognizer` block.
*   `grouping` merges neighbouring boxes into one speech bubble: `{"method": "none"}` or `{"method": "dbscan", "eps": 80}` (`eps` is in thousandths of the longer page side).

Using the YOLO manga text detector (one box per text region, so no grouping):

```json
"ocr": {
    "pipeline": "detect_recognize",
    "detector": {"name": "yolo", "repo": "lordtrilink/manga-text-detector-v0", "conf": 0.05, "iou": 0.7, "imgsz": 1024},
    "recognizer": {"name": "paddleocr_vl", "model": "PaddlePaddle/PaddleOCR-VL-1.5", "max_tokens": 128, "crop_padding": 10},
    "grouping": {"method": "none"}
}
```

The YOLO weights (`lordtrilink/manga-text-detector-v0`) download automatically on first use and are licensed CC BY-NC-SA 4.0 (non-commercial use only).

Changing anything in the `ocr` block re-runs OCR for pages already cached in the temp directory.
````

Run: `PYTEST tests -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add inference.py config.json README.md tests/test_inference_resume.py tests/test_cache_extension.py
git commit -m "feat: drive OCR through config-selected pluggable pipelines" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git status --short
```
Expected after the commit: `ocr_utils.py` deletion is included (it was staged by `git rm`); `git status --short` shows nothing tracked as modified (an untracked `uv.lock` is fine).

- [ ] **Step 7: Manual verification (human, not automated)**

The tests mock all models. To verify against real models, from the worktree root:

```bash
uv sync --extra dev
```

Then temporarily switch `config.json` to the `detect_recognize` + `yolo` block from the README and run `uv run python inference.py --input-dir <pages> --output-dir <out> --temp-dir <tmp>` on the pages that previously had missed text. Compare the translated output with the previous run. Things to look at: missed blocks, duplicate or overlapping boxes (YOLO's NMS can leave a box inside another one, and with grouping `none` both would be recognized and translated), and the SFX class (`include_sfx`). Report what you see; do not change `config.json`'s committed default without asking.
