# Single-Model Detection + OCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace RTDetr (text detection) + PaddleOCR-VL (per-crop OCR) + SAM3 (free-text cleaning) with a single PaddleOCR-VL `Spotting:` pass per page.

**Architecture:** PaddleOCR-VL in spotting mode returns all text lines with normalized bounding boxes in one pass. Lines are grouped into dialogue clusters via DBSCAN, sorted in manga reading order, and the cluster boxes are used for both cleaning (rectangle fill) and translation insertion. RTDetr, SAM3, and the bubble/free-text distinction are removed entirely.

**Tech Stack:** Python, PyTorch, HuggingFace Transformers (`AutoModelForImageTextToText`), scikit-learn (DBSCAN), PIL, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `ocr_utils.py` | **Create** | Parse spotting output, cluster lines into bubbles, convert clusters to box format |
| `tests/test_ocr_utils.py` | **Create** | Unit tests for `ocr_utils.py` |
| `inference.py` | **Modify** | Replace two-model pipeline with spotting; simplify `clean_page`; remove `extract_text`, `create_masks`, `clean_bubble_free_text` |
| `data_model.py` | **Modify** | Remove `BubbleType` enum |
| `img_utils.py` | **Modify** | Remove `match_text_to_bubbles`, `get_expanded_insertion_box`, `get_text_insertion_boxes`, `BubbleType` import |
| `config.json` | **Modify** | Add `spotting_cluster_eps`, `spotting_max_tokens`; remove `text_detection_model_path` |
| `tests/conftest.py` | **Modify** | Remove `text_detection` mock |
| `tests/test_cache_extension.py` | **Modify** | Update tests to match new cache schema |
| `text_detection.py` | **Delete** | No longer used |

---

## Task 1: Create `ocr_utils.py` (TDD)

**Files:**
- Create: `ocr_utils.py`
- Create: `tests/test_ocr_utils.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ocr_utils.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/harikrishnan-c/projects/manga-polyglot
python -m pytest tests/test_ocr_utils.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'ocr_utils'`

- [ ] **Step 3: Implement `ocr_utils.py`**

Create `ocr_utils.py`:

```python
import re
import numpy as np
from sklearn.cluster import DBSCAN


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


def cluster_into_bubbles(boxes: list[dict], eps: float) -> list[list[dict]]:
    if not boxes:
        return []
    centers = np.array([
        [(b['x_min'] + b['x_max']) / 2, (b['y_min'] + b['y_max']) / 2]
        for b in boxes
    ])
    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(centers)
    groups: dict[int, list[dict]] = {}
    for label, box in zip(labels, boxes):
        groups.setdefault(int(label), []).append(box)
    return list(groups.values())


def boxes_from_clusters(groups: list[list[dict]]) -> list[dict]:
    result = []
    for group in groups:
        text = ' '.join(b['text'] for b in group)
        x_min = min(b['x_min'] for b in group)
        y_min = min(b['y_min'] for b in group)
        x_max = max(b['x_max'] for b in group)
        y_max = max(b['y_max'] for b in group)
        result.append({
            'text': text,
            'insertion_polygon': [x_min, y_min, x_max, y_max],
        })
    return result
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_ocr_utils.py -v
```

Expected: all 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ocr_utils.py tests/test_ocr_utils.py
git commit -m "feat: add ocr_utils with spotting parser, DBSCAN clustering, and cluster box builder"
```

---

## Task 2: Remove `BubbleType` from `data_model.py`

**Files:**
- Modify: `data_model.py`

- [ ] **Step 1: Remove `BubbleType` from `data_model.py`**

In `data_model.py`, delete the `BubbleType` class (lines 10-12):

```python
class BubbleType(str, Enum):
    FREE = "free"
    FIXED = "fixed"
```

Also remove the `Enum` import since nothing else uses it. The top of the file becomes:

```python
from typing import Literal
from pydantic import BaseModel, field_validator
```

- [ ] **Step 2: Verify no import errors**

```bash
python -c "from data_model import SessionMemory, Translation, EntityEntry, CharacterEntry; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add data_model.py
git commit -m "refactor: remove BubbleType enum (no longer needed after single-model OCR)"
```

---

## Task 3: Remove RTDetr-specific functions from `img_utils.py`

**Files:**
- Modify: `img_utils.py`

- [ ] **Step 1: Remove dead functions and import**

In `img_utils.py`, delete:
- Line 6: `from data_model import BubbleType`
- The entire `match_text_to_bubbles` function (lines 110–141)
- The entire `get_expanded_insertion_box` function (lines 144–168)
- The entire `get_text_insertion_boxes` function (lines 213–250)

`sort_manga_reading_order` (lines 171–210) is kept unchanged.

- [ ] **Step 2: Verify no import errors and that `sort_manga_reading_order` still works**

```bash
python -c "from img_utils import sort_manga_reading_order, fill_bubble_with_estimated_color; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add img_utils.py
git commit -m "refactor: remove RTDetr-specific bubble-matching and box-expansion helpers from img_utils"
```

---

## Task 4: Update `config.json`

**Files:**
- Modify: `config.json`

- [ ] **Step 1: Update config**

Replace the contents of `config.json` with:

```json
{
    "ocr_model": "PaddlePaddle/PaddleOCR-VL-1.5",
    "llm_name": "translategemma:12b",
    "font_path": "./fonts/animeace2_bld.otf",
    "image_enabled": false,
    "json_enabled": false,
    "memory_enabled": false,
    "spotting_cluster_eps": 80,
    "spotting_max_tokens": 2048
}
```

(`text_detection_model_path` is removed.)

- [ ] **Step 2: Commit**

```bash
git add config.json
git commit -m "config: add spotting_cluster_eps and spotting_max_tokens; remove text_detection_model_path"
```

---

## Task 5: Rewrite `inference.py`

**Files:**
- Modify: `inference.py`

- [ ] **Step 1: Replace the full contents of `inference.py`**

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
from transformers import AutoProcessor, AutoModelForImageTextToText
from ocr_utils import parse_spotting_output, cluster_into_bubbles, boxes_from_clusters
from text_utils import translate, update_session_memory
from data_model import SessionMemory
from memory_utils import load_memory


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def spot_text(img_path: str, model, processor, max_tokens: int = 2048) -> str:
    image = Image.open(img_path).convert("RGB")
    orig_w, orig_h = image.size
    if orig_w < 1500 and orig_h < 1500:
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        image = image.resize((orig_w * 2, orig_h * 2), resample)
    max_pixels = 2048 * 28 * 28
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
        images_kwargs={
            "size": {
                "shortest_edge": processor.image_processor.min_pixels,
                "longest_edge": max_pixels,
            }
        },
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])


def clean_page(img_path: str, temp_dir: str, boxes: list[dict]) -> str:
    pil_image = Image.open(img_path).convert("RGB")
    file_name = os.path.basename(img_path)
    cleaned_file_path = os.path.join(temp_dir, file_name)
    for box in boxes:
        pil_image = fill_bubble_with_estimated_color(pil_image, box["insertion_polygon"])
    pil_image.save(cleaned_file_path)
    return cleaned_file_path


def _cluster_boxes(spotting_raw: str, img_path: str, cluster_eps: int) -> list[dict]:
    img = Image.open(img_path)
    img_w, img_h = img.size
    lines = parse_spotting_output(spotting_raw, img_w, img_h)
    eps_pixels = int(cluster_eps / 1000 * max(img_w, img_h))
    groups = cluster_into_bubbles(lines, eps=eps_pixels)
    boxes = boxes_from_clusters(groups)
    return sort_manga_reading_order(boxes)


def driver(input_dir, temp_dir, output_dir, config, source_language, target_language):
    ocr_model_id = config["ocr_model"]
    llm_name = config["llm_name"]
    font_path = config["font_path"]
    image_enabled = config.get("image_enabled", False)
    json_enabled = config.get("json_enabled", True)
    memory_enabled = config.get("memory_enabled", True)
    cluster_eps = config.get("spotting_cluster_eps", 80)
    max_tokens = config.get("spotting_max_tokens", 2048)

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    memory_path = os.path.join(temp_dir, "memory.md")
    session_memory = load_memory(memory_path) if memory_enabled else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ocr_model = (
        AutoModelForImageTextToText.from_pretrained(
            ocr_model_id, torch_dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(ocr_model_id)

    img_paths = sorted(os.listdir(input_dir))
    computed = {}

    for img_name in tqdm(img_paths):
        img_path = os.path.join(input_dir, img_name)
        cache_path = os.path.join(temp_dir, img_name + ".ocr.json")
        current_hash = _file_hash(img_path)
        clean_img_path = os.path.join(temp_dir, img_name)

        spotting_raw = None

        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("hash") == current_hash:
                if cached.get("cluster_eps") == cluster_eps and os.path.exists(clean_img_path):
                    computed[img_path] = {
                        "texts": cached["texts"],
                        "text_boxes": cached["text_boxes"],
                        "page_context": cached["page_context"],
                        "clean_img_path": clean_img_path,
                        "cache_path": cache_path,
                        "hash": current_hash,
                    }
                    continue
                if "spotting_raw" in cached:
                    spotting_raw = cached["spotting_raw"]

        if spotting_raw is None:
            spotting_raw = spot_text(img_path, ocr_model, processor, max_tokens)

        boxes = _cluster_boxes(spotting_raw, img_path, cluster_eps)
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
                "spotting_raw": spotting_raw,
                "cluster_eps": cluster_eps,
            }, f)

    del ocr_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    lookback_pages = 2
    lookahead_pages = 2
    n_pages = len(img_paths)

    for i, img_name in enumerate(tqdm(img_paths)):
        img_path = os.path.join(input_dir, img_name)
        out_path = os.path.join(output_dir, img_name)

        cache_path = computed[img_path]["cache_path"]
        with open(cache_path) as f:
            cache_data = json.load(f)
        if cache_data.get("hash") == computed[img_path]["hash"] and cache_data.get("translated"):
            continue

        translations = []
        precomputed_vals = computed[img_path]
        cleaned_file_path = computed[img_path]["clean_img_path"]

        context_parts = []
        for j in range(max(0, i - lookback_pages), i):
            prev_img_path = os.path.join(input_dir, img_paths[j])
            ctx = computed[prev_img_path]["page_context"]
            if ctx:
                context_parts.append(f"[Page {j+1}] {ctx}")
        context_parts.append(f"[Current Page] {precomputed_vals['page_context']}")
        for j in range(i + 1, min(n_pages, i + 1 + lookahead_pages)):
            next_img_path = os.path.join(input_dir, img_paths[j])
            ctx = computed[next_img_path]["page_context"]
            if ctx:
                context_parts.append(f"[Page {j+1} ahead] {ctx}")
        context = "\n\n".join(context_parts).strip()

        for text, text_box in zip(precomputed_vals["texts"], precomputed_vals["text_boxes"]):
            image = None
            if image_enabled:
                img = Image.open(img_path)
                image = img.crop(text_box)

            previous_translations = [
                {"original": p["original"], "translated": p["translated"]}
                for p in translations
            ]

            translated = translate(
                text,
                llm_name,
                context=context,
                source_language=source_language,
                target_language=target_language,
                image=image,
                previous_translations=previous_translations,
                session_memory=session_memory,
                use_json=json_enabled,
            )
            translations.append(
                {"original": text, "translated": translated, "polygon": text_box}
            )

        if translations and memory_enabled:
            session_memory = update_session_memory(
                translations, session_memory, llm_name, temp_dir
            )

        translated_image = replace_text_with_translation(
            cleaned_file_path, font_path, translations
        )
        translated_image.save(out_path)

        cache_data["translated"] = True
        cache_data["translations"] = [
            {"original": t["original"], "translated": t["translated"]}
            for t in translations
        ]
        with open(computed[img_path]["cache_path"], "w") as f:
            json.dump(cache_data, f)


def main():
    parser = argparse.ArgumentParser(description="Inputs to translate")
    parser.add_argument("--input-dir", type=str, help="the directory containing images")
    parser.add_argument("--output-dir", type=str, help="the directory to which translated images are stored")
    parser.add_argument("--source-lang", type=str, default="Japanese")
    parser.add_argument("--target-lang", type=str, default="English")
    parser.add_argument("--temp-dir", type=str, default="./temp")
    args = parser.parse_args()

    config_path = "./config.json"
    with open(config_path) as f:
        config = json.load(f)

    driver(args.input_dir, args.temp_dir, args.output_dir, config, args.source_lang, args.target_lang)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import works (with mocked heavy deps)**

```bash
python -c "
import sys
from unittest.mock import MagicMock, patch
mocks = {'torch': MagicMock(), 'transformers': MagicMock(), 'PIL': MagicMock(), 'PIL.Image': MagicMock(), 'tqdm': MagicMock(), 'img_utils': MagicMock(), 'text_utils': MagicMock(), 'data_model': MagicMock(), 'memory_utils': MagicMock(), 'ocr_utils': MagicMock()}
with patch.dict(sys.modules, mocks):
    import inference
print('ok')
"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add inference.py
git commit -m "feat: replace RTDetr+SAM3+per-crop OCR with PaddleOCR-VL spotting pipeline"
```

---

## Task 6: Update existing tests

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_cache_extension.py`

- [ ] **Step 1: Remove `text_detection` mock from `conftest.py`**

In `tests/conftest.py`, remove `'text_detection': MagicMock(),` from the `_MOCKS` dict. The file becomes:

```python
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

_MOCKS = {
    'cv2': MagicMock(),
    'torch': MagicMock(),
    'numpy': MagicMock(),
    'transformers': MagicMock(),
    'PIL': MagicMock(),
    'PIL.Image': MagicMock(),
    'tqdm': MagicMock(),
    'img_utils': MagicMock(),
    'text_utils': MagicMock(),
    'data_model': MagicMock(),
    'memory_utils': MagicMock(),
    'ocr_utils': MagicMock(),
}

with patch.dict(sys.modules, _MOCKS):
    import inference

sys.modules['inference'] = inference
```

- [ ] **Step 2: Update `test_cache_extension.py`**

Replace the contents of `tests/test_cache_extension.py` with:

```python
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_cache_write_includes_spotting_raw_and_cluster_eps():
    """First-pass cache must include spotting_raw and cluster_eps."""
    cache_data = {
        "hash": "abc123",
        "texts": ["FROM MY TEACHER"],
        "text_boxes": [[498, 80, 583, 111]],
        "page_context": "FROM MY TEACHER",
        "spotting_raw": "FROM MY<|LOC_498|><|LOC_80|><|LOC_580|><|LOC_80|><|LOC_580|><|LOC_93|><|LOC_498|><|LOC_93|>",
        "cluster_eps": 80,
    }
    assert "spotting_raw" in cache_data
    assert "cluster_eps" in cache_data
    assert cache_data["cluster_eps"] == 80


def test_cache_write_includes_translations():
    """After translation, the cache must include a 'translations' key."""
    translations = [
        {"original": "こんにちは", "translated": "Hello!", "polygon": [10, 10, 50, 30]},
    ]
    cache_data = {"hash": "abc", "texts": ["こんにちは"], "text_boxes": [[10, 10, 50, 30]], "page_context": ""}

    cache_data["translations"] = [
        {"original": t["original"], "translated": t["translated"]}
        for t in translations
    ]
    cache_data["translated"] = True

    assert cache_data["translations"] == [{"original": "こんにちは", "translated": "Hello!"}]
    assert cache_data["translated"] is True


def test_cache_hit_requires_matching_cluster_eps():
    """Cache is only a full hit when cluster_eps matches config."""
    cached = {"hash": "abc", "cluster_eps": 80, "texts": [], "text_boxes": [], "page_context": ""}
    current_hash = "abc"
    config_eps = 100  # different from cached

    full_hit = (
        cached.get("hash") == current_hash
        and cached.get("cluster_eps") == config_eps
    )
    assert full_hit is False


def test_cache_recluster_uses_spotting_raw_when_eps_changes():
    """When hash matches but eps differs, spotting_raw is available for re-clustering."""
    cached = {
        "hash": "abc",
        "cluster_eps": 80,
        "spotting_raw": "SOME TEXT<|LOC_1|><|LOC_2|><|LOC_3|><|LOC_4|><|LOC_5|><|LOC_6|><|LOC_7|><|LOC_8|>",
    }
    current_hash = "abc"
    config_eps = 120

    full_hit = cached.get("hash") == current_hash and cached.get("cluster_eps") == config_eps
    can_recluster = not full_hit and cached.get("hash") == current_hash and "spotting_raw" in cached

    assert full_hit is False
    assert can_recluster is True
```

- [ ] **Step 3: Run all tests to confirm they pass**

```bash
python -m pytest tests/ -v --ignore=tests/test_inference_resume.py
```

Expected: all tests PASS (skip `test_inference_resume.py` if it depends on removed symbols — fix or delete separately if broken)

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_cache_extension.py
git commit -m "test: update cache tests for spotting schema; remove text_detection mock"
```

---

## Task 7: Delete `text_detection.py`

**Files:**
- Delete: `text_detection.py`

- [ ] **Step 1: Delete the file**

```bash
git rm text_detection.py
```

- [ ] **Step 2: Confirm no remaining imports**

```bash
grep -r "text_detection" /home/harikrishnan-c/projects/manga-polyglot --include="*.py" --exclude-dir=__pycache__
```

Expected: no output

- [ ] **Step 3: Run full test suite one final time**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 4: Final commit**

```bash
git commit -m "chore: delete text_detection.py (replaced by PaddleOCR-VL spotting)"
```
