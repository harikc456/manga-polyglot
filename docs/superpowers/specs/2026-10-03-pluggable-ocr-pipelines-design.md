# Pluggable OCR Pipelines Design

Baseline: `main` (not PR #19 / `feat/lama-inpainting`). Nothing from that branch (LaMa inpainting, bubble expansion, adaptive cleaning, Paddle `TextDetection` detector, two-stage OCR) is assumed or included.

## Goal

Make text detection and recognition pluggable so the project can use better detectors (starting with `lordtrilink/manga-text-detector-v0`, a YOLO11s manga text detector) without editing the driver. Support two pipeline types:

1. **detect_recognize**: Detector -> (optional grouping) -> Recognizer. Detection and OCR are separate models.
2. **spot**: Spotter -> (optional grouping). One model detects and reads text in a single pass.

Motivation: detection is the weak step today; recognition and cleaning are acceptable and stay unchanged.

## Current state (main)

`inference.py::driver()` has a single hard-coded path:

- `spot_text` runs PaddleOCR-VL with the "Spotting:" prompt on the whole page.
- `_cluster_boxes` parses the output (`parse_spotting_output`), groups line boxes with DBSCAN (`cluster_into_bubbles`), and merges groups (`boxes_from_clusters`) into `[{text, insertion_polygon: [x_min, y_min, x_max, y_max]}]`, then `sort_manga_reading_order`.
- Cleaning is `clean_page` -> `fill_bubble_with_estimated_color` per box (color fill).
- OCR config is flat: `ocr_model`, `spotting_cluster_eps`, `spotting_max_tokens`.
- Per-page cache `<page>.ocr.json` stores `hash`, `texts`, `text_boxes`, `page_context`, `spotting_raw`, `cluster_eps`. A cache hit requires matching hash and `cluster_eps` and an existing cleaned image; if only `cluster_eps` changed, `spotting_raw` is re-clustered without re-running the model.
- `ocr_utils.py` holds the parser, clustering and box merge; there are no tests for it. `tests/test_inference_resume.py` and `tests/test_cache_extension.py` mock `ocr_utils` and `inference.spot_text`/`_cluster_boxes`.

## Non-goals

- Cleaning, inpainting, bubble expansion, translation, memory, text drawing.
- The Paddle `TextDetection` detector (exists only on PR #19; not ported).
- Backward compatibility with the old flat OCR config keys (removed).
- Real-model tests in the automated suite.

## Design

### Package layout

New package `ocr/` replaces `ocr_utils.py` (removed):

```
ocr/
  __init__.py          # build_pipeline(ocr_config)
  base.py              # Detector, Recognizer, Spotter, Pipeline interfaces
  registry.py          # register(), build(), available()
  grouping.py          # make_grouper(): "none" | "dbscan"; union_box()
  pipelines.py         # DetectRecognize, Spot
  _paddleocr_vl.py     # shared: get_min_pixels(), load_model_and_processor()
  detectors/yolo.py    # YoloTextDetector (new)
  recognizers/paddleocr_vl.py   # PaddleOCRVLRecognizer + read_crop_text (new, "OCR:" prompt on a crop)
  spotters/paddleocr_vl.py      # PaddleOCRVLSpotter + spot_text + parse_spotting_output (moved)
```

### Interfaces (`ocr/base.py`)

```python
class Detector:
    def detect(self, pil_image) -> list[dict]:   # [{x_min, y_min, x_max, y_max}] pixels
    def close(self) -> None: ...
class Recognizer:
    def read(self, crop) -> str: ...
    def close(self) -> None: ...
class Spotter:
    def spot(self, pil_image) -> list[dict]:     # [{text, x_min, y_min, x_max, y_max}] pixels
    def close(self) -> None: ...
class Pipeline:
    def run(self, img_path: str) -> list[dict]:  # [{text, insertion_polygon: [x0,y0,x1,y1]}]
    def close(self) -> None: ...
```

`close()` drops model references; `driver()` calls `pipeline.close()` then runs its existing `gc.collect()` / CUDA cache release so memory is freed before translation, as models are freed today.

### Registry (`ocr/registry.py`)

`@register("detector", "yolo")` registers a class. Kinds: `detector`, `recognizer`, `spotter`. Built-in names map to modules that are imported lazily on `build`, so a config that doesn't use YOLO never imports `ultralytics`. `build(kind, name, **params)` rejects unknown kinds/names (listing valid ones) and unknown constructor params (listing accepted ones), as `ValueError`.

### Pipelines (`ocr/pipelines.py`)

- **DetectRecognize**: open image, convert to RGB -> `detector.detect` -> grouper -> for each group: union box, expand by `crop_padding`, clamp to image, skip if zero-area, crop, `recognizer.read` -> `{text, insertion_polygon}`.
- **Spot**: open image, convert to RGB -> `spotter.spot` -> grouper -> for each group: union box, join texts with a space -> `{text, insertion_polygon}`.

Empty detection/spotting returns `[]`.

### Grouping (`ocr/grouping.py`)

- `none`: each box is its own group (YOLO already gives one box per text region).
- `dbscan`: existing `cluster_into_bubbles` behaviour; `eps` is in thousandths of the longer image side (same unit as today's `spotting_cluster_eps`), with a minimum of 1 pixel.

### YOLO detector (`ocr/detectors/yolo.py`)

- Weights via `huggingface_hub.hf_hub_download(repo, filename)` (defaults: `lordtrilink/manga-text-detector-v0`, `best.pt`), loaded with `ultralytics.YOLO`; cached by Hugging Face, nothing committed.
- Predict args: `imgsz` 1024, `conf` 0.05, `iou` 0.7 (model card values), `verbose=False`.
- Keeps class 0 (`text-region`); `include_sfx: true` also keeps class 1 (`sfx`). Default false.
- Boxes are clamped to the image; boxes with no area after clamping are dropped.
- The image is passed as-is, **without** `ImageOps.exif_transpose`: the cleaning and drawing code opens pages without transposing, so boxes must stay in that coordinate frame.
- License is CC BY-NC-SA 4.0 (non-commercial); noted in the module docstring and README.

### Config

Single `ocr` block; old flat OCR keys (`ocr_model`, `spotting_cluster_eps`, `spotting_max_tokens`) are removed. Other keys unchanged.

```json
"ocr": {
  "pipeline": "detect_recognize",
  "detector":   {"name": "yolo", "repo": "lordtrilink/manga-text-detector-v0",
                 "conf": 0.05, "iou": 0.7, "imgsz": 1024},
  "recognizer": {"name": "paddleocr_vl", "model": "PaddlePaddle/PaddleOCR-VL-1.5",
                 "max_tokens": 128, "crop_padding": 10},
  "grouping":   {"method": "none"}
}
```

`crop_padding` is written in the `recognizer` block for convenience but belongs to the pipeline; `build_pipeline` removes it before constructing the recognizer.

Spot pipeline (behaviour-preserving; this is what the repo's `config.json` ships with, so merging changes nothing until the user switches):

```json
"ocr": {
  "pipeline": "spot",
  "spotter":  {"name": "paddleocr_vl", "model": "PaddlePaddle/PaddleOCR-VL-1.5", "max_tokens": 512},
  "grouping": {"method": "dbscan", "eps": 80}
}
```

Validation (`ValueError` listing valid options): missing/non-dict `ocr`; unknown `pipeline`; `detect_recognize` without `detector`/`recognizer`; `spot` without `spotter`; blocks that don't belong to the chosen pipeline; component block without `name`; unknown component names/params; unknown grouping method/params. Validation runs before any model loads.

### Driver and cache

`driver()` builds the pipeline once (`build_pipeline(config.get("ocr"))`), calls `pipeline.run(img_path)` on a cache miss, then `sort_manga_reading_order` and `clean_page` as today.

Cache (`<page>.ocr.json`):

- Stores the `ocr` block and the final `ocr_boxes` (`[{text, insertion_polygon}]`, before sorting).
- OCR cache is valid when hash matches, `cached["ocr"] == ocr_config`, and `ocr_boxes` exists. Old-format caches have no `ocr` key and are treated as a miss (no crash).
- Cached page is reused as-is when OCR cache is valid and the cleaned image exists. If the cleaned image is missing, `ocr_boxes` are reused and only cleaning re-runs.
- Behaviour change vs main: changing any OCR setting (including `eps`) re-runs the model instead of re-clustering cached `spotting_raw`.

### Dependencies

`ultralytics` and `huggingface_hub` added via `uv add`.

### Testing

- Registry: register/build, unknown kind/name/param, built-in names listed.
- Grouping: `none`, `dbscan` near/far boxes, bad method/params.
- Pipelines with fake Detector/Recognizer/Spotter: contract, grouping, crop padding clamping, zero-area skip, RGB conversion, empty result, `close()`; `build_pipeline` success and each validation error.
- YOLO detector with `ultralytics`/`hf_hub_download` mocked: class filtering, `include_sfx`, clamping, degenerate boxes, predict args.
- PaddleOCR-VL spotter/recognizer: moved parser tests, `read_crop_text` prompt/strip, no-resize guard (ported from `test_inference_resume.py`).
- Migrate `tests/test_inference_resume.py` and `tests/test_cache_extension.py` to the new config shape and mocks; add driver cache tests (hit, miss on `ocr` change, old-format cache, cleaning-only re-run).
- Real-model check is manual: run the YOLO probe script on a problem page.

## Merge note

PR #19 also rewrites `inference.py` (two-stage OCR, expansion, LaMa). Whichever of the two merges second will need a manual merge in `driver()`.
