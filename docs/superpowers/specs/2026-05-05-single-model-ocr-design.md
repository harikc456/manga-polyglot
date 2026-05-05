# Single-Model Detection + OCR via PaddleOCR-VL Spotting

**Date:** 2026-05-05
**Status:** Approved

## Problem

The pipeline currently uses two separate models for text detection and OCR:

- `ogkalu/comic-text-and-bubble-detector` (RTDetrV2) — detects bounding boxes and classifies them as `bubble`, `text_bubble`, or `text_free`
- `PaddlePaddle/PaddleOCR-VL-1.5` — runs OCR on each cropped region individually

PaddleOCR-VL-1.5 supports a `spotting` task that performs text detection and recognition in a single pass over the full image, making the RTDetr model redundant.

## Goal

Replace the two-model pipeline with a single PaddleOCR-VL spotting pass per page. Also remove SAM3 (used only for free-text cleaning), simplifying the cleaning step. The bubble/free-text distinction is intentionally dropped.

## Architecture & Model Changes

**Removed:**
- `RTDetrV2ForObjectDetection` + `RTDetrImageProcessor` (`ogkalu/comic-text-and-bubble-detector`)
- `Sam3Model` + `Sam3Processor` (`jetjodh/sam3`)
- `text_detection.py` (now unused, deleted)
- `BubbleType` enum (no longer needed)

**Kept:**
- `AutoModelForImageTextToText` + `AutoProcessor` (PaddleOCR-VL-1.5, same weights, no new model downloads)

**New config fields in `config.json`:**
- `spotting_cluster_eps` (int, default `80`) — DBSCAN distance threshold on the 0–1000 normalized coordinate scale
- `spotting_max_tokens` (int, default `2048`) — `max_new_tokens` for the spotting generation pass

## Data Flow

Per page:

1. **Spotting** — run PaddleOCR-VL with `"Spotting:"` prompt on the full page image. If either image dimension is below 1500px, upscale 2× with LANCZOS before passing to the model (model input only; original dimensions used for all box math). Use `max_pixels = 2048 * 28 * 28`.

2. **Parse** — `parse_spotting_output(raw, img_w, img_h)` converts the LOC-token string into per-line dicts:
   ```
   {text, x_min, y_min, x_max, y_max, quad}
   ```
   Coordinates are denormalized: `x_pixel = (loc_val / 1000) * img_w`. Malformed lines (not matching the `TEXT + 8 LOC tokens` pattern) are silently skipped.

3. **Cluster** — `cluster_into_bubbles(boxes, eps)` runs DBSCAN (`min_samples=1`) on line center points. Each cluster becomes one translation unit. Text lines within a cluster are joined with a space.

4. **Sort** — `sort_manga_reading_order()` (existing, unchanged) is applied to the cluster axis-aligned bounding boxes.

5. **Clean** — `clean_page()` is simplified: for every cluster box, call `fill_bubble_with_estimated_color()`. No SAM3, no type branching.

6. **Translate & render** — unchanged. Each cluster's joined text is translated; the cluster bounding box is used as both `text_box` and insertion `polygon`.

## New Module: `ocr_utils.py`

`parse_spotting_output` and `cluster_into_bubbles` are extracted into a new `ocr_utils.py` module (ported and adapted from `paddle_ocr_spotting.ipynb`). This keeps `inference.py` focused on orchestration.

## Cache Schema Changes

Two new fields added to the per-page `.ocr.json` cache:

- `spotting_raw` (str) — raw model output string, preserved for debugging
- `cluster_eps` (int) — eps value used when clustering

**Cache invalidation rule:** If `cluster_eps` in the cache differs from the current config value, `spotting_raw` is re-used (skipping the expensive model call) but clustering and all downstream steps are re-run from the raw string.

## Edge Cases

| Situation | Handling |
|---|---|
| No text detected / empty spotting output | Page skipped; original image copied to output |
| Single isolated text line | DBSCAN `min_samples=1` creates a single-box cluster; no special case |
| Dense page truncating at `max_new_tokens` | Increase `spotting_max_tokens` in config (default 2048) |
| Changing `cluster_eps` | Re-clusters from cached `spotting_raw`; no re-inference |

## Testing

- **`parse_spotting_output`** — unit test with a fixed LOC-token string. Assert correct text, denormalized coordinates, and that malformed lines are skipped.
- **`cluster_into_bubbles`** — unit test with synthetic boxes: two tight clusters and one isolated box. Assert correct number of groups.
- **Integration smoke test** — run `driver()` on a single test image. Assert output image created and cache JSON contains `texts`, `text_boxes`, and `spotting_raw`.
- **No model mocks** — GPU-dependent tests use `pytest.mark.skipif(not torch.cuda.is_available(), ...)`.

## Files Changed

| File | Change |
|---|---|
| `inference.py` | Remove RTDetr/SAM3 loading; replace `detect_text` + `extract_text` calls with spotting; simplify `clean_page` |
| `ocr_utils.py` | New — `parse_spotting_output`, `cluster_into_bubbles` |
| `text_detection.py` | Deleted |
| `data_model.py` | Remove `BubbleType` enum |
| `img_utils.py` | Remove `get_text_insertion_boxes`, `match_text_to_bubbles`, `get_expanded_insertion_box` (RTDetr-specific); keep everything else |
| `config.json` | Add `spotting_cluster_eps`, `spotting_max_tokens`; remove `text_detection_model_path` |
| `tests/` | Add unit tests for `ocr_utils.py`; update integration test |
