# LaMa Neural Inpainting Integration

**Date:** 2026-05-06
**Status:** Approved

## Overview

Replace the current flat color-fill text removal with LaMa neural inpainting, controlled by a config flag. When enabled, inpainting uses a manga-tuned LaMa ONNX model that generates semantically coherent background fill instead of a solid color. The existing color-fill path remains unchanged as the default.

---

## Motivation

Current `fill_bubble_with_estimated_color()` in `img_utils.py` samples the 12px border of each bounding box, takes the modal pixel color, and floods the box. This works for solid white speech bubbles but fails on:
- Gradient or textured backgrounds
- Sound effects overlaid on screentone or art
- Translucent / semi-transparent bubbles
- Partially overlapping or irregular text regions

LaMa (Large Mask inpainting, CVPR 2022) uses Fourier convolutions to generate plausible background texture — a significant quality improvement for non-solid backgrounds.

---

## Architecture

```
config.json
  └── "inpainting_engine": "lama" | "color_fill"

inference.py: clean_page()
  ├── color_fill → img_utils.fill_bubble_with_estimated_color() [unchanged]
  └── lama → inpainting.inpaint_page(pil_image, boxes)
                  ├── build_text_mask(boxes, img_w, img_h)
                  │     └── morphological dilation (elliptical, scale-relative kernel)
                  ├── cv2.connectedComponentsWithStats()  ← split mask into blobs
                  ├── for each blob (largest first):
                  │     ├── crop tile centered on blob centroid + 32px padding
                  │     ├── reflect-pad to multiple of 8 (cv2.BORDER_REFLECT)
                  │     ├── LamaInpainter.infer(tile, mask_tile)
                  │     └── blend result back, mark region processed
                  └── return inpainted PIL image

inpainting.py  (new file)
  ├── ensure_model()       → download lama-manga.onnx if absent
  ├── build_text_mask()    → boxes → dilated binary mask
  ├── LamaInpainter        → ONNX session + infer()
  └── inpaint_page()       → orchestration
```

---

## New File: `inpainting.py`

### `ensure_model() -> Path`

- Model: `lama-manga.onnx` from `mayocream/lama-manga-onnx` on HuggingFace
- Download URL: `https://huggingface.co/mayocream/lama-manga-onnx/resolve/main/lama-manga.onnx`
- Cache path: `~/.manga-polyglot/models/lama-manga.onnx`
- Downloads with `urllib.request` + `tqdm` progress bar only if file is absent
- Returns local `Path` to model file

### `build_text_mask(boxes, img_w, img_h) -> np.ndarray`

- Input: `boxes` list (each has `insertion_polygon: [x_min, y_min, x_max, y_max]`), image dimensions
- Creates blank uint8 mask (H×W, all zeros)
- Fills each box with 255 using `cv2.rectangle()`
- Applies morphological dilation:
  - Kernel size: `max(11, int(0.04 * max(img_w, img_h)))`, rounded up to next odd number
  - Kernel shape: `cv2.MORPH_ELLIPSE`
  - Iterations: 2
- Returns binary mask (values 0 or 255)

### `class LamaInpainter`

**`__init__(model_path: Path)`**
- Creates `onnxruntime.InferenceSession` with `CUDAExecutionProvider`, falls back to `CPUExecutionProvider` silently
- Session reused across all blobs on a page (instantiated once per `clean_page()` call)

**`infer(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray`**
- Input image: HWC uint8 RGB
- Input mask: HW uint8 (0/255)
- Normalize image to `[0,1]` float32, transpose HWC→CHW, add batch dim → `[1,3,H,W]`
- Mask: float32 `[0,1]`, reshape to `[1,1,H,W]`
- Run ONNX session
- Denormalize output → clip to `[0,255]` → uint8
- Returns inpainted tile, same shape as input

### `inpaint_page(pil_image: Image, boxes: list) -> Image`

1. Convert PIL → numpy BGR (for OpenCV ops)
2. Call `build_text_mask()` → dilated binary mask
3. `cv2.connectedComponentsWithStats()` with 8-connectivity
4. Filter blobs < 5px area; sort remaining by area descending (largest first)
5. Instantiate `LamaInpainter` (once)
6. For each blob:
   - Skip if centroid already in processed-region mask
   - Compute tile: blob bounding box + 32px padding on each side, clamped to image bounds
   - Reflect-pad tile to next multiple of 8: `cv2.copyMakeBorder(..., cv2.BORDER_REFLECT)`
   - Slice tile + mask tile from image and mask
   - `LamaInpainter.infer(tile, mask_tile)`
   - Unpad result (remove reflection padding)
   - Blend into output array (overwrite pixels where mask == 255)
   - Mark processed region in tracking mask
7. Convert output numpy → PIL RGB and return

---

## Changes to Existing Files

### `config.json`
Add key:
```json
"inpainting_engine": "color_fill"
```
Valid values: `"color_fill"` (default) or `"lama"`. Invalid value raises `ValueError` with clear message at startup.

### `inference.py: clean_page()`
- Read `inpainting_engine` from config
- `"color_fill"`: existing per-box loop calling `fill_bubble_with_estimated_color()` — no change
- `"lama"`: call `inpainting.inpaint_page(pil_image, boxes)`, save result

### `pyproject.toml`
Add `"onnxruntime"` to dependencies. GPU users can swap to `onnxruntime-gpu` manually; the CUDA→CPU fallback in `LamaInpainter` handles both transparently.

---

## Files Unchanged

- `img_utils.py` — color-fill path untouched
- `ocr_utils.py` — no changes
- `text_utils.py` — no changes
- `memory_utils.py` — no changes

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| `inpainting_engine` invalid value | `ValueError` at startup with valid options listed |
| Model download fails (no internet) | Exception propagates with clear message; color-fill remains available |
| CUDA unavailable | Silent fallback to CPU inside `LamaInpainter.__init__()` |
| Blob too small (<5px) | Skipped silently |
| All blobs already processed (overlap) | Loop exits cleanly, no re-processing |

---

## Out of Scope

- GUI or review UI changes
- Per-blob model switching
- Support for other inpainting models (only LaMa)
- Automatic model version updates
