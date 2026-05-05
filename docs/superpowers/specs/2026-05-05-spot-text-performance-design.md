# spot_text Performance: Remove Redundant Upscale and Lower max_new_tokens

**Date:** 2026-05-05
**Status:** Approved

## Problem

Pages with little or no text (background art, environment panels, panels with few speech bubbles) take ~20 minutes to process on an RTX 5060Ti 16GB. Two inefficiencies in `spot_text()` in `inference.py` compound to cause this:

1. **Redundant 2× upscale.** Images under 1500px on either dimension are doubled before being passed to the processor. The processor immediately applies its own `longest_edge` cap (`max_pixels = 2048 * 28 * 28 ≈ 1.6M pixels`), so the upscale is discarded. The cost is not: the processor must resize a larger-than-necessary image, increasing pre-processing time proportionally to the number of pixels added.

2. **`max_new_tokens=2048` always.** Generation time scales linearly with tokens produced. Typical manga spotting output is well under 200 tokens per page. A cap of 2048 wastes generation budget on every page and causes the model to continue searching for tokens on complex background pages with no text.

## Goal

Reduce per-page OCR time for low-text and text-free pages without affecting spotting accuracy on text-heavy pages.

## Design

### Change 1: Remove the 2× upscale

In `spot_text()`, delete the upscale block entirely:

```python
# Remove this:
if orig_w < 1500 and orig_h < 1500:
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    image = image.resize((orig_w * 2, orig_h * 2), resample)
```

The image is opened and converted to RGB, then passed directly to `processor.apply_chat_template`. The processor's `longest_edge` constraint handles resolution uniformly for all input sizes.

### Change 2: Lower the default max_new_tokens

Change the default value of `spotting_max_tokens` in `config.json` from 2048 to 512.

The `driver()` function already reads this from config via `config.get("spotting_max_tokens", 2048)` and passes it through to `spot_text()`. No code changes are needed beyond updating the config default.

**Truncation tradeoff:** If a page has so many text boxes that the spotting output exceeds 512 tokens, the output is cut off and text boxes late in reading order (bottom-right of the page) may be missed. This is unlikely in practice — a typical dense manga page produces 50–150 tokens of spotting output. If truncation becomes an issue for a specific chapter, `spotting_max_tokens` can be raised per-run in `config.json`.

## Scope

- 1 block deleted from `inference.py` (`spot_text`)
- 1 value changed in `config.json` (`spotting_max_tokens`: 2048 → 512)
- 1 new test in `tests/test_inference_resume.py` asserting the image passed to `processor.apply_chat_template` has the same dimensions as the source image (i.e., was not upscaled)

## Out of Scope

- Lightweight speech bubble pre-screening (separate initiative)
- Changing `longest_edge` or `min_pixels` processor parameters
- Any changes to clustering, translation, or the cache layer
