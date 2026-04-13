# OCR + Translation Cache Design

**Date:** 2026-04-13
**Status:** Approved

## Problem

The `driver()` function in `inference.py` processes manga pages in two sequential phases:

1. **OCR phase** — detect text boxes, clean pages, extract text (GPU-intensive, results stored in-memory `computed` dict)
2. **Translation phase** — call LLM per page, save translated images to `output_dir`

Both phases are restarted from scratch on resume:
- OCR re-runs for all pages even if nothing changed.
- Translation skips already-output pages by output file existence, but does not verify the input image hasn't been swapped.

## Goal

- Persist OCR results to disk so resumed runs skip OCR for unchanged pages.
- Tie translation skipping to the input image hash so swapped input images always trigger a full re-run.

## Design

### Cache File Format

For each input image, a single sidecar file is written to `temp_dir`:

```
temp_dir/<img_name>.ocr.json
```

Contents:
```json
{
  "hash": "<sha256 hex digest of input image>",
  "texts": ["extracted text 1", "extracted text 2"],
  "text_boxes": [[x1, y1, x2, y2], ...],
  "page_context": "extracted text 1\n\nextracted text 2",
  "translated": true
}
```

- `page_context` is stored to avoid recomputing the join on load.
- The cleaned image (`<img_name>` in `temp_dir`) is already persisted by `clean_page()` and reused on cache hit.
- `translated` is written as `true` only after `translated_image.save(out_path)` succeeds. It is absent (or `false`) if the run was interrupted before translation completed.

### Cache Invalidation

Cache validity is determined by SHA-256 hash of the input image file. A hash mismatch (or missing cache file) invalidates both the OCR cache and the translation result, even if the output image exists on disk.

### Phase 1 — OCR Loop

For each page:

1. Compute SHA-256 of the input image.
2. If `temp_dir/<img_name>.ocr.json` exists and its `hash` matches:
   - **Cache hit:** load `texts`, `text_boxes`, `page_context` from JSON; set `clean_img_path` to existing cleaned image. Skip `detect_text`, `clean_page`, `extract_text`.
3. Otherwise:
   - **Cache miss:** run full pipeline, populate `computed`, write cache file (with `translated` absent).

```python
import hashlib

def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# In the OCR loop:
for img_name in tqdm(img_paths):
    img_path = os.path.join(input_dir, img_name)
    cache_path = os.path.join(temp_dir, img_name + ".ocr.json")
    current_hash = _file_hash(img_path)

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("hash") == current_hash:
            computed[img_path] = {
                "texts": cached["texts"],
                "text_boxes": cached["text_boxes"],
                "page_context": cached["page_context"],
                "clean_img_path": os.path.join(temp_dir, img_name),
                "cache_path": cache_path,
                "hash": current_hash,
            }
            continue

    # Cache miss: run full pipeline
    results = detect_text(img_path, det_model, image_processor)
    boxes = get_text_insertion_boxes(results, expand_ratio=0.8)
    cleaned_file_path = clean_page(img_path, temp_dir, boxes, segmentation_model, segmentation_processor)
    texts, text_boxes = extract_text(img_path, boxes, ocr_model, processor)
    page_context = "\n\n".join(texts)

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
        }, f)
```

### Phase 2 — Translation Loop

Replace the current `os.path.exists(out_path)` check with a hash + `translated` flag check:

```python
for i, img_name in enumerate(tqdm(img_paths)):
    img_path = os.path.join(input_dir, img_name)
    out_path = os.path.join(output_dir, img_name)
    cached = computed[img_path]

    # Skip if already translated for this exact input image
    cache_path = cached["cache_path"]
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache_data = json.load(f)
        if cache_data.get("hash") == cached["hash"] and cache_data.get("translated"):
            continue

    # ... translation logic unchanged ...

    translated_image.save(out_path)

    # Mark translation complete in cache
    with open(cache_path) as f:
        cache_data = json.load(f)
    cache_data["translated"] = True
    with open(cache_path, "w") as f:
        json.dump(cache_data, f)
```

### Scope

- `_file_hash()` helper added to `inference.py`
- `computed` dict extended with `cache_path` and `hash` fields (internal only)
- OCR loop modified to check/write per-page cache files
- Translation loop skip condition changed from output-file existence to hash + `translated` flag
- No new files, CLI flags, or external dependencies
- `hashlib` is stdlib; `json` already imported

## Out of Scope

- Caching detection boxes separately from OCR text
- Cache expiry by age
- Explicit `--no-cache` flag to force re-OCR
