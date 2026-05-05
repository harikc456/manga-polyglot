# Review UI Design

**Date:** 2026-05-05  
**Status:** Approved

## Overview

A standalone FastAPI web app (`review_ui.py`) for reviewing the manga translation pipeline's output stage by stage. It reads from existing temp/input/output directories — it does not invoke any pipeline code. The user can approve or flag each page and add free-text notes to document shortcomings. Results are logged to `review_log.json` in the temp dir. The pipeline is never blocked; the UI is for auditing.

## Running the Tool

```bash
python review_ui.py \
  --input-dir ./input \
  --temp-dir  ./temp \
  --output-dir ./output
```

Launches a uvicorn server at `http://localhost:8765` and prints the URL. The browser opens automatically (via `webbrowser.open`).

## UI Layout

**Sidebar + 3-stage detail panel.**

- **Top bar**: pipeline summary (N approved / N flagged / N unseen) + "Export review_log.json" button.
- **Left sidebar**: scrollable list of all pages. Each entry shows filename, a 30px-wide thumbnail (served from `/image/thumbnail/{name}`, which is the original image), and a color-coded status icon (green = approved, red = flagged, grey = unseen). Clicking a page loads it into the detail panel.
- **Main panel**: three equal-width columns, one per stage:
  1. **Detection** — original image with bounding boxes drawn server-side (PIL). Red = `FIXED` bubble, orange = `FREE` text. Each box is labelled with its confidence score.
  2. **Cleaning** — cleaned image from temp dir (text inpainted out).
  3. **Translation** — final output image from output dir, with OCR→translation pairs shown below as a text table.
- **Bottom bar**: free-text notes textarea + "Approve" and "Flag page" buttons. Notes persist to `review_log.json` on each save.
- **Panel header**: shows filename, bubble count badge, current status badge, and Prev/Next navigation.

## Architecture

### Files

| File | Role |
|------|------|
| `review_ui.py` | FastAPI app + uvicorn entry point |
| `review_frontend/index.html` | Single-page frontend (no build step) |
| `review_frontend/app.js` | Sidebar logic, image loading, review save |
| `review_frontend/style.css` | Dark theme matching mockup |

### API Routes

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/api/pages` | List all pages (scanned from temp dir `*.ocr.json`), each with review status from `review_log.json` |
| `GET` | `/api/page/{name}` | Box metadata, OCR texts, translations for one page |
| `GET` | `/image/detection/{name}` | Original image with boxes drawn (PIL, server-side) |
| `GET` | `/image/cleaned/{name}` | Cleaned image from temp dir |
| `GET` | `/image/output/{name}` | Final translated image from output dir |
| `POST` | `/api/review/{name}` | Save `{status, notes, timestamp}` to `review_log.json` |
| `GET` | `/api/export` | Returns `review_log.json` as a file download |
| `GET` | `/image/thumbnail/{name}` | Original image CSS-scaled to 30px wide for sidebar |
| `GET` | `/` | Serves `review_frontend/index.html` |

### Data Flow

1. On startup, the server scans `--temp-dir` for `*.ocr.json` files to build the page list.
2. `review_log.json` (in temp dir) is loaded if it exists; otherwise initialized as `{}`.
3. When the user clicks a page in the sidebar, the frontend calls `/api/page/{name}` which reads the OCR cache and returns box data + texts.
4. The frontend sets the `<img>` srcs to the three image endpoints. Detection image is rendered server-side on each request.
5. Approve/Flag button calls `POST /api/review/{name}`, which updates `review_log.json` in memory and writes it to disk.
6. "Export" button triggers `GET /api/export` which returns `review_log.json` as a file download.

## Cache Schema Extension

The current OCR cache (`*.ocr.json`) stores only `texts`, `text_boxes` (insertion polygons), `page_context`, and `hash`. To support the detection panel (confidence scores + bubble types), the cache schema is extended:

**Current:**
```json
{
  "hash": "...",
  "texts": ["...", "..."],
  "text_boxes": [[x1,y1,x2,y2], ...],
  "page_context": "..."
}
```

**Extended:**
```json
{
  "hash": "...",
  "texts": ["...", "..."],
  "text_boxes": [[x1,y1,x2,y2], ...],
  "page_context": "...",
  "boxes": [
    {
      "original_text_box": [x1,y1,x2,y2],
      "insertion_polygon": [x1,y1,x2,y2],
      "confidence": 0.94,
      "type": "fixed"
    }
  ],
  "translated": false
}
```

`inference.py` must be updated to write `boxes` to the cache when it first processes a page. The review UI reads `boxes` from the cache; if `boxes` is absent (old cache), the detection panel falls back to drawing `text_boxes` without confidence labels.

## Output: `review_log.json`

```json
{
  "001.jpg": {
    "status": "approved",
    "notes": "",
    "timestamp": "2026-05-05T14:23:01"
  },
  "003.jpg": {
    "status": "flagged",
    "notes": "sfx bubble: TELEA left ink residue. Detection missed small bubble top-left.",
    "timestamp": "2026-05-05T14:25:44"
  }
}
```

## Dependencies

- `fastapi` + `uvicorn` — web server
- `pillow` — server-side bounding box rendering (already a project dependency)
- No new frontend libraries — plain HTML/CSS/JS

Add `fastapi` and `uvicorn` to `pyproject.toml`.

## Out of Scope

- Editing translations in the UI (view only)
- Re-running pipeline stages from the UI
- Per-bubble approval (per-page only)
- Authentication or multi-user support
