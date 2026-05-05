# spot_text Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the redundant 2× upscale from `spot_text()` and lower `max_new_tokens` to 512 to reduce per-page OCR time for low-text and text-free pages.

**Architecture:** Two independent changes — delete the upscale block in `inference.py:spot_text()` (the processor already caps resolution via `longest_edge`), and lower the `spotting_max_tokens` default in `config.json` from 2048 to 512. One new test verifies the upscale no longer happens.

**Tech Stack:** Python, PIL, unittest.mock, pytest

---

### Task 1: Add a failing test for the no-upscale behaviour

**Files:**
- Modify: `tests/test_inference_resume.py`

- [ ] **Step 1: Write the failing test**

Add this test at the bottom of `tests/test_inference_resume.py`:

```python
def test_spot_text_does_not_resize_image():
    """spot_text passes the image to the processor at its original size."""
    mock_img = MagicMock()
    mock_converted = MagicMock()
    mock_converted.size = (800, 1200)   # both dims < 1500 — triggers old upscale path
    mock_img.convert.return_value = mock_converted

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value.to.return_value = MagicMock()
    mock_processor.decode.return_value = ""
    mock_model = MagicMock()
    mock_model.device = "cpu"

    with patch.object(inference.Image, "open", return_value=mock_img):
        inference.spot_text("fake/path.jpg", mock_model, mock_processor)

    mock_converted.resize.assert_not_called()
```

The test creates a mock image with size (800, 1200) — both under 1500 — which is the condition that triggers the upscale. It then asserts `resize` was never called on the converted image.

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd /home/harikrishnan-c/projects/manga-polyglot
uv run pytest tests/test_inference_resume.py::test_spot_text_does_not_resize_image -v
```

Expected:
```
FAILED tests/test_inference_resume.py::test_spot_text_does_not_resize_image
AssertionError: Expected 'resize' to not have been called. Called 1 times.
```

---

### Task 2: Remove the upscale block and commit

**Files:**
- Modify: `inference.py:29-62` (`spot_text` function)

- [ ] **Step 1: Delete the upscale block**

In `inference.py`, replace the body of `spot_text` so it reads:

```python
def spot_text(img_path: str, model, processor, max_tokens: int = 2048) -> str:
    image = Image.open(img_path).convert("RGB")
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
```

The removed lines are `orig_w, orig_h = image.size` and the entire `if orig_w < 1500 and orig_h < 1500:` block (lines 31–37 in the original).

- [ ] **Step 2: Run the new test to confirm it passes**

```bash
uv run pytest tests/test_inference_resume.py::test_spot_text_does_not_resize_image -v
```

Expected:
```
PASSED
```

- [ ] **Step 3: Run the full suite to check for regressions**

```bash
uv run pytest -v
```

Expected: all 51 tests pass.

- [ ] **Step 4: Commit**

```bash
git add inference.py tests/test_inference_resume.py
git commit -m "perf: remove redundant 2x upscale from spot_text"
```

---

### Task 3: Lower the default max_new_tokens

**Files:**
- Modify: `config.json`

- [ ] **Step 1: Update the config value**

In `config.json`, change `spotting_max_tokens` from `2048` to `512`:

```json
{
    "ocr_model": "PaddlePaddle/PaddleOCR-VL-1.5",
    "llm_name": "translategemma:12b",
    "font_path": "./fonts/animeace2_bld.otf",
    "image_enabled": false,
    "json_enabled": false,
    "memory_enabled": false,
    "spotting_cluster_eps": 80,
    "spotting_max_tokens": 512
}
```

- [ ] **Step 2: Run the full suite to confirm nothing broke**

```bash
uv run pytest -v
```

Expected: all 51 tests pass.

- [ ] **Step 3: Commit**

```bash
git add config.json
git commit -m "perf: lower spotting_max_tokens default to 512"
```
