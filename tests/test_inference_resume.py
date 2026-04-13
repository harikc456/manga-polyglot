import hashlib
import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Mock heavy dependencies before importing inference
sys.modules['cv2'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
sys.modules['tqdm'] = MagicMock()
sys.modules['img_utils'] = MagicMock()
sys.modules['text_detection'] = MagicMock()
sys.modules['text_utils'] = MagicMock()
sys.modules['data_model'] = MagicMock()
sys.modules['memory_utils'] = MagicMock()

import inference
from inference import driver, _file_hash

def _make_driver_deps():
    """Return the minimal mock set needed to run driver() without real models."""
    # Patch all heavy imports inside inference
    patches = {
        "inference.Sam3Model": MagicMock(),
        "inference.Sam3Processor": MagicMock(),
        "inference.RTDetrV2ForObjectDetection": MagicMock(),
        "inference.RTDetrImageProcessor": MagicMock(),
        "inference.AutoModelForImageTextToText": MagicMock(),
        "inference.AutoProcessor": MagicMock(),
        "inference.detect_text": MagicMock(return_value=[]),
        "inference.get_text_insertion_boxes": MagicMock(return_value=[]),
        "inference.clean_page": MagicMock(side_effect=lambda img_path, temp_dir, *a, **kw: img_path),
        "inference.extract_text": MagicMock(return_value=(["Hello"], [(0, 0, 10, 10)])),
        "inference.translate": MagicMock(return_value="こんにちは"),
        "inference.update_session_memory": MagicMock(return_value=MagicMock()),
        "inference.replace_text_with_translation": MagicMock(return_value=MagicMock()),
        "inference.load_memory": MagicMock(return_value=MagicMock()),
        "inference.tqdm": MagicMock(side_effect=lambda x: x),
        "torch.cuda.is_available": MagicMock(return_value=False),
    }
    return patches


def test_translate_not_called_for_translated_cache(tmp_path):
    """Pages with translated:true in their cache are skipped — translate() not called for them."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir(); output_dir.mkdir(); temp_dir.mkdir()

    page1_bytes = b"fake page 1"
    page2_bytes = b"fake page 2"
    (input_dir / "page_001.jpg").write_bytes(page1_bytes)
    (input_dir / "page_002.jpg").write_bytes(page2_bytes)

    # page_001 already translated — write its cache with translated:true
    page1_hash = hashlib.sha256(page1_bytes).hexdigest()
    cache_001 = {
        "hash": page1_hash,
        "texts": ["Hello"],
        "text_boxes": [[0, 0, 10, 10]],
        "page_context": "Hello",
        "translated": True,
    }
    (temp_dir / "page_001.jpg.ocr.json").write_text(json.dumps(cache_001))
    # Also write the cleaned image so the cache hit is valid
    (temp_dir / "page_001.jpg").write_bytes(b"cleaned")

    config = {
        "text_detection_model_path": "d", "ocr_model": "d",
        "llm_name": "d", "font_path": "d", "image_enabled": False,
    }

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), config, "Japanese", "English")

    # translate() called exactly once — for page_002 only
    assert patches["inference.translate"].call_count == 1


def test_translate_reruns_when_input_image_changes(tmp_path):
    """Translation re-runs when the input image hash doesn't match, even if an output file exists."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir(); output_dir.mkdir(); temp_dir.mkdir()

    (input_dir / "page_001.jpg").write_bytes(b"new content")
    # Output file exists — should be ignored because hash won't match
    (output_dir / "page_001.jpg").write_bytes(b"old translated output")

    stale_cache = {
        "hash": "a" * 64,  # wrong hash
        "texts": ["old"],
        "text_boxes": [[0, 0, 10, 10]],
        "page_context": "old",
        "translated": True,
    }
    (temp_dir / "page_001.jpg.ocr.json").write_text(json.dumps(stale_cache))

    config = {
        "text_detection_model_path": "d", "ocr_model": "d",
        "llm_name": "d", "font_path": "d", "image_enabled": False,
    }

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), config, "Japanese", "English")

    assert patches["inference.translate"].call_count == 1


def test_all_pages_translated_when_no_output_exists(tmp_path):
    """When output_dir is empty, translate() is called for every page."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir()
    output_dir.mkdir()
    temp_dir.mkdir()

    (input_dir / "page_001.jpg").write_bytes(b"fake")
    (input_dir / "page_002.jpg").write_bytes(b"fake")

    config = {
        "text_detection_model_path": "dummy",
        "ocr_model": "dummy",
        "llm_name": "dummy",
        "font_path": "dummy",
        "image_enabled": False,
    }

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), \
         patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), config, "Japanese", "English")

    translate_mock = patches["inference.translate"]
    assert translate_mock.call_count == 2


def test_file_hash_consistent(tmp_path):
    f = tmp_path / "img.jpg"
    f.write_bytes(b"fake image data")
    assert _file_hash(str(f)) == _file_hash(str(f))


def test_file_hash_is_64_chars(tmp_path):
    f = tmp_path / "img.jpg"
    f.write_bytes(b"fake image data")
    assert len(_file_hash(str(f))) == 64


def test_file_hash_differs_for_different_content(tmp_path):
    f1 = tmp_path / "a.jpg"
    f2 = tmp_path / "b.jpg"
    f1.write_bytes(b"content A")
    f2.write_bytes(b"content B")
    assert _file_hash(str(f1)) != _file_hash(str(f2))


def test_ocr_cache_written_after_run(tmp_path):
    """A .ocr.json cache file is written to temp_dir for each page after a run."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir(); output_dir.mkdir(); temp_dir.mkdir()

    (input_dir / "page_001.jpg").write_bytes(b"fake image")

    config = {
        "text_detection_model_path": "d", "ocr_model": "d",
        "llm_name": "d", "font_path": "d", "image_enabled": False,
    }

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), config, "Japanese", "English")

    cache_path = tmp_path / "temp" / "page_001.jpg.ocr.json"
    assert cache_path.exists()
    data = json.loads(cache_path.read_text())
    assert len(data["hash"]) == 64
    assert data["texts"] == ["Hello"]
    assert data["text_boxes"] == [[0, 0, 10, 10]]
    assert data["page_context"] == "Hello"


def test_ocr_cache_hit_skips_ocr(tmp_path):
    """detect_text, clean_page, extract_text are not called when a valid OCR cache exists."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir(); output_dir.mkdir(); temp_dir.mkdir()

    img_bytes = b"fake image"
    (input_dir / "page_001.jpg").write_bytes(img_bytes)

    # Pre-populate the cleaned image so the cache hit condition is satisfied
    (temp_dir / "page_001.jpg").write_bytes(img_bytes)

    img_hash = hashlib.sha256(img_bytes).hexdigest()
    cache = {
        "hash": img_hash,
        "texts": ["cached text"],
        "text_boxes": [[0, 0, 5, 5]],
        "page_context": "cached text",
    }
    (temp_dir / "page_001.jpg.ocr.json").write_text(json.dumps(cache))

    config = {
        "text_detection_model_path": "d", "ocr_model": "d",
        "llm_name": "d", "font_path": "d", "image_enabled": False,
    }

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), config, "Japanese", "English")

    patches["inference.detect_text"].assert_not_called()
    patches["inference.clean_page"].assert_not_called()
    patches["inference.extract_text"].assert_not_called()


def test_ocr_cache_miss_on_hash_mismatch(tmp_path):
    """OCR runs when cache exists but hash doesn't match (input image changed)."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir(); output_dir.mkdir(); temp_dir.mkdir()

    (input_dir / "page_001.jpg").write_bytes(b"new content")

    stale_cache = {
        "hash": "a" * 64,
        "texts": ["old text"],
        "text_boxes": [],
        "page_context": "old text",
    }
    (temp_dir / "page_001.jpg.ocr.json").write_text(json.dumps(stale_cache))

    config = {
        "text_detection_model_path": "d", "ocr_model": "d",
        "llm_name": "d", "font_path": "d", "image_enabled": False,
    }

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), config, "Japanese", "English")

    patches["inference.extract_text"].assert_called_once()
