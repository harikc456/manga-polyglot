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


def test_translate_not_called_for_existing_output(tmp_path):
    """Pages with an existing output file are skipped — translate() is never called for them."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir()
    output_dir.mkdir()
    temp_dir.mkdir()

    # Create two input images
    (input_dir / "page_001.jpg").write_bytes(b"fake")
    (input_dir / "page_002.jpg").write_bytes(b"fake")

    # page_001 is already translated
    (output_dir / "page_001.jpg").write_bytes(b"done")

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
    # translate() should have been called exactly once — for page_002 only
    assert translate_mock.call_count == 1


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


import json

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
