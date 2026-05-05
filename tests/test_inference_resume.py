import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

_MOCKS = {
    'cv2': MagicMock(),
    'torch': MagicMock(),
    'numpy': MagicMock(),
    'transformers': MagicMock(),
    'PIL': MagicMock(),
    'PIL.Image': MagicMock(),
    'tqdm': MagicMock(),
    'img_utils': MagicMock(),
    'ocr_utils': MagicMock(),
    'text_utils': MagicMock(),
    'data_model': MagicMock(),
    'memory_utils': MagicMock(),
}

with patch.dict(sys.modules, _MOCKS):
    import inference
    from inference import driver, _file_hash

sys.modules['inference'] = inference

_FAKE_SPOTTING_RAW = "HELLO<|LOC_100|><|LOC_100|><|LOC_200|><|LOC_100|><|LOC_200|><|LOC_200|><|LOC_100|><|LOC_200|>"
_FAKE_BOXES = [{"text": "Hello", "insertion_polygon": [0, 0, 10, 10]}]


def _make_driver_deps():
    patches = {
        "inference.AutoModelForImageTextToText": MagicMock(),
        "inference.AutoProcessor": MagicMock(),
        "inference.spot_text": MagicMock(return_value=_FAKE_SPOTTING_RAW),
        "inference._cluster_boxes": MagicMock(return_value=_FAKE_BOXES),
        "inference.clean_page": MagicMock(side_effect=lambda img_path, temp_dir, *a, **kw: img_path),
        "inference.translate": MagicMock(return_value="こんにちは"),
        "inference.update_session_memory": MagicMock(return_value=MagicMock()),
        "inference.replace_text_with_translation": MagicMock(return_value=MagicMock()),
        "inference.load_memory": MagicMock(return_value=MagicMock()),
        "inference.tqdm": MagicMock(side_effect=lambda x: x),
    }
    return patches


_BASE_CONFIG = {
    "ocr_model": "d",
    "llm_name": "d",
    "font_path": "d",
    "image_enabled": False,
}


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

    page1_hash = hashlib.sha256(page1_bytes).hexdigest()
    cache_001 = {
        "hash": page1_hash,
        "texts": ["Hello"],
        "text_boxes": [[0, 0, 10, 10]],
        "page_context": "Hello",
        "spotting_raw": _FAKE_SPOTTING_RAW,
        "cluster_eps": 80,
        "translated": True,
    }
    (temp_dir / "page_001.jpg.ocr.json").write_text(json.dumps(cache_001))
    (temp_dir / "page_001.jpg").write_bytes(b"cleaned")

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), _BASE_CONFIG, "Japanese", "English")

    assert patches["inference.translate"].call_count == 1


def test_translate_reruns_when_input_image_changes(tmp_path):
    """Translation re-runs when the input image hash doesn't match, even if an output file exists."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir(); output_dir.mkdir(); temp_dir.mkdir()

    (input_dir / "page_001.jpg").write_bytes(b"new content")
    (output_dir / "page_001.jpg").write_bytes(b"old translated output")

    stale_cache = {
        "hash": "a" * 64,
        "texts": ["old"],
        "text_boxes": [[0, 0, 10, 10]],
        "page_context": "old",
        "spotting_raw": _FAKE_SPOTTING_RAW,
        "cluster_eps": 80,
        "translated": True,
    }
    (temp_dir / "page_001.jpg.ocr.json").write_text(json.dumps(stale_cache))

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), _BASE_CONFIG, "Japanese", "English")

    assert patches["inference.translate"].call_count == 1


def test_all_pages_translated_when_no_output_exists(tmp_path):
    """When output_dir is empty, translate() is called for every page."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir(); output_dir.mkdir(); temp_dir.mkdir()

    (input_dir / "page_001.jpg").write_bytes(b"fake")
    (input_dir / "page_002.jpg").write_bytes(b"fake")

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), _BASE_CONFIG, "Japanese", "English")

    assert patches["inference.translate"].call_count == 2


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

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), _BASE_CONFIG, "Japanese", "English")

    cache_path = tmp_path / "temp" / "page_001.jpg.ocr.json"
    assert cache_path.exists()
    data = json.loads(cache_path.read_text())
    assert len(data["hash"]) == 64
    assert data["texts"] == ["Hello"]
    assert data["text_boxes"] == [[0, 0, 10, 10]]
    assert data["page_context"] == "Hello"
    assert "spotting_raw" in data
    assert "cluster_eps" in data


def test_ocr_cache_hit_skips_spotting(tmp_path):
    """spot_text is not called when a valid OCR cache with matching cluster_eps exists."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir(); output_dir.mkdir(); temp_dir.mkdir()

    img_bytes = b"fake image"
    (input_dir / "page_001.jpg").write_bytes(img_bytes)
    (temp_dir / "page_001.jpg").write_bytes(img_bytes)

    img_hash = hashlib.sha256(img_bytes).hexdigest()
    cache = {
        "hash": img_hash,
        "texts": ["cached text"],
        "text_boxes": [[0, 0, 5, 5]],
        "page_context": "cached text",
        "spotting_raw": _FAKE_SPOTTING_RAW,
        "cluster_eps": 80,
    }
    (temp_dir / "page_001.jpg.ocr.json").write_text(json.dumps(cache))

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), _BASE_CONFIG, "Japanese", "English")

    patches["inference.spot_text"].assert_not_called()
    patches["inference.clean_page"].assert_not_called()
    patches["inference.translate"].assert_called_once()


def test_cache_marked_translated_after_save(tmp_path):
    """Cache file has translated:true after a page is successfully translated."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir(); output_dir.mkdir(); temp_dir.mkdir()

    (input_dir / "page_001.jpg").write_bytes(b"fake image")

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), _BASE_CONFIG, "Japanese", "English")

    cache_path = tmp_path / "temp" / "page_001.jpg.ocr.json"
    data = json.loads(cache_path.read_text())
    assert data.get("translated") is True


def test_ocr_cache_miss_on_hash_mismatch(tmp_path):
    """spot_text runs when cache exists but hash doesn't match (input image changed)."""
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
        "spotting_raw": _FAKE_SPOTTING_RAW,
        "cluster_eps": 80,
    }
    (temp_dir / "page_001.jpg.ocr.json").write_text(json.dumps(stale_cache))

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), _BASE_CONFIG, "Japanese", "English")

    patches["inference.spot_text"].assert_called_once()


def test_memory_not_loaded_or_updated_when_memory_disabled(tmp_path):
    """When memory_enabled=False, load_memory and update_session_memory are never called."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir(); output_dir.mkdir(); temp_dir.mkdir()

    (input_dir / "page_001.jpg").write_bytes(b"fake image")

    config = {**_BASE_CONFIG, "memory_enabled": False}

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), config, "Japanese", "English")

    patches["inference.load_memory"].assert_not_called()
    patches["inference.update_session_memory"].assert_not_called()


def test_translate_called_with_use_json_false_when_json_disabled(tmp_path):
    """When json_enabled=False, translate() is called with use_json=False for every page."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir(); output_dir.mkdir(); temp_dir.mkdir()

    (input_dir / "page_001.jpg").write_bytes(b"fake image")

    config = {**_BASE_CONFIG, "json_enabled": False}

    patches = _make_driver_deps()
    with patch.multiple("inference", **{k.replace("inference.", ""): v for k, v in patches.items() if k.startswith("inference.")}), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
        driver(str(input_dir), str(temp_dir), str(output_dir), config, "Japanese", "English")

    translate_mock = patches["inference.translate"]
    assert translate_mock.call_count == 1
    for call in translate_mock.call_args_list:
        assert call.kwargs.get("use_json") is False
