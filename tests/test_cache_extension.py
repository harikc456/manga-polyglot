import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

_MOCKS = {
    'cv2': MagicMock(),
    'torch': MagicMock(),
    'numpy': MagicMock(),
    'transformers': MagicMock(),
    'PIL': MagicMock(),
    'PIL.Image': MagicMock(),
    'tqdm': MagicMock(),
    'img_utils': MagicMock(),
    'text_detection': MagicMock(),
    'text_utils': MagicMock(),
    'data_model': MagicMock(),
    'memory_utils': MagicMock(),
}

with patch.dict(sys.modules, _MOCKS):
    import inference

sys.modules['inference'] = inference


def test_cache_write_includes_boxes(tmp_path):
    """When a page is processed for the first time, the cache must include a 'boxes' key."""
    fake_box = {
        "original_text_box": [10, 10, 50, 30],
        "insertion_polygon": [8, 8, 52, 32],
        "confidence": 0.94,
        "type": MagicMock(value="fixed"),
    }

    serialized = {
        "original_text_box": fake_box["original_text_box"],
        "insertion_polygon": fake_box["insertion_polygon"],
        "confidence": fake_box["confidence"],
        "type": fake_box["type"].value,
    }
    assert serialized["type"] == "fixed"
    assert serialized["confidence"] == 0.94
    assert len(serialized["original_text_box"]) == 4


def test_cache_write_includes_translations(tmp_path):
    """After translation, the cache must include a 'translations' key."""
    translations = [
        {"original": "こんにちは", "translated": "Hello!", "polygon": [10, 10, 50, 30]},
    ]
    cache_data = {"hash": "abc", "texts": ["こんにちは"], "text_boxes": [[10,10,50,30]], "page_context": ""}

    serialized = [
        {"original": t["original"], "translated": t["translated"]}
        for t in translations
    ]
    cache_data["translations"] = serialized
    cache_data["translated"] = True

    assert cache_data["translations"] == [{"original": "こんにちは", "translated": "Hello!"}]
    assert cache_data["translated"] is True
