import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

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
    'ocr_utils': MagicMock(),
    'text_utils': MagicMock(),
    'data_model': MagicMock(),
    'memory_utils': MagicMock(),
}

with patch.dict(sys.modules, _MOCKS):
    import inference

sys.modules['inference'] = inference


def test_cache_write_includes_spotting_raw_and_cluster_eps():
    """First-pass cache must include spotting_raw and cluster_eps."""
    cache_data = {
        "hash": "abc123",
        "texts": ["FROM MY TEACHER"],
        "text_boxes": [[498, 80, 583, 111]],
        "page_context": "FROM MY TEACHER",
        "spotting_raw": "FROM MY<|LOC_498|><|LOC_80|><|LOC_580|><|LOC_80|><|LOC_580|><|LOC_93|><|LOC_498|><|LOC_93|>",
        "cluster_eps": 80,
    }
    assert "spotting_raw" in cache_data
    assert "cluster_eps" in cache_data
    assert cache_data["cluster_eps"] == 80


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
