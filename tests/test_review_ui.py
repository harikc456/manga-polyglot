import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture()
def dirs(tmp_path):
    input_dir = tmp_path / "input"
    temp_dir = tmp_path / "temp"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    temp_dir.mkdir()
    output_dir.mkdir()

    # A tiny real image so PIL can open it
    img = Image.new("RGB", (100, 150), color=(240, 240, 240))
    img.save(str(input_dir / "001.jpg"))

    # OCR cache
    cache = {
        "hash": "abc123",
        "texts": ["こんにちは"],
        "text_boxes": [[10, 10, 50, 30]],
        "page_context": "こんにちは",
        "boxes": [
            {
                "original_text_box": [10, 10, 50, 30],
                "insertion_polygon": [8, 8, 52, 32],
                "confidence": 0.94,
                "type": "fixed",
            }
        ],
        "translations": [{"original": "こんにちは", "translated": "Hello!"}],
        "translated": True,
    }
    (temp_dir / "001.jpg.ocr.json").write_text(json.dumps(cache))

    # Cleaned image
    cleaned = Image.new("RGB", (100, 150), color=(255, 255, 255))
    cleaned.save(str(temp_dir / "001.jpg"))

    # Output image
    output = Image.new("RGB", (100, 150), color=(220, 220, 220))
    output.save(str(output_dir / "001.jpg"))

    return input_dir, temp_dir, output_dir


@pytest.fixture()
def client(dirs):
    input_dir, temp_dir, output_dir = dirs
    from review_ui import create_app
    app = create_app(input_dir, temp_dir, output_dir)
    return TestClient(app)


def test_pages_returns_list(client):
    resp = client.get("/api/pages")
    assert resp.status_code == 200
    pages = resp.json()
    assert isinstance(pages, list)
    assert len(pages) == 1
    assert pages[0]["name"] == "001.jpg"
    assert pages[0]["status"] == "unseen"


def test_pages_reflects_review_log(dirs):
    input_dir, temp_dir, output_dir = dirs
    review_log = {"001.jpg": {"status": "flagged", "notes": "bad clean", "timestamp": "2026-05-05T10:00:00"}}
    (temp_dir / "review_log.json").write_text(json.dumps(review_log))

    from review_ui import create_app
    app = create_app(input_dir, temp_dir, output_dir)
    c = TestClient(app)
    resp = c.get("/api/pages")
    assert resp.json()[0]["status"] == "flagged"
