import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock
import pytest
import numpy as np


def test_ensure_model_returns_correct_path():
    """ensure_model() returns a Path ending in lama-manga.onnx under ~/.manga-polyglot/models/."""
    from inpainting import ensure_model, MODEL_PATH
    with patch("inpainting.MODEL_PATH") as mock_path:
        mock_path.exists.return_value = True
        mock_path.__str__ = lambda self: str(Path.home() / ".manga-polyglot" / "models" / "lama-manga.onnx")
        result = ensure_model()
        assert str(result).endswith("lama-manga.onnx")


def test_ensure_model_skips_download_if_file_exists(tmp_path):
    """ensure_model() does not call urlretrieve when the model file already exists."""
    fake_model = tmp_path / "lama-manga.onnx"
    fake_model.write_bytes(b"fake")
    with patch("inpainting.MODEL_PATH", fake_model), \
         patch("urllib.request.urlretrieve") as mock_dl:
        from inpainting import ensure_model
        ensure_model()
        mock_dl.assert_not_called()


def test_ensure_model_downloads_when_missing(tmp_path):
    """ensure_model() calls urlretrieve with the correct URL when the model file is absent."""
    fake_model = tmp_path / "lama-manga.onnx"
    with patch("inpainting.MODEL_PATH", fake_model), \
         patch("urllib.request.urlretrieve") as mock_dl:
        mock_dl.side_effect = lambda url, path, reporthook: Path(path).write_bytes(b"fake")
        from inpainting import ensure_model, MODEL_URL
        result = ensure_model()
        mock_dl.assert_called_once()
        assert mock_dl.call_args[0][0] == MODEL_URL
        assert result == fake_model


def test_build_text_mask_shape():
    """build_text_mask returns a uint8 mask of the correct image dimensions."""
    from inpainting import build_text_mask
    boxes = [{"insertion_polygon": [10, 10, 50, 50]}]
    mask = build_text_mask(boxes, img_w=100, img_h=80)
    assert mask.shape == (80, 100)
    assert mask.dtype == np.uint8


def test_build_text_mask_covers_box():
    """Mask has 255 inside the box region before dilation."""
    from inpainting import build_text_mask
    # Use a box far from edges so dilation doesn't complicate this check
    boxes = [{"insertion_polygon": [40, 40, 60, 60]}]
    mask = build_text_mask(boxes, img_w=200, img_h=200)
    # Centre of box must be white
    assert mask[50, 50] == 255


def test_build_text_mask_is_dilated():
    """Mask extends beyond the original box due to morphological dilation."""
    from inpainting import build_text_mask
    boxes = [{"insertion_polygon": [50, 50, 100, 100]}]
    mask = build_text_mask(boxes, img_w=300, img_h=300)
    # A pixel just outside the box should be white (dilated)
    assert mask[48, 75] == 255  # 2 pixels above top edge
    assert mask[75, 48] == 255  # 2 pixels left of left edge


def test_build_text_mask_empty_boxes():
    """build_text_mask with no boxes returns an all-zero mask."""
    from inpainting import build_text_mask
    mask = build_text_mask([], img_w=100, img_h=100)
    assert mask.max() == 0
