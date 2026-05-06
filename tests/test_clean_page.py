import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image


def _make_rgb_image(w=100, h=100):
    return Image.fromarray(np.ones((h, w, 3), dtype=np.uint8) * 200)


def test_clean_page_color_fill_calls_fill_bubble(tmp_path):
    """clean_page() with 'color_fill' calls fill_bubble_with_estimated_color for each box."""
    img_path = tmp_path / "page.png"
    _make_rgb_image().save(img_path)
    boxes = [{"insertion_polygon": [10, 10, 50, 50]}]

    with patch("inference.fill_bubble_with_estimated_color", return_value=_make_rgb_image()) as mock_fill:
        from inference import clean_page
        clean_page(str(img_path), str(tmp_path), boxes, inpainting_engine="color_fill")
        mock_fill.assert_called_once()


def test_clean_page_lama_calls_inpaint_page(tmp_path):
    """clean_page() with 'lama' calls inpaint_page() instead of fill_bubble."""
    img_path = tmp_path / "page.png"
    _make_rgb_image().save(img_path)
    boxes = [{"insertion_polygon": [10, 10, 50, 50]}]

    with patch("inference.inpaint_page", return_value=_make_rgb_image()) as mock_inpaint, \
         patch("inference.fill_bubble_with_estimated_color") as mock_fill:
        from inference import clean_page
        clean_page(str(img_path), str(tmp_path), boxes, inpainting_engine="lama")
        mock_inpaint.assert_called_once()
        mock_fill.assert_not_called()


def test_clean_page_invalid_engine_raises(tmp_path):
    """clean_page() raises ValueError for an unknown inpainting_engine value."""
    img_path = tmp_path / "page.png"
    _make_rgb_image().save(img_path)

    with pytest.raises(ValueError, match="inpainting_engine"):
        from inference import clean_page
        clean_page(str(img_path), str(tmp_path), [], inpainting_engine="magic")


def test_clean_page_default_engine_is_color_fill(tmp_path):
    """clean_page() defaults to color_fill when inpainting_engine is omitted."""
    img_path = tmp_path / "page.png"
    _make_rgb_image().save(img_path)

    with patch("inference.fill_bubble_with_estimated_color", return_value=_make_rgb_image()) as mock_fill, \
         patch("inference.inpaint_page") as mock_inpaint:
        from inference import clean_page
        clean_page(str(img_path), str(tmp_path), [], inpainting_engine="color_fill")
        mock_inpaint.assert_not_called()
