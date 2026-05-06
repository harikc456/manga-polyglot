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


def _make_fake_ort_session(output_shape):
    """Return a mock onnxruntime.InferenceSession that outputs a fixed array."""
    session = MagicMock()
    fake_input_0 = MagicMock()
    fake_input_0.name = "image"
    fake_input_1 = MagicMock()
    fake_input_1.name = "mask"
    session.get_inputs.return_value = [fake_input_0, fake_input_1]
    # output[0] shape: [1, 3, H, W] — normalized float32
    h, w = output_shape
    session.run.return_value = [np.ones((1, 3, h, w), dtype=np.float32) * 0.5]
    return session


def test_lama_inpainter_infer_output_shape(tmp_path):
    """LamaInpainter.infer() returns an array with the same HxW as the input."""
    fake_model = tmp_path / "lama-manga.onnx"
    fake_model.write_bytes(b"fake")

    with patch("onnxruntime.InferenceSession", return_value=_make_fake_ort_session((64, 64))):
        from inpainting import LamaInpainter
        inpainter = LamaInpainter(fake_model)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = inpainter.infer(img, mask)
        assert result.shape == (64, 64, 3)


def test_lama_inpainter_infer_output_dtype(tmp_path):
    """LamaInpainter.infer() returns uint8 values in [0, 255]."""
    fake_model = tmp_path / "lama-manga.onnx"
    fake_model.write_bytes(b"fake")

    with patch("onnxruntime.InferenceSession", return_value=_make_fake_ort_session((32, 32))):
        from inpainting import LamaInpainter
        inpainter = LamaInpainter(fake_model)
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        result = inpainter.infer(img, mask)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255


from PIL import Image as PILImage


def _fake_inpainter_passthrough(img_rgb, mask):
    """Identity inpainter — returns the tile unchanged."""
    return img_rgb.copy()


def test_inpaint_page_returns_pil_image(tmp_path):
    """inpaint_page() returns a PIL Image of the same size as the input."""
    img = PILImage.fromarray(np.ones((200, 200, 3), dtype=np.uint8) * 200)
    boxes = [{"insertion_polygon": [50, 50, 150, 150]}]
    fake_model = tmp_path / "lama-manga.onnx"
    fake_model.write_bytes(b"fake")

    with patch("inpainting.ensure_model", return_value=fake_model), \
         patch("inpainting.LamaInpainter") as MockInpainter:
        MockInpainter.return_value.infer.side_effect = _fake_inpainter_passthrough
        from inpainting import inpaint_page
        result = inpaint_page(img, boxes)
        assert isinstance(result, PILImage.Image)
        assert result.size == img.size


def test_inpaint_page_processes_text_regions(tmp_path):
    """inpaint_page() calls LamaInpainter.infer() at least once when boxes exist."""
    img = PILImage.fromarray(np.ones((200, 200, 3), dtype=np.uint8) * 200)
    boxes = [{"insertion_polygon": [50, 50, 150, 150]}]
    fake_model = tmp_path / "lama-manga.onnx"
    fake_model.write_bytes(b"fake")

    with patch("inpainting.ensure_model", return_value=fake_model), \
         patch("inpainting.LamaInpainter") as MockInpainter:
        mock_inpainter = MockInpainter.return_value
        mock_inpainter.infer.side_effect = _fake_inpainter_passthrough
        from inpainting import inpaint_page
        inpaint_page(img, boxes)
        assert mock_inpainter.infer.call_count >= 1


def test_inpaint_page_no_boxes_returns_unchanged(tmp_path):
    """inpaint_page() with empty boxes returns image without calling infer."""
    arr = np.ones((100, 100, 3), dtype=np.uint8) * 128
    img = PILImage.fromarray(arr)
    fake_model = tmp_path / "lama-manga.onnx"
    fake_model.write_bytes(b"fake")

    with patch("inpainting.ensure_model", return_value=fake_model), \
         patch("inpainting.LamaInpainter") as MockInpainter:
        mock_inpainter = MockInpainter.return_value
        from inpainting import inpaint_page
        result = inpaint_page(img, [])
        mock_inpainter.infer.assert_not_called()
        assert np.array_equal(np.array(result), arr)
