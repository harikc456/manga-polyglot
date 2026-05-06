import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock
import pytest


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
