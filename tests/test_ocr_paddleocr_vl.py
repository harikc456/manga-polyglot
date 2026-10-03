from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from ocr._paddleocr_vl import get_min_pixels
from ocr.recognizers import paddleocr_vl as recognizer_mod
from ocr.spotters import paddleocr_vl as spotter_mod

_RAW = "HELLO<|LOC_100|><|LOC_100|><|LOC_200|><|LOC_100|><|LOC_200|><|LOC_200|><|LOC_100|><|LOC_200|>"


def _model_processor(decoded="hello"):
    processor = MagicMock()
    processor.image_processor.min_pixels = 256
    inputs = MagicMock()
    inputs.__getitem__ = lambda self, key: MagicMock(shape=[-1, 4])
    inputs.to.return_value = inputs
    processor.apply_chat_template.return_value = inputs
    processor.decode.return_value = decoded
    model = MagicMock()
    model.device = "cpu"
    model.generate.return_value = [list(range(10))]
    return model, processor


def _prompt_texts(processor):
    messages = processor.apply_chat_template.call_args[0][0]
    return [c["text"] for c in messages[0]["content"] if c.get("type") == "text"]


# --- shared helpers -------------------------------------------------------

def test_get_min_pixels_prefers_min_pixels_attribute():
    processor = SimpleNamespace(image_processor=SimpleNamespace(min_pixels=123, size={"shortest_edge": 99}))
    assert get_min_pixels(processor) == 123


def test_get_min_pixels_falls_back_to_shortest_edge():
    processor = SimpleNamespace(image_processor=SimpleNamespace(min_pixels=None, size={"shortest_edge": 99}))
    assert get_min_pixels(processor) == 99


# --- spotter --------------------------------------------------------------

def test_parse_spotting_output_scales_to_pixels():
    boxes = spotter_mod.parse_spotting_output(_RAW, 2000, 1000)
    assert boxes == [{"text": "HELLO", "x_min": 200, "y_min": 100, "x_max": 400, "y_max": 200}]


def test_parse_spotting_output_skips_malformed_and_blank_lines():
    raw = "no coordinates here\n\n" + _RAW + "\nTOO FEW<|LOC_1|><|LOC_2|>"
    boxes = spotter_mod.parse_spotting_output(raw, 1000, 1000)
    assert [b["text"] for b in boxes] == ["HELLO"]


def test_spot_text_uses_spotting_prompt_and_does_not_resize():
    model, processor = _model_processor(decoded="raw")
    image = Image.new("RGB", (800, 1200))
    result = spotter_mod.spot_text(image, model, processor)
    assert "Spotting:" in _prompt_texts(processor)
    messages = processor.apply_chat_template.call_args[0][0]
    assert messages[0]["content"][0]["image"] is image
    assert result == "raw"


def test_spotter_spot_returns_pixel_boxes():
    model, processor = _model_processor()
    with patch.object(spotter_mod, "load_model_and_processor", return_value=(model, processor)), \
         patch.object(spotter_mod, "spot_text", return_value=_RAW) as spot:
        spotter = spotter_mod.PaddleOCRVLSpotter(model="m", max_tokens=77)
        boxes = spotter.spot(Image.new("RGB", (1000, 1000)))
    assert boxes == [{"text": "HELLO", "x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200}]
    assert spot.call_args[0][3] == 77


def test_spotter_close_drops_model_references():
    model, processor = _model_processor()
    with patch.object(spotter_mod, "load_model_and_processor", return_value=(model, processor)):
        spotter = spotter_mod.PaddleOCRVLSpotter()
    spotter.close()
    assert spotter._model is None and spotter._processor is None


# --- recognizer -----------------------------------------------------------

def test_read_crop_text_uses_ocr_prompt():
    model, processor = _model_processor()
    recognizer_mod.read_crop_text(Image.new("RGB", (64, 64)), model, processor)
    assert "OCR:" in _prompt_texts(processor)


def test_read_crop_text_returns_stripped_string():
    model, processor = _model_processor(decoded="  hello world  ")
    assert recognizer_mod.read_crop_text(Image.new("RGB", (64, 64)), model, processor) == "hello world"


def test_recognizer_read_passes_max_tokens():
    model, processor = _model_processor(decoded="hi")
    with patch.object(recognizer_mod, "load_model_and_processor", return_value=(model, processor)):
        recognizer = recognizer_mod.PaddleOCRVLRecognizer(model="m", max_tokens=33)
    assert recognizer.read(Image.new("RGB", (64, 64))) == "hi"
    assert model.generate.call_args.kwargs["max_new_tokens"] == 33


def test_recognizer_close_drops_model_references():
    model, processor = _model_processor()
    with patch.object(recognizer_mod, "load_model_and_processor", return_value=(model, processor)):
        recognizer = recognizer_mod.PaddleOCRVLRecognizer()
    recognizer.close()
    assert recognizer._model is None and recognizer._processor is None
