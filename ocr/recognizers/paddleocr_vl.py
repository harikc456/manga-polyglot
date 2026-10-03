from ocr._paddleocr_vl import DEFAULT_MODEL, get_min_pixels, load_model_and_processor
from ocr.base import Recognizer
from ocr.registry import register

MAX_PIXELS = 512 * 28 * 28


def read_crop_text(crop, model, processor, max_tokens: int = 128) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop},
                {"type": "text", "text": "OCR:"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "images_kwargs": {
                "size": {
                    "shortest_edge": get_min_pixels(processor),
                    "longest_edge": MAX_PIXELS,
                }
            },
        },
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1]).strip()


@register("recognizer", "paddleocr_vl")
class PaddleOCRVLRecognizer(Recognizer):
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 128):
        self._model, self._processor = load_model_and_processor(model)
        self._max_tokens = max_tokens

    def read(self, crop) -> str:
        return read_crop_text(crop, self._model, self._processor, self._max_tokens)

    def close(self) -> None:
        self._model = None
        self._processor = None
