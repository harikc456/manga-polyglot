import re

from ocr._paddleocr_vl import DEFAULT_MODEL, get_min_pixels, load_model_and_processor
from ocr.base import Spotter
from ocr.registry import register

# Dense pages (many sound effects/narration boxes) can approach max_tokens; raise it in config if spotting looks incomplete.
MAX_PIXELS = 2048 * 28 * 28


def parse_spotting_output(raw: str, img_w: int, img_h: int) -> list[dict]:
    pattern = r'(.+?)((?:<\|LOC_\d+\|>){8})'
    boxes = []
    for line in raw.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        match = re.match(pattern, line)
        if not match:
            continue
        text = match.group(1).strip()
        loc_tokens = re.findall(r'<\|LOC_(\d+)\|>', match.group(2))
        if len(loc_tokens) != 8:
            continue
        coords = list(map(int, loc_tokens))
        xs = coords[0::2]
        ys = coords[1::2]
        boxes.append({
            'text': text,
            'x_min': int(min(xs) / 1000 * img_w),
            'y_min': int(min(ys) / 1000 * img_h),
            'x_max': int(max(xs) / 1000 * img_w),
            'y_max': int(max(ys) / 1000 * img_h),
        })
    return boxes


def spot_text(image, model, processor, max_tokens: int = 512) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Spotting:"},
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
    return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])


@register("spotter", "paddleocr_vl")
class PaddleOCRVLSpotter(Spotter):
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 512):
        self._model, self._processor = load_model_and_processor(model)
        self._max_tokens = max_tokens

    def spot(self, pil_image) -> list[dict]:
        raw = spot_text(pil_image, self._model, self._processor, self._max_tokens)
        img_w, img_h = pil_image.size
        return parse_spotting_output(raw, img_w, img_h)

    def close(self) -> None:
        self._model = None
        self._processor = None
