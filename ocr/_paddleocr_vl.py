DEFAULT_MODEL = "PaddlePaddle/PaddleOCR-VL-1.5"


def get_min_pixels(processor) -> int:
    """min_pixels was replaced by size.shortest_edge in newer transformers releases."""
    ip = processor.image_processor
    min_pixels = getattr(ip, "min_pixels", None)
    if min_pixels is not None:
        return min_pixels
    return ip.size["shortest_edge"]


def load_model_and_processor(model_id: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (
        AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor
