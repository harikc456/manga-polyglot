import torch
from PIL import Image


def detect_text(img_path, model, processor):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.5
    )
    results = parse_detection_results(results, model)
    return results


def parse_detection_results(results, model):
    bubbles = []
    text_bubbles = []
    text_free = []
    for result in results:
        for score, label_id, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            score, label = score.item(), label_id.item()
            box = [round(i, 2) for i in box.tolist()]

            label = model.config.id2label[label]
            if label == "bubble":
                bubbles.append({"conf": score, "box": box})
            elif label == "text_bubble":
                text_bubbles.append({"conf": score, "box": box})
            else:
                text_free.append({"conf": score, "box": box})

    return {"text_bubbles": text_bubbles, "bubbles": bubbles, "text_free": text_free}
