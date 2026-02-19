import os
import gc
import time
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from collections import deque
from img_utils import (
    replace_text_with_translation,
    fill_bubble_with_estimated_color,
    get_text_insertion_boxes,
)
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from transformers import AutoProcessor, AutoModelForImageTextToText
from text_detection import detect_text

from text_utils import translate


def clean_page(img_path, temp_dir, boxes):
    pil_image = Image.open(img_path).convert("RGB")
    file_name = os.path.basename(img_path)
    cleaned_file_path = os.path.join(temp_dir, file_name)
    for box in boxes:
        pil_image = fill_bubble_with_estimated_color(
            pil_image, box["original_text_box"]
        )

    pil_image.save(cleaned_file_path)
    return cleaned_file_path


def extract_text(img_path, boxes, model, processor):
    max_pixels = 1280 * 28 * 28
    texts = []
    text_boxes = []
    for box in boxes:
        box = box["insertion_polygon"]
        img = Image.open(img_path)
        cropped_img = img.crop(box)
        cropped_img = cropped_img.convert("L").convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": cropped_img},
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
            images_kwargs={
                "size": {
                    "shortest_edge": processor.image_processor.min_pixels,
                    "longest_edge": max_pixels,
                }
            },
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=256)
        result = processor.decode(outputs[0][inputs["input_ids"].shape[-1] : -1])
        texts.append(result)
        text_boxes.append(box)
    return texts, text_boxes


def driver(input_dir, temp_dir, output_dir, config, source_language, target_language):
    context_pages = 3
    ocr_model_id = config["ocr_model"]
    model_id = config["text_detection_model_path"]
    llm_name = config["llm_name"]
    font_path = config["font_path"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_processor = RTDetrImageProcessor.from_pretrained(model_id)
    det_model = RTDetrV2ForObjectDetection.from_pretrained(model_id)

    ocr_model = (
        AutoModelForImageTextToText.from_pretrained(
            ocr_model_id, torch_dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(ocr_model_id)

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    img_paths = sorted(os.listdir(input_dir))
    computed = {}
    for img_name in tqdm(img_paths):
        img_path = os.path.join(input_dir, img_name)

        computed[img_path] = {}

        results = detect_text(img_path, det_model, image_processor)

        boxes = get_text_insertion_boxes(results, expand_ratio=0.8)

        ## clean the image to remove the texts
        cleaned_file_path = clean_page(img_path, temp_dir, boxes)

        # Extract texts from the bounding boxes
        texts, text_boxes = extract_text(img_path, boxes, ocr_model, processor)

        computed[img_path]["texts"] = texts
        computed[img_path]["text_boxes"] = text_boxes
        computed[img_path]["page_context"] = "\n\n".join(texts)
        computed[img_path]["clean_img_path"] = cleaned_file_path

    ## Removing models from GPU to make space for the LLM

    del ocr_model
    del det_model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)

    context_list = deque(maxlen=context_pages)
    for img_name in tqdm(img_paths):

        img_path = os.path.join(input_dir, img_name)
        out_path = os.path.join(output_dir, img_name)

        # Replace original text with the translated ones

        translations = []
        precomputed_vals = computed[img_path]
        cleaned_file_path = computed[img_path]["clean_img_path"]
        context_list.append(computed[img_path]["page_context"])
        context = "\n".join(context_list)

        for text, text_box in zip(
            precomputed_vals["texts"], precomputed_vals["text_boxes"]
        ):
            translated = translate(
                text,
                llm_name,
                context=context,
                source_language=source_language,
                target_language=target_language,
            )
            translations.append(
                {"original": text, "translated": translated, "polygon": text_box}
            )

        translated_image = replace_text_with_translation(
            cleaned_file_path, font_path, translations
        )
        translated_image.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Inputs to translate")
    parser.add_argument("--input-dir", type=str, help="the directory containing images")

    parser.add_argument(
        "--output-dir",
        type=str,
        help="the directory to which translated images are stored",
    )

    parser.add_argument(
        "--source-lang",
        type=str,
        help="the directory to which translated images are stored",
        default="English",
    )

    parser.add_argument(
        "--target-lang",
        type=str,
        help="the directory to which translated images are stored",
        default="English",
    )

    parser.add_argument(
        "--temp-dir",
        type=str,
        help="the directory to which translated images are stored",
        default="./temp",
    )

    args = parser.parse_args()
    config_path = "./config.json"
    with open(config_path) as f:
        config = json.load(f)

    input_dir = args.input_dir
    output_dir = args.output_dir
    temp_dir = args.temp_dir
    source_language = args.source_lang
    target_language = args.target_lang

    driver(input_dir, temp_dir, output_dir, config, source_language, target_language)


if __name__ == "__main__":
    main()
