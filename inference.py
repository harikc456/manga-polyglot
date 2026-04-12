import os
import gc
import cv2
import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from img_utils import (
    imread,
    replace_text_with_translation,
    fill_bubble_with_estimated_color,
    get_text_insertion_boxes,
)
from transformers import Sam3Processor, Sam3Model
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from transformers import AutoProcessor, AutoModelForImageTextToText
from text_detection import detect_text

from text_utils import translate, update_session_memory
from data_model import BubbleType, SessionMemory
from memory_utils import load_memory



def clean_text_blocks(img, mask):
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    inpainted_telea = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return inpainted_telea


def clean_bubble_free_text(pil_image, box, masked_block):
    
    numpy_image = np.array(pil_image)
    img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    masked_block = masked_block.cpu().numpy()

    masked_block = (masked_block * 255).astype(np.uint8)
    
    x1, y1, x2, y2 = list(map(int, box))
    cropped_img = img[y1:y2, x1:x2]
    cleaned_block = clean_text_blocks(cropped_img, masked_block)
    filtered_block = cv2.medianBlur(cleaned_block, 25)
    img[y1:y2, x1:x2] = filtered_block

    color_converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_converted_image)
    return pil_image

def create_masks(pil_image, box: dict, segmentation_model, segmentation_processor):
    box = box["original_text_box"]
    box = list(map(int, box))

    cropped_image = pil_image.crop(box)
    inputs = segmentation_processor(images=cropped_image, text="manga dialogue letters", return_tensors="pt").to(segmentation_model.device)

    with torch.no_grad():
        outputs = segmentation_model(**inputs)

    # Post-process results
    results = segmentation_processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    return results['masks']

def clean_page(img_path: str, temp_dir: str, boxes: list[dict], segmentation_model, segmentation_processor):
    pil_image = Image.open(img_path).convert("RGB")
    file_name = os.path.basename(img_path)
    cleaned_file_path = os.path.join(temp_dir, file_name)
    for box in boxes:
        if box["type"] == BubbleType.FIXED:
            pil_image = fill_bubble_with_estimated_color(
                pil_image, box["original_text_box"]
            )
        elif box["type"] == BubbleType.FREE:
            masks = create_masks(pil_image, box, segmentation_model, segmentation_processor)
            for mask in masks:
                pil_image = clean_bubble_free_text(pil_image, box["original_text_box"], mask)

    pil_image.save(cleaned_file_path)
    return cleaned_file_path


def extract_text(img_path: str, boxes: list[dict], model, processor):
    max_pixels = 1280 * 28 * 28
    texts = []
    text_boxes = []
    img = Image.open(img_path)
    for box in boxes:
        box = box["insertion_polygon"]
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
    ocr_model_id = config["ocr_model"]
    model_id = config["text_detection_model_path"]
    llm_name = config["llm_name"]
    font_path = config["font_path"]
    image_enabled = config.get("image_enabled", False)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    memory_path = os.path.join(temp_dir, "memory.md")
    session_memory = load_memory(memory_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    segmentation_model = Sam3Model.from_pretrained("jetjodh/sam3").to(device)
    segmentation_processor = Sam3Processor.from_pretrained("jetjodh/sam3")

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

    img_paths = sorted(os.listdir(input_dir))
    computed = {}
    for img_name in tqdm(img_paths):
        img_path = os.path.join(input_dir, img_name)

        computed[img_path] = {}

        results = detect_text(img_path, det_model, image_processor)

        boxes = get_text_insertion_boxes(results, expand_ratio=0.8)

        ## clean the image to remove the texts
        cleaned_file_path = clean_page(img_path, temp_dir, boxes, segmentation_model, segmentation_processor)

        # Extract texts from the bounding boxes
        texts, text_boxes = extract_text(img_path, boxes, ocr_model, processor)

        computed[img_path]["texts"] = texts
        computed[img_path]["text_boxes"] = text_boxes
        computed[img_path]["page_context"] = "\n\n".join(texts)
        computed[img_path]["clean_img_path"] = cleaned_file_path

    ## Removing models from GPU to make space for the LLM

    del ocr_model
    del det_model
    del segmentation_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    lookback_pages = 2
    lookahead_pages = 2
    n_pages = len(img_paths)

    for i, img_name in enumerate(tqdm(img_paths)):

        img_path = os.path.join(input_dir, img_name)
        out_path = os.path.join(output_dir, img_name)

        # Replace original text with the translated ones

        translations = []
        precomputed_vals = computed[img_path]
        cleaned_file_path = computed[img_path]["clean_img_path"]

        # Build context: lookback + current (optional) + lookahead
        context_parts = []

        # Lookback (previous pages)
        for j in range(max(0, i - lookback_pages), i):
            prev_img_path = os.path.join(input_dir, img_paths[j])
            ctx = computed[prev_img_path]["page_context"]
            if ctx:
                context_parts.append(f"[Page {j+1}] {ctx}")

        # Optionally include current page context (many teams exclude it)
        context_parts.append(f"[Current Page] {precomputed_vals['page_context']}")

        # Lookahead (future pages)
        for j in range(i + 1, min(n_pages, i + 1 + lookahead_pages)):
            next_img_path = os.path.join(input_dir, img_paths[j])
            ctx = computed[next_img_path]["page_context"]
            if ctx:
                context_parts.append(f"[Page {j+1} ahead] {ctx}")

        context = "\n\n".join(context_parts).strip()

        for text, text_box in zip(
            precomputed_vals["texts"], precomputed_vals["text_boxes"]
        ):
            # Optionally crop and pass image to the translation model
            image = None
            if image_enabled:
                img = Image.open(img_path)
                image = img.crop(text_box)

            # Build previous translations for consistency
            previous_translations = []
            if translations:
                for prev in translations:
                    previous_translations.append({
                        "original": prev["original"],
                        "translated": prev["translated"]
                    })

            translated = translate(
                text,
                llm_name,
                context=context,
                source_language=source_language,
                target_language=target_language,
                image=image,
                previous_translations=previous_translations,
                session_memory=session_memory,
            )
            translations.append(
                {"original": text, "translated": translated, "polygon": text_box}
            )

        if translations:
            session_memory = update_session_memory(
                translations, session_memory, llm_name, temp_dir
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
        default="Japanese",
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
