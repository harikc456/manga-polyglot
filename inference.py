import os
import cv2
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from img_utils import imread, replace_text_with_translation, get_img_hash
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

from text_detector import TextDetector
from text_utils import post_process, translate


def clean_text_blocks(img, mask):
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    inpainted_telea = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return inpainted_telea


def clean_page(img_path, temp_dir, blk_list, mask_refined):
    img = imread(img_path)
    for blk in blk_list:
        x1, y1, x2, y2 = blk.xyxy
        cropped_img = img[y1:y2, x1:x2]
        masked_block = mask_refined[y1:y2, x1:x2]
        cleaned_block = clean_text_blocks(cropped_img, masked_block)
        filtered_block = cv2.medianBlur(cleaned_block, 25)
        img[y1:y2, x1:x2] = filtered_block

    file_name = os.path.basename(img_path)
    cleaned_file_path = os.path.join(temp_dir, file_name)
    cv2.imwrite(cleaned_file_path, img)
    return cleaned_file_path


def extract_text(img_path, blk_list, ocr_model, processor, tokenizer):
    texts = []
    text_boxes = []
    for blk in blk_list:
        box = blk.xyxy
        img = Image.open(img_path)
        cropped_img = img.crop(box)
        cropped_img = cropped_img.convert("L").convert("RGB")
        x = processor(cropped_img, return_tensors="pt").pixel_values.squeeze(0)
        x = ocr_model.generate(x[None].to(ocr_model.device), max_length=300)[0].cpu()
        x = tokenizer.decode(x, skip_special_tokens=True)
        x = post_process(x)
        texts.append(x)
        text_boxes.append(box)
    return texts, text_boxes


def driver(input_dir, temp_dir, output_dir, ocr_model_id, llm_name, font_path, model_path, target_language):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det_model = TextDetector(model_path=model_path, input_size=1024, device=device, act="leaky")
    processor = ViTImageProcessor.from_pretrained(ocr_model_id)
    tokenizer = AutoTokenizer.from_pretrained(ocr_model_id)
    ocr_model = VisionEncoderDecoderModel.from_pretrained(ocr_model_id)
    ocr_model = ocr_model.to(device)

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    with open("computed.json") as f:
        computed = json.load(f)

    img_paths = os.listdir(input_dir)
    for img_name in tqdm(img_paths):
        img_path = os.path.join(input_dir, img_name)
        out_path = os.path.join(output_dir, img_name)

        img_hash = get_img_hash(img_path)

        if img_hash in computed:
            # TODO Reusing previously computed data
            pass

        # Read image
        img = imread(img_path)

        # Perform text detection (bounding box prediction)
        _, mask_refined, blk_list = det_model(img)

        ## clean the image to remove the texts
        cleaned_file_path = clean_page(img_path, temp_dir, blk_list, mask_refined)

        # Extract texts from the bounding boxes
        texts, text_boxes = extract_text(img_path, blk_list, ocr_model, processor, tokenizer)

        # Translate all the texts extracted from the page
        context = "\n".join(texts)
        translated_texts = [translate(text, llm_name, context, target_language) for text in texts]

        # Replace original text with the translated ones
        translated_image = replace_text_with_translation(cleaned_file_path, font_path, translated_texts, text_boxes)
        translated_image.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Inputs to translate")
    parser.add_argument("--input-dir", type=str, help="the directory containing images")

    parser.add_argument("--output-dir", type=str, help="the directory to which translated images are stored")

    parser.add_argument(
        "--target-lang", type=str, help="the directory to which translated images are stored", default="English"
    )

    parser.add_argument(
        "--temp-dir", type=str, help="the directory to which translated images are stored", default="./temp"
    )

    args = parser.parse_args()
    config_path = "./config.json"
    with open(config_path) as f:
        config = json.load(f)

    input_dir = args.input_dir
    output_dir = args.output_dir
    temp_dir = args.temp_dir
    ocr_model_id = config["ocr_model"]
    model_path = config["text_detection_model_path"]
    llm_name = config["llm_name"]
    font_path = config["font_path"]
    target_language = args.target_lang

    driver(input_dir, temp_dir, output_dir, ocr_model_id, llm_name, font_path, model_path, target_language)

    


if __name__ == "__main__":
    main()
