import os
import json
import torch
import argparse
from PIL import Image
from tqdm.notebook import tqdm
from img_utils import imread, replace_text_with_translation
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

from text_detector import TextDetector
from text_utils import post_process, translate


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


def drive(img_path, ocr_model_id, llm_name, font_path, model_path, out_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det_model = TextDetector(model_path=model_path, input_size=1024, device=device, act="leaky")
    processor = ViTImageProcessor.from_pretrained(ocr_model_id)
    tokenizer = AutoTokenizer.from_pretrained(ocr_model_id)
    ocr_model = VisionEncoderDecoderModel.from_pretrained(ocr_model_id)
    ocr_model = ocr_model.to(device)

    # Read image
    img = imread(img_path)

    # Perform text detection (bounding box prediction)
    _, _, blk_list = det_model(img)

    # Extract texts from the bounding boxes
    texts, text_boxes = extract_text(img_path, blk_list, ocr_model, processor, tokenizer)

    # Translate all the texts extracted from the page
    context = "\n".join(texts)
    translated_texts = [translate(text, llm_name, context) for text in texts]

    # Replace original text with the translated ones
    translated_image = replace_text_with_translation(img_path, font_path, translated_texts, text_boxes)
    translated_image.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Inputs to translate")
    parser.add_argument("--input-dir", type=str, help="the directory containing images")

    parser.add_argument("--output-dir", type=str, help="the directory to which translated images are stored")

    args = parser.parse_args()
    config_path = "./config.json"
    with open(config_path) as f:
        config = json.load(f)
    input_dir = args.input_dir
    output_dir = args.output_dir
    ocr_model_id = config["ocr_model"]
    model_path = config["text_detection_model_path"]
    llm_name = config["llm_name"]
    font_path = config["font_path"]

    img_paths = os.listdir(input_dir)
    for img_name in tqdm(img_paths):
        img_path = os.path.join(input_dir, img_name)
        out_path = os.path.join(output_dir, img_name)
        drive(img_path, ocr_model_id, llm_name, font_path, model_path, out_path)


if __name__ == "__main__":
    main()
