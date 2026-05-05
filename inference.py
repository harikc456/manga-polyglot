import os
import gc
import json
import torch
import argparse
import hashlib
from PIL import Image
from tqdm import tqdm
from img_utils import (
    replace_text_with_translation,
    fill_bubble_with_estimated_color,
    sort_manga_reading_order,
)
from transformers import AutoProcessor, AutoModelForImageTextToText
from ocr_utils import parse_spotting_output, cluster_into_bubbles, boxes_from_clusters
from text_utils import translate, update_session_memory
from data_model import SessionMemory
from memory_utils import load_memory


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def spot_text(img_path: str, model, processor, max_tokens: int = 2048) -> str:
    image = Image.open(img_path).convert("RGB")
    orig_w, orig_h = image.size
    if orig_w < 1500 and orig_h < 1500:
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        image = image.resize((orig_w * 2, orig_h * 2), resample)
    max_pixels = 2048 * 28 * 28
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
        images_kwargs={
            "size": {
                "shortest_edge": processor.image_processor.min_pixels,
                "longest_edge": max_pixels,
            }
        },
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])


def clean_page(img_path: str, temp_dir: str, boxes: list[dict]) -> str:
    pil_image = Image.open(img_path).convert("RGB")
    file_name = os.path.basename(img_path)
    cleaned_file_path = os.path.join(temp_dir, file_name)
    for box in boxes:
        pil_image = fill_bubble_with_estimated_color(pil_image, box["insertion_polygon"])
    pil_image.save(cleaned_file_path)
    return cleaned_file_path


def _cluster_boxes(spotting_raw: str, img_path: str, cluster_eps: int) -> list[dict]:
    img = Image.open(img_path)
    img_w, img_h = img.size
    lines = parse_spotting_output(spotting_raw, img_w, img_h)
    eps_pixels = int(cluster_eps / 1000 * max(img_w, img_h))
    groups = cluster_into_bubbles(lines, eps=eps_pixels)
    boxes = boxes_from_clusters(groups)
    return sort_manga_reading_order(boxes)


def driver(input_dir, temp_dir, output_dir, config, source_language, target_language):
    ocr_model_id = config["ocr_model"]
    llm_name = config["llm_name"]
    font_path = config["font_path"]
    image_enabled = config.get("image_enabled", False)
    json_enabled = config.get("json_enabled", True)
    memory_enabled = config.get("memory_enabled", True)
    cluster_eps = config.get("spotting_cluster_eps", 80)
    max_tokens = config.get("spotting_max_tokens", 2048)

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    memory_path = os.path.join(temp_dir, "memory.md")
    session_memory = load_memory(memory_path) if memory_enabled else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        cache_path = os.path.join(temp_dir, img_name + ".ocr.json")
        current_hash = _file_hash(img_path)
        clean_img_path = os.path.join(temp_dir, img_name)

        spotting_raw = None

        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("hash") == current_hash:
                if cached.get("cluster_eps") == cluster_eps and os.path.exists(clean_img_path):
                    computed[img_path] = {
                        "texts": cached["texts"],
                        "text_boxes": cached["text_boxes"],
                        "page_context": cached["page_context"],
                        "clean_img_path": clean_img_path,
                        "cache_path": cache_path,
                        "hash": current_hash,
                    }
                    continue
                if "spotting_raw" in cached:
                    spotting_raw = cached["spotting_raw"]

        if spotting_raw is None:
            spotting_raw = spot_text(img_path, ocr_model, processor, max_tokens)

        boxes = _cluster_boxes(spotting_raw, img_path, cluster_eps)
        texts = [b["text"] for b in boxes]
        text_boxes = [b["insertion_polygon"] for b in boxes]
        page_context = "\n\n".join(texts)
        cleaned_file_path = clean_page(img_path, temp_dir, boxes)

        computed[img_path] = {
            "texts": texts,
            "text_boxes": text_boxes,
            "page_context": page_context,
            "clean_img_path": cleaned_file_path,
            "cache_path": cache_path,
            "hash": current_hash,
        }

        with open(cache_path, "w") as f:
            json.dump({
                "hash": current_hash,
                "texts": texts,
                "text_boxes": [list(b) for b in text_boxes],
                "page_context": page_context,
                "spotting_raw": spotting_raw,
                "cluster_eps": cluster_eps,
            }, f)

    del ocr_model
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

        cache_path = computed[img_path]["cache_path"]
        with open(cache_path) as f:
            cache_data = json.load(f)
        if cache_data.get("hash") == computed[img_path]["hash"] and cache_data.get("translated"):
            continue

        translations = []
        precomputed_vals = computed[img_path]
        cleaned_file_path = computed[img_path]["clean_img_path"]

        context_parts = []
        for j in range(max(0, i - lookback_pages), i):
            prev_img_path = os.path.join(input_dir, img_paths[j])
            ctx = computed[prev_img_path]["page_context"]
            if ctx:
                context_parts.append(f"[Page {j+1}] {ctx}")
        context_parts.append(f"[Current Page] {precomputed_vals['page_context']}")
        for j in range(i + 1, min(n_pages, i + 1 + lookahead_pages)):
            next_img_path = os.path.join(input_dir, img_paths[j])
            ctx = computed[next_img_path]["page_context"]
            if ctx:
                context_parts.append(f"[Page {j+1} ahead] {ctx}")
        context = "\n\n".join(context_parts).strip()

        for text, text_box in zip(precomputed_vals["texts"], precomputed_vals["text_boxes"]):
            image = None
            if image_enabled:
                img = Image.open(img_path)
                image = img.crop(text_box)

            previous_translations = [
                {"original": p["original"], "translated": p["translated"]}
                for p in translations
            ]

            translated = translate(
                text,
                llm_name,
                context=context,
                source_language=source_language,
                target_language=target_language,
                image=image,
                previous_translations=previous_translations,
                session_memory=session_memory,
                use_json=json_enabled,
            )
            translations.append(
                {"original": text, "translated": translated, "polygon": text_box}
            )

        if translations and memory_enabled:
            session_memory = update_session_memory(
                translations, session_memory, llm_name, temp_dir
            )

        translated_image = replace_text_with_translation(
            cleaned_file_path, font_path, translations
        )
        translated_image.save(out_path)

        cache_data["translated"] = True
        cache_data["translations"] = [
            {"original": t["original"], "translated": t["translated"]}
            for t in translations
        ]
        with open(computed[img_path]["cache_path"], "w") as f:
            json.dump(cache_data, f)


def main():
    parser = argparse.ArgumentParser(description="Inputs to translate")
    parser.add_argument("--input-dir", type=str, help="the directory containing images")
    parser.add_argument("--output-dir", type=str, help="the directory to which translated images are stored")
    parser.add_argument("--source-lang", type=str, default="Japanese")
    parser.add_argument("--target-lang", type=str, default="English")
    parser.add_argument("--temp-dir", type=str, default="./temp")
    args = parser.parse_args()

    config_path = "./config.json"
    with open(config_path) as f:
        config = json.load(f)

    driver(args.input_dir, args.temp_dir, args.output_dir, config, args.source_lang, args.target_lang)


if __name__ == "__main__":
    main()
