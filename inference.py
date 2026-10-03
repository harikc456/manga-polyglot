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
from ocr import build_pipeline
from text_utils import translate, update_session_memory
from data_model import SessionMemory
from memory_utils import load_memory


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _translation_settings(config: dict, source_language: str, target_language: str) -> dict:
    """Everything that changes the translations; a cached translation is reused only if these match."""
    return {
        "llm_name": config["llm_name"],
        "source_language": source_language,
        "target_language": target_language,
        "image_enabled": config.get("image_enabled", False),
        "json_enabled": config.get("json_enabled", True),
        "memory_enabled": config.get("memory_enabled", True),
    }


def clean_page(img_path: str, temp_dir: str, boxes: list[dict]) -> str:
    pil_image = Image.open(img_path).convert("RGB")
    file_name = os.path.basename(img_path)
    cleaned_file_path = os.path.join(temp_dir, file_name)
    for box in boxes:
        pil_image = fill_bubble_with_estimated_color(pil_image, box["insertion_polygon"])
    pil_image.save(cleaned_file_path)
    return cleaned_file_path


def driver(input_dir, temp_dir, output_dir, config, source_language, target_language):
    llm_name = config["llm_name"]
    font_path = config["font_path"]
    image_enabled = config.get("image_enabled", False)
    json_enabled = config.get("json_enabled", True)
    memory_enabled = config.get("memory_enabled", True)
    ocr_config = config.get("ocr")
    translation_settings = _translation_settings(config, source_language, target_language)

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    memory_path = os.path.join(temp_dir, "memory.md")
    session_memory = load_memory(memory_path) if memory_enabled else None

    pipeline = build_pipeline(ocr_config)

    img_paths = sorted(os.listdir(input_dir))
    computed = {}

    for img_name in tqdm(img_paths):
        img_path = os.path.join(input_dir, img_name)
        cache_path = os.path.join(temp_dir, img_name + ".ocr.json")
        current_hash = _file_hash(img_path)
        clean_img_path = os.path.join(temp_dir, img_name)

        boxes_raw = None

        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            ocr_ok = (
                cached.get("hash") == current_hash
                and cached.get("ocr") == ocr_config
                and "ocr_boxes" in cached
            )
            if ocr_ok:
                if os.path.exists(clean_img_path):
                    computed[img_path] = {
                        "texts": cached["texts"],
                        "text_boxes": cached["text_boxes"],
                        "page_context": cached["page_context"],
                        "clean_img_path": clean_img_path,
                        "cache_path": cache_path,
                        "hash": current_hash,
                    }
                    continue
                boxes_raw = cached["ocr_boxes"]

        if boxes_raw is None:
            boxes_raw = pipeline.run(img_path)

        boxes = sort_manga_reading_order(boxes_raw)
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
                "ocr": ocr_config,
                "ocr_boxes": boxes_raw,
            }, f)

    pipeline.close()
    del pipeline
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
        precomputed_vals = computed[img_path]
        cleaned_file_path = computed[img_path]["clean_img_path"]

        if (
            cache_data.get("hash") == computed[img_path]["hash"]
            and cache_data.get("translated")
            and cache_data.get("translation_settings") == translation_settings
        ):
            if not os.path.exists(out_path):
                # Translations are still valid; only the rendered page is missing.
                cached = [
                    {**t, "polygon": box}
                    for t, box in zip(cache_data.get("translations", []), precomputed_vals["text_boxes"])
                ]
                replace_text_with_translation(cleaned_file_path, font_path, cached).save(out_path)
            continue

        translations = []

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
        cache_data["translation_settings"] = translation_settings
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
