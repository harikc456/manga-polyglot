"""Report, per text box, whether its bubble was found for text expansion and what font size it gets.

Prints only measurements (sizes, colours, ratios), never page content or text, so the output can be
shared even when the pages cannot. With --overlay, also saves annotated copies of the pages locally.

    python diagnose_layout.py --temp-dir ./temp [--page page_001.jpg] [--overlay ./diagnose]
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from img_utils import (
    MAX_FONT_SIZE,
    _fit_in,
    _max_font_size,
    diagnose_text_region,
    layout_in_region,
    renderable_text,
    text_areas_in_region,
)

_COLOURS = {"ok": (0, 170, 0), "region reaches page edge": (220, 0, 0)}


def _plan(text, polygon, region, draw, font_path, cap):
    """The method and font size draw_wrapped_text would pick (before any fallback growth)."""
    if region is None:
        size, _, fits = _fit_in(text, draw, font_path, polygon, _max_font_size(polygon, False, cap))
        return "box only", size, fits
    best_size = -1
    for area in text_areas_in_region(region) + [list(polygon)]:
        size, _, fits = _fit_in(text, draw, font_path, area, _max_font_size(area, True, cap))
        if fits and size > best_size:
            best_size = size
    shaped = layout_in_region(text, region, font_path, cap)
    if shaped is not None and shaped.size >= best_size:
        return "shaped", shaped.size, True
    return ("rectangle", best_size, True) if best_size > 0 else ("rectangle", 0, False)


def diagnose_page(temp_dir, page, font_path, cap, overlay_dir=None):
    with open(os.path.join(temp_dir, page + ".ocr.json")) as f:
        cache = json.load(f)
    boxes = cache["text_boxes"]
    texts = [t["translated"] for t in cache.get("translations", [])] or ["" for _ in boxes]
    cleaned = Image.open(os.path.join(temp_dir, page)).convert("RGB")
    draw = ImageDraw.Draw(cleaned.copy())
    print(f"\n{page}  ({cleaned.width}x{cleaned.height}, {len(boxes)} boxes)")

    overlay = cleaned.copy() if overlay_dir else None
    for i, (box, text) in enumerate(zip(boxes, texts)):
        others = boxes[:i] + boxes[i + 1:]
        region, info = diagnose_text_region(cleaned, box, others)
        text = renderable_text(text, font_path) if text else ""
        method, size, fits = _plan(text, box, region, draw, font_path, cap) if text else ("no text", 0, True)

        w, h = int(box[2] - box[0]), int(box[3] - box[1])
        details = {k: v for k, v in info.items() if k != "reason"}
        if region is not None:
            ys, xs = np.nonzero(region.mask)
            details["region_size"] = f"{xs.max() - xs.min() + 1}x{ys.max() - ys.min() + 1}"
        fit = "" if fits else " (does not fit)"
        print(f"  #{i:<2} box {w}x{h:<5} {info['reason']:<34} {method:<9} font {size}{fit}  {details}")

        if overlay is not None:
            colour = _COLOURS.get(info["reason"], (255, 140, 0))
            if region is not None:
                tint = Image.new("RGB", region.mask.shape[::-1], (0, 200, 0))
                mask = Image.fromarray((region.mask * 70).astype(np.uint8))
                overlay.paste(tint, region.origin, mask)
            ImageDraw.Draw(overlay).rectangle(box, outline=colour, width=3)
            ImageDraw.Draw(overlay).text((box[0] + 3, box[1] + 3), f"#{i}", fill=colour)

    if overlay is not None:
        os.makedirs(overlay_dir, exist_ok=True)
        overlay.save(os.path.join(overlay_dir, page))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--temp-dir", default="./temp")
    parser.add_argument("--page", help="one page file name (default: every page with an OCR cache)")
    parser.add_argument("--overlay", help="directory to save pages annotated with boxes and found bubbles")
    parser.add_argument("--config", default="./config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    cap = config.get("max_font_size") or MAX_FONT_SIZE
    pages = [args.page] if args.page else sorted(
        name[: -len(".ocr.json")] for name in os.listdir(args.temp_dir) if name.endswith(".ocr.json")
    )
    print(f"font {config['font_path']}, max_font_size {cap}")
    print("box colours in overlays: green = bubble found, red = reaches page edge, orange = other")
    for page in pages:
        if os.path.exists(os.path.join(args.temp_dir, page)):
            diagnose_page(args.temp_dir, page, config["font_path"], cap, args.overlay)
        else:
            print(f"\n{page}: cleaned image missing, skipped")


if __name__ == "__main__":
    main()
