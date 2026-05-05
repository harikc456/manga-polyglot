import cv2
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

def imread(imgpath, read_type=cv2.IMREAD_COLOR):
    """Read an image from a file path (supports non-ASCII paths) using OpenCV."""
    return cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), read_type)

def draw_wrapped_text(image, draw, polygon, text, font_path, font_scale=1.2):
    x_min, y_min, x_max, y_max = polygon

    box_width = x_max - x_min
    box_width = int(0.9 * box_width)
    box_height = y_max - y_min
    box_height = int(0.9 * box_height)

    font_size, wrapped = find_max_fontsize(
        text, draw, font_path, box_width, box_height, min_size=4, max_size=20
    )

    font = ImageFont.truetype(font_path, math.ceil(font_size * font_scale))
    bbox = draw.textbbox((0, 0), wrapped, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = x_min + (box_width - text_w) / 2
    y = y_min + (box_height - text_h) / 2
    background_color = get_background_color(image, x_min, y_min, x_max, y_max)
    fill = get_text_fill_color(background_color)
    draw.text((x, y), wrapped, font=font, fill=fill, align="center")


def estimate_bubble_bg_color(pil_image, outer_box, border_thickness=12):
    """
    Sample color from a frame near the inside edge of the bubble box.
    Avoids text, avoids outer black border.
    """
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = map(int, outer_box)

    h, w = img_cv.shape[:2]

    # Create a mask for the border strip only
    full_roi = img_cv[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
    if full_roi.size == 0:
        return (255, 255, 255)

    roi_h, roi_w = full_roi.shape[:2]

    # Mask = 255 only in the outer border ring of this ROI
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.rectangle(
        mask,
        (border_thickness, border_thickness),
        (roi_w - border_thickness, roi_h - border_thickness),
        0,
        -1,
    )  # hole in center
    cv2.rectangle(
        mask, (0, 0), (roi_w - 1, roi_h - 1), 255, border_thickness // 2
    )  # outer frame

    # Get pixels in that border
    border_pixels = full_roi[mask == 255]

    if len(border_pixels) < 50:
        return (255, 255, 255)  # fallback

    # Most common color (mode) — robust against outlines / artifacts
    pixels_list = [tuple(p) for p in border_pixels]
    most_common = Counter(pixels_list).most_common(1)[0][0]

    # Or median (sometimes smoother)
    # most_common = np.median(border_pixels, axis=0).astype(np.uint8)

    return tuple(int(c) for c in most_common)  # BGR → RGB later if needed


def fill_bubble_with_estimated_color(pil_image, outer_box):
    bg_color = estimate_bubble_bg_color(pil_image, outer_box)
    result = pil_image.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle(outer_box, fill=bg_color)
    return result


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = box_area(boxA)
    boxBArea = box_area(boxB)
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def sort_manga_reading_order(boxes):
    """Sort insertion boxes in manga reading order: right-to-left columns, top-to-bottom within each column."""
    if not boxes:
        return boxes

    def get_poly(box):
        return box["insertion_polygon"]

    def cx(box):
        x1, _, x2, _ = get_poly(box)
        return (x1 + x2) / 2

    def cy(box):
        _, y1, _, y2 = get_poly(box)
        return (y1 + y2) / 2

    widths = sorted(get_poly(b)[2] - get_poly(b)[0] for b in boxes)
    tolerance = widths[len(widths) // 2]

    sorted_boxes = sorted(boxes, key=lambda b: -cx(b))

    columns = []
    col_cx_vals = []

    for box in sorted_boxes:
        box_cx = cx(box)
        assigned = False
        for i, col_cx_val in enumerate(col_cx_vals):
            if abs(box_cx - col_cx_val) <= tolerance:
                columns[i].append(box)
                assigned = True
                break
        if not assigned:
            columns.append([box])
            col_cx_vals.append(box_cx)

    result = []
    for col in columns:
        result.extend(sorted(col, key=cy))
    return result


def add_discoloration(color, strength):
    r, g, b = color[:3]
    r = max(0, min(255, r + strength))
    g = max(0, min(255, g + strength))
    b = max(0, min(255, b + strength))

    if r == 255 and g == 255 and b == 255:
        r, g, b = 245, 245, 245

    return (r, g, b)


def get_background_color(image, x_min, y_min, x_max, y_max):
    image = image.convert("RGBA")  # Handle transparency

    margin = 10
    edge_region = image.crop(
        (
            max(x_min - margin, 0),
            max(y_min - margin, 0),
            min(x_max + margin, image.width),
            min(y_max + margin, image.height),
        )
    )

    pixels = list(edge_region.getdata())
    opaque_pixels = [pixel[:3] for pixel in pixels if pixel[3] > 0]

    if not opaque_pixels:
        background_color = (255, 255, 255)  # fallback if all pixels are transparent
    else:
        from collections import Counter

        most_common = Counter(opaque_pixels).most_common(1)[0][0]
        background_color = most_common

    background_color = add_discoloration(background_color, 40)
    return background_color


def get_text_fill_color(background_color):
    # Calculate the luminance of the background color
    luminance = (
        0.299 * background_color[0]
        + 0.587 * background_color[1]
        + 0.114 * background_color[2]
    ) / 255

    # Determine the text color based on the background luminance
    if luminance > 0.5:
        return "black"  # Use black text for light backgrounds
    else:
        return "white"  # Use white text for dark backgrounds


def wrap_text(text, font, box_w, hyphenate_only_words_ge=8):
    """
    Wrap text into lines that fit inside box_w.
    Only hyphenate words that have >= hyphenate_only_words_ge characters.
    """
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        # Try adding the whole word to current line
        trial = (current_line + " " + word).strip() if current_line else word
        
        if font.getlength(trial) <= box_w:
            current_line = trial
            continue

        # Word doesn't fit → decide what to do
        if current_line:
            lines.append(current_line)
            current_line = ""

        # Now: does the word itself fit on its own line?
        if font.getlength(word) <= box_w:
            current_line = word
        else:
            # Word is too long to fit even alone
            if len(word) >= hyphenate_only_words_ge:
                # Hyphenate long words
                partial = ""
                for ch in word:
                    candidate = partial + ch + "-"
                    if font.getlength(candidate) <= box_w:
                        partial += ch
                    else:
                        if partial:
                            lines.append(partial + "-")
                        partial = ch
                if partial:
                    current_line = partial
            else:
                # Short word but still doesn't fit → have to put it anyway
                # (this case is rare after line break, but we don't break it)
                current_line = word

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)


def fits_in_box(text, draw, font, box_w, box_h):
    wrapped = wrap_text(text, font, box_w * 0.92, hyphenate_only_words_ge=9)
    bbox = draw.textbbox((0, 0), wrapped, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    required_h = text_h * 1.15
    return text_w <= box_w and required_h <= box_h, wrapped


def find_max_fontsize(text, draw, font_path, box_w, box_h, min_size=1, max_size=200):
    """Find the largest font size where wrapped text fits inside box."""
    best_size, best_wrapped = min_size, text
    while min_size <= max_size:
        mid = (min_size + max_size) // 2
        font = ImageFont.truetype(font_path, mid)
        fits, wrapped = fits_in_box(text, draw, font, box_w, box_h)
        if fits:
            best_size, best_wrapped = mid, wrapped
            min_size = mid + 1  # try bigger
        else:
            max_size = mid - 1  # too big
    return best_size, best_wrapped


def replace_text_with_translation(image_path, font_path, translations):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for translation in translations:
        polygon = translation["polygon"]
        translated_text = translation["translated"]
        if translated_text:
            draw_wrapped_text(image, draw, polygon, translated_text, font_path)
    return image
