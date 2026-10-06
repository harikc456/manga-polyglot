import cv2
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

def imread(imgpath, read_type=cv2.IMREAD_COLOR):
    """Read an image from a file path (supports non-ASCII paths) using OpenCV."""
    return cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), read_type)

GROW_STEPS = (1.0, 1.3, 1.6, 2.0)


def _scaled_area(area, scale, image_size):
    """Grow area about its centre by scale, clamped to the page."""
    x0, y0, x1, y1 = area
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w, half_h = (x1 - x0) * scale / 2, (y1 - y0) * scale / 2
    return (
        max(0, cx - half_w),
        max(0, cy - half_h),
        min(image_size[0], cx + half_w),
        min(image_size[1], cy + half_h),
    )


def draw_wrapped_text(image, draw, polygon, text, font_path, text_area=None):
    """Draw text fitted into text_area (defaults to polygon); colours are taken from around polygon.

    When the text cannot be set at a legible size, the area is grown (up to 2x about its centre) so
    tiny source boxes do not force one-character lines.
    """
    base = tuple(text_area or polygon)
    # An expanded area may take a bigger font, scaled to its height but capped so it stays readable.
    max_size = 20 if text_area is None else min(36, max(20, (base[3] - base[1]) // 4))

    for scale in GROW_STEPS:
        x_min, y_min, x_max, y_max = _scaled_area(base, scale, image.size)
        # Keep a small margin from the edge of the area.
        box_width = int(0.94 * (x_max - x_min))
        box_height = int(0.94 * (y_max - y_min))
        font_size, wrapped, fits = fit_text(
            text, draw, font_path, box_width, box_height, min_size=MIN_LEGIBLE_SIZE, max_size=max_size
        )
        if fits:
            break

    font = ImageFont.truetype(font_path, font_size)
    stroke = stroke_width_for(font_size)
    bbox = draw.textbbox((0, 0), wrapped, font=font, align="center", stroke_width=stroke)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # Centre in the full box; subtract the bbox origin so glyph side bearings don't shift the text.
    x = x_min + (x_max - x_min - text_w) / 2 - bbox[0]
    y = y_min + (y_max - y_min - text_h) / 2 - bbox[1]
    background_color = get_background_color(image, *polygon)
    fill = get_text_fill_color(background_color)
    outline = "white" if fill == "black" else "black"
    draw.text(
        (x, y), wrapped, font=font, fill=fill, align="center", stroke_width=stroke, stroke_fill=outline
    )


def find_text_area(image, box, other_boxes=(), tolerance=12, edge_gap=4, min_inside=0.98):
    """Grow a cleaned text box into the empty bubble around it.

    Flood-fills the bubble colour outward from the box on the cleaned page and grows the box while it
    stays inside that region. Returns box unchanged unless the region is clearly enclosed: it must not
    reach the edge of the search window (open background or art) or contain another text box.
    """
    x_min, y_min, x_max, y_max = map(int, box)
    box_w, box_h = x_max - x_min, y_max - y_min
    if box_w <= 0 or box_h <= 0:
        return box

    img = np.asarray(image.convert("RGB"))
    img_h, img_w = img.shape[:2]
    margin = max(3 * max(box_w, box_h), 100)
    cx0, cy0 = max(0, x_min - margin), max(0, y_min - margin)
    cx1, cy1 = min(img_w, x_max + margin), min(img_h, y_max + margin)
    crop = img[cy0:cy1, cx0:cx1].astype(np.int16)
    bx0, by0, bx1, by1 = x_min - cx0, y_min - cy0, x_max - cx0, y_max - cy0

    # The cleaning step paints the box with the bubble colour, so the box's median is that colour.
    bubble_color = np.median(crop[by0:by1, bx0:bx1].reshape(-1, 3), axis=0)
    similar = (np.abs(crop - bubble_color).max(axis=2) <= tolerance).astype(np.uint8)
    # Close small holes such as leftover anti-aliased text specks.
    similar = cv2.morphologyEx(similar, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    _, labels = cv2.connectedComponents(similar, connectivity=4)
    box_labels = labels[by0:by1, bx0:bx1]
    box_labels = box_labels[box_labels > 0]
    if box_labels.size == 0:
        return box
    region = labels == np.bincount(box_labels).argmax()

    if region[0, :].any() or region[-1, :].any() or region[:, 0].any() or region[:, -1].any():
        return box
    for other in other_boxes:
        ox, oy = int((other[0] + other[2]) / 2) - cx0, int((other[1] + other[3]) / 2) - cy0
        if 0 <= ox < region.shape[1] and 0 <= oy < region.shape[0] and region[oy, ox]:
            return box

    # Keep the text a few pixels away from the bubble outline.
    region = cv2.erode(region.astype(np.uint8), np.ones((2 * edge_gap + 1,) * 2, np.uint8))
    # Treat the original box as free space even where erosion or specks nibbled it.
    region[by0:by1, bx0:bx1] = 1
    integral = cv2.integral(region)

    def inside(x0, y0, x1, y1):
        if x0 < 0 or y0 < 0 or x1 > region.shape[1] or y1 > region.shape[0]:
            return False
        total = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        return total >= min_inside * (x1 - x0) * (y1 - y0)

    step = 2
    x0, y0, x1, y1 = bx0, by0, bx1, by1
    grew = True
    while grew:
        grew = False
        if inside(x0 - step, y0, x0, y1):
            x0 -= step
            grew = True
        if inside(x1, y0, x1 + step, y1):
            x1 += step
            grew = True
        if inside(x0, y0 - step, x1, y0):
            y0 -= step
            grew = True
        if inside(x0, y1, x1, y1 + step):
            y1 += step
            grew = True

    return [x0 + cx0, y0 + cy0, x1 + cx0, y1 + cy0]


def estimate_bubble_bg_color(pil_image, outer_box, border_thickness=12):
    """
    Sample color from a frame near the inside edge of the bubble box.
    Avoids text, avoids outer black border.
    """
    img = np.array(pil_image.convert("RGB"))
    x1, y1, x2, y2 = map(int, outer_box)

    h, w = img.shape[:2]

    # Create a mask for the border strip only
    full_roi = img[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
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

    return tuple(int(c) for c in most_common)  # RGB, ready for PIL


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
    """Sort insertion boxes in manga reading order: rows top-to-bottom, right-to-left within each row.

    A box joins a row when it overlaps the row's first (topmost) box vertically by at least half of
    the shorter box's height, so bubbles that are only slightly staggered still share a row.
    """
    if not boxes:
        return boxes

    def get_poly(box):
        return box["insertion_polygon"]

    def cx(box):
        x1, _, x2, _ = get_poly(box)
        return (x1 + x2) / 2

    def same_row(anchor, box):
        _, a_y1, _, a_y2 = get_poly(anchor)
        _, b_y1, _, b_y2 = get_poly(box)
        overlap = min(a_y2, b_y2) - max(a_y1, b_y1)
        shorter = min(a_y2 - a_y1, b_y2 - b_y1)
        return shorter > 0 and overlap >= 0.5 * shorter

    rows = []
    for box in sorted(boxes, key=lambda b: get_poly(b)[1]):
        for row in rows:
            if same_row(row[0], box):
                row.append(box)
                break
        else:
            rows.append([box])

    result = []
    for row in rows:
        result.extend(sorted(row, key=lambda b: -cx(b)))
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


MIN_LEGIBLE_SIZE = 10  # below this, grow the text area (or overflow) rather than shrink the font
MIN_FRAGMENT = 3  # shortest piece left on a line when a long word is hyphenated


def stroke_width_for(font_size):
    """Outline thickness that keeps text readable without clogging small glyphs."""
    return max(1, round(font_size / 14))


def _split_long_word(word, font, box_w):
    """Break a word that is wider than box_w into hyphenated pieces of >= MIN_FRAGMENT characters.

    Returns [word] unchanged when the box is too narrow to hold a sensible fragment.
    """
    pieces, start = [], 0
    while start < len(word):
        if font.getlength(word[start:]) <= box_w:
            pieces.append(word[start:])
            break
        n = 0
        while start + n < len(word) and font.getlength(word[start : start + n + 1] + "-") <= box_w:
            n += 1
        if n < MIN_FRAGMENT:
            return [word]
        pieces.append(word[start : start + n] + "-")
        start += n
    # Avoid a dangling single character on the last line.
    if len(pieces) > 1 and len(pieces[-1]) < 2 and len(pieces[-2]) > MIN_FRAGMENT + 1:
        pieces[-1] = pieces[-2][-2] + pieces[-1]
        pieces[-2] = pieces[-2][:-2] + "-"
    return pieces


def _greedy_lines(words, font, box_w):
    lines, current = [], ""
    for word in words:
        trial = f"{current} {word}" if current else word
        if not current or font.getlength(trial) <= box_w:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def wrap_text(text, font, box_w, hyphenate_only_words_ge=8):
    """Wrap text into balanced lines that fit inside box_w.

    Words wider than box_w are hyphenated only when they have >= hyphenate_only_words_ge characters
    (None disables hyphenation); otherwise they stay whole and overflow. Lines are balanced: the line
    count is the greedy minimum, but the width is narrowed as far as that count allows, which avoids
    orphan words and gives rounder, bubble-shaped blocks.
    """
    words = []
    for word in text.split():
        if (
            hyphenate_only_words_ge is not None
            and len(word) >= hyphenate_only_words_ge
            and font.getlength(word) > box_w
        ):
            words.extend(_split_long_word(word, font, box_w))
        else:
            words.append(word)
    if not words:
        return ""

    lines = _greedy_lines(words, font, box_w)
    count = len(lines)
    if count > 1:
        # Narrowest width that still wraps into the same number of lines.
        lo = max(font.getlength(w) for w in words)
        hi = box_w
        while hi - lo > 1:
            mid = (lo + hi) / 2
            if len(_greedy_lines(words, font, mid)) <= count:
                hi = mid
            else:
                lo = mid
        lines = _greedy_lines(words, font, hi)
    return "\n".join(lines)


def fits_in_box(text, draw, font, box_w, box_h, hyphenate=True):
    wrapped = wrap_text(text, font, box_w, hyphenate_only_words_ge=7 if hyphenate else None)
    stroke = stroke_width_for(font.size)
    bbox = draw.textbbox((0, 0), wrapped, font=font, align="center", stroke_width=stroke)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    return text_w <= box_w and text_h * 1.1 <= box_h, wrapped


def _largest_fitting(text, draw, font_path, box_w, box_h, lo, hi, hyphenate):
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        fits, wrapped = fits_in_box(text, draw, ImageFont.truetype(font_path, mid), box_w, box_h, hyphenate)
        if fits:
            best = (mid, wrapped)
            lo = mid + 1  # try bigger
        else:
            hi = mid - 1  # too big
    return best


def fit_text(text, draw, font_path, box_w, box_h, min_size=1, max_size=200):
    """Find the largest font size where text fits; returns (size, wrapped, fits).

    Words are never broken if a legible size exists without breaking them; hyphenation is a fallback
    that still keeps the font >= min_size. If nothing fits, returns min_size (fits=False).
    """
    best = _largest_fitting(text, draw, font_path, box_w, box_h, min_size, max_size, hyphenate=False)
    if best is None:
        best = _largest_fitting(text, draw, font_path, box_w, box_h, min_size, max_size, hyphenate=True)
    if best is not None:
        return best[0], best[1], True
    _, wrapped = fits_in_box(text, draw, ImageFont.truetype(font_path, min_size), box_w, box_h)
    return min_size, wrapped, False


def find_max_fontsize(text, draw, font_path, box_w, box_h, min_size=1, max_size=200):
    """Find the largest font size where wrapped text fits inside box.

    If nothing fits, returns min_size with the text wrapped at that size (it may still overflow).
    """
    size, wrapped, _ = fit_text(text, draw, font_path, box_w, box_h, min_size, max_size)
    return size, wrapped


def replace_text_with_translation(image_path, font_path, translations, expand_text_area=True):
    image = Image.open(image_path)
    # Measure free bubble space on the cleaned page before any translation is drawn onto it.
    cleaned = image.copy()
    draw = ImageDraw.Draw(image)
    polygons = [t["polygon"] for t in translations]
    for i, translation in enumerate(translations):
        polygon = translation["polygon"]
        translated_text = translation["translated"]
        if translated_text:
            text_area = None
            if expand_text_area:
                others = polygons[:i] + polygons[i + 1 :]
                area = find_text_area(cleaned, polygon, others)
                text_area = area if list(area) != list(polygon) else None
            draw_wrapped_text(image, draw, polygon, translated_text, font_path, text_area)
    return image
