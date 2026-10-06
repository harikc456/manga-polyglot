import cv2
import math
import re
import unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from fontTools.ttLib import TTFont

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


# Largest font for text in an expanded area (short text in a big bubble); overridable per page.
MAX_FONT_SIZE = 64


def _max_font_size(area, expanded, cap=MAX_FONT_SIZE):
    # An expanded area may take a bigger font, scaled to its height but capped so it stays readable.
    return min(cap, max(20, (area[3] - area[1]) // 4)) if expanded else 20


def _fit_in(text, draw, font_path, area, max_size):
    x_min, y_min, x_max, y_max = area
    # Keep a small margin from the edge of the area.
    box_width = int(0.94 * (x_max - x_min))
    box_height = int(0.94 * (y_max - y_min))
    return fit_text(text, draw, font_path, box_width, box_height, min_size=MIN_LEGIBLE_SIZE, max_size=max_size)


def _text_colours(image, polygon):
    fill = get_text_fill_color(get_background_color(image, *polygon))
    return fill, "white" if fill == "black" else "black"


def _draw_shaped(image, draw, polygon, layout, font_path):
    font = ImageFont.truetype(font_path, layout.size)
    stroke = stroke_width_for(layout.size)
    fill, outline = _text_colours(image, polygon)
    for line in layout.lines:
        bbox = draw.textbbox((0, 0), line.text, font=font, stroke_width=stroke)
        # Centre on the line; subtract the bbox origin so glyph side bearings don't shift the text.
        x = (line.left + line.right) / 2 - (bbox[2] - bbox[0]) / 2 - bbox[0]
        draw.text((x, line.top + stroke), line.text, font=font, fill=fill, stroke_width=stroke, stroke_fill=outline)


def draw_wrapped_text(
    image, draw, polygon, text, font_path, text_area=None, text_areas=None, text_region=None, max_font_size=MAX_FONT_SIZE
):
    """Draw text fitted into text_area (defaults to polygon); colours are taken from around polygon.

    With text_region (the bubble from find_text_region), lines follow the bubble's shape, unless one of
    its rectangles takes a bigger font. With text_areas (candidate rectangles), the one that takes the
    largest font is used. When the text cannot be set at a legible size, the area is grown (up to 2x
    about its centre) so tiny source boxes do not force one-character lines.
    """
    if text_region is not None and not text_areas:
        # The cleaned box itself is offered last, in case its own shape suits the text best.
        text_areas = text_areas_in_region(text_region) + [list(polygon)]
    expanded = bool(text_areas) or text_area is not None
    candidates = [tuple(a) for a in text_areas] if text_areas else [tuple(text_area or polygon)]
    base, best_size = candidates[0], -1
    if len(candidates) > 1:
        for area in candidates:
            size, _, fits = _fit_in(text, draw, font_path, area, _max_font_size(area, expanded, max_font_size))
            if fits and size > best_size:
                base, best_size = area, size
    max_size = _max_font_size(base, expanded, max_font_size)

    if text_region is not None:
        shaped = layout_in_region(text, text_region, font_path, max_font_size)
        if shaped is not None and shaped.size >= best_size:
            _draw_shaped(image, draw, polygon, shaped, font_path)
            return

    for scale in GROW_STEPS:
        x_min, y_min, x_max, y_max = _scaled_area(base, scale, image.size)
        font_size, wrapped, fits = _fit_in(text, draw, font_path, (x_min, y_min, x_max, y_max), max_size)
        if fits:
            break

    font = ImageFont.truetype(font_path, font_size)
    stroke = stroke_width_for(font_size)
    bbox = draw.textbbox((0, 0), wrapped, font=font, align="center", stroke_width=stroke)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # Centre in the full box; subtract the bbox origin so glyph side bearings don't shift the text.
    x = x_min + (x_max - x_min - text_w) / 2 - bbox[0]
    y = y_min + (y_max - y_min - text_h) / 2 - bbox[1]
    fill, outline = _text_colours(image, polygon)
    draw.text(
        (x, y), wrapped, font=font, fill=fill, align="center", stroke_width=stroke, stroke_fill=outline
    )


def _box_distance(xs, ys, box):
    """Distance from each (xs, ys) point to the rectangle box (0 inside it)."""
    x0, y0, x1, y1 = box
    dx = np.maximum(np.maximum(x0 - xs, 0), xs - x1)
    dy = np.maximum(np.maximum(y0 - ys, 0), ys - y1)
    return np.hypot(dx, dy)


@dataclass
class TextRegion:
    """Free space in a bubble around a cleaned text box, in a window of the page.

    mask is 1 where text may go (bubble interior, kept a few pixels from its outline); origin is the
    window's top-left on the page; centre is the text box centre in window coordinates; rows is the
    free run [top, bottom) of the centre column.
    """
    mask: np.ndarray
    origin: tuple
    box: tuple
    centre: tuple
    rows: tuple
    column_sums: np.ndarray

    def span(self, y0, y1, min_inside=0.98):
        """Free columns [x0, x1) around the centre for rows [y0, y1), in window coordinates, or None."""
        if y0 < 0 or y1 > self.mask.shape[0] or y1 <= y0:
            return None
        free = (self.column_sums[y1] - self.column_sums[y0]) >= min_inside * (y1 - y0)
        ccx = self.centre[0]
        if not free[ccx]:
            return None
        blocked_left = np.flatnonzero(~free[:ccx])
        blocked_right = np.flatnonzero(~free[ccx:])
        x0 = blocked_left[-1] + 1 if blocked_left.size else 0
        x1 = ccx + blocked_right[0] if blocked_right.size else self.mask.shape[1]
        return int(x0), int(x1)


def find_text_region(image, box, other_boxes=(), tolerance=12, edge_gap=4):
    """The empty bubble around a cleaned text box, or None if no bubble colour surrounds the box.

    Flood-fills the bubble colour (taken from just around the box) outward on the cleaned page. A region
    that runs on to the page edge is limited to OPEN_REGION_GROWTH times the box around it. Another text
    box in the same bubble gets the part of the bubble nearer to it.
    """
    return diagnose_text_region(image, box, other_boxes, tolerance, edge_gap)[0]


# When the bubble colour runs on to the page edge (a bubble breaking out of its panel into the gutter,
# a bubble cut by the page edge, or text on open background), expansion is limited to a square this
# many times the box's longer side, around the box.
OPEN_REGION_GROWTH = 1.5
# Width of the ring around the box whose colour is taken as the bubble's; the box itself was repainted by
# the cleaning step and may be slightly off.
_RING = 6


def diagnose_text_region(image, box, other_boxes=(), tolerance=12, edge_gap=4):
    """find_text_region plus a dict of measurements saying why the region was or was not found."""
    info = {"reason": "ok"}
    x_min, y_min, x_max, y_max = map(int, box)
    box_w, box_h = x_max - x_min, y_max - y_min
    if box_w <= 0 or box_h <= 0:
        return None, {"reason": "empty box"}
    img = np.asarray(image.convert("RGB"))
    img_h, img_w = img.shape[:2]
    margin = max(3 * max(box_w, box_h), 100)
    while True:
        cx0, cy0 = max(0, x_min - margin), max(0, y_min - margin)
        cx1, cy1 = min(img_w, x_max + margin), min(img_h, y_max + margin)
        crop = img[cy0:cy1, cx0:cx1].astype(np.int16)
        bx0, by0, bx1, by1 = x_min - cx0, y_min - cy0, x_max - cx0, y_max - cy0

        ring = np.zeros(crop.shape[:2], bool)
        ring[max(0, by0 - _RING) : by1 + _RING, max(0, bx0 - _RING) : bx1 + _RING] = True
        ring[by0:by1, bx0:bx1] = False
        bubble_color = np.median(crop[ring].reshape(-1, 3), axis=0)
        similar = (np.abs(crop - bubble_color).max(axis=2) <= tolerance).astype(np.uint8)
        info["bubble_color"] = [int(c) for c in bubble_color]
        # How evenly the cleaning painted the box: the share of its pixels matching the bubble colour.
        info["box_match"] = round(float(similar[by0:by1, bx0:bx1].mean()), 3)
        # Close small holes such as leftover anti-aliased text specks.
        similar = cv2.morphologyEx(similar, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        _, labels = cv2.connectedComponents(similar, connectivity=4)
        ring_labels = labels[ring]
        ring_labels = ring_labels[ring_labels > 0]
        if ring_labels.size == 0:
            return None, {**info, "reason": "no bubble colour around the box"}
        region = labels == np.bincount(ring_labels).argmax()

        # Reaching the page edge means the bubble is not closed off (open background, a gutter, the page
        # edge); reaching only the window edge means a big bubble around an off-centre box, so look further.
        touches_page = (
            (cy0 == 0 and region[0, :].any()) or (cy1 == img_h and region[-1, :].any())
            or (cx0 == 0 and region[:, 0].any()) or (cx1 == img_w and region[:, -1].any())
        )
        touches_window = region[0, :].any() or region[-1, :].any() or region[:, 0].any() or region[:, -1].any()
        if touches_page:
            edges = [
                name for name, hit in [
                    ("top", cy0 == 0 and region[0, :].any()), ("bottom", cy1 == img_h and region[-1, :].any()),
                    ("left", cx0 == 0 and region[:, 0].any()), ("right", cx1 == img_w and region[:, -1].any()),
                ] if hit
            ]
            # A large share means the fill ran into the page background (gutter, gap in the outline).
            info.update(reason="open, limited", edges=edges, region_page_share=round(float(region.sum()) / (img_w * img_h), 3))
            half = int(OPEN_REGION_GROWTH * max(box_w, box_h)) // 2
            ccx, ccy = (bx0 + bx1) // 2, (by0 + by1) // 2
            limit = np.zeros_like(region)
            limit[max(0, ccy - half) : ccy + half, max(0, ccx - half) : ccx + half] = True
            region &= limit
            break
        if not touches_window:
            break
        margin *= 2

    own = (bx0, by0, bx1, by1)
    ys, xs = np.mgrid[0 : region.shape[0], 0 : region.shape[1]]
    own_distance = None
    for other in other_boxes:
        ob = (other[0] - cx0, other[1] - cy0, other[2] - cx0, other[3] - cy0)
        ox, oy = int((ob[0] + ob[2]) / 2), int((ob[1] + ob[3]) / 2)
        if 0 <= ox < region.shape[1] and 0 <= oy < region.shape[0] and region[oy, ox]:
            if own_distance is None:
                own_distance = _box_distance(xs, ys, own)
            region &= own_distance < _box_distance(xs, ys, ob)

    # The box is free space even if the cleaning painted it a slightly different colour than the bubble.
    # The +1 covers the right and bottom edge that ImageDraw.rectangle also paints when cleaning.
    region[by0 : by1 + 1, bx0 : bx1 + 1] = True
    # Keep the text a few pixels away from the bubble outline (and from a neighbour's share).
    region = cv2.erode(region.astype(np.uint8), np.ones((2 * edge_gap + 1,) * 2, np.uint8))
    # Treat the original box as free space even where erosion or specks nibbled it.
    region[by0:by1, bx0:bx1] = 1

    column_sums = np.vstack([np.zeros((1, region.shape[1]), np.int32), np.cumsum(region, axis=0, dtype=np.int32)])
    ccx, ccy = (bx0 + bx1) // 2, (by0 + by1) // 2
    top = ccy
    while top > 0 and region[top - 1, ccx]:
        top -= 1
    bottom = ccy + 1
    while bottom < region.shape[0] and region[bottom, ccx]:
        bottom += 1
    info["region_box_ratio"] = round(float(region.sum()) / (box_w * box_h), 2)
    return TextRegion(region, (cx0, cy0), (bx0, by0, bx1, by1), (ccx, ccy), (top, bottom), column_sums), info


# Aspect ratio (width / height) buckets; the best area of each shape is offered to the renderer.
_ASPECT_BUCKETS = (0.5, 0.8, 1.25, 2.0)


def text_areas_in_region(region, max_bands=48):
    """The largest rectangle of each shape (tall and narrow to short and wide) around the box centre, largest first.

    A tall box of vertical Japanese text is a poor shape for horizontal translated text, so the height
    is free to shrink while the width grows.
    """
    ccx, ccy = region.centre
    top, bottom = region.rows
    step = max(2, (bottom - top) // max_bands)
    best = {}  # aspect bucket -> (area, rect)
    for y0 in range(ccy, top - 1, -step):
        for y1 in range(ccy + 1, bottom + 1, step):
            span = region.span(y0, y1)
            if span is None:
                continue
            x0, x1 = span
            area = (x1 - x0) * (y1 - y0)
            bucket = int(np.searchsorted(_ASPECT_BUCKETS, (x1 - x0) / (y1 - y0)))
            if area > best.get(bucket, (0,))[0]:
                best[bucket] = (area, (x0, y0, x1, y1))
    ox, oy = region.origin
    return [[int(x0 + ox), int(y0 + oy), int(x1 + ox), int(y1 + oy)] for _, (x0, y0, x1, y1) in sorted(best.values(), reverse=True)]


def find_text_areas(image, box, other_boxes=(), **kwargs):
    """Candidate rectangles for the text inside the empty bubble around a cleaned text box, largest first.

    Returns [box] when the box is not clearly inside a bubble; otherwise the box itself is offered last,
    in case its own shape suits the text best.
    """
    region = find_text_region(image, box, other_boxes, **kwargs)
    if region is None:
        return [box]
    areas = text_areas_in_region(region)
    if list(box) not in areas:
        areas.append(list(box))
    return areas


def find_text_area(image, box, other_boxes=(), **kwargs):
    """The largest rectangle from find_text_areas."""
    return find_text_areas(image, box, other_boxes, **kwargs)[0]


@dataclass
class ShapedLine:
    """One line of shaped text; left/right is the text's own extent, top/bottom its band, in page coordinates."""
    text: str
    left: float
    top: int
    right: float
    bottom: int


@dataclass
class ShapedLayout:
    size: int
    lines: list


def _line_pitch(font, size):
    ascent, descent = font.getmetrics()
    height = ascent + descent + 2 * stroke_width_for(size)
    return height, height + max(1, size // 8)


def _shape_lines(words, region, font, size, n_lines):
    """Fill words into n_lines lines centred on the bubble, each as wide as the bubble is at its height."""
    line_h, pitch = _line_pitch(font, size)
    stroke = stroke_width_for(size)
    top_run, bottom_run = region.rows
    block_top = (top_run + bottom_run) // 2 - (n_lines * pitch - (pitch - line_h)) // 2
    # Breathing room from the bubble outline, growing with the font so big text does not crowd it.
    pad = size // 3
    lines, i = [], 0
    for n in range(n_lines):
        y0 = block_top + n * pitch
        span = region.span(y0 - pad, y0 + line_h + pad)
        if span is None:
            return None
        x0, x1 = span
        margin = max(0.03 * (x1 - x0), pad)
        width = (x1 - x0) - 2 * margin - 2 * stroke
        line = []
        while i < len(words) and font.getlength(" ".join(line + [words[i]])) <= width:
            line.append(words[i])
            i += 1
        if not line:
            return None  # a word wider than this line: the rectangle layout can hyphenate it
        lines.append((" ".join(line), (x0 + x1) / 2, y0))
        if i == len(words):
            break
    return lines if i == len(words) else None


def layout_in_region(text, region, font_path, max_font_size=MAX_FONT_SIZE):
    """Lines that follow the bubble's shape (short at a round bubble's top and bottom, long in its middle).

    Returns the layout with the largest font, or None if the text cannot be set at a legible size.
    """
    words = text.split()
    if not words:
        return None
    top_run, bottom_run = region.rows
    max_size = min(max_font_size, max(20, (bottom_run - top_run) // 4))

    def attempt(size):
        font = ImageFont.truetype(font_path, size)
        _, pitch = _line_pitch(font, size)
        for n_lines in range(1, min(len(words), (bottom_run - top_run) // pitch) + 1):
            lines = _shape_lines(words, region, font, size, n_lines)
            if lines is not None:
                return font, lines
        return None

    best, lo, hi = None, MIN_LEGIBLE_SIZE, max_size
    while lo <= hi:
        mid = (lo + hi) // 2
        found = attempt(mid)
        if found:
            best, lo = (mid, *found), mid + 1
        else:
            hi = mid - 1
    if best is None:
        return None

    size, font, lines = best
    line_h, _ = _line_pitch(font, size)
    ox, oy = region.origin
    shaped = []
    for line, centre_x, y0 in lines:
        half = font.getlength(line) / 2 + stroke_width_for(size)
        shaped.append(ShapedLine(line, centre_x - half + ox, int(y0 + oy), centre_x + half + ox, int(y0 + line_h + oy)))
    return ShapedLayout(size, shaped)


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


# Stand-ins for symbols that comic fonts often lack; anything else the font lacks (emoji, ♥, ♪) is dropped.
_GLYPH_SUBSTITUTES = {
    "〜": "~", "～": "~", "—": "-", "–": "-", "―": "-", "…": "...",
    "★": "*", "☆": "*", "✨": "*",
}


@lru_cache(maxsize=None)
def _font_charset(font_path: str) -> frozenset:
    with TTFont(font_path, lazy=True) as font:
        return frozenset(font.getBestCmap())


def renderable_text(text: str, font_path: str) -> str:
    """Replace or drop characters the font has no glyph for, so they are not drawn as empty boxes."""
    charset = _font_charset(font_path)

    def supported(s: str) -> bool:
        return bool(s) and all(c.isspace() or ord(c) in charset for c in s)

    out = []
    for char in text:
        if supported(char):
            out.append(char)
            continue
        for candidate in (unicodedata.normalize("NFKC", char), _GLYPH_SUBSTITUTES.get(char, "")):
            if supported(candidate):
                out.append(candidate)
                break
    return re.sub(r"\s{2,}", " ", "".join(out)).strip()


def replace_text_with_translation(image_path, font_path, translations, expand_text_area=True, max_font_size=None):
    """Draw each translation on the cleaned page; max_font_size caps text in expanded areas (default MAX_FONT_SIZE)."""
    max_font_size = max_font_size or MAX_FONT_SIZE
    image = Image.open(image_path)
    # Measure free bubble space on the cleaned page before any translation is drawn onto it.
    cleaned = image.copy()
    draw = ImageDraw.Draw(image)
    polygons = [t["polygon"] for t in translations]
    for i, translation in enumerate(translations):
        polygon = translation["polygon"]
        translated_text = renderable_text(translation["translated"], font_path)
        if translated_text:
            region = None
            if expand_text_area:
                others = polygons[:i] + polygons[i + 1 :]
                region = find_text_region(cleaned, polygon, others)
            draw_wrapped_text(
                image, draw, polygon, translated_text, font_path, text_region=region, max_font_size=max_font_size
            )
    return image
