from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from img_utils import (
    draw_wrapped_text,
    fill_bubble_with_estimated_color,
    find_max_fontsize,
    find_text_area,
    replace_text_with_translation,
    wrap_text,
    sort_manga_reading_order,
)

FONT_PATH = str(Path(__file__).parent.parent / "fonts" / "animeace2_bld.otf")


def test_fill_uses_rgb_order_on_coloured_page():
    img = Image.new("RGB", (200, 200), (255, 0, 0))
    out = fill_bubble_with_estimated_color(img, [50, 50, 150, 150])
    assert out.getpixel((100, 100)) == (255, 0, 0)


def test_fill_samples_background_not_text():
    img = Image.new("RGB", (200, 200), (10, 120, 200))
    ImageDraw.Draw(img).rectangle([80, 80, 120, 120], fill=(0, 0, 0))
    out = fill_bubble_with_estimated_color(img, [50, 50, 150, 150])
    assert out.getpixel((100, 100)) == (10, 120, 200)


def _box(name, x, y, w=100, h=60):
    return {"text": name, "insertion_polygon": [x, y, x + w, y + h]}


def test_reading_order_is_rows_top_to_bottom_right_to_left():
    boxes = [
        _box("top-left", 100, 100),
        _box("bot-right", 600, 700),
        _box("top-right", 600, 100),
        _box("bot-left", 100, 700),
    ]
    order = [b["text"] for b in sort_manga_reading_order(boxes)]
    assert order == ["top-right", "top-left", "bot-right", "bot-left"]


def test_reading_order_groups_slightly_staggered_bubbles_into_one_row():
    boxes = [_box("left", 100, 130), _box("right", 600, 100)]
    order = [b["text"] for b in sort_manga_reading_order(boxes)]
    assert order == ["right", "left"]


def test_reading_order_empty():
    assert sort_manga_reading_order([]) == []


def _ink_bbox(img, background=(255, 255, 255)):
    bg = Image.new("RGB", img.size, background)
    from PIL import ImageChops

    return ImageChops.difference(img, bg).getbbox()


def test_drawn_text_stays_inside_box():
    img = Image.new("RGB", (300, 300), (255, 255, 255))
    box = [100, 100, 190, 190]
    draw_wrapped_text(img, ImageDraw.Draw(img), box, "Hello there, how are you", FONT_PATH)
    x0, y0, x1, y1 = _ink_bbox(img)
    # Text may grow the area up to 2x about its centre, but never beyond that.
    assert x0 >= box[0] - 45 and y0 >= box[1] - 45 and x1 <= box[2] + 45 and y1 <= box[3] + 45


def test_drawn_text_is_centred_in_box():
    img = Image.new("RGB", (400, 400), (255, 255, 255))
    box = [100, 100, 300, 300]
    draw_wrapped_text(img, ImageDraw.Draw(img), box, "Hi", FONT_PATH)
    x0, y0, x1, y1 = _ink_bbox(img)
    assert abs((x0 + x1) / 2 - 200) <= 3
    assert abs((y0 + y1) / 2 - 200) <= 3


def test_find_max_fontsize_wraps_when_nothing_fits():
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    size, wrapped = find_max_fontsize(
        "one two three four five six", draw, FONT_PATH, 30, 10, min_size=4, max_size=20
    )
    assert size == 4
    assert "\n" in wrapped


def _bubble_page(size=(600, 600), ellipse=(100, 100, 500, 400), outline=4):
    """A cleaned page: grey screentone-ish background with a white outlined oval bubble."""
    img = Image.new("RGB", size, (200, 200, 200))
    ImageDraw.Draw(img).ellipse(ellipse, fill=(255, 255, 255), outline=(0, 0, 0), width=outline)
    return img


def _inside_ellipse(x, y, ellipse):
    ex0, ey0, ex1, ey1 = ellipse
    cx, cy = (ex0 + ex1) / 2, (ey0 + ey1) / 2
    rx, ry = (ex1 - ex0) / 2, (ey1 - ey0) / 2
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1


def test_text_area_expands_inside_big_bubble():
    ellipse = (100, 100, 500, 400)
    img = _bubble_page(ellipse=ellipse)
    box = [270, 220, 330, 280]
    x0, y0, x1, y1 = find_text_area(img, box)
    assert x0 <= box[0] and y0 <= box[1] and x1 >= box[2] and y1 >= box[3]
    assert (x1 - x0) * (y1 - y0) >= 3 * (box[2] - box[0]) * (box[3] - box[1])
    for corner in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
        assert _inside_ellipse(*corner, ellipse)


def test_text_area_unchanged_on_open_background():
    img = Image.new("RGB", (600, 600), (255, 255, 255))
    box = [270, 220, 330, 280]
    assert find_text_area(img, box) == box


def test_text_area_unchanged_when_bubble_holds_another_text_box():
    img = _bubble_page()
    box = [180, 220, 240, 280]
    other = [360, 220, 420, 280]
    assert find_text_area(img, box, [other]) == box


def test_text_area_barely_grows_in_tight_bubble():
    ellipse = (240, 190, 360, 310)
    img = _bubble_page(ellipse=ellipse)
    box = [265, 215, 335, 285]
    x0, y0, x1, y1 = find_text_area(img, box)
    assert x0 >= ellipse[0] and y0 >= ellipse[1] and x1 <= ellipse[2] and y1 <= ellipse[3]


def test_text_area_ignores_leftover_text_specks():
    img = _bubble_page()
    draw = ImageDraw.Draw(img)
    for x in range(275, 330, 9):
        draw.point((x, 250), fill=(30, 30, 30))
    box = [270, 220, 330, 280]
    x0, y0, x1, y1 = find_text_area(img, box)
    assert (x1 - x0) > (box[2] - box[0]) * 1.5


def test_render_uses_bigger_font_in_big_bubble(tmp_path):
    img = _bubble_page()
    page = tmp_path / "page.png"
    img.save(page)
    box = [270, 220, 330, 280]
    translations = [{"translated": "This is a fairly long line of dialogue", "polygon": box}]

    plain = replace_text_with_translation(str(page), FONT_PATH, translations, expand_text_area=False)
    grown = replace_text_with_translation(str(page), FONT_PATH, translations, expand_text_area=True)

    plain_ink = _diff_bbox(plain, img)
    grown_ink = _diff_bbox(grown, img)
    assert plain_ink[0] >= box[0] - 30 and plain_ink[2] <= box[2] + 30
    assert (grown_ink[2] - grown_ink[0]) > (plain_ink[2] - plain_ink[0])
    for x, y in [(grown_ink[0], grown_ink[1]), (grown_ink[2], grown_ink[3])]:
        assert _inside_ellipse(x, y, (100, 100, 500, 400))


def _diff_bbox(a, b):
    from PIL import ImageChops

    return ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox()


def test_wrap_text_balances_lines_without_orphan():
    font = ImageFont.truetype(FONT_PATH, 20)
    wrapped = wrap_text("I will never forgive you ever", font, 260)
    lines = wrapped.split("\n")
    assert len(lines) >= 2
    assert len(lines[-1].split()) > 1 or len(lines[-1]) >= len(lines[0]) * 0.5


def test_wrap_text_hyphenates_in_chunks_of_three_or_more():
    font = ImageFont.truetype(FONT_PATH, 20)
    wrapped = wrap_text("Extraordinarily", font, 90, hyphenate_only_words_ge=7)
    pieces = wrapped.split("\n")
    assert len(pieces) > 1
    assert all(len(p.rstrip("-")) >= 3 for p in pieces)


def test_long_word_in_narrow_box_does_not_become_one_char_per_line():
    img = Image.new("RGB", (400, 400), (255, 255, 255))
    box = [190, 190, 220, 230]  # tiny source box, long translation
    draw_wrapped_text(img, ImageDraw.Draw(img), box, "Unbelievable", FONT_PATH)
    x0, y0, x1, y1 = _ink_bbox(img)
    # Area grows past the box instead of stacking single characters: wide and not absurdly tall.
    assert (x1 - x0) > (box[2] - box[0])
    assert (y1 - y0) < 6 * (x1 - x0)


def test_text_has_contrasting_outline():
    img = Image.new("RGB", (300, 300), (255, 255, 255))
    draw_wrapped_text(img, ImageDraw.Draw(img), [50, 50, 250, 250], "Hi", FONT_PATH)
    colours = {c for _, c in img.getcolors(maxcolors=100000)}
    assert (0, 0, 0) in colours
