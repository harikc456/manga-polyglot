from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from img_utils import (
    draw_wrapped_text,
    fill_bubble_with_estimated_color,
    find_max_fontsize,
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
    assert x0 >= box[0] and y0 >= box[1] and x1 <= box[2] and y1 <= box[3]


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
