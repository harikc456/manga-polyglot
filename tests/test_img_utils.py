import pytest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from img_utils import (
    renderable_text,
    draw_wrapped_text,
    fill_bubble_with_estimated_color,
    find_max_fontsize,
    find_text_area,
    find_text_areas,
    find_text_region,
    layout_in_region,
    text_areas_in_region,
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


def test_text_area_on_open_background_grows_only_within_limit():
    img = Image.new("RGB", (600, 600), (255, 255, 255))
    box = [270, 220, 330, 280]
    x0, y0, x1, y1 = find_text_area(img, box)
    assert x1 - x0 > 60 and y1 - y0 > 60
    assert x1 - x0 <= 1.5 * 60 + 2 and y1 - y0 <= 1.5 * 60 + 2


def test_text_area_shares_bubble_with_another_text_box():
    """Two boxes in one bubble each grow into their own half instead of neither growing."""
    img = _bubble_page()
    box = [180, 220, 240, 280]
    other = [360, 220, 420, 280]
    x0, y0, x1, y1 = find_text_area(img, box, [other])
    assert x1 <= 300 + 2  # stays on its side of the midpoint between the boxes
    assert (x1 - x0) * (y1 - y0) >= 2 * 60 * 60
    ox0, _, ox1, _ = find_text_area(img, other, [box])
    assert ox0 >= 300 - 2 and ox1 - ox0 > 60


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


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello!", "Hello!"),
        ("Wait〜!", "Wait~!"),
        ("Wait～!", "Wait~!"),
        ("Well—fine.", "Well-fine."),
        ("★Special★", "*Special*"),
        ("I love you♥", "I love you"),
        ("La la ♪ la", "La la la"),
        ("Thanks 😊!", "Thanks !"),
        ("Hi ❤️ there", "Hi there"),
        ("😊", ""),
    ],
)
def test_renderable_text_replaces_or_drops_missing_glyphs(text, expected):
    assert renderable_text(text, FONT_PATH) == expected


def test_renderable_text_keeps_glyphs_the_font_has():
    # Arial has ♥ and ♪, so they are kept there.
    arial = str(Path(FONT_PATH).parent / "ARIAL.TTF")
    assert renderable_text("I love you♥ ♪", arial) == "I love you♥ ♪"


def test_render_skips_text_with_no_drawable_glyphs(tmp_path):
    page = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(page)
    out = replace_text_with_translation(str(page), FONT_PATH, [{"polygon": [10, 10, 190, 90], "translated": "😊"}])
    assert out.getextrema() == ((255, 255),) * 3  # nothing drawn


def test_tall_box_widens_into_round_bubble():
    """A tall vertical-Japanese box must not keep its full height and stay narrow."""
    ellipse = (200, 150, 600, 650)
    img = _bubble_page(size=(800, 800), ellipse=ellipse)
    box = [370, 230, 430, 570]
    areas = find_text_areas(img, box)
    widest = max(areas, key=lambda a: a[2] - a[0])
    assert widest[2] - widest[0] >= 250
    for x0, y0, x1, y1 in areas:
        for corner in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
            assert _inside_ellipse(*corner, ellipse)


def test_text_areas_offer_different_shapes():
    img = _bubble_page(size=(800, 800), ellipse=(200, 150, 600, 650))
    areas = find_text_areas(img, [370, 230, 430, 570])
    aspects = [(a[2] - a[0]) / (a[3] - a[1]) for a in areas]
    assert min(aspects) < 0.8 and max(aspects) > 1.2


def test_render_tall_box_uses_bigger_font_than_box_alone(tmp_path):
    img = _bubble_page(size=(800, 800), ellipse=(200, 150, 600, 650))
    page = tmp_path / "page.png"
    img.save(page)
    translations = [{"translated": "Wait for me, I said I'm coming with you!", "polygon": [370, 230, 430, 570]}]

    plain = _diff_bbox(replace_text_with_translation(str(page), FONT_PATH, translations, expand_text_area=False), img)
    grown = _diff_bbox(replace_text_with_translation(str(page), FONT_PATH, translations, expand_text_area=True), img)

    assert grown[2] - grown[0] > 2 * (plain[2] - plain[0])  # uses the bubble's width
    assert (grown[2] - grown[0]) * (grown[3] - grown[1]) > 2 * (plain[2] - plain[0]) * (plain[3] - plain[1])  # bigger font


def test_off_centre_box_in_big_bubble_expands():
    """The bubble reaches past the initial search window around a small box; that is not open background."""
    img = _bubble_page(size=(1200, 900), ellipse=(100, 100, 1100, 800))
    box = [180, 420, 220, 460]
    x0, y0, x1, y1 = find_text_area(img, box)
    assert (x1 - x0) * (y1 - y0) >= 10 * 40 * 40


_LONG_LINE = "Wait for me, I said I'm coming with you no matter what happens!"


def _best_rect_size(img, region, text):
    from img_utils import _fit_in, _max_font_size
    draw = ImageDraw.Draw(img)
    sizes = [
        _fit_in(text, draw, FONT_PATH, a, _max_font_size(a, True))
        for a in text_areas_in_region(region)
    ]
    return max(size for size, _, fits in sizes if fits)


def _manga_page():
    """White page (gutters) with a grey panel; one bubble breaks out of the panel into the gutter."""
    img = Image.new("RGB", (1200, 1700), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((40, 40, 1160, 820), fill=(150, 150, 150), outline="black", width=5)
    d.ellipse((700, 540, 1000, 940), fill="white", outline="black", width=4)
    return img


def test_bubble_leaking_into_gutter_still_expands_within_limit():
    img = _manga_page()
    box = [820, 620, 880, 860]
    x0, y0, x1, y1 = find_text_area(img, box)
    assert x1 - x0 >= 150  # much wider than the 60px box
    assert x1 - x0 <= 1.5 * 240 + 2 and y1 - y0 <= 1.5 * 240 + 2


def test_bubble_cut_by_page_edge_still_expands():
    img = Image.new("RGB", (800, 800), (150, 150, 150))
    ImageDraw.Draw(img).ellipse((500, 300, 900, 800), fill="white", outline="black", width=4)
    box = [660, 400, 720, 640]
    x0, y0, x1, y1 = find_text_area(img, box)
    assert x1 - x0 >= 120


def test_off_colour_cleaning_fill_still_finds_bubble():
    """The cleaning step painted the box light grey inside a white bubble."""
    img = _bubble_page(size=(800, 800), ellipse=(200, 150, 600, 650))
    box = [370, 230, 430, 570]
    ImageDraw.Draw(img).rectangle(box, fill=(232, 232, 232))
    x0, y0, x1, y1 = find_text_area(img, box)
    assert x1 - x0 >= 200


def test_shaped_layout_lines_follow_round_bubble():
    ellipse = (200, 150, 600, 650)
    img = _bubble_page(size=(800, 800), ellipse=ellipse)
    region = find_text_region(img, [370, 230, 430, 570])
    layout = layout_in_region(_LONG_LINE, region, FONT_PATH)

    assert layout is not None
    assert " ".join(line.text for line in layout.lines) == _LONG_LINE
    widths = [line.right - line.left for line in layout.lines]
    middle = widths[len(widths) // 2]
    assert middle > widths[0] and middle > widths[-1]  # narrow at the top and bottom, wide in the middle
    for line in layout.lines:
        for x, y in [(line.left, line.top), (line.right, line.top), (line.left, line.bottom), (line.right, line.bottom)]:
            assert _inside_ellipse(x, y, ellipse)


def test_shaped_layout_takes_a_font_at_least_as_big_as_any_rectangle():
    img = _bubble_page(size=(800, 800), ellipse=(200, 150, 600, 650))
    region = find_text_region(img, [370, 230, 430, 570])
    assert layout_in_region(_LONG_LINE, region, FONT_PATH).size >= _best_rect_size(img, region, _LONG_LINE)


def test_shaped_layout_is_none_when_a_word_cannot_fit():
    img = _bubble_page(ellipse=(240, 190, 360, 310))
    region = find_text_region(img, [265, 215, 335, 285])
    assert layout_in_region("Supercalifragilisticexpialidocious", region, FONT_PATH) is None


def test_render_shaped_text_stays_inside_bubble(tmp_path):
    ellipse = (200, 150, 600, 650)
    img = _bubble_page(size=(800, 800), ellipse=ellipse)
    page = tmp_path / "page.png"
    img.save(page)
    out = replace_text_with_translation(str(page), FONT_PATH, [{"translated": _LONG_LINE, "polygon": [370, 230, 430, 570]}])
    x0, y0, x1, y1 = _diff_bbox(out, img)
    assert x1 - x0 > 200
    import numpy as np
    from PIL import ImageChops
    ys, xs = np.nonzero(np.asarray(ImageChops.difference(out.convert("RGB"), img).convert("L")))
    assert all(_inside_ellipse(x, y, ellipse) for x, y in zip(xs, ys))


def test_short_text_in_big_bubble_goes_past_old_cap():
    img = _bubble_page(size=(800, 800), ellipse=(200, 150, 600, 650))
    region = find_text_region(img, [370, 230, 430, 570])
    assert layout_in_region("Huh?!", region, FONT_PATH).size > 36


def test_max_font_size_caps_short_text():
    img = _bubble_page(size=(800, 800), ellipse=(200, 150, 600, 650))
    region = find_text_region(img, [370, 230, 430, 570])
    assert layout_in_region("Huh?!", region, FONT_PATH, max_font_size=30).size == 30


def test_render_max_font_size_changes_text_size(tmp_path):
    img = _bubble_page(size=(800, 800), ellipse=(200, 150, 600, 650))
    page = tmp_path / "page.png"
    img.save(page)
    tr = [{"translated": "Huh?!", "polygon": [370, 230, 430, 570]}]
    small = _diff_bbox(replace_text_with_translation(str(page), FONT_PATH, tr, max_font_size=30), img)
    big = _diff_bbox(replace_text_with_translation(str(page), FONT_PATH, tr, max_font_size=90), img)
    assert big[2] - big[0] > 2 * (small[2] - small[0])


def test_shaped_text_keeps_clear_of_outline_at_large_sizes():
    ellipse = (200, 150, 600, 650)
    img = _bubble_page(size=(800, 800), ellipse=ellipse)
    region = find_text_region(img, [370, 230, 430, 570])
    layout = layout_in_region(_LONG_LINE, region, FONT_PATH, max_font_size=96)
    gap = layout.size // 3
    for line in layout.lines:
        for x, y in [(line.left - gap, line.top - gap), (line.right + gap, line.top - gap),
                     (line.left - gap, line.bottom + gap), (line.right + gap, line.bottom + gap)]:
            assert _inside_ellipse(x, y, ellipse)


def test_off_colour_cleaning_fill_does_not_block_any_side():
    img = _bubble_page(size=(800, 800), ellipse=(200, 150, 600, 650))
    box = [370, 230, 430, 570]
    ImageDraw.Draw(img).rectangle(box, fill=(232, 232, 232))  # inclusive, like fill_bubble_with_estimated_color
    x0, y0, x1, y1 = find_text_area(img, box)
    assert x0 < 300 and x1 > 500
