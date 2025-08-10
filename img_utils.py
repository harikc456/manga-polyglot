import cv2
import torch
import numpy as np
from typing import Union
from utils.yolov5_utils import non_max_suppression
from utils.imgproc_utils import letterbox
from PIL import Image, ImageDraw, ImageFont


def imread(imgpath, read_type=cv2.IMREAD_COLOR):
    """Read an image from a file path (supports non-ASCII paths) using OpenCV."""
    return cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), read_type)


def preprocess_img(img, input_size=(1024, 1024), device="cpu", bgr2rgb=True, half=False, to_tensor=True):
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in, ratio, (dw, dh) = letterbox(img, new_shape=input_size, auto=False, stride=64)
    if to_tensor:
        img_in = img_in.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255
        if to_tensor:
            img_in = torch.from_numpy(img_in).to(device)
            if half:
                img_in = img_in.half()
    return img_in, ratio, int(dw), int(dh)


def postprocess_mask(img: Union[torch.Tensor, np.ndarray], thresh=None):
    if isinstance(img, torch.Tensor):
        img = img.squeeze_()
        if img.device != "cpu":
            img = img.detach_().cpu()
        img = img.numpy()
    else:
        img = img.squeeze()
    if thresh is not None:
        img = img > thresh
    img = img * 255
    return img.astype(np.uint8)


def postprocess_yolo(det, conf_thresh, nms_thresh, resize_ratio, sort_func=None):
    det = non_max_suppression(det, conf_thresh, nms_thresh)[0]
    # bbox = det[..., 0:4]
    if det.device != "cpu":
        det = det.detach_().cpu().numpy()
    det[..., [0, 2]] = det[..., [0, 2]] * resize_ratio[0]
    det[..., [1, 3]] = det[..., [1, 3]] * resize_ratio[1]
    if sort_func is not None:
        det = sort_func(det)

    blines = det[..., 0:4].astype(np.int32)
    confs = np.round(det[..., 4], 3)
    cls = det[..., 5].astype(np.int32)
    return blines, cls, confs


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
    luminance = (0.299 * background_color[0] + 0.587 * background_color[1] + 0.114 * background_color[2]) / 255

    # Determine the text color based on the background luminance
    if luminance > 0.5:
        return "black"  # Use black text for light backgrounds
    else:
        return "white"  # Use white text for dark backgrounds


def get_text_width(text, font):
    left, top, right, bottom = font.getbbox(text.strip())
    width = right - left
    return width


def get_text_height(text, font):
    left, top, right, bottom = font.getbbox(text.strip())
    height = bottom - top
    return height


def text_wrap(text, font, max_width):
    lines = []
    width = get_text_width(text, font)
    if width <= max_width:
        lines.append(text)
    else:
        words = text.split()
        line = ""
        for i, word in enumerate(words):
            word = word.strip()
            width = get_text_width(word, font)
            temp_line = line + " " + word
            if get_text_width(temp_line, font) > max_width:
                lines.append(line)
                line = word
            else:
                line = line + " " + word
        if line:
            lines.append(line)
    lines = [line.strip() for line in lines if line]
    return lines


def find_text_box_coordinates(max_line_width: int, max_line_height: int, center_x: int, center_y: int):
    text_box_x_min = center_x - max_line_width // 2 + 1
    text_box_y_min = center_y - max_line_height // 2 + 1

    text_box_x_max = center_x + max_line_width // 2 + 1
    text_box_y_max = center_y + max_line_height // 2 + 1

    return text_box_x_min, text_box_y_min, text_box_x_max, text_box_y_max


def is_out_of_bounds(x_min, y_min, x_max, y_max, text_box_x_min, text_box_y_min, text_box_x_max, text_box_y_max):
    if text_box_x_min < x_min:
        return True
    if text_box_y_min < y_min:
        return True
    if text_box_x_max > x_max:
        return True
    if text_box_y_max > y_max:
        return True
    return False


def determine_font_size(
    font_path: str,
    text: str,
    max_width: int,
    max_height: int,
    center_x: int,
    center_y: int,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    size: int = 16,
    disable_increment: bool = False,
):
    if size < 11:
        return size
    font = ImageFont.truetype(font_path, size=size)
    lines = text_wrap(text, font, max_width - 10)
    max_line_width = max([get_text_width(line, font) for line in lines])
    max_line_height = max([get_text_height(line, font) for line in lines])
    text_box_area = 2 * max_line_width * max_line_height
    bounding_box_area = 2 * max_width * max_height

    text_box_x_min, text_box_y_min, text_box_x_max, text_box_y_max = find_text_box_coordinates(
        max_line_width, max_line_height, center_x, center_y
    )

    out_of_bounds = is_out_of_bounds(
        x_min, y_min, x_max, y_max, text_box_x_min, text_box_y_min, text_box_x_max, text_box_y_max
    )

    if out_of_bounds:
        size = determine_font_size(
            font_path, text, max_width, max_height, center_x, center_y, x_min, y_min, x_max, y_max, size - 1, True
        )

    if bounding_box_area / text_box_area > 16 and not disable_increment:
        size = determine_font_size(
            font_path, text, max_width, max_height, center_x, center_y, x_min, y_min, x_max, y_max, size + 1
        )
    return size


def replace_text_with_translation(image_path, font_path, translated_texts, text_boxes):
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    x_pad, y_pad = 5, 5

    # Replace each text box with translated text
    for text_box, translated in zip(text_boxes, translated_texts):

        # Set initial values
        x_min, y_min = text_box[0], text_box[1]
        x_max, y_max = text_box[2], text_box[3]

        max_width = x_max - x_min - 2 * x_pad
        max_height = y_max - y_min - 2 * y_pad

        center_x, center_y = x_min + max_width // 2, y_min + max_height // 2

        # Load a font
        font_size = determine_font_size(
            font_path, translated, max_width, max_height, center_x, center_y, x_min, y_min, x_max, y_max
        )
        font = ImageFont.truetype(font_path, size=font_size)

        lines = text_wrap(translated, font, max_width - 10)
        max_line_width = max([get_text_width(line, font) for line in lines])
        max_line_height = max([get_text_height(line, font) for line in lines])

        text_box_x_min = center_x - max_line_width // 2 + 1
        text_box_y_min = center_y - max_line_height // 2 + 1
        text_box_x_max = center_x + max_line_width // 2 + 1
        text_box_y_max = center_y + max_line_height // 2 + 1

        out_of_bounds = is_out_of_bounds(
            x_min, y_min, x_max, y_max, text_box_x_min, text_box_y_min, text_box_x_max, text_box_y_max
        )

        # Find the most common color in the text region
        background_color = get_background_color(image, x_min, y_min, x_max, y_max)

        # Draw a rectangle to cover the text region with the original background color
        draw.rectangle(((x_min, y_min), (x_max, y_max)), fill=background_color)

        fill = get_text_fill_color(background_color)
        x = x_min if out_of_bounds else text_box_x_min
        y = y_min if out_of_bounds else text_box_y_min

        for line in lines:
            draw.text((x, y), line, fill=fill, font=font)
            y = y + max_line_height + y_pad

    return image
