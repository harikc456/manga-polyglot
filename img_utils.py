import cv2
import math
import torch
import hashlib
import numpy as np
from typing import Union
from utils.yolov5_utils import non_max_suppression
from utils.imgproc_utils import letterbox
from PIL import Image, ImageDraw, ImageFont


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


def get_img_hash(img_path: str) -> str:
    md5hash = hashlib.md5(Image.open(img_path).tobytes())
    return md5hash.hexdigest()


def imread(imgpath, read_type=cv2.IMREAD_COLOR):
    """Read an image from a file path (supports non-ASCII paths) using OpenCV."""
    return cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), read_type)


def preprocess_img(
    img, input_size=(1024, 1024), device="cpu", bgr2rgb=True, half=False, to_tensor=True
):
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in, ratio, (dw, dh) = letterbox(
        img, new_shape=input_size, auto=False, stride=64
    )
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


def wrap_text(text, font, box_w):
    """Wrap text into lines that fit inside box_w, with hyphenation if needed."""
    words = text.split()
    lines, line = [], ""

    for word in words:
        trial = (line + " " + word).strip()
        if font.getlength(trial) <= box_w:
            line = trial
        else:
            if line:  # push previous line
                lines.append(line)
            # check if single word itself is too long → hyphenate
            if font.getlength(word) > 1.1 * box_w:
                partial = ""
                for ch in word:
                    if font.getlength(partial + ch + "-") <= box_w:
                        partial += ch
                    else:
                        lines.append(partial + "-")
                        partial = ch
                line = partial
            else:
                line = word
    if line:
        lines.append(line)
    return "\n".join(lines)


def fits_in_box(text, draw, font, box_w, box_h):
    """Check if wrapped text fits inside box with given font."""
    wrapped = wrap_text(text, font, box_w)
    # left, top, right, bottom = font.getbbox_multiline(wrapped)
    left, top, right, bottom = draw.textbbox((0, 0), wrapped, font=font)
    text_w, text_h = right - left, bottom - top
    return text_w <= box_w and text_h <= box_h, wrapped


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
