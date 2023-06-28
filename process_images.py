import cv2
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path


def _to_array(img):
    img = np.array(img)
    return img


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def save_image(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _to_pil(img).save(str(path))


def _dilate_image(img, kernel_size, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(kernel_size, kernel_size))
    img = cv2.dilate(src=img, kernel=kernel, iterations=iterations)
    return img
