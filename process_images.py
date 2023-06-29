import cv2
import numpy as np
from PIL import Image
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


def _get_width_and_height(img):
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w, _ = img.shape
    return w, h


def show_image(img):
    copied_img = img.copy()
    copied_img = _to_pil(copied_img)
    copied_img.show()


def save_image(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _to_pil(img).save(str(path), quality=100, subsampling=0)
