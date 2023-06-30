import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from itertools import product


def load_image(img_path):
    img_path = str(img_path)
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    return img


def _to_array(img):
    img = np.array(img)
    return img


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def _get_canvas_same_size_as_image(img, black=False):
    if black:
        return np.zeros_like(img).astype("uint8")
    else:
        return (np.ones_like(img) * 255).astype("uint8")


def _repaint_segmentation_map(seg_map, n_color_values=3):
    canvas_r = _get_canvas_same_size_as_image(seg_map, black=True)
    canvas_g = _get_canvas_same_size_as_image(seg_map, black=True)
    canvas_b = _get_canvas_same_size_as_image(seg_map, black=True)

    color_vals = list(range(50, 255 + 1, 255 // n_color_values))
    perm = list(product(color_vals, color_vals, color_vals))[1:]
    perm = perm[:: 2] + perm[1:: 2]

    remainder_map = seg_map % len(perm) + 1
    for remainder, (r, g, b) in enumerate(perm, start=1):
        canvas_r[remainder_map == remainder] = r
        canvas_g[remainder_map == remainder] = g
        canvas_b[remainder_map == remainder] = b
    canvas_r[seg_map == 0] = 0
    canvas_g[seg_map == 0] = 0
    canvas_b[seg_map == 0] = 0

    dstacked = np.dstack([canvas_r, canvas_g, canvas_b])
    return dstacked


def _preprocess_image(img):
    if img.dtype == "int32":
        img = _repaint_segmentation_map(img)
    return img


def show_image(img1, img2=None, alpha=0.5):
    img1 = _to_pil(_preprocess_image(_to_array(img1)))
    if img2 is None:
        img1.show()
    else:
        img2 = _to_pil(_preprocess_image(_to_array(img2)))
        img_blended = Image.blend(im1=img1, im2=img2, alpha=alpha)
        img_blended.show()


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


def save_image(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _to_pil(img).save(str(path), quality=100, subsampling=0)
