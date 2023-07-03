import numpy as np
import cv2
import torch
import torchvision.transforms as T
from pathlib import Path

from craft.torch_utils import _get_state_dict
from image_utils import _get_width_and_height
from craft.craft import CRAFT
from craft.link_refiner import LinkRefiner

TRANSFORM = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def load_craft_checkpoint(cuda=False):
    craft = CRAFT()
    if cuda:
        craft = craft.to("cuda")

    ckpt_path = Path(__file__).parent/"craft_mlt_25k.pth"
    state_dict = _get_state_dict(ckpt_path=ckpt_path, include="module.", delete="module.", cuda=cuda)
    craft.load_state_dict(state_dict=state_dict, strict=True)
    craft.eval()

    print(f"Loaded pre-trained parameters for 'CRAFT'\n    from checkpoint '{ckpt_path}'.")
    return craft


def load_link_refiner_checkpoint(cuda=False):
    link_refiner = LinkRefiner()
    if cuda:
        link_refiner = link_refiner.to("cuda")

    ckpt_path = Path(__file__).parent/"craft_refiner_CTW1500.pth"
    state_dict = _get_state_dict(ckpt_path=ckpt_path, include="module.", delete="module.", cuda=cuda)
    link_refiner.load_state_dict(state_dict=state_dict, strict=True)
    link_refiner.eval()

    print(f"Loaded pre-trained parameters for 'LinkRefiner'\n    from checkpoint '{ckpt_path}'.")
    return link_refiner


def _resize_image_for_craft_input(img):
    ### Resize the image so that the width and the height are multiples of 32 each. ###
    width, height = _get_width_and_height(img)

    height32, width32 = height, width
    if height % 32 != 0:
        height32 = height + (32 - height % 32)
    if width % 32 != 0:
        width32 = width + (32 - width % 32)

    canvas = np.zeros(shape=(height32, width32, img.shape[2]), dtype=np.uint8)
    resized_img = cv2.resize(src=img, dsize=(width, height), interpolation=cv2.INTER_LANCZOS4)
    canvas[: height, : width, :] = resized_img
    return canvas


def _normalize_score_map(score_map):
    score_map = np.clip(a=score_map, a_min=0, a_max=1)
    score_map *= 255
    score_map = score_map.astype(np.uint8)
    return score_map


def _postprocess_score_map(z, ori_width, ori_height, resized_width, resized_height):
    resized_z = cv2.resize(src=z, dsize=(resized_width, resized_height))
    resized_z = resized_z[: ori_height, : ori_width]
    score_map = _normalize_score_map(resized_z)
    return score_map


def _infer_using_craft(img, craft, transform=TRANSFORM, cuda=False):
    z = transform(img)
    z = z.unsqueeze(0)
    if cuda:
        z = z.to("cuda")

    craft.eval()
    with torch.no_grad():
        z, feature = craft(z)
    return z, feature


def get_score_maps(img, craft, link_refiner=None, cuda=False):
    ori_width, ori_height = _get_width_and_height(img)

    resized_img = _resize_image_for_craft_input(img)
    resized_width, resized_height = _get_width_and_height(resized_img)

    z, feature = _infer_using_craft(img=resized_img, craft=craft, cuda=cuda)

    z0 = z[0, :, :, 0].detach().cpu().numpy()
    region_score_map = _postprocess_score_map(
        z=z0, ori_width=ori_width, ori_height=ori_height, resized_width=resized_width, resized_height=resized_height
    )

    z1 = z[0, :, :, 1].detach().cpu().numpy()
    affinity_score_map = _postprocess_score_map(
        z=z1, ori_width=ori_width, ori_height=ori_height, resized_width=resized_width, resized_height=resized_height
    )

    line_score_map = None
    if link_refiner is not None:
        with torch.no_grad():
            z = link_refiner(z, feature)
        refined_z0 = z[0, :, :, 0].detach().cpu().numpy()
        line_score_map = _postprocess_score_map(
            z=refined_z0,
            ori_width=ori_width,
            ori_height=ori_height,
            resized_width=resized_width,
            resized_height=resized_height
        )
    return region_score_map, affinity_score_map, line_score_map
