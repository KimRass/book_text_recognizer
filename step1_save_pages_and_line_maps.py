import torch
from pdf2image import convert_from_path
from pathlib import Path
from tqdm.auto import tqdm
import argparse

from craft.detect_text import load_craft_checkpoint, load_link_refiner_checkpoint, get_score_maps
from process_images import _to_array, save_image


def get_args():
    parser = argparse.ArgumentParser("OCR")

    parser.add_argument("--pdf_path")

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    cuda = torch.cuda.is_available()

    craft = load_craft_checkpoint(cuda=cuda)
    link_refiner = load_link_refiner_checkpoint(cuda=cuda)

    # pdf_path = "/Users/jongbeomkim/Downloads/document_text_recognition/scanned2.pdf"
    for idx, img in enumerate(tqdm(list(convert_from_path(args.pdf_path)))):
        img = _to_array(img)
        save_image(
            img=img, path=Path(args.pdf_path).parent/f"pages/{str(idx).zfill(3)}.png"
        )
        _, _, line_score_map = get_score_maps(img, craft=craft, link_refiner=link_refiner, cuda=cuda)
        save_image(
            img=line_score_map, path=Path(args.pdf_path).parent/f"line_score_maps/{str(idx).zfill(3)}.png"
        )


if __name__ == "__main__":
    main()
