from pdf2image import convert_from_path
from pathlib import Path
from tqdm.auto import tqdm

from libs.image.craft_utilities import (
    load_craft_checkpoint,
    load_craft_refiner_checkpoint,
    get_text_map_link_map_and_line_map
)
from process_images import _to_array, save_image
from book_text_recognizer.utils import get_arguments


def main():
    args = get_arguments()

    text_detector = load_craft_checkpoint(cuda=args.cuda)
    refiner = load_craft_refiner_checkpoint(cuda=args.cuda)

    for idx, img in enumerate(
        tqdm(
            list(convert_from_path(args.pdf))
        )
    ):
        img = _to_array(img)
        save_image(img=img, path=Path(args.pdf).parent/f"image/{str(idx).zfill(3)}.png")

        _, _, map_line = get_text_map_link_map_and_line_map(
            img=img, text_detector=text_detector, refiner=refiner, cuda=args.cuda
        )
        save_image(img=map_line, path=Path(args.pdf).parent/f"line_map/{str(idx).zfill(3)}.png")


if __name__ == "__main__":
    main()
