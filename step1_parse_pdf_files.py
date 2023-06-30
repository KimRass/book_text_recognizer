from pdf2image import convert_from_path
from pathlib import Path
from tqdm.auto import tqdm

from utils import get_args
from process_images import _to_array, save_image


def main():
    args = get_args()

    for idx, img in enumerate(tqdm(list(convert_from_path(args.pdf_path)))):
        img = _to_array(img)
        filename = str(idx).zfill(4)
        save_image(img=img, path=Path(args.save_dir)/f"pages/{filename}.png")

if __name__ == "__main__":
    main()
