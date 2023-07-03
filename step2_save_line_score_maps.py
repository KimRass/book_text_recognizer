import torch
from pathlib import Path
from tqdm.auto import tqdm

from utils import get_args
from craft.detect_text import load_craft_checkpoint, load_link_refiner_checkpoint, get_score_maps
from image_utils import load_image, save_image


def main():
    args = get_args()

    cuda = torch.cuda.is_available()

    craft = load_craft_checkpoint(cuda=cuda)
    link_refiner = load_link_refiner_checkpoint(cuda=cuda)

    for img_path in tqdm(sorted((Path(args.save_dir)/"pages").glob("*.png"))):
        img = load_image(img_path)
        _, _, line_score_map = get_score_maps(img, craft=craft, link_refiner=link_refiner, cuda=cuda)
        save_image(line_score_map, Path(args.save_dir)/"line_score_maps"/img_path.name)


if __name__ == "__main__":
    main()
