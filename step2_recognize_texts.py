import pandas as pd
import numpy as np
import json
import uuid
import json
import time
import cv2
import requests
from pathlib import Path
import json
from tqdm.auto import tqdm

from book_text_recognizer.utils import get_arguments
from process_images import _dilate_image


def get_clova_ocr_api_response_body(path):
    api_url = open("api_url_txt", mode="r").read()

    headers = {"X-OCR-SECRET": open("./secret_key.txt", mode="r").read()}

    request_json = {
        "images": [
            {
                "format": "png",
                "name": "ocr"
            }
        ],
        "requestId": str(uuid.uuid4()),
        "version": "V2",
        "timestamp": int(round(time.time() * 1000))
    }
    payload = {"message": json.dumps(request_json).encode("UTF-8")}

    files = [("file", open(path, mode="rb"))]
    
    resp = requests.request(
        method="POST", url=api_url, headers=headers, data=payload, files=files
    )
    resp_body = resp.json()
    return resp_body


def get_line_segmentation_map(map_line):
    _, mask_line = cv2.threshold(src=map_line, thresh=70, maxval=255, type=cv2.THRESH_BINARY)
    _, segmap_line = cv2.connectedComponents(image=mask_line, connectivity=4)
    return segmap_line


def get_block_segmentation_map(map_line):
    _, mask_line = cv2.threshold(src=map_line, thresh=60, maxval=255, type=cv2.THRESH_BINARY)

    mask_block = _dilate_image(img=mask_line, kernel_shape=(40, 40))
    _, segmap_block = cv2.connectedComponents(image=mask_block, connectivity=4)
    return segmap_block


def get_center(ls):
    x = int((ls[0]["x"] + ls[2]["x"]) // 2)
    y = int((ls[0]["y"] + ls[2]["y"]) // 2)
    return (x, y)


def get_df(resp_body, segmap_block, segmap_line):
    df_texts = pd.DataFrame(
        [i for i in resp_body["images"][0]["fields"]]
    )
    df_texts.drop(["boundingPoly"], axis=1, inplace=True)

    df_centers = pd.DataFrame(
        [get_center(i["boundingPoly"]["vertices"]) for i in resp_body["images"][0]["fields"]],
        columns=["x", "y"]
    ).astype("int")

    df_concated = pd.concat([df_texts, df_centers], axis=1)

    df_concated["block"] = df_concated.apply(
        lambda i: segmap_block[i["y"], i["x"]], axis=1
    )
    df_concated["line"] = df_concated.apply(
        lambda i: segmap_line[i["y"], i["x"]], axis=1
    )
    df_concated.sort_values(["block", "line", "x"], inplace=True)
    
    df_gby = df_concated.groupby(["block", "line"])["inferText"].apply(list).apply(lambda x: " ".join(x))
    df_gby = df_gby.reset_index()
    return df_gby


def main():
    args = get_arguments()

    for path_img in tqdm(
        sorted(
            list((Path(args.pdf).parent/"image").glob("*.png"))
        )
    ):
        map_line = cv2.imread(
            str(Path(args.pdf).parent/"line_map"/f"{path_img.stem}.png"),
            flags=cv2.IMREAD_GRAYSCALE
        )

        segmap_block = get_block_segmentation_map(map_line)
        segmap_line = get_line_segmentation_map(map_line)

        resp_body = get_clova_ocr_api_response_body(path_img)
        df = get_df(resp_body=resp_body, segmap_block=segmap_block, segmap_line=segmap_line)

        save_dir = Path(args.pdf).parent/"text_recognition"
        save_dir.mkdir(parents=True, exist_ok=True)
        df.to_excel(save_dir/f"{path_img.stem}.xlsx", index=False)


if __name__ == "__main__":
    main()
