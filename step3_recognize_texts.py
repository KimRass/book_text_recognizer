import pandas as pd
import json
import uuid
import json
import time
import cv2
import requests
from pathlib import Path
import json
from tqdm.auto import tqdm

from book_text_recognizer.utils import get_args
from process_images import load_image, _dilate_image


def get_clova_ocr_api_response_body(path):
    api_url = "https://90tduwg1y9.apigw.ntruss.com/custom/v1/19391/0bda7bcad5ab17846484e70d684f4a944832a94a043d05c34dfa7e37efe2345c/general"

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


def get_line_segmentation_map(line_score_map):
    _, line_mask = cv2.threshold(src=line_score_map, thresh=70, maxval=255, type=cv2.THRESH_BINARY)
    _, line_seg_map = cv2.connectedComponents(image=line_mask[..., 0], connectivity=4)
    return line_seg_map


def get_block_segmentation_map(line_score_map):
    _, line_mask = cv2.threshold(src=line_score_map, thresh=60, maxval=255, type=cv2.THRESH_BINARY)
    block_mask = _dilate_image(img=line_mask, kernel_size=40)
    _, block_seg_map = cv2.connectedComponents(image=block_mask[..., 0], connectivity=4)
    return block_seg_map


def get_center(ls):
    x = int((ls[0]["x"] + ls[2]["x"]) // 2)
    y = int((ls[0]["y"] + ls[2]["y"]) // 2)
    return (x, y)


def response_body_to_df(resp_body, block_seg_map, line_seg_map):
    df_texts = pd.DataFrame([i for i in resp_body["images"][0]["fields"]])
    df_texts.drop(["boundingPoly"], axis=1, inplace=True)

    df_centers = pd.DataFrame(
        [get_center(i["boundingPoly"]["vertices"]) for i in resp_body["images"][0]["fields"]],
        columns=["x", "y"]
    ).astype("int")

    df_concated = pd.concat([df_texts, df_centers], axis=1)
    df_concated["block"] = df_concated.apply(lambda x: block_seg_map[x["y"], x["x"]], axis=1)
    df_concated["line"] = df_concated.apply(lambda x: line_seg_map[x["y"], x["x"]], axis=1)
    df_concated.sort_values(["block", "line", "x"], inplace=True)
    
    df_gby = df_concated.groupby(["block", "line"])["inferText"].apply(list).apply(lambda x: " ".join(x))
    df_gby = df_gby.reset_index()
    return df_gby


def main():
    args = get_args()

    save_dir = "/Users/jongbeomkim/Downloads/document_text_recognition"
    for img_path in tqdm(sorted((Path(save_dir)/"pages").glob("*.png"))):
        line_score_map = load_image(Path(save_dir)/"line_score_maps"/img_path.name)

        line_seg_map = get_line_segmentation_map(line_score_map)
        block_seg_map = get_block_segmentation_map(line_score_map)

        resp_body = get_clova_ocr_api_response_body(img_path)
        df = response_body_to_df(resp_body=resp_body, block_seg_map=block_seg_map, line_seg_map=line_seg_map)

        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        df.to_excel(Path(args.save_dir)/f"{img_path.stem}.xlsx", index=False)


if __name__ == "__main__":
    main()
