import argparse


def get_arguments():
    parser = argparse.ArgumentParser("OCR")

    parser.add_argument("--pdf")
    parser.add_argument("--cuda", default=False, action="store_true")

    args = parser.parse_args()
    return args
