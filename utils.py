import argparse


def get_args():
    parser = argparse.ArgumentParser("Book Text Recognizer")

    parser.add_argument("--pdf_path")
    parser.add_argument("--save_dir")

    args = parser.parse_args()
    return args
