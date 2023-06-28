import pandas as pd
from pathlib import Path
import re

from book_text_recognizer.utilities import get_arguments

pd.options.display.max_colwidth = 100


def get_page(text):
    return int(
        re.search(pattern=r"^\d{3}|\d{3}?", string=text).group()
    )


def exclude_block_with_one_row(df):
    block2cnts = df["block"].value_counts()
    block_nonzero = block2cnts[block2cnts != 1].index.tolist()
    df = df[df["block"].isin(block_nonzero)]
    return df


def concatenate_df(dir):
    max_block = 0
    ls_df = list()
    for path in sorted(list(Path(dir).glob("*.xlsx"))):
        if path.stem[: 2] != "~$":
            df = pd.read_excel(path)

            page = get_page(df.iloc[-1]["inferText"])
            df["page"] = page

            df = df.iloc[: len(df) - 1]

            df["block"] = df["block"] + max_block
            max_block = df["block"].max() + 1

            ls_df.append(df)
    df_concated = pd.concat(ls_df, axis=0, ignore_index=True)

    df_concated.rename({"inferText": "text"}, axis=1, inplace=True)
    df_concated = df_concated[["page", "block", "text"]]
    return df_concated


def merge_by_block(df):
    df = df.groupby(["page", "block"])["text"].apply(list).apply(lambda x: "".join(x))
    df = df.reset_index()
    return df


def exclude_titles(df):
    df_copied = df[~df["content"].str.startswith("□")]
    df_copied = df[~df["content"].str.contains(r"^\d\.")]
    df_copied = df_copied[~df_copied["content"].str.contains(r"^[가-나]\. ")]
    df_copied = df_copied[~df_copied["content"].str.contains(f"^\d\)")]
    return df_copied


def replace_dot_with_at_sign(texts):
    return re.sub(
        pattern=r"([a-zA-Z0-9])\.",
        repl=r"\1@",
        string=texts
    )


def split_into_sentences(text):
    pattern=r"""[ㄱ-ㄴㅏ-ㅣ가-힣a-zA-Z0-9,:@()'" ]*[ㄱ-ㄴㅏ-ㅣ가-힣]\.|[ㄱ-ㄴㅏ-ㅣ가-힣a-zA-Z0-9,:@()'" ]*[ㄱ-ㄴㅏ-ㅣ가-힣]\?|[ㄱ-ㄴㅏ-ㅣ가-힣a-zA-Z0-9,:@()'" ]*[ㄱ-ㄴㅏ-ㅣ가-힣]\!"""
    if re.search(pattern=pattern, string=text):
        return re.findall(pattern=pattern, string=text)
    else:
        return [text]


def replace_at_sign_with_dot(texts):
    return re.sub(
        pattern=r"([a-zA-Z0-9])@",
        repl=r"\1.",
        string=texts
    )


def get_speaker_and_content(text):
    match = re.search(
        pattern=r"""(^[가-힣 ]+:)([ㄱ-ㄴㅏ-ㅣ가-힣a-zA-Z0-9,:@()'".!? ]+)""",
        string=text
    )
    if match:
        speaker = match.group(1).replace(":", "").strip()
        speaker = {
            "내": "아내",
            "아 내": "아내",
            "검 사": "검사",
            "남 편": "남편",
            "원 고": "원고",
            "원 변": "원변",
            "피 고": "피고",
            "피 변": "피변",
            "증 인": "증인",
            "원 증 인": "원증인",
            "원 원 고": "원원고"
        }.get(speaker, speaker)

        return pd.Series([speaker, match.group(2)])
    else:
        return pd.Series([None, text])


def main():
    args = get_arguments()

    df = concatenate_df(Path(args.pdf).parent/"text_recognition")
    df = df[~df["text"].str.startswith("□")]

    df = merge_by_block(df)
    
    df["text"] = df["text"].apply(replace_dot_with_at_sign)
    df["text"] = df["text"].apply(lambda x: x.strip())

    df["spoken_or_written"] = df["text"].apply(
        lambda x: "spoken" if re.search(pattern=r"[가-힣]+: ", string=x) else "written"
    )

    ls_row = list()
    for page, _, text, spoken_or_written in df.values:
        ls_sentence = split_into_sentences(text)
        for sentence in ls_sentence:
            sentence = replace_at_sign_with_dot(sentence)

            ls_row.append(
                (page, sentence, spoken_or_written)
            )
    df = pd.DataFrame(ls_row, columns=["page", "text", "spoken_or_written"])
    
    df[["speaker", "content"]] = df["text"].apply(get_speaker_and_content)
    df["speaker"] = df["speaker"].fillna(method="ffill")
    df.loc[df["spoken_or_written"] == "written", "speaker"] = "-"
    
    df["content"] = df["content"].apply(lambda x: x.strip())
    
    df.drop(["text", "spoken_or_written"], axis=1, inplace=True)

    df.to_excel(Path(args.pdf).parent/"ocr_result.xlsx", index=False)
    
    
if __name__ == "__main__":
    main()
