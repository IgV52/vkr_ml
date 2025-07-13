from re import findall

import pandas as pd

from cl_okpd2.constants import DEFAULT_VALUE, MAP_GROUP_CODE


def valid_okpd(value: str) -> bool:
    return (
        value != DEFAULT_VALUE
        and value.replace(".", "").isdigit()
        and bool(findall(r"^\d{1,2}\.\d{1,2}$", value))
    )


def valid_text(value: str) -> bool:
    return value != DEFAULT_VALUE and not value.replace(".", "").isdigit() and len(value) >= 3


def extract_draft_data(path: str, new_path: str, chunk_size: int) -> None:
    for chunk in pd.read_csv(path, chunksize=chunk_size, names=list(range(300))):
        chunk = chunk.dropna(axis=1, how="all")
        chunk[25] = chunk[25].astype(str)
        chunk[26] = chunk[26].astype(str)
        chunk = chunk[chunk[25].apply(valid_text)]
        chunk = chunk[chunk[26].apply(valid_okpd)]

        for k, v in MAP_GROUP_CODE.items():
            for i in v:
                mask = chunk[26].str.split(".").str[0] == str(i)
                chunk.loc[mask, 26] = k
        chunk = chunk[[25, 26]]
        chunk = chunk.drop_duplicates()

        if not chunk.empty:
            chunk.to_csv(
                new_path,
                mode="a",
                header=False,
                index=False,
                encoding="utf-8",
            )


def extract_uniq_data(path: str, new_path: str) -> None:
    header = ["text", "code"]
    draft_data = pd.read_csv(path, names=header)
    draft_data = draft_data.drop_duplicates()
    counts = draft_data["code"].value_counts()
    draft_data = draft_data[draft_data["code"].isin(counts[counts >= 100].index)]

    if not draft_data.empty:
        draft_data.to_csv(
            new_path,
            mode="a",
            header=header,
            index=False,
            encoding="utf-8",
        )
