import os
import json
import pandas as pd


DATA_VERSION = "V1"
GLOBAL_BASE_PATH = "../data/train"
LOCAL_BASE_PATH = "./data/train"


def tsv2json(file_name):
    file_path = os.path.join(GLOBAL_BASE_PATH, file_name)
    df = pd.read_csv(file_path, delimiter="\t")

    item2idx_path = os.path.join(LOCAL_BASE_PATH, "item2idx.json")

    with open(item2idx_path) as file:
        item2idx = json.load(file)

    df["item"] = df["item"].apply(lambda x: item2idx[str(x)])

    attr_col = df.columns.values[1]

    df[attr_col] = df[attr_col].astype("category")

    json_dict = {}
    for item, group in df.groupby("item"):
        attrs = group[attr_col].cat.codes.to_list()
        json_dict[item] = attrs

    json_path = os.path.join(LOCAL_BASE_PATH, f"{DATA_VERSION}_item2{file_name[:-4]}.json")
    with open(json_path, "w") as f:
        json.dump(json_dict, f)


if __name__ == "__main__":
    for file_name in os.listdir(GLOBAL_BASE_PATH):
        if not file_name.endswith(".tsv"):
            continue

        tsv2json(file_name)
