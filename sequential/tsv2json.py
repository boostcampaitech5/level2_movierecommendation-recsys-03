import os
import json
import pandas as pd


DATA_VERSION = "V1"
BASE_PATH = "../data/train"


def tsv2json(file_name):
    file_path = os.path.join(BASE_PATH, file_name)
    df = pd.read_csv(file_path, delimiter="\t")
    attr_col = df.columns.values[1]

    df[attr_col] = df[attr_col].astype("category")

    json_dict = {}
    for item, group in df.groupby("item"):
        attrs = group[attr_col].cat.codes.to_list()
        json_dict[item] = attrs

    json_path = os.path.join(BASE_PATH, f"{DATA_VERSION}_item2{file_name[:-4]}.json")
    with open(json_path, "w") as f:
        json.dump(json_dict, f)


if __name__ == "__main__":
    for file_name in os.listdir(BASE_PATH):
        if not file_name.endswith(".tsv"):
            continue

        tsv2json(file_name)
