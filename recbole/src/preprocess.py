import os
import pandas as pd
from .utils import existence, get_path


def create_inter(data_file: str, data: pd.DataFrame) -> None:
    inter_table = list(data.values)
    print("--------------------Creating train.inter files--------------------")
    with open(data_file, "w") as f:
        f.write("user:token\titem:token\ttime:float\n")
        for row in inter_table:
            f.write("\t".join([str(x) for x in row]) + "\n")

    print("--------------------Creating dummy.inter files--------------------")
    dummy_file = data_file.replace("train", "dummy")
    dummy_table = inter_table[:1000]
    with open(dummy_file, "w") as f:
        f.write("user:token\titem:token\ttime:float\n")
        for row in dummy_table:
            f.write("\t".join([str(x) for x in row]) + "\n")

    print("--------------------Creating submission dummy files--------------------")
    users = data[:1000].user.unique()
    dummy_sub = pd.DataFrame({"user": [], "item": []})
    dummy_sub["user"] = users.repeat(10)
    dummy_sub["item"] = [0] * (10 * len(users))

    dummy_sub_path = os.path.join("../data/eval/dummy.csv")
    dummy_sub.to_csv(dummy_sub_path, index=False)


def check_and_create_data(config: dict) -> None:
    if not existence(config):
        data_file = get_path(config)
        data_path = config["data_path"]
        data_path = os.path.join(data_path, "train/train_ratings.csv")
        data = pd.read_csv(data_path)
        create_inter(data_file, data)
