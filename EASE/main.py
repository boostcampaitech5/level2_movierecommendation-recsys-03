import os
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from data import load_data, preprocess
from model import EASE


def main(config: DictConfig = None) -> None:
    print(">>> load and preprocess data ... ")
    data = load_data(config)
    data = preprocess(data, config.rel_colname, config.rel_val)

    model = EASE(config)
    model.fit(data)
    submit = model.predict(data, data["user"].unique(), data["item"].unique(), config.k)

    try:
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
    except OSError:
        print("Error: Creating directory. " + config.save_path)

    file_name = "ease_" + str(config.lambda_) + "_" + str(config.k) + ".csv"
    submit.to_csv(os.path.join(config.save_path, file_name), index=False)

    print(f">>> {file_name} is successfully saved !!! ")


@hydra.main(version_base="1.2", config_path="configs", config_name="default.yaml")
def main_hydra(config: DictConfig = None) -> None:
    main(config)


if __name__ == "__main__":
    main_hydra()
