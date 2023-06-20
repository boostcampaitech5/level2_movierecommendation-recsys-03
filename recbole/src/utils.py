import os
import yaml
from ray import tune
from typing import Dict, Any


def existence(config: dict) -> bool:
    path = os.path.join(config["data_path"], "train")
    data_file = os.path.join(path, "train.inter")
    return os.path.isfile(data_file)


def get_path(config: dict) -> str:
    path = os.path.join(config["data_path"], "train")
    data_file = os.path.join(path, "train.inter")
    return data_file


def read_yaml(file: str) -> Dict[str, Any]:
    with open(file) as f:
        result = yaml.load(f, Loader=yaml.FullLoader)
    return result


def read_config(config_file: str) -> Dict[str, Any]:
    config = read_yaml(config_file)
    for p in config.keys():
        config[p] = eval(config[p])
    return config
