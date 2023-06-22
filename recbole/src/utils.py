import os
import yaml
import wandb
import dotenv
from ray import tune
from datetime import datetime
from typing import Dict, Any, Tuple
from ray.tune import ExperimentAnalysis


def existence(config: dict, data_type: str) -> bool:
    path = os.path.join(config["data_path"], "train")
    data_file = os.path.join(path, "train." + data_type)
    return os.path.isfile(data_file)


def get_path(config: dict, data_type: str) -> str:
    path = os.path.join(config["data_path"], "train")
    data_file = os.path.join(path, ("train." + data_type))
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


def init_wandb(config: dict, timestamp: str) -> None:
    dotenv.load_dotenv()
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)

    run = wandb.init(project="MovieRec_recbole", entity="recsys01", name=("work" + timestamp))
    run.tags = [config["model"]]


def log_result(best_valid_score: float, test_result: float) -> None:
    wandb.log({"best_valid_recall": best_valid_score, "test_recall": test_result})


def get_timestamp() -> str:
    timestamp = datetime.now()
    timestamp = timestamp.strftime("%d_%H%M%S")
    return timestamp


def get_result(tuner: ExperimentAnalysis) -> Tuple[dict, str]:
    best_trial = tuner.get_best_trial("recall", "max", "last")
    best_config = best_trial.config

    result = best_trial.last_result
    test_result = result["test_result"]["recall@10"]
    best_valid_score = result["best_valid_result/recall@10"]["recall@10"]
    log_result(best_valid_score, test_result)

    best_dir = os.path.join(tuner.get_best_logdir("recall", "max"), "saved")
    model_file = os.listdir(best_dir)[0]
    best_model = os.path.join(best_dir, model_file)
    return best_config, best_model


def set_path(config: dict, base_path: str) -> None:
    base_path = base_path.removesuffix("recbole")
    config["data_path"] = os.path.join(base_path, config["data_path"])
    config["output_path"] = os.path.join(base_path, config["output_path"])
    config["submission_path"] = os.path.join(base_path, config["submission_path"])
