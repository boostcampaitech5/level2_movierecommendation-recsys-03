import os
import torch
import wandb
import dotenv
import random
import numpy as np
import pandas as pd
from datetime import datetime


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_timestamp(date_format: str = "%d_%H%M%S") -> str:
    timestamp = datetime.now()
    return timestamp.strftime(date_format)


def login_wandb(config) -> None:
    dotenv.load_dotenv()
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)


def init_wandb(config, k=None, group=None) -> None:
    wandb.init(
        project=config.wandb.project + "AutoEncoder",
        entity=config.wandb.entity,
        name=config.wandb.name + f"_fold_{k}" if k is not None else config.wandb.name,
        tags=[config.model.model_name],
        group=group,
    )


def generate_submission_file(config, preds: np.ndarray, idx2item: dict):
    sample_sub_path = os.path.join(config.path.eval_dir, config.path.eval_file)

    _check_dir(config.path.output_dir)

    sub_path = os.path.join(config.path.output_dir, f"{config.timestamp}_{config.model.model_name}_submit.csv")

    sub_df = pd.read_csv(sample_sub_path)

    items = np.vstack(preds)
    items = items.reshape(-1)

    sub_df.loc[:, "item"] = items
    sub_df["item"] = sub_df["item"].apply(lambda x: idx2item[x])
    sub_df.to_csv(sub_path, index=False)


def _check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def predict_topk(x_hat: np.ndarray, k: int) -> np.ndarray:
    ind = np.argpartition(x_hat, -k)[:, -k:]
    arr_ind = x_hat[np.arange(len(x_hat))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(x_hat)), ::-1]
    pred_topk = ind[np.arange(len(x_hat))[:, None], arr_ind_argsort]
    return pred_topk
