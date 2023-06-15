import os
import wandb
import hydra
import dotenv
from omegaconf import DictConfig

from tabular.data import TabularDataModule, TabularDataset
from tabular.trainer import cv_trainer, trainer
from tabular.utils import set_seeds, get_timestamp


def main(config: DictConfig = None) -> None:
    # setting
    print(f"----------------- Setting -----------------")
    config.timestamp = get_timestamp()
    config.wandb.name = f"work-{get_timestamp()}"
    set_seeds(config.seed)

    # # wandb init
    # dotenv.load_dotenv()
    # WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    # wandb.login(key=WANDB_API_KEY)

    # run = wandb.init(
    #     project=config.wandb.project,
    #     entity=config.wandb.entity,
    #     name=config.wandb.name,
    # )

    if config.trainer.cv_strategy == "kfold":
        cv_trainer(config)
    elif config.trainer.cv_strategy == "holdout":
        trainer(config)
    else:
        raise Exception("Invalid cv strategy is entered")

    # wandb.finish()


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main_hydra(config: DictConfig = None) -> None:
    main(config)


if __name__ == "__main__":
    main_hydra()
