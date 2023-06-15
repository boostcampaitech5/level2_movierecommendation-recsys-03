import hydra
import wandb
from src.data.datamodule import DataModule
from src.model.multivae import MultiVAE
from src.model.recommender import Recommender
from src.trainer import Trainer
from src.utils import (
    set_seed,
    get_timestamp,
    init_wandb,
    generate_submission_file,
)


def main(config) -> None:
    set_seed(config.seed)
    config.timestamp = get_timestamp()
    config.wandb.name = f"work-{config.timestamp}"
    init_wandb(config)

    model = MultiVAE(config)
    datamodule = DataModule(config)
    recommender = Recommender(config, model)
    trainer = Trainer(config, recommender, datamodule)

    trainer.train()
    trainer.test()
    pred = trainer.predict()

    generate_submission_file(config, pred, datamodule.indexer["idx2item"])

    wandb.finish()


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main_hydra(config) -> None:
    main(config)


if __name__ == "__main__":
    main_hydra()
