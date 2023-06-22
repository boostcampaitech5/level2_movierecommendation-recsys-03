import hydra
import wandb
import numpy as np
from src.data.datamodule import DataModule
from src.model.multivae import MultiVAE
from src.model.recommender import Recommender
from src.trainer import Trainer
from src.utils import (
    set_seed,
    get_timestamp,
    login_wandb,
    init_wandb,
    generate_submission_file,
    predict_topk,
)


def main(config) -> None:
    set_seed(config.seed)
    config.timestamp = get_timestamp()
    config.wandb.name = f"work-{config.timestamp}"
    login_wandb(config)

    if config.trainer.strategy == "holdout":
        init_wandb(config)
        model = MultiVAE(config)
        datamodule = DataModule(config)
        recommender = Recommender(config, model)
        trainer = Trainer(config, recommender, datamodule)

        trainer.train()
        trainer.test()
        pred = trainer.predict()

        wandb.finish()

        topk_pred = [predict_topk(pred[i], k=10) for i in range(len(pred))]
        generate_submission_file(config, topk_pred, datamodule.idx2item)

    elif config.trainer.strategy == "kfold":
        idx2item = None
        results = []

        for k in range(5):
            init_wandb(config, k=k, group=config.wandb.name)

            model = MultiVAE(config)
            datamodule = DataModule(config, k)
            recommender = Recommender(config, model)
            trainer = Trainer(config, recommender, datamodule, k)

            trainer.train()
            trainer.test()
            pred = trainer.predict()
            results.append(np.concatenate(pred))
            idx2item = datamodule.idx2item

            wandb.finish()

        mean_pred = np.array(results).mean(axis=0)
        topk_pred = predict_topk(mean_pred, 10)
        generate_submission_file(config, topk_pred, idx2item)

    else:
        raise NotImplementedError


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main_hydra(config) -> None:
    main(config)


if __name__ == "__main__":
    main_hydra()
