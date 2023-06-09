import hydra
import torch
import os
from src.config import Config
from src.utils import get_timestamp, set_seed
from src.trainers import HoldoutTrainer, PretrainTrainer
from src.models import S3Rec, SASRec
from src.dataloaders import S3RecDataModule, SASRecDataModule


def get_trainer(config: Config):
    if config.trainer.is_pretrain:
        datamodule = S3RecDataModule(config)
        datamodule.prepare_data()
        model = S3Rec(config)

        pretrain_file = config.trainer.pretrain_version + "_" + config.path.pretrain_file + ".pt"
        pretrain_path = os.path.join(config.path.output_dir, pretrain_file)

        return PretrainTrainer(config, model, datamodule, "avg_sp_loss", "min", pretrain_path)
    if config.model.model_name == "SASRec":
        datamodule = SASRecDataModule(config)
        datamodule.prepare_data()
        model = SASRec(config, datamodule.valid_matrix, datamodule.test_matrix, datamodule.submission_matrix)

        if config.trainer.use_pretrain:
            pretrain_file = config.trainer.pretrain_version + "_" + config.path.pretrain_file + ".pt"
            pretrain_path = os.path.join(config.path.output_dir, pretrain_file)

            model.load_pretrained_module(pretrain_path)

        return HoldoutTrainer(config, model=model, data_module=datamodule, metric="NDCG@10", mode="max")


def main(config: Config = None) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    set_seed(config.seed)

    config.timestamp = get_timestamp()
    config.cuda_condition = torch.cuda.is_available() and not config.no_cuda

    trainer = get_trainer(config)
    trainer.train()

    if not config.trainer.is_pretrain:
        trainer.test()
        trainer.predict()


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main_hydra(config: Config) -> None:
    main(config)


if __name__ == "__main__":
    main_hydra()
