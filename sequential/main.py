import hydra
import torch
import wandb
import os
from src.config import Config
from src.utils import get_timestamp, set_seed, init_wandb, log_parameters, generate_submission_file
from src.trainers import HoldoutTrainer, PretrainTrainer, KFoldTrainer
from src.models import S3Rec, SASRec
from src.dataloaders import S3RecDataModule, SASRecDataModule, SASRecKFoldDataModuleContainer
import warnings

warnings.filterwarnings("ignore")


def get_trainer(config: Config):
    if config.trainer.is_pretrain:
        datamodule = S3RecDataModule(config)
        datamodule.prepare_data()

        model = S3Rec(config, datamodule.name2attr_size)

        pretrain_file = config.trainer.pretrain_version + "_" + config.path.pretrain_file + ".pt"
        pretrain_path = os.path.join(config.path.output_dir, pretrain_file)

        return PretrainTrainer(config, model, datamodule, "avg_joint_loss", "min", pretrain_path)
    if config.model.model_name == "SASRec":
        datamodule = SASRecDataModule(config)
        datamodule.prepare_data()
        model = SASRec(config, datamodule.valid_matrix, datamodule.test_matrix, datamodule.submission_matrix)

        if config.trainer.use_pretrain:
            pretrain_file = config.trainer.pretrain_version + "_" + config.path.pretrain_file + ".pt"
            pretrain_path = os.path.join(config.path.output_dir, pretrain_file)

            model.load_pretrained_module(pretrain_path)

        if config.trainer.cv:
            kfold_data_module_container = SASRecKFoldDataModuleContainer(config)

            return KFoldTrainer(config, model=model, kfold_data_module_container=kfold_data_module_container, metric="Recall@10", mode="max")
        else:
            return HoldoutTrainer(config, model=model, data_module=datamodule, metric="Recall@10", mode="max")


def main(config: Config = None) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    set_seed(config.seed)

    print(f"----------------- Setting -----------------")
    config.timestamp = get_timestamp()
    config.wandb.name = "work-" + config.timestamp
    config.cuda_condition = torch.cuda.is_available() and not config.no_cuda

    init_wandb(config.trainer.is_pretrain, config)
    log_parameters(config.trainer.is_pretrain, config)

    trainer = get_trainer(config)
    trainer.train()

    if not config.trainer.is_pretrain:
        trainer.test()
        preds = trainer.predict()

        generate_submission_file(config, preds)

    wandb.finish()


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main_hydra(config: Config) -> None:
    main(config)


if __name__ == "__main__":
    main_hydra()
