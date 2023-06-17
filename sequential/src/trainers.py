from src.config import Config
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from src.dataloaders import KFoldDataModuleContainer
from src.models import S3Rec
import torch
import lightning as L
import copy
import numpy as np
import wandb


class HoldoutTrainer:
    def __init__(self, config: Config, model: L.LightningModule, data_module: L.LightningDataModule, metric: str, mode="max") -> None:
        self.config = config
        self.model = model
        self.data_module = data_module

        self.device = torch.device("cuda" if config.cuda_condition else "cpu")

        self.early_stop = EarlyStopping(monitor=metric, patience=10, verbose=True, mode=mode)

        if config.trainer.is_pretrain:
            checkpoint_file = f"{config.timestamp}_{config.model.model_name}_pretrain_{{{metric}:.2f}}"
        else:
            checkpoint_file = f"{config.timestamp}_{config.model.model_name}_{{{metric}:.2f}}"

        self.checkpoint = ModelCheckpoint(monitor=metric, mode=mode, dirpath=config.path.output_dir, filename=checkpoint_file)

        if config.cuda_condition:
            self.trainer = L.Trainer(max_epochs=self.config.trainer.epoch, callbacks=[self.early_stop, self.checkpoint], accelerator="cuda")
        else:
            self.trainer = L.Trainer(max_epochs=self.config.trainer.epoch, callbacks=[self.early_stop, self.checkpoint], accelerator="cpu")

    def train(self):
        print("start training")
        self.trainer.fit(self.model, datamodule=self.data_module)

    def predict(self):
        # inference
        preds = self.trainer.predict(self.model, datamodule=self.data_module)

        return torch.concatenate(preds)

    def test(self):
        self.trainer.test(self.model, datamodule=self.data_module)


class PretrainTrainer(HoldoutTrainer):
    def __init__(self, config: Config, model: S3Rec, data_module: L.LightningDataModule, metric: str, mode: str, pretrain_path: str) -> None:
        super().__init__(config, model, data_module, metric, mode)
        self.pretrain_path = pretrain_path

    def train(self):
        super().train()

        self.save_best_pretrained_module()

    def save_best_pretrained_module(self):
        # load and
        self.model.load_from_checkpoint(self.checkpoint.best_model_path, config=self.config, name2attr_size=self.model.name2attr_size)
        self.model.save_pretrained_module(self.pretrain_path)


class KFoldTrainer:
    def __init__(
        self, config: Config, model: L.LightningModule, kfold_data_module_container: KFoldDataModuleContainer, metric: str, mode: str
    ) -> None:
        self.config = config
        self.n_fold = config.trainer.k

        self.kfold_data_module_container = kfold_data_module_container

        self.fold_trainers: list[HoldoutTrainer] = self.__fold_trainers(self.n_fold, config, model, kfold_data_module_container, metric, mode)
        self.sub_result_csv_list = []
        self.val_result_csv_list = []

    def __fold_trainers(
        self, n_fold: int, config: Config, model: L.LightningModule, kfold_data_module_container: KFoldDataModuleContainer, metric: str, mode: str
    ) -> list[HoldoutTrainer]:
        fold_trainers = []

        for fold in range(n_fold):
            fold_model = copy.deepcopy(model)
            kfold_data_module = kfold_data_module_container.kfold_data_module(fold)

            trainer = HoldoutTrainer(config, fold_model, kfold_data_module, metric, mode)
            fold_trainers.append(trainer)

        return fold_trainers

    def train(self):
        cv_score = 0.0

        for fold, fold_trainer in enumerate(self.fold_trainers):
            print(f"------------- Train Fold {fold} -------------")

            fold_trainer.train()
            fold_model = fold_trainer.model

            print(
                "check tr_result, val_result: ",
                len(fold_model.tr_result),
                len(fold_model.val_result),
            )
            tr_avg_loss = torch.stack([x["rec_avg_loss"] for x in fold_model.tr_result]).mean()
            tr_cur_loss = torch.stack([x["rec_cur_loss"] for x in fold_model.tr_result]).mean()

            val_recall = fold_model.val_result.mean()

            print(f">>> tr_avg_loss: {tr_avg_loss},tr_cur_loss: {tr_cur_loss}, val_recall@10: {val_recall}")
            cv_score += val_recall

        cv_score /= self.n_fold
        print(f"-----------------cv_recall@10_score: {cv_score}-----------------")
        wandb.log({"cv_score": cv_score})

        return cv_score

    def predict(self):
        fold = 0
        while self.fold_trainers:
            fold_trainer = self.fold_trainers.pop(0)

            print(f"-------------  Predict Fold {fold} -------------")
            if fold == 0:
                output = fold_trainer.predict()
            else:
                output = output + fold_trainer.predict()

            fold += 1

        rating_pred = np.array(output / self.n_fold)

        ind = np.argpartition(rating_pred, -10)[:, -10:]
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
        oof_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        return oof_pred_list

    def test(self):
        pass
