from src.config import Config
from src.utils import generate_submission_file
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from src.dataloaders import KFoldDataModule
import torch
import lightning as L
from sklearn.model_selection import KFold, TimeSeriesSplit
import numpy as np


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
        generate_submission_file(self.config, preds)

    def test(self):
        self.trainer.test(self.model, datamodule=self.data_module)


class PretrainTrainer(HoldoutTrainer):
    def __init__(
        self, config: Config, model: L.LightningModule, data_module: L.LightningDataModule, metric: str, mode: str, pretrain_path: str
    ) -> None:
        super().__init__(config, model, data_module, metric, mode)
        self.pretrain_path = pretrain_path

    def train(self):
        super().train()

        self.save_best_pretrained_module()

    def save_best_pretrained_module(self):
        # load and
        self.model.load_from_checkpoint(self.checkpoint.best_model_path, config=self.config)
        self.model.save_pretrained_module(self.pretrain_path)


class KFoldTrainer(HoldoutTrainer):
    def __init__(self, config: Config, model: L.LightningModule, data_module: L.LightningDataModule, metric: str, mode: str) -> None:
        super().__init__(config, model, data_module, metric, mode)
        self.cv_score = 0
        self.model = model
        self.sub_result_csv_list = []
        self.val_result_csv_list = []
        self.data_module = data_module
        self.data_module.prepare_data()

    def split_data(self, data: str, idx: int) -> dict:
        if idx == 0:
            new_data = {
                "input_ids": [user[:idx] for user in data["input_ids"]],
                "target_pos": [user[:idx] for user in data["target_pos"]],
                "answers": data["answers"],
            }
        else:
            new_data = {
                "input_ids": [user[:idx] for user in data["input_ids"]],
                "target_pos": [user[:idx] for user in data["target_pos"]],
                "answers": [[user[idx]] for user in data["input_ids"]],
            }

        return new_data

    def train(self):
        for fold, idx in enumerate(range(1, self.config.trainer.k + 1)):
            train_idx = -1 * idx
            valid_idx = idx - 1
            print(f"------------- Fold {fold}  :  train {train_idx}, val {valid_idx} -------------")
            # set data for training and validation in fold

            self.fold_model = self.model
            self.fold_trainer = L.Trainer(max_epochs=self.config.trainer.epoch, callbacks=[self.early_stop, self.checkpoint], accelerator="cuda")

            self.fold_dm = KFoldDataModule(self.config, self.data_module.user_seq)

            self.fold_dm.setup()

            self.fold_dm.valid_data = self.split_data(self.fold_dm.train_data, valid_idx)
            self.fold_dm.train_data = self.split_data(self.fold_dm.train_data, train_idx)
            self.fold_dm.submission_data = self.fold_dm.submission_data

            # train n validation
            self.fold_trainer.fit(self.fold_model, datamodule=self.fold_dm)

            print(
                "check tr_result, val_result: ",
                len(self.fold_model.tr_result),
                len(self.fold_model.val_result),
            )
            tr_avg_loss = torch.stack([x["rec_avg_loss"] for x in self.fold_model.tr_result]).mean()
            tr_cur_loss = torch.stack([x["rec_cur_loss"] for x in self.fold_model.tr_result]).mean()

            val_recall = self.fold_model.val_result.mean()

            print(f">>> >>> tr_avg_loss: {tr_avg_loss},tr_cur_loss: {tr_cur_loss}, val_recall@10: {val_recall}")
            self.cv_score += val_recall / self.config.trainer.k
            # self.cv_predict(fold)

        print(f"-----------------cv_recall@10_score: {self.cv_score}-----------------")

    def cv_predict(self, fold: int):
        sub_predictions = self.fold_trainer.predict(self.fold_model, datamodule=self.fold_dm)
        # generate_submission_file(self.config, sub_predictions, fold)
