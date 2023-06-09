from src.config import Config
from src.utils import generate_submission_file
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import torch
import lightning as L


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
