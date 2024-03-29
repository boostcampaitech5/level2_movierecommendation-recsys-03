from typing import Optional
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


class Trainer:
    def __init__(self, config, model: L.LightningModule, datamodule: L.LightningDataModule, k: Optional[int] = None):
        self.config = config
        self.accelerator = config.trainer.accelerator
        self.devices = config.trainer.devices
        self.max_epochs = config.trainer.max_epochs
        self.output_dir = config.path.output_dir
        self.name = config.wandb.name + f"_fold_{k}" if k is not None else config.wandb.name
        self.project = config.wandb.project

        self.model = model
        self.datamodule = datamodule
        self.trainer = self._build_trainer()

    def train(self) -> None:
        self.trainer.fit(self.model, self.datamodule)

    def test(self) -> None:
        self.trainer.test(self.model, self.datamodule, ckpt_path="best")

    def predict(self) -> np.ndarray:
        pred = self.trainer.predict(self.model, self.datamodule, ckpt_path="best")
        return pred

    def _build_trainer(self) -> L.Trainer:
        trainer = L.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            logger=self._build_wandb_logger(),
            callbacks=[self._build_ckpt_callback()],
            max_epochs=self.max_epochs,
        )
        return trainer

    def _build_ckpt_callback(self) -> ModelCheckpoint:
        ckpt_callback = ModelCheckpoint(
            dirpath=self.output_dir,
            filename=self.name,
            monitor="val_recall@10",
            save_last=False,
            save_top_k=1,
            mode="max",
            verbose=True,
        )
        return ckpt_callback

    def _build_wandb_logger(self) -> WandbLogger:
        wandb_logger = WandbLogger(
            name=self.name,
            project=self.project,
        )
        return wandb_logger
