from src.config import Config
from src.utils import generate_submission_file
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import torch
import lightning as L


class HoldoutTrainer:
    def __init__(
        self, config: Config, model: L.LightningModule, data_module: L.LightningDataModule, metric: str, checkpoint_file: str, mode="max"
    ) -> None:
        self.config = config
        self.model = model
        self.data_module = data_module

        self.device = torch.device("cuda" if config.cuda_condition else "cpu")

        early_stop = EarlyStopping(monitor=metric, patience=10, verbose=True, mode=mode)
        checkpoint = ModelCheckpoint(monitor=metric, mode=mode, dirpath=config.path.output_dir, filename=checkpoint_file)

        if config.cuda_condition:
            self.trainer = L.Trainer(max_epochs=self.config.trainer.epoch, callbacks=[early_stop, checkpoint], accelerator="cuda")
        else:
            self.trainer = L.Trainer(max_epochs=self.config.trainer.epoch, callbacks=[early_stop, checkpoint], accelerator="cpu")

    def train(self):
        print("start training")
        self.trainer.fit(self.model, datamodule=self.data_module)

    def predict(self):
        # inference
        preds = self.trainer.predict(self.model, datamodule=self.data_module)
        generate_submission_file(self.config, preds)

    def test(self):
        self.trainer.test(self.model, datamodule=self.data_module)
