from typing import Optional
import os
import pandas as pd
import lightning as L
from torch.utils.data import DataLoader
from src.data.dataset import TrainDataset, EvalDataset
from src.data.utils import (
    generate_user_item_indexer,
    split_user_ids,
    numerize,
)


class DataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dir = config.path.train_dir
        self.train_file = config.path.train_file
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers

        self.raw_data: Optional[pd.DataFrame] = None

        self.indexer = None

        self.train_dataset: Optional[TrainDataset] = None
        self.valid_dataset: Optional[EvalDataset] = None
        self.test_dataset: Optional[EvalDataset] = None
        self.predict_dataset: Optional[TrainDataset] = None

    def prepare_data(self):
        self.raw_data = pd.read_csv(os.path.join(self.train_dir, self.train_file))
        self.indexer: dict = generate_user_item_indexer(self.raw_data)

    def setup(self, stage: Optional[str] = None):
        train_user_ids, valid_user_ids, test_user_ids = split_user_ids(self.raw_data)

        train_data = self.raw_data.loc[self.raw_data["user"].isin(train_user_ids)]
        valid_data = self.raw_data.loc[self.raw_data["user"].isin(valid_user_ids)]
        test_data = self.raw_data.loc[self.raw_data["user"].isin(test_user_ids)]

        train_data_numerized = numerize(train_data, self.indexer["user2idx"], self.indexer["item2idx"])
        valid_data_numerized = numerize(valid_data, self.indexer["user2idx"], self.indexer["item2idx"])
        test_data_numerized = numerize(test_data, self.indexer["user2idx"], self.indexer["item2idx"])
        raw_data_numerized = numerize(self.raw_data, self.indexer["user2idx"], self.indexer["item2idx"])

        self.train_dataset = TrainDataset(train_data_numerized)
        self.valid_dataset = EvalDataset(valid_data_numerized)
        self.test_dataset = EvalDataset(test_data_numerized)
        self.predict_dataset = TrainDataset(raw_data_numerized)

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader

    def predict_dataloader(self) -> DataLoader:
        loader = DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader
