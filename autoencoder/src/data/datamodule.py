from typing import Optional
import os
import pandas as pd
import lightning as L
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from src.data.dataset import TrainDataset, EvalDataset
from src.data.utils import (
    get_user_seqs,
    train_val_test_split,
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
        self.user_seq: Optional[list] = None
        self.num_items: Optional[int] = None
        self.raw_matrix: Optional[csr_matrix] = None
        self.idx2item: Optional[dict] = None

        self.train_dataset: Optional[TrainDataset] = None
        self.val_dataset: Optional[EvalDataset] = None
        self.test_dataset: Optional[EvalDataset] = None
        self.predict_dataset: Optional[TrainDataset] = None

    def prepare_data(self):
        self.raw_data = pd.read_csv(os.path.join(self.train_dir, self.train_file))
        self.user_seq, self.num_items, self.raw_matrix, self.idx2item = get_user_seqs(self.raw_data)

    def setup(self, stage: Optional[str] = None):
        train_matrix, val_matrix, test_matrix, train_seq, val_seq, test_seq = train_val_test_split(self.raw_matrix, self.user_seq)

        self.train_dataset = TrainDataset(train_matrix)
        self.val_dataset = EvalDataset(val_matrix, val_seq)
        self.test_dataset = EvalDataset(test_matrix, test_seq)
        self.predict_dataset = TrainDataset(self.raw_matrix)

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader

    def predict_dataloader(self) -> DataLoader:
        loader = DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader
