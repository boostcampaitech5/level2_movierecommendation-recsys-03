import torch
import lightning as pl
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from src.config import Config
from utils import (
    neg_sample,
    get_user_seqs,
)


class SASRecDataset(Dataset):
    def __init__(self, config: Config, data: dict, user_seq: list, test_neg_items=None):
        self.config = config
        self.user_seq = user_seq
        self.data = data
        self.test_neg_items = test_neg_items
        self.max_len = self.config.data.max_seq_length

    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]
        input_ids = self.data["input_ids"][index]
        target_pos = self.data["target_pos"][index]
        answer = self.data["answer"][index]

        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.config.data.item_size))

        # padding
        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        # slicing
        input_ids = input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:  # remove?
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing [1]
                torch.tensor(input_ids, dtype=torch.long),  # item_id [seqlen]
                torch.tensor(target_pos, dtype=torch.long),  # target_pos [seqlen]
                torch.tensor(target_neg, dtype=torch.long),  # target_neg [seqlen]
                torch.tensor(answer, dtype=torch.long),  # answer [1]
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing [1]
                torch.tensor(input_ids, dtype=torch.long),  # item_id [seqlen]
                torch.tensor(target_pos, dtype=torch.long),  # target_pos [seqlen]
                torch.tensor(target_neg, dtype=torch.long),  # target_neg [seqlen]
                torch.tensor(answer, dtype=torch.long),  # answer [1]
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)


class SASRecDataModule(pl.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.data_path = self.config.path.data_dir + self.config.path.data_file
        self.batch_size = self.config.data.batch_size
        self.user_seq = None
        self.max_item = None

        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.submission_data = None

        self.valid_matrix = None
        self.test_matrix = None
        self.submission_matrix = None
        self.n_user = None  # TODO

    # load and feature_engineering dataset
    def prepare_data(self):
        self.user_seq, self.max_item, self.valid_matrix, self.test_matrix, self.submission_matrix = get_user_seqs(self.data_path)
        self.config.data.item_size = self.max_item + 2
        self.config.data.mask_id = self.max_item + 1

    # preprocess and set dataset on train/test case
    def setup(self, stage=None):
        self.n_user = len(self.user_seq)

        if stage == "fit" or stage is None:
            # train
            self.train_data = {"input_ids": self.user_seq[:][:-3], "target_pos": self.user_seq[:][1:-2], "answer": [[0] * self.n_user]}

            # valid
            self.valid_data = {"input_ids": self.user_seq[:][:-2], "target_pos": self.user_seq[:][1:-1], "answer": [self.user_seq[:][-2]]}

        elif stage == "test" or stage is None:
            # test
            self.test_data = {"input_ids": self.user_seq[:][:-1], "target_pos": self.user_seq[:][1:], "answer": [self.user_seq[:][-1]]}

        elif stage == "predict" or stage is None:
            self.submission_data = {"input_ids": self.user_seq[:][:], "target_pos": self.user_seq[:][:], "answer": [[] * self.n_user]}

    def train_dataloader(self) -> DataLoader:
        train_dataset = SASRecDataset(config=self.config, data=self.train_data, user_seq=self.user_seq)
        train_sampler = RandomSampler(train_dataset)
        return DataLoader(train_dataset, sampler=train_sampler, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        valid_dataset = SASRecDataset(config=self.config, data=self.valid_data, user_seq=self.user_seq)
        valid_sampler = SequentialSampler(valid_dataset)
        return DataLoader(valid_dataset, sampler=valid_sampler, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        test_dataset = SASRecDataset(config=self.config, data=self.test_data, user_seq=self.user_seq)
        test_sampler = SequentialSampler(test_dataset)
        return DataLoader(test_dataset, sampler=test_sampler, batch_size=self.batch_size)

    def predict_dataloader(self) -> DataLoader:
        submission_dataset = SASRecDataset(config=self.config, data=self.submission_data, user_seq=self.user_seq)
        submission_sampler = SequentialSampler(submission_dataset)
        return DataLoader(submission_dataset, sampler=submission_sampler, batch_size=self.batch_size)
