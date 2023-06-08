import random
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import neg_sample
from src.config import Config


class S3RecDataset(Dataset):
    def __init__(self, config: Config, user_seq, long_seq):
        self.config = config
        self.user_seq = user_seq
        self.long_seq = long_seq
        self.max_len = config.data.max_seq_length
        self.mask_id = config.data.mask_id
        self.item_size = config.data.item_size
        self.attr_size = config.data.attr_size
        self.item2attr = self.attr_encoding(config.data.item2attr)
        self.part_seq = []
        self.split_seq()

    def split_seq(self):
        for seq in self.user_seq:
            input_ids = seq[-(self.max_len + 2) : -2]  # keeping same as train set
            for i in range(len(input_ids)):
                self.part_seq.append(input_ids[: i + 1])

    def attr_encoding(self, item2attr):
        item2encoded_attr = {}
        for item_id, attrs in item2attr.items():
            item2encoded_attr[item_id] = np.zeros((self.attr_size))
            item2encoded_attr[item_id][attrs] = 1
            item2encoded_attr[item_id] = item2encoded_attr[item_id].tolist()
        return item2encoded_attr

    def __len__(self):
        return len(self.part_seq)

    def __getitem__(self, index):
        seq = self.part_seq[index]  # pos_items
        # sample neg item for every masked item
        masked_item_seq = []
        neg_items = []
        # Masked Item Prediction
        item_set = set(seq)
        for item in seq[:-1]:
            prob = random.random()
            if prob < self.config.data.mask_p:
                masked_item_seq.append(self.mask_id)
                neg_items.append(neg_sample(item_set, self.item_size))
            else:
                masked_item_seq.append(item)
                neg_items.append(item)

        # add mask at the last position
        masked_item_seq.append(self.mask_id)
        neg_items.append(neg_sample(item_set, self.item_size))

        # Segment Prediction
        if len(seq) < 2:
            masked_segment_seq = seq
            pos_segment = seq
            neg_segment = seq
        else:
            sample_length = random.randint(1, len(seq) // 2)
            start_id = random.randint(0, len(seq) - sample_length)
            neg_start_id = random.randint(0, len(self.long_seq) - sample_length)
            pos_segment = seq[start_id : start_id + sample_length]
            neg_segment = self.long_seq[neg_start_id : neg_start_id + sample_length]
            masked_segment_seq = seq[:start_id] + [self.mask_id] * sample_length + seq[start_id + sample_length :]
            pos_segment = [self.mask_id] * start_id + pos_segment + [self.mask_id] * (len(seq) - (start_id + sample_length))
            neg_segment = [self.mask_id] * start_id + neg_segment + [self.mask_id] * (len(seq) - (start_id + sample_length))

        assert len(masked_segment_seq) == len(seq)
        assert len(pos_segment) == len(seq)
        assert len(neg_segment) == len(seq)

        # padding sequence
        pad_len = self.max_len - len(seq)
        masked_item_seq = [0] * pad_len + masked_item_seq
        pos_items = [0] * pad_len + seq
        neg_items = [0] * pad_len + neg_items
        masked_segment_seq = [0] * pad_len + masked_segment_seq
        pos_segment = [0] * pad_len + pos_segment
        neg_segment = [0] * pad_len + neg_segment

        masked_item_seq = masked_item_seq[-self.max_len :]
        pos_items = pos_items[-self.max_len :]
        neg_items = neg_items[-self.max_len :]

        masked_segment_seq = masked_segment_seq[-self.max_len :]
        pos_segment = pos_segment[-self.max_len :]
        neg_segment = neg_segment[-self.max_len :]

        # Associated Attribute Prediction
        # Masked Attribute Prediction
        attrs = []
        for item in pos_items:
            if item in self.item2attr:
                attrs.append(self.item2attr[item])
            else:
                attrs.append([0] * self.attr_size)

        assert len(attrs) == self.max_len
        assert len(masked_item_seq) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len
        assert len(masked_segment_seq) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len

        cur_tensors = (
            torch.tensor(attrs, dtype=torch.long),
            torch.tensor(masked_item_seq, dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long),
            torch.tensor(masked_segment_seq, dtype=torch.long),
            torch.tensor(pos_segment, dtype=torch.long),
            torch.tensor(neg_segment, dtype=torch.long),
        )
        return cur_tensors


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
