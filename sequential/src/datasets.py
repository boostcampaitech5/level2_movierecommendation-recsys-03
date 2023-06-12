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
    def __init__(self, config: Config, data: dict, user_seq: list):
        self.config = config
        self.user_seq = user_seq
        self.data = data
        self.max_len = self.config.data.max_seq_length
        self.tensors = self.__prepare_data(data, user_seq)

    def __prepare_data(self, data: dict, user_seq: list):
        user_num = len(user_seq)

        user_id_list = []
        input_ids_list = []
        target_pos_list = []
        target_neg_list = []
        answers_list = []

        for user_id in range(user_num):
            items = user_seq[user_id]
            input_ids = data["input_ids"][user_id]
            target_pos = data["target_pos"][user_id]
            answers = data["answers"][user_id]

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

            user_id_list.append(user_id)
            input_ids_list.append(input_ids)
            target_pos_list.append(target_pos)
            target_neg_list.append(target_neg)
            answers_list.append(answers)

        tensors = {
            "user_id": torch.tensor(user_id_list, dtype=torch.long),  # user_id for testing [user_num, 1]
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),  # item_id [user_num, seqlen]
            "target_pos": torch.tensor(target_pos_list, dtype=torch.long),  # target_pos [user_num, seqlen]
            "target_neg": torch.tensor(target_neg_list, dtype=torch.long),  # target_neg [user_num, seqlen]
            "answers": torch.tensor(answers_list, dtype=torch.long),  # answer [user_num, 1]
        }

        return tensors

    def __getitem__(self, index):
        return (
            self.tensors["user_id"][index],
            self.tensors["input_ids"][index],
            self.tensors["target_pos"][index],
            self.tensors["target_neg"][index],
            self.tensors["answers"][index],
        )

    def __len__(self):
        return len(self.data["input_ids"])
