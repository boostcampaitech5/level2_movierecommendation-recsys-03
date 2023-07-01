import random
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset

from src.utils import neg_sample
from src.config import Config


class S3RecDataset(Dataset):
    def __init__(self, config: Config, user_seq, long_seq, attr_name2item2attr: dict[str, dict[int, list[int]]], attr_name2attr_size: dict[str, int]):
        self.config = config
        self.long_seq = long_seq
        self.max_len = config.data.max_seq_length
        self.mask_id = config.data.mask_id
        self.mask_p = config.data.mask_p
        self.item_size = config.data.item_size

        self.attr_name2item2enc = self.__attr_name2item2enc(attr_name2item2attr, attr_name2attr_size)
        self.attr_name2attr_size = attr_name2attr_size

        self.part_seq = self.split_seq(user_seq, config.data.pre_data_augmentation)
        self.data = {}

        if config.data.prepare_data:
            self.data = self.__prepare_data(self.config.data.mask_p)

    def split_seq(self, user_seq, data_augmentation: bool):
        part_seq = []

        for seq in user_seq:
            input_ids = seq[-(self.max_len + 2) : -2]  # keeping same as train set

            if data_augmentation:
                for i in range(len(input_ids)):
                    part_seq.append(input_ids[: i + 1])
            else:
                part_seq.append(input_ids)

        return part_seq

    def __attr_name2item2enc(self, attr_name2item2attr: dict[str, dict[int, list[int]]], attr_name2attr_size: dict[str, int]):
        attr_name2item2enc = {}

        for attr_name in attr_name2item2attr.keys():
            item2attr = attr_name2item2attr[attr_name]
            attr_size = attr_name2attr_size[attr_name]

            attr_name2item2enc[attr_name] = self.attr_encoding(item2attr, attr_size)

        return attr_name2item2enc

    def attr_encoding(self, item2attr, attr_size):
        item2encoded_attr = {}
        for item_id, attrs in item2attr.items():
            item2encoded_attr[item_id] = np.zeros((attr_size))
            item2encoded_attr[item_id][attrs] = 1
            item2encoded_attr[item_id] = item2encoded_attr[item_id].tolist()
        return item2encoded_attr

    def __prepare_row(self, seq, mask_p):
        # sample neg item for every masked item
        masked_item_seq = []
        neg_items = []
        # Masked Item Prediction
        item_set = set(seq)

        for idx in range(len(seq) - 1):
            item = seq[idx]
            prob = random.random()

            if prob < mask_p:
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
        attr_name2attrs = {}

        for name in self.attr_name2item2enc.keys():
            item2enc = self.attr_name2item2enc[name]
            attr_size = self.attr_name2attr_size[name]

            attrs = []

            for item in pos_items:
                if item in item2enc:
                    attrs.append(item2enc[item])
                else:
                    attrs.append([0] * attr_size)

            attr_name2attrs[name] = attrs

        for attrs in attr_name2attrs.values():
            assert len(attrs) == self.max_len
        assert len(masked_item_seq) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len
        assert len(masked_segment_seq) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len

        return attr_name2attrs, masked_item_seq, pos_items, neg_items, masked_segment_seq, pos_segment, neg_segment

    def __prepare_data(self, mask_p):
        attr_name2attrs_list = defaultdict(list)
        masked_item_seq_list = []
        pos_items_list = []
        neg_items_list = []
        masked_segment_seq_list = []
        pos_segment_list = []
        neg_segment_list = []

        for seq in self.part_seq:  # pos_items
            attr_name2attrs, masked_item_seq, pos_items, neg_items, masked_segment_seq, pos_segment, neg_segment = self.__prepare_row(seq, mask_p)

            for name, attrs in attr_name2attrs.items():
                attr_name2attrs_list[name].append(attrs)
            masked_item_seq_list.append(masked_item_seq)
            pos_items_list.append(pos_items)
            neg_items_list.append(neg_items)
            masked_segment_seq_list.append(masked_segment_seq)
            pos_segment_list.append(pos_segment)
            neg_segment_list.append(neg_segment)

        data = {
            "attrs": {name: torch.tensor(attrs, dtype=torch.long) for name, attrs in attr_name2attrs_list.items()},
            "masked_item_seq": torch.tensor(masked_item_seq_list, dtype=torch.long),
            "pos_items": torch.tensor(pos_items_list, dtype=torch.long),
            "neg_items": torch.tensor(neg_items_list, dtype=torch.long),
            "masked_segment_seq": torch.tensor(masked_segment_seq_list, dtype=torch.long),
            "pos_segment": torch.tensor(pos_segment_list, dtype=torch.long),
            "neg_segment": torch.tensor(neg_segment_list, dtype=torch.long),
        }

        return data

    def __getitem__(self, index):
        if self.data:
            return (
                {name: attrs[index] for name, attrs in self.data["attrs"].items()},
                self.data["masked_item_seq"][index],
                self.data["pos_items"][index],
                self.data["neg_items"][index],
                self.data["masked_segment_seq"][index],
                self.data["pos_segment"][index],
                self.data["neg_segment"][index],
            )

        attr_name2attrs, masked_item_seq, pos_items, neg_items, masked_segment_seq, pos_segment, neg_segment = self.__prepare_row(
            self.part_seq[index], self.mask_p
        )

        return (
            {name: torch.tensor(attrs, dtype=torch.long) for name, attrs in attr_name2attrs.items()},
            torch.tensor(masked_item_seq, dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long),
            torch.tensor(masked_segment_seq, dtype=torch.long),
            torch.tensor(pos_segment, dtype=torch.long),
            torch.tensor(neg_segment, dtype=torch.long),
        )

    def __len__(self):
        return len(self.part_seq)


class SASRecDataset(Dataset):
    def __init__(self, config: Config, data: dict, user_seq: list):
        self.config = config
        self.user_seq = user_seq
        self.data = data
        self.item_size = self.config.data.item_size
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
                target_neg.append(neg_sample(seq_set, self.item_size))

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


class BERT4RecValidDataset(Dataset):
    def __init__(self, config: Config, data: dict):
        self.config = config
        self.data = data
        self.max_len = self.config.data.max_seq_length
        self.tensors = self.__prepare_data(data)

    def __prepare_data(self, data):
        user_num = len(data["input_ids"])

        user_id_list = []
        tokens_list = []
        answers_list = []

        for user_id in range(user_num):
            tokens = data["input_ids"][user_id]
            answers = data["answers"][user_id]

            tokens = tokens[-self.max_len :]
            mask_len = self.max_len - len(tokens)
            # zero padding
            tokens = [0] * mask_len + tokens

            user_id_list.append(user_id)
            tokens_list.append(tokens)
            answers_list.append(answers)

        tensors = {
            "user_ids": torch.tensor(user_id_list, dtype=torch.long),
            "tokens": torch.tensor(tokens_list, dtype=torch.long),
            "answers": torch.tensor(answers_list, dtype=torch.long),
        }

        return tensors

    def __getitem__(self, index):
        return (
            self.tensors["user_ids"][index],
            self.tensors["tokens"][index],
            self.tensors["answers"][index],
        )

    def __len__(self):
        return len(self.data["input_ids"])


class BERT4RecTrainDataset(Dataset):
    def __init__(self, config: Config, data: dict):
        self.config = config
        self.data = data
        self.max_len = self.config.data.max_seq_length
        self.mask_p = self.config.data.mask_p
        self.mask_id = self.config.data.mask_id
        self.tensors = self.__prepare_data(data)

    def __prepare_data(self, data):
        user_num = len(data["input_ids"])

        user_id_list = []
        tokens_list = []
        labels_list = []

        for user_id in range(user_num):
            seq = data["input_ids"][user_id]
            tokens = []
            labels = []

            for s in seq:
                prob = np.random.random()
                if prob < self.mask_p:
                    prob /= self.mask_p

                    # BERT 학습
                    if prob < 0.8:
                        # masking
                        tokens.append(self.mask_id)  # mask_index: item_size + 1, 0: pad, 1~item_size: item index
                    elif prob < 0.9:
                        tokens.append(np.random.randint(1, self.mask_id - 1))  # item random sampling
                    else:
                        tokens.append(s)
                    labels.append(s)  # 학습에 사용
                else:
                    tokens.append(s)
                    labels.append(0)  # 학습에 사용 X, trivial

            tokens = tokens[-self.max_len :]
            labels = labels[-self.max_len :]
            mask_len = self.max_len - len(tokens)

            # zero padding
            tokens = [0] * mask_len + tokens
            labels = [0] * mask_len + labels

            user_id_list.append(user_id)
            tokens_list.append(tokens)
            labels_list.append(labels)

        tensors = {
            "user_ids": torch.tensor(user_id_list, dtype=torch.long),
            "tokens": torch.tensor(tokens_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }

        return tensors

    def __getitem__(self, index):
        return (
            self.tensors["user_ids"][index],
            self.tensors["tokens"][index],
            self.tensors["labels"][index],
        )

    def __len__(self):
        return len(self.data["input_ids"])
