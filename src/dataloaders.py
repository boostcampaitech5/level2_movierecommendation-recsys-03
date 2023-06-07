import lightning as L
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from src.config import Config
from src.utils import get_user_seqs_long, get_item2attr_json
from src.datasets import S3RecDataset, SASRecDataset


class S3RecDataModule(L.LightningDataModule):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config

        # data_file = config.path.data_dir + config.path.data_version + '.txt'
        self.data_file = config.path.train_dir + config.path.train_file
        self.item2attr_file = config.path.train_dir + config.data.data_version + "_" + config.path.atrr_file

    def prepare_data(self) -> None:
        # concat all user_seq get a long sequence, from which sample neg segment for SP
        self.user_seq, max_item, self.long_seq = get_user_seqs_long(self.config.path.train_dir, self.config.path.train_file)
        item2attr, attr_size = get_item2attr_json(self.item2attr_file)

        self.config.data.item_size = max_item + 2
        self.config.data.mask_id = max_item + 1
        self.config.data.attr_size = attr_size + 1
        self.config.data.item2attr = item2attr

    def train_dataloader(self) -> DataLoader:
        trainset = S3RecDataset(self.config, self.user_seq, self.long_seq)
        sampler = RandomSampler(trainset)
        return DataLoader(trainset, sampler=sampler, batch_size=self.config.data.pre_batch_size)


class SASRecDataModule(L.LightningDataModule):
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
