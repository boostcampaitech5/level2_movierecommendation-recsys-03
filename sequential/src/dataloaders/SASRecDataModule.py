import lightning as L
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from src.config import Config
from src.utils import get_user_seqs, get_user_seqs_long, get_item2attr_json
from src.datasets import SASRecDataset
from src.dataloaders.common import KFoldDataModule, KFoldDataModuleContainer


class SASRecDataModule(L.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
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
        self.n_user = None

        self.train_dir = self.config.path.train_dir
        self.train_file = self.config.path.train_file
        self.attr_file = config.data.data_version + "_" + config.path.attr_file

    # load and feature_engineering dataset
    def prepare_data(self):
        self.user_seq, self.max_item, self.valid_matrix, self.test_matrix, self.submission_matrix = get_user_seqs(self.train_dir, self.train_file)
        self.user_seq, _, self.long_seq = get_user_seqs_long(self.train_dir, self.train_file)
        self.item2attr, self.attr_size = get_item2attr_json(self.train_dir, self.attr_file)

        self.config.data.item_size = self.max_item + 2
        self.config.data.mask_id = self.max_item + 1
        self.config.data.attr_size = self.attr_size + 1
        self.config.data.item2attr = self.item2attr

    # preprocess and set dataset on train/test case
    def setup(self, stage=None):
        self.n_user = len(self.user_seq)

        if stage == "fit" or stage is None:
            # train
            self.train_data = {
                "input_ids": [seq[:-3] for seq in self.user_seq],
                "target_pos": [seq[1:-2] for seq in self.user_seq],
                "answers": [[0] for _ in range(self.n_user)],
            }

            # valid
            self.valid_data = {
                "input_ids": [seq[:-2] for seq in self.user_seq],
                "target_pos": [seq[1:-1] for seq in self.user_seq],
                "answers": [[seq[-2]] for seq in self.user_seq],
            }

        elif stage == "test" or stage is None:
            # test
            self.test_data = {
                "input_ids": [seq[:-1] for seq in self.user_seq],
                "target_pos": [seq[1:] for seq in self.user_seq],
                "answers": [[seq[-1]] for seq in self.user_seq],
            }

        elif stage == "predict" or stage is None:
            self.submission_data = {"input_ids": self.user_seq, "target_pos": self.user_seq, "answers": [[0] for _ in range(self.n_user)]}

    def train_dataloader(self) -> DataLoader:
        train_dataset = SASRecDataset(config=self.config, data=self.train_data, user_seq=self.user_seq)
        train_sampler = RandomSampler(train_dataset)
        return DataLoader(train_dataset, sampler=train_sampler, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        valid_dataset = SASRecDataset(config=self.config, data=self.valid_data, user_seq=self.user_seq)
        valid_sampler = SequentialSampler(valid_dataset)
        return DataLoader(valid_dataset, sampler=valid_sampler, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self) -> DataLoader:
        test_dataset = SASRecDataset(config=self.config, data=self.test_data, user_seq=self.user_seq)
        test_sampler = SequentialSampler(test_dataset)
        return DataLoader(test_dataset, sampler=test_sampler, batch_size=self.batch_size, num_workers=0)

    def predict_dataloader(self) -> DataLoader:
        submission_dataset = SASRecDataset(config=self.config, data=self.submission_data, user_seq=self.user_seq)
        submission_sampler = SequentialSampler(submission_dataset)
        return DataLoader(submission_dataset, sampler=submission_sampler, batch_size=self.batch_size, num_workers=0)


class SASRecKFoldDataModule(SASRecDataModule, KFoldDataModule):
    def __init__(self, config: Config, fold: int):
        SASRecDataModule.__init__(self, config)
        self.fold = fold

    def __split_data(self, data: str, idx: int) -> dict:
        if idx == 0:
            new_data = {
                "input_ids": [user[:idx] for user in data["input_ids"]],
                "target_pos": [user[:idx] for user in data["target_pos"]],
                "answers": data["answers"],
            }
        else:
            new_data = {
                "input_ids": [user[:idx] for user in data["input_ids"]],
                "target_pos": [user[:idx] for user in data["target_pos"]],
                "answers": [[user[idx]] for user in data["input_ids"]],
            }

        return new_data

    def setup(self, stage=None):
        self.n_user = len(self.user_seq)

        if stage == "fit" or stage is None:
            idx = self.fold + 1
            train_idx = -1 * idx
            valid_idx = idx - 1

            # train
            self.train_data = {
                "input_ids": [seq[:-2] for seq in self.user_seq],
                "target_pos": [seq[1:-1] for seq in self.user_seq],
                "answers": [[seq[-2]] for seq in self.user_seq],
            }

            self.valid_data = self.__split_data(self.train_data, valid_idx)
            self.train_data = self.__split_data(self.train_data, train_idx)

        elif stage == "test" or stage is None:
            # test
            self.test_data = {
                "input_ids": [seq[:-1] for seq in self.user_seq],
                "target_pos": [seq[1:] for seq in self.user_seq],
                "answers": [[seq[-1]] for seq in self.user_seq],
            }

        elif stage == "predict" or stage is None:
            self.submission_data = {"input_ids": self.user_seq, "target_pos": self.user_seq, "answers": [[0] for _ in range(self.n_user)]}


class SASRecKFoldDataModuleContainer(KFoldDataModuleContainer):
    def __init__(self, config: Config):
        self.config = config

    def kfold_data_module(self, fold: int) -> KFoldDataModule:
        return SASRecKFoldDataModule(self.config, fold)
