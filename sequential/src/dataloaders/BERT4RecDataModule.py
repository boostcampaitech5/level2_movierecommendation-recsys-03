import lightning as L
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from src.config import Config
from src.utils import get_user_seqs, get_user_seqs_long
from src.datasets import BERT4RecDataset
from src.dataloaders.common import KFoldDataModule, KFoldDataModuleContainer


class BERT4RecDataModule(L.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.batch_size = self.config.data.batch_size
        self.user_seq = None
        self.item_size = None
        self.mask_id = None

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

    # load and feature_engineering dataset
    def prepare_data(self):
        self.user_seq, self.item_size, self.valid_matrix, self.test_matrix, self.submission_matrix = get_user_seqs(self.train_dir, self.train_file)
        self.user_seq, _, self.long_seq = get_user_seqs_long(self.train_dir, self.train_file)

        self.config.data.item_size = self.item_size + 2
        self.config.data.mask_id = self.item_size + 1
        self.mask_id = self.config.data.mask_id

    # preprocess and set dataset on train/test case
    def setup(self, stage=None):
        self.n_user = len(self.user_seq)

        if stage == "fit" or stage is None:
            # train
            self.train_data = {
                "input_ids": [seq[:-2] for seq in self.user_seq],
                "answers": [[0] for _ in range(self.n_user)],
            }

            # valid
            self.valid_data = {
                "input_ids": [seq[:-2] + [self.mask_id] for seq in self.user_seq],
                "answers": [[seq[-2]] for seq in self.user_seq],
            }

        elif stage == "test" or stage is None:
            # test
            self.test_data = {
                "input_ids": [seq[:-1] + [self.mask_id] for seq in self.user_seq],
                "answers": [[seq[-1]] for seq in self.user_seq],
            }

        elif stage == "predict" or stage is None:
            self.submission_data = {"input_ids": [seq[:] + [self.mask_id] for seq in self.user_seq], "answers": [[0] for _ in range(self.n_user)]}

    def train_dataloader(self) -> DataLoader:
        train_dataset = BERT4RecDataset(config=self.config, data=self.train_data)
        train_sampler = RandomSampler(train_dataset)
        return DataLoader(train_dataset, sampler=train_sampler, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        valid_dataset = BERT4RecDataset(config=self.config, data=self.valid_data)
        valid_sampler = SequentialSampler(valid_dataset)
        return DataLoader(valid_dataset, sampler=valid_sampler, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self) -> DataLoader:
        test_dataset = BERT4RecDataset(config=self.config, data=self.test_data)
        test_sampler = SequentialSampler(test_dataset)
        return DataLoader(test_dataset, sampler=test_sampler, batch_size=self.batch_size, num_workers=0)

    def predict_dataloader(self) -> DataLoader:
        submission_dataset = BERT4RecDataset(config=self.config, data=self.submission_data)
        submission_sampler = SequentialSampler(submission_dataset)
        return DataLoader(submission_dataset, sampler=submission_sampler, batch_size=self.batch_size, num_workers=0)


class BERT4RecKFoldDataModule(BERT4RecDataModule, KFoldDataModule):
    def __init__(self, config: Config, fold: int):
        BERT4RecDataModule.__init__(self, config)
        self.fold = fold

    def __split_data(self, data: str, idx: int) -> dict:
        if idx == 0:
            new_data = {
                "input_ids": [user[:idx] for user in data["input_ids"]],
                "answers": data["answers"],
            }
        else:
            new_data = {
                "input_ids": [user[:idx] for user in data["input_ids"]],
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
                "answers": [[seq[-2]] for seq in self.user_seq],
            }

            self.train_data = self.__split_data(self.train_data, train_idx)
            self.valid_data = self.train_data

            self.valid_data["input_ids"] = [seq[:] + [self.mask_id] for seq in self.valid_data["input_ids"]]

        elif stage == "test" or stage is None:
            # test
            self.test_data = {
                "input_ids": [seq[:-1] + [self.mask_id] for seq in self.user_seq],
                "answers": [[seq[-1]] for seq in self.user_seq],
            }

        elif stage == "predict" or stage is None:
            self.submission_data = {"input_ids": [seq[:] + [self.mask_id] for seq in self.user_seq], "answers": [[0] for _ in range(self.n_user)]}


class BERT4RecKFoldDataModuleContainer(KFoldDataModuleContainer):
    def __init__(self, config: Config):
        self.config = config

    def kfold_data_module(self, fold: int) -> KFoldDataModule:
        return BERT4RecKFoldDataModule(self.config, fold)
