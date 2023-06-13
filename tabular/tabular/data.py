import os
import pandas as pd
from omegaconf import OmegaConf
from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold


class TabularDataModule:
    def __init__(self, config: DictConfig):
        self.config = config

        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def prepare_data(self):
        """
        - load data file from csv file
        """
        self.data = pd.read_csv(os.path.join(self.config.path.train_dir, self.config.data.training_version + self.config.path.train_file))

    def setup(self):
        """
        - Split data by user
        - Create new feature through FE
        - Save datamoudle instance in variable
        """
        splitter = GroupKFold(n_splits=self.config.trainer.k)

        if self.config.trainer.cv_strategy == "kfold":
            train_kfold, valid_kfold = [], []
            for train_idx, valid_idx in splitter.split(self.data, groups=self.data["user"]):
                train_kfold.append(self.data.loc[train_idx])
                valid_kfold.append(self.data.loc[valid_idx])

            self.train_data = [TabularDataset(self.config, df) for df in train_kfold]
            self.valid_data = [TabularDataset(self.config, df) for df in valid_kfold]
        else:
            raise Exception("Invalid cv strategy is entered")


class TabularDataset:
    def __init__(self, config: DictConfig, df: pd.DataFrame, is_test=False, is_val=False):
        self.X = df.drop(columns="item")
        self.y = df["item"]
        self.user = df["user"]
