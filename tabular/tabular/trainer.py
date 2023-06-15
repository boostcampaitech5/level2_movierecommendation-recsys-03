import os
import pandas as pd
from omegaconf import DictConfig
from catboost import CatBoostClassifier

from .data import TabularDataModule, TabularDataset


def cv_trainer(config: DictConfig):
    dm = TabularDataModule(config)

    model = []

    # for trainset, validset in zip(dm.train_data, dm.valid_data):
    #     model = CatBoostClassifier(iterations = 2,
    #                                depth = 2,
    #                                learning_rate = 1,
    #                                loss_function='MultiClass',
    #                                verbose=True)

    #     model.fit(trainset.X, trainset.y)


def trainer(config: DictConfig):
    dm = TabularDataModule(config)
