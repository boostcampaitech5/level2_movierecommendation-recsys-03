import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig
import lightgbm as lgb
from catboost import CatBoostClassifier

from .data import TabularDataModule, TabularDataset

pd.set_option("mode.chained_assignment", None)  # warning off


def cv_trainer(config: DictConfig):
    dm = TabularDataModule(config)

    models = []
    for idx, (train_set, valid_set) in enumerate(zip(dm.train_data, dm.valid_data)):
        print(f"model {idx+1} is training ... ")
        model = create_model(config.model.name)
        model.fit(
            X=train_set.X,
            y=train_set.y,
            eval_set=[(valid_set.X, valid_set.y)],
            verbose=10,
            early_stopping_rounds=50,
        )
        models.append(model)

    submit_df = inference(dm=dm, is_cv=True, k=config.trainer.rank_k, models=models)

    createFolder(config.path.output_dir)
    output_name = config.wandb.name + "_" + "CV_" + config.model.name + ".csv"
    submit_df.to_csv(os.path.join(config.path.output_dir, output_name), index=False)

    # for trainset, validset in zip(dm.train_data, dm.valid_data):
    #     model = CatBoostClassifier(iterations = 2,
    #                                depth = 2,
    #                                learning_rate = 1,
    #                                loss_function='MultiClass',
    #                                verbose=True)

    #     model.fit(trainset.X, trainset.y)


def trainer(config: DictConfig):
    dm = TabularDataModule(config)

    createFolder(config.path.output_dir)


def create_model(model_name: str):
    if model_name == "LGBMClassifier":
        model = lgb.LGBMClassifier()
    elif model_name == "CatBoost":
        model = CatBoostClassifier()
    else:
        raise Exception("Invalid model_name is entered :", model_name)

    return model


def mean_cv_prob(k: int, probs):
    probs_pd = [pd.DataFrame(prob) for prob in probs]
    mean_pd = pd.concat(probs_pd, axis=1)
    return mean_pd.mean(axis=1)


def inference(dm: TabularDataModule, is_cv: False, k: int, models=None) -> pd.DataFrame:
    total_df = dm.total_df
    pred_frame_df = dm.pred_frame_df
    submits = []

    for user in tqdm(sorted(total_df["user"].unique())):
        user_df = total_df.query("user==@user")[["item", "user", "relevance"]]
        pred_df = pd.merge(pred_frame_df, user_df, how="outer", on=["item"])

        pred_df["user"] = [user] * pred_df.shape[0]
        pred_df["user"] = pred_df["user"].astype("category")
        pred_df["item"] = pred_df["item"].astype("category")

        pred_df = pred_df.loc[pred_df.loc[pred_df.relevance != 1].index]

        if is_cv:
            probs = []
            for model in models:
                preds_proba = model.predict_proba(pred_df[dm.features])
                probs.append(preds_proba[:, 1])
            cv_proba = mean_cv_prob(k=k, probs=probs)

            topk_idx = np.argsort(cv_proba)[::-1][:k]
        else:
            preds_proba = model.predict_proba(pred_df[dm.features])
            topk_idx = np.argsort(preds_proba[:, 1])[::-1][:k]

        recommend_df = pred_df.iloc[topk_idx].reset_index(drop=True)
        tmp_df = recommend_df[["user", "item"]]
        tmp_df["user"] = [user] * k

        submits.append(tmp_df)

    submit_df = pd.concat(submits, ignore_index=True)
    submit_df["user"] = submit_df["user"].astype("int64")
    submit_df["item"] = submit_df["item"].astype("int64")

    return submit_df


def createFolder(directory: str):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        raise Exception("Error: Creating directory : " + directory)
