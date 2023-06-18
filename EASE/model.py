import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import starmap
from omegaconf import DictConfig
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count


class EASE:
    def __init__(self, config: DictConfig):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

        self.config = config
        self.lambda_ = self.config.lambda_

    def fit(self, df: pd.DataFrame):
        print(">>> model is on fit task ...")
        users = self.user_encoder.fit_transform(df.loc[:, "user"])
        items = self.item_encoder.fit_transform(df.loc[:, "item"])
        values = np.ones(df.shape[0])

        X = csr_matrix((values, (users, items)))  # change to csr matrix

        G = X.T @ X.toarray()  # G = X'X (gram matrix, item-to-item matrix, co-occurrence matrix)
        diag_indices = np.diag_indices(G.shape[0])

        G[diag_indices] += self.lambda_  # X;X + λI
        P = np.linalg.inv(G)  # get sim matrix P of inverse mat G
        B = P / (-np.diag(P))  # P_{ij} / - P_{jj} if i ≠ j
        B[diag_indices] = 0  # for diag = 0

        self.pred = X.dot(B)

    def predict(self, data, users, items, k) -> pd.DataFrame:
        print(">>> model is on predict task ...")
        items = self.item_encoder.transform(items)

        filtered_data = data.loc[data.user.isin(users)]
        filtered_data["encoded_item"] = self.item_encoder.transform(filtered_data.item)
        filtered_data["encoded_user"] = self.user_encoder.transform(filtered_data.user)

        g = filtered_data.groupby("encoded_user")

        user_preds = list(starmap(self.predict_each_user, [(user, group, self.pred[user, :], items, k) for user, group in tqdm(g)]))

        print(">>> create submit file by concat total prediction df ...")
        result = pd.concat(user_preds)
        result["item"] = self.item_encoder.inverse_transform(result["item"])
        result["user"] = self.user_encoder.inverse_transform(result["user"])
        result = result.drop("score", axis=1)

        return result

    def predict_each_user(self, user, group, pred, items, k):
        watched = set(group["encoded_item"])
        candidates = [item for item in items if item not in watched]
        pred = np.take(pred, candidates)
        res = np.argpartition(pred, -k)[-k:]
        return pd.DataFrame(
            {
                "user": [user] * len(res),
                "item": np.take(candidates, res),
                "score": np.take(pred, res),
            }
        ).sort_values("score", ascending=False)
