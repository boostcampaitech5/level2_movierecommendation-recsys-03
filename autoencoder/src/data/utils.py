from typing import Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def generate_user_item_indexer(data: pd.DataFrame) -> dict:
    user2idx = {v: k for k, v in enumerate(data["user"].unique())}
    item2idx = {v: k for k, v in enumerate(data["item"].unique())}

    idx2user = {v: k for k, v in user2idx.items()}
    idx2item = {v: k for k, v in item2idx.items()}

    indexer = {
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2user": idx2user,
        "idx2item": idx2item,
    }

    return indexer


def split_user_ids(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_ids = data["user"].unique()
    num_users = data["user"].nunique()

    train_cut = int(num_users * 0.8)
    valid_cut = train_cut + int(num_users * 0.1)

    shuffled_user_ids = user_ids[np.random.permutation(num_users)]

    train_user_ids = shuffled_user_ids[:train_cut]
    valid_user_ids = shuffled_user_ids[train_cut:valid_cut]
    test_user_ids = shuffled_user_ids[valid_cut:]

    assert set(train_user_ids) & set(valid_user_ids) & set(test_user_ids) == set()

    return train_user_ids, valid_user_ids, test_user_ids


def numerize(data: pd.DataFrame, user2idx: dict, item2idx: dict) -> pd.DataFrame:
    user_id = data["user"].apply(lambda x: user2idx[x])
    item_id = data["item"].apply(lambda x: item2idx[x])
    return pd.DataFrame(data={"user_id": user_id, "item_id": item_id}, columns=["user_id", "item_id"])


def generate_sparse_matrix(df: pd.DataFrame) -> csr_matrix:
    num_users = df["user_id"].max() + 1
    num_items = 6807

    row = df["user_id"].values
    col = df["item_id"].values
    data = np.ones_like(row)

    sparse_matrix = csr_matrix((data, (row, col)), dtype="float64", shape=(num_users, num_items))

    return sparse_matrix


def split_sparse_matrix_stratified(sparse_matrix: csr_matrix, ratio: float = 0.2) -> Tuple[csr_matrix, csr_matrix]:
    numpy_matrix: np.ndarray = sparse_matrix.toarray()

    input_matrix = numpy_matrix.copy()
    target_matrix = numpy_matrix.copy()

    num_rated: np.ndarray = np.sum(input_matrix != 0, axis=1)  # 유저 별 평가 횟수 계산
    test_size: np.ndarray = (num_rated * ratio).astype(int)  # 유저 별 test_size 계산

    for user_id in range(numpy_matrix.shape[0]):
        idx: list = np.asarray(np.where(input_matrix[user_id] != 0))[0].tolist()
        idx_test: list = np.random.choice(idx, test_size[user_id], replace=False).tolist()  # test_size만큼 비복원추출된 아이템 리스트
        idx_train: list = list(set(idx) - (set(idx_test)))  # 추출되지 않은 나머지 아이템 리스트

        input_matrix[user_id, idx_test] = 0  # test로 추출된 아이템 0으로 변경
        target_matrix[user_id, idx_train] = 0  # train에서 이미 평가된 아이템 0으로 변경

    assert input_matrix.shape == target_matrix.shape

    return csr_matrix(input_matrix), csr_matrix(target_matrix)
