from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def generate_rating_matrix(user_seq: list, num_users: int, num_items: int) -> csr_matrix:
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list:
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def get_user_seqs(rating_df: pd.DataFrame) -> Tuple[list, int, csr_matrix, dict]:
    item2idx = {v: k for k, v in enumerate(rating_df["item"].unique())}
    idx2item = {v: k for k, v in item2idx.items()}

    rating_df["item"] = rating_df["item"].apply(lambda x: item2idx[x])

    lines = rating_df.groupby("user")["item"].apply(list)
    user_seq = []
    item_set = set()
    for line in lines:
        items = line
        user_seq.append(items)
        item_set = item_set | set(items)

    num_users = len(lines)
    num_items = len(item_set)

    rating_matrix = generate_rating_matrix(user_seq, num_users, num_items)

    return (
        user_seq,
        num_items,
        rating_matrix,
        idx2item,
    )


def train_val_test_split(sparse_matrix: csr_matrix, user_seq: list, ratio_list=[0.8, 0.1, 0.1], k: Optional[int] = None) -> Tuple[csr_matrix, list]:
    num_users = sparse_matrix.shape[0]

    train_cut = int(num_users * ratio_list[0])
    val_cut = train_cut + int(num_users * ratio_list[1])

    train_matrix = sparse_matrix[:train_cut]
    val_matrix = sparse_matrix[train_cut:val_cut]
    test_matrix = sparse_matrix[val_cut:]

    train_seq = user_seq[:train_cut]
    val_seq = user_seq[train_cut:val_cut]
    test_seq = user_seq[val_cut:]

    return (train_matrix, val_matrix, test_matrix, train_seq, val_seq, test_seq)


def input_target_split(sparse_matrix: csr_matrix, ratio: float = 0.2) -> Tuple[csr_matrix, csr_matrix]:
    numpy_matrix: np.ndarray = sparse_matrix.toarray()

    input_matrix = numpy_matrix.copy()
    target_matrix = numpy_matrix.copy()

    num_rated: np.ndarray = np.sum(input_matrix != 0, axis=1)
    target_size: np.ndarray = (num_rated * ratio).astype(int)

    for user_id in range(numpy_matrix.shape[0]):
        idx = np.asarray(np.where(input_matrix[user_id] != 0))[0].tolist()
        idx_target = np.random.choice(idx, target_size[user_id], replace=False).tolist()
        idx_input = list(set(idx) - set(idx_target))

        input_matrix[user_id, idx_target] = 0
        target_matrix[user_id, idx_input] = 0

    assert input_matrix.shape == target_matrix.shape

    return csr_matrix(input_matrix), csr_matrix(target_matrix)


def input_target_mix_split(sparse_matrix: csr_matrix, user_seq: list, ratio: float = 0.2) -> Tuple[csr_matrix, csr_matrix]:
    numpy_matrix: np.ndarray = sparse_matrix.toarray()

    input_matrix = numpy_matrix.copy()
    target_matrix = numpy_matrix.copy()

    num_rated: np.ndarray = np.sum(input_matrix != 0, axis=1)
    half_target_size: np.ndarray = (num_rated * ratio // 2).astype(int)

    for user_id in range(numpy_matrix.shape[0]):
        cut = -half_target_size[user_id]

        idx = np.asarray(np.where(input_matrix[user_id] != 0))[0].tolist()
        idx_static = np.random.choice(user_seq[user_id][:cut], half_target_size[user_id], replace=False).tolist()
        idx_dynamic = user_seq[user_id][cut:]
        idx_target = idx_static + idx_dynamic
        idx_input = list(set(idx) - set(idx_target))

        input_matrix[user_id, idx_target] = 0
        target_matrix[user_id, idx_input] = 0

    assert input_matrix.shape == target_matrix.shape

    return csr_matrix(input_matrix), csr_matrix(target_matrix)


def input_target_mix_leave_n_out_split(sparse_matrix: csr_matrix, user_seq: list, n: int = 1) -> Tuple[csr_matrix, csr_matrix]:
    numpy_matrix: np.ndarray = sparse_matrix.toarray()

    input_matrix = numpy_matrix.copy()
    target_matrix = numpy_matrix.copy()

    for user_id in range(numpy_matrix.shape[0]):
        idx = np.asarray(np.where(input_matrix[user_id] != 0))[0].tolist()
        idx_static = np.random.choice(user_seq[user_id][:-n], n, replace=False).tolist()
        idx_dynamic = user_seq[user_id][-n:]
        idx_target = idx_static + idx_dynamic
        idx_input = list(set(idx) - set(idx_target))

        input_matrix[user_id, idx_target] = 0
        target_matrix[user_id, idx_input] = 0

    assert input_matrix.shape == target_matrix.shape

    return csr_matrix(input_matrix), csr_matrix(target_matrix)


def kfold_train_val_test_split(sparse_matrix: csr_matrix, user_seq: list, k: int, num_folds=5) -> Tuple[csr_matrix, list]:
    numpy_matrix = sparse_matrix.toarray()
    num_users = sparse_matrix.shape[0]
    num_items = sparse_matrix.shape[1]
    fold_size = num_users // num_folds

    ratio_list = [1.0 - 1.0 / num_folds, 0.5 / num_folds, 0.5 / num_folds]
    train_cut = int(num_users * ratio_list[0])
    val_cut = train_cut + int(num_users * ratio_list[1])

    numpy_matrix = numpy_matrix.reshape(num_folds, num_users // num_folds, num_items)
    numpy_matrix[[4, k]] = numpy_matrix[[k, 4]]
    numpy_matrix = numpy_matrix.reshape(num_users, num_items)

    user_seq[4 * fold_size : 5 * fold_size], user_seq[k * fold_size : (k + 1) * fold_size] = (
        user_seq[k * fold_size : (k + 1) * fold_size],
        user_seq[4 * fold_size : 5 * fold_size],
    )

    train_matrix = numpy_matrix[:train_cut]
    val_matrix = numpy_matrix[train_cut:val_cut]
    test_matrix = numpy_matrix[val_cut:]

    train_seq = user_seq[:train_cut]
    val_seq = user_seq[train_cut:val_cut]
    test_seq = user_seq[val_cut:]

    return (csr_matrix(train_matrix), csr_matrix(val_matrix), csr_matrix(test_matrix), train_seq, val_seq, test_seq)
