from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.data.utils import (
    generate_sparse_matrix,
    split_sparse_matrix_stratified,
)


class TrainDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.matrix = generate_sparse_matrix(self.data)

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, idx) -> torch.Tensor:
        cur_tensors = torch.tensor(self.matrix[idx].toarray(), dtype=torch.float).squeeze()
        return cur_tensors


class EvalDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.matrix = generate_sparse_matrix(self.data)
        self.input_matrix, self.target_matrix = split_sparse_matrix_stratified(self.matrix)

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        cur_tensors = (
            torch.tensor(self.input_matrix[idx].toarray(), dtype=torch.float).squeeze(),
            torch.tensor(self.target_matrix[idx].toarray(), dtype=torch.float).squeeze(),
        )
        return cur_tensors
