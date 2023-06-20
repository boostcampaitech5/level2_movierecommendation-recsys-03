from typing import Tuple
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import Dataset
from src.data.utils import input_target_mix_split


class TrainDataset(Dataset):
    def __init__(self, matrix: csr_matrix):
        self.matrix = matrix

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, idx) -> torch.Tensor:
        cur_tensors = torch.tensor(self.matrix[idx].toarray(), dtype=torch.float).squeeze()
        return cur_tensors


class EvalDataset(Dataset):
    def __init__(self, matrix: csr_matrix, seq):
        self.matrix = matrix
        self.seq = seq
        self.input_matrix, self.target_matrix = input_target_mix_split(self.matrix, self.seq)

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        cur_tensors = (
            torch.tensor(self.input_matrix[idx].toarray(), dtype=torch.float).squeeze(),
            torch.tensor(self.target_matrix[idx].toarray(), dtype=torch.float).squeeze(),
        )
        return cur_tensors
