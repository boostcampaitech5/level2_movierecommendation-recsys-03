import torch
import torch.nn as nn
import numpy as np
from src.config import Config
from src.modules import LayerNorm, Encoder


class BERT4Rec(nn.Module):
    def __init__(self, config: Config):
        super(BERT4Rec, self).__init__()
        self.config = config
        self.item_size = config.data.item_size
        self.hidden_size = config.model.hidden_size
        self.num_layers = config.model.num_hidden_layers
        self.initializer_range = config.model.initializer_range
        self.max_len = config.data.max_seq_length
        self.dropout_rate = config.model.hidden_dropout_prob
        self.cuda_condition = config.cuda_condition

        self.item_embeddings = nn.Embedding(self.item_size, self.hidden_size, padding_idx=0)  # TODO2: mask와 padding을 고려하여 embedding을 생성해보세요.
        self.position_embeddings = nn.Embedding(self.max_len, self.hidden_size)  # learnable positional encoding
        self.dropout = nn.Dropout(self.dropout_rate)
        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12)

        self.item_encoder = Encoder(self.config)

        # self.item_encoder = nn.ModuleList([BERT4RecBlock(self.config) for _ in range(self.num_layers)])
        # item_emb을 가져와 내적을 하는 것이 올바른 방법(shared 구조)이나 구현의 편의상 output을 위한 새로운 linear layer 활용
        self.out = nn.Linear(self.hidden_size, self.item_size)  # TODO3: 예측을 위한 output layer를 구현해보세요. (self.item_size 주의)

    def save(self, file_name: str) -> None:
        torch.save(self.state_dict(), file_name)

    def load(self, file_name: str) -> None:
        self.load_state_dict(torch.load(file_name))

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def make_seq_embedding(self, seq: list):
        seq_length = seq.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(seq)
        item_embeddings = self.item_embeddings(seq)
        position_embeddings = self.position_embeddings(position_ids)
        seq_emb = item_embeddings + position_embeddings
        seq_emb = self.LayerNorm(seq_emb)
        seq_emb = self.dropout(seq_emb)

        return seq_emb

    def init_weights(self, module):
        """
        Initialize the weights.
        cf https://github.com/pytorch/pytorch/pull/5617
        """

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, log_seqs):
        if self.cuda_condition:
            seqs = self.item_embeddings(torch.tensor(log_seqs, dtype=torch.long).cuda())
            positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
            seqs += self.position_embeddings(torch.tensor(positions, dtype=torch.long).cuda())
        else:
            seqs = self.item_embeddings(torch.tensor(log_seqs, dtype=torch.long))
            positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
            seqs += self.position_embeddings(torch.tensor(positions, dtype=torch.long))

        seqs = self.LayerNorm(self.dropout(seqs))

        mask = torch.tensor(log_seqs > 0, dtype=torch.bool).unsqueeze(1).repeat(1, log_seqs.shape[1], 1).unsqueeze(1)  # mask for zero pad
        item_encoded_layers = self.item_encoder(seqs, mask, output_all_encoded_layers=True)
        seqs = item_encoded_layers[-1]

        out = self.out(seqs)

        return seqs, out
