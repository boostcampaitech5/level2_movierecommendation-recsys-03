import torch
import torch.nn as nn
from src.config import Config
from src.modules import Encoder, LayerNorm
from src.utils import ndcg_k, recall_at_k


class SASRecModule(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.item_size = self.config.data.item_size
        self.hidden_size = self.config.model.hidden_size
        self.hidden_dropout_prob = self.config.model.hidden_dropout_prob
        self.initializer_range = self.config.model.initializer_range
        self.max_seq_length = self.config.data.max_seq_length

        self.item_embeddings = nn.Embedding(self.item_size, self.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.item_encoder = Encoder(self.config)
        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def save(self, file_name: str) -> None:
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name: str) -> None:
        self.load_state_dict(torch.load(file_name))

    def get_full_sort_score(self, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))

        return [recall[0], ndcg[0], recall[1], ndcg[1]]

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

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

    def forward(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.config.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        seq_emb = self.make_seq_embedding(input_ids)

        item_encoded_layers = self.item_encoder(seq_emb, extended_attention_mask, output_all_encoded_layers=True)

        seq_output = item_encoded_layers[-1]
        return seq_output
