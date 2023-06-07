import torch
import torch.nn as nn
import numpy as np
import tqdm
import lightning as pl
from scipy.sparse import csr_matrix
from src.config import config
from torch.optim import Adam
from src.modules import Encoder, LayerNorm
from src.utils import ndcg_k, recall_at_k


class BaseModule(pl.LightningModule):
    def __init__(self, config: config):
        super(BaseModule, self).__init__()
        self.config = config
        # cant find in config
        self.item_size = self.config.model.item_size
        self.hidden_size = self.config.model.hidden_size
        self.hidden_dropout_prob = self.config.model.hidden_dropout_prob
        self.initializer_range = self.config.model.initializer_range
        self.max_seq_length = self.config.data.max_seq_length

        self.item_embeddings = nn.Embedding(self.item_size, self.hidden_size, padding_idx=0)
        self.attribute_embeddings = nn.Embedding(self.attribute_size, self.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.item_encoder = Encoder(self)
        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # add unique dense layer for 4 losses respectively
        self.aap_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.mip_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.map_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.sp_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

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

    def make_sequence_embedding(self, seq: list):
        seq_length = seq.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(seq)
        item_embeddings = self.item_embeddings(seq)
        position_embeddings = self.position_embeddings(position_ids)
        seq_emb = item_embeddings + position_embeddings
        seq_emb = self.LayerNorm(seq_emb)
        seq_emb = self.dropout(seq_emb)

        return seq_emb

    def compute_loss(self):
        """
        Not essential
        """
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_train_epoch_end(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError


## sasrec for finetune
class SASRec(BaseModule):
    def __init__(self, config: config, valid_matrix: csr_matrix, test_matrix: csr_matrix, submission_matrix: csr_matrix):
        super().__init__(config)
        self.pred_list = None
        self.answer_list = None
        self.rec_avg_loss = 0.0
        self.rec_cur_loss = 0.0
        self.valid_matrix = valid_matrix
        self.test_matrix = test_matrix
        self.submission_matrix = submission_matrix

    def save(self, file_name: str) -> None:
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name: str) -> None:
        self.load_state_dict(torch.load(file_name))

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "RECALL@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "RECALL@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
        }
        print(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix)

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def compute_loss(self, seq_output, target_pos, target_neg):
        # [batch seq_len hidden_size]
        pos_emb = self.item_embeddings(target_pos)
        neg_emb = self.item_embeddings(target_neg)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_output.view(-1, self.hidden_size)  # [batch*seq_len hidden_size]

        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (target_pos > 0).view(target_pos.size(0) * self.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def configure_optimizers(self):
        betas = (self.config.trainer.adam_beta1, self.config.trainer.adam_beta2)
        return Adam(
            self.parameters(),
            lr=self.config.trainer.lr,
            betas=betas,
            weight_decay=self.config.trainer.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        _, input_ids, target_pos, target_neg, _ = batch  # predict
        seq_output = self(input_ids)
        loss = self.compute_loss(seq_output, target_pos, target_neg)

        self.rec_avg_loss += loss.item()
        self.rec_cur_loss = loss.item()

        return loss

    def train_epoch_end(self):
        self.rec_avg_loss = 0.0
        self.rec_cur_loss = 0.0
        """
        add loggig
        post_fix = {
                "rec_avg_loss": "{:.4f}".format(self.rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": "{:.4f}".format(self.rec_cur_loss),
            }
        """

    def validation_step(self, batch, batch_idx):
        user_ids, input_ids, _, _, answers = batch
        seq_output = self(input_ids)
        seq_output = seq_output[:, -1, :]
        rating_pred = self.predict_full(seq_output)

        rating_pred = rating_pred.cpu().data.numpy().copy()
        batch_user_index = user_ids.cpu().numpy()
        rating_pred[self.valid_matrix[batch_user_index].toarray() > 0] = 0

        ind = np.argpartition(rating_pred, -10)[:, -10:]

        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        if self.pred_list is None:
            self.pred_list = batch_pred_list
            self.answer_list = answers.cpu().data.numpy()
        else:
            self.pred_list = np.append(self.pred_list, batch_pred_list, axis=0)
            self.answer_list = np.append(self.answer_list, answers.cpu().data.numpy(), axis=0)

    def validation_epoch_end(self):
        self.pred_list = None
        self.answer_list = None
        """
        logging
        self.get_full_sort_score(epoch, self.answer_list, self.pred_list)
        """

    def test_step(self, batch, batch_idx):
        user_ids, input_ids, _, _, answers = batch
        seq_output = self(input_ids)
        seq_output = seq_output[:, -1, :]
        rating_pred = self.predict_full(seq_output)

        rating_pred = rating_pred.cpu().data.numpy().copy()
        batch_user_index = user_ids.cpu().numpy()
        rating_pred[self.test_matrix[batch_user_index].toarray() > 0] = 0

        ind = np.argpartition(rating_pred, -10)[:, -10:]

        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        if self.pred_list is None:
            self.pred_list = batch_pred_list
            self.answer_list = answers.cpu().data.numpy()
        else:
            self.pred_list = np.append(self.pred_list, batch_pred_list, axis=0)
            self.answer_list = np.append(self.answer_list, answers.cpu().data.numpy(), axis=0)

    def test_epoch_end(self):
        """
        self.get_full_sort_score(epoch, self.answer_list, self.pred_list)

        """

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        user_ids, input_ids, _, _, answers = batch
        seq_output = self(input_ids)
        seq_output = seq_output[:, -1, :]
        rating_pred = self.predict_full(seq_output)

        rating_pred = rating_pred.cpu().data.numpy().copy()
        batch_user_index = user_ids.cpu().numpy()
        rating_pred[self.submission_matrix[batch_user_index].toarray() > 0] = 0

        ind = np.argpartition(rating_pred, -10)[:, -10:]

        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        if self.pred_list is None:
            self.pred_list = batch_pred_list
            self.answer_list = answers.cpu().data.numpy()
        else:
            self.pred_list = np.append(self.pred_list, batch_pred_list, axis=0)
            self.answer_list = np.append(self.answer_list, answers.cpu().data.numpy(), axis=0)
        return self.pred_list

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

        seq_emb = self.make_sequence_embedding(input_ids)

        item_encoded_layers = self.item_encoder(seq_emb, extended_attention_mask, output_all_encoded_layers=True)

        seq_output = item_encoded_layers[-1]
        return seq_output
