from typing import Any
import torch
import torch.nn as nn
import numpy as np
import lightning as L
from torch.optim import Adam
from torch.optim import Optimizer
from torch.optim import Adam
from scipy.sparse import csr_matrix
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


class S3Rec(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.sasrec = SASRecModule(config)
        self.mask_id = config.data.mask_id
        self.attr_size = self.config.data.attr_size
        self.training_step_outputs = []

        self.attr_embeddings = nn.Embedding(self.attr_size, self.sasrec.hidden_size, padding_idx=0)

        # add unique dense layer for 4 losses respectively
        self.aap_norm = nn.Linear(self.sasrec.hidden_size, self.sasrec.hidden_size)
        self.mip_norm = nn.Linear(self.sasrec.hidden_size, self.sasrec.hidden_size)
        self.map_norm = nn.Linear(self.sasrec.hidden_size, self.sasrec.hidden_size)
        self.sp_norm = nn.Linear(self.sasrec.hidden_size, self.sasrec.hidden_size)
        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.sasrec.init_weights)

    # AAP
    def associated_attr_prediction(self, seq_output, attr_embedding) -> torch.Tensor:
        """
        :param seq_output: [B L H]
        :param attr_embedding: [arribute_num H]
        :return: scores [B*L tag_num]
        """
        seq_output = self.aap_norm(seq_output)  # [B L H]
        seq_output = seq_output.view([-1, self.sasrec.hidden_size, 1])  # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attr_embedding, seq_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L tag_num]

    # MIP sample neg items
    def masked_item_prediction(self, seq_output, target_item) -> torch.Tensor:
        """
        :param seq_output: [B L H]
        :param target_item: [B L H]
        :return: scores [B*L]
        """
        seq_output = self.mip_norm(seq_output.view([-1, self.sasrec.hidden_size]))  # [B*L H]
        target_item = target_item.view([-1, self.sasrec.hidden_size])  # [B*L H]
        score = torch.mul(seq_output, target_item)  # [B*L H]
        return torch.sigmoid(torch.sum(score, -1))  # [B*L]

    # MAP
    def masked_attr_prediction(self, seq_output, attr_embedding) -> torch.Tensor:
        seq_output = self.map_norm(seq_output)  # [B L H]
        seq_output = seq_output.view([-1, self.sasrec.hidden_size, 1])  # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attr_embedding, seq_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L tag_num]

    # SP sample neg segment
    def segment_prediction(self, context, segment) -> torch.Tensor:
        """
        :param context: [B H]
        :param segment: [B H]
        :return:
        """
        context = self.sp_norm(context)
        score = torch.mul(context, segment)  # [B H]
        return torch.sigmoid(torch.sum(score, dim=-1))  # [B]

    def forward(
        self,
        masked_item_seq,
        pos_items,
        neg_items,
        masked_segment_seq,
        pos_segment,
        neg_segment,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode masked seq
        seq_emb = self.sasrec.make_seq_embedding(masked_item_seq)
        seq_mask = (masked_item_seq == 0).float() * -1e8
        seq_mask = torch.unsqueeze(torch.unsqueeze(seq_mask, 1), 1)

        encoded_layers = self.sasrec.item_encoder(seq_emb, seq_mask, output_all_encoded_layers=True)
        # [B L H]
        seq_output = encoded_layers[-1]

        attr_embeddings = self.attr_embeddings.weight
        # AAP
        aap_score = self.associated_attr_prediction(seq_output, attr_embeddings)

        # MIP
        pos_item_embs = self.sasrec.item_embeddings(pos_items)
        neg_item_embs = self.sasrec.item_embeddings(neg_items)
        pos_score = self.masked_item_prediction(seq_output, pos_item_embs)
        neg_score = self.masked_item_prediction(seq_output, neg_item_embs)
        mip_distance = torch.sigmoid(pos_score - neg_score)

        # MAP
        map_score = self.masked_attr_prediction(seq_output, attr_embeddings)

        # SP
        # segment context
        segment_context = self.sasrec.make_seq_embedding(masked_segment_seq)
        segment_mask = (masked_segment_seq == 0).float() * -1e8
        segment_mask = torch.unsqueeze(torch.unsqueeze(segment_mask, 1), 1)
        segment_encoded_layers = self.sasrec.item_encoder(segment_context, segment_mask, output_all_encoded_layers=True)

        # take the last position hidden as the context
        segment_context = segment_encoded_layers[-1][:, -1, :]  # [B H]
        # pos_segment
        pos_segment_emb = self.sasrec.make_seq_embedding(pos_segment)
        pos_segment_mask = (pos_segment == 0).float() * -1e8
        pos_segment_mask = torch.unsqueeze(torch.unsqueeze(pos_segment_mask, 1), 1)
        pos_segment_encoded_layers = self.sasrec.item_encoder(pos_segment_emb, pos_segment_mask, output_all_encoded_layers=True)
        pos_segment_emb = pos_segment_encoded_layers[-1][:, -1, :]

        # neg_segment
        neg_segment_emb = self.sasrec.make_seq_embedding(neg_segment)
        neg_segment_mask = (neg_segment == 0).float() * -1e8
        neg_segment_mask = torch.unsqueeze(torch.unsqueeze(neg_segment_mask, 1), 1)
        neg_segment_encoded_layers = self.sasrec.item_encoder(neg_segment_emb, neg_segment_mask, output_all_encoded_layers=True)
        neg_segment_emb = neg_segment_encoded_layers[-1][:, -1, :]  # [B H]

        pos_segment_score = self.segment_prediction(segment_context, pos_segment_emb)
        neg_segment_score = self.segment_prediction(segment_context, neg_segment_emb)

        sp_distance = torch.sigmoid(pos_segment_score - neg_segment_score)

        return aap_score, mip_distance, map_score, sp_distance

    def compute_loss(self, aap_score, mip_distance, map_score, sp_distance, attrs, masked_item_seq):
        ## AAP
        aap_loss = self.criterion(aap_score, attrs.view(-1, self.attr_size).float())
        # only compute loss at non-masked position
        aap_mask = (masked_item_seq != self.mask_id).float() * (masked_item_seq != 0).float()
        aap_loss = torch.sum(aap_loss * aap_mask.flatten().unsqueeze(-1))

        ## MIP
        mip_loss = self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32))
        mip_mask = (masked_item_seq == self.mask_id).float()
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())

        ## MAP
        map_loss = self.criterion(map_score, attrs.view(-1, self.attr_size).float())
        map_mask = (masked_item_seq == self.mask_id).float()
        map_loss = torch.sum(map_loss * map_mask.flatten().unsqueeze(-1))

        ## SP
        sp_loss = torch.sum(self.criterion(sp_distance, torch.ones_like(sp_distance, dtype=torch.float32)))

        return aap_loss, mip_loss, map_loss, sp_loss

    def configure_optimizers(self) -> Optimizer:
        betas = (self.config.trainer.adam_beta1, self.config.trainer.adam_beta2)
        optimizer = Adam(self.parameters(), lr=self.config.trainer.lr, betas=betas, weight_decay=self.config.trainer.weight_decay)

        return optimizer

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        attrs, masked_item_seq, pos_items, neg_items, masked_segment_seq, pos_segment, neg_segment = batch

        # get score
        aap_score, mip_distance, map_score, sp_distance = self.forward(
            masked_item_seq, pos_items, neg_items, masked_segment_seq, pos_segment, neg_segment
        )
        # get loss
        aap_loss, mip_loss, map_loss, sp_loss = self.compute_loss(aap_score, mip_distance, map_score, sp_distance, attrs, masked_item_seq)

        joint_loss = (
            self.config.trainer.aap_weight * aap_loss
            + self.config.trainer.mip_weight * mip_loss
            + self.config.trainer.map_weight * map_loss
            + self.config.trainer.sp_weight * sp_loss
        )

        self.training_step_outputs.append(
            {"aap_loss": aap_loss.detach(), "mip_loss": mip_loss.detach(), "map_loss": map_loss.detach(), "sp_loss": sp_loss.detach()}
        )

        return joint_loss

    def on_train_epoch_end(self) -> None:
        avg_aap_loss = torch.stack([x["aap_loss"] for x in self.training_step_outputs]).mean()
        avg_mip_loss = torch.stack([x["mip_loss"] for x in self.training_step_outputs]).mean()
        avg_map_loss = torch.stack([x["map_loss"] for x in self.training_step_outputs]).mean()
        avg_sp_loss = torch.stack([x["sp_loss"] for x in self.training_step_outputs]).mean()

        self.log("avg_aap_loss", avg_aap_loss)
        self.log("avg_mip_loss", avg_mip_loss)
        self.log("avg_map_loss", avg_map_loss)
        self.log("avg_sp_loss", avg_sp_loss)

        self.training_step_outputs.clear()


## sasrec for finetune
class SASRec(L.LightningModule):
    def __init__(self, config: Config, valid_matrix: csr_matrix, test_matrix: csr_matrix, submission_matrix: csr_matrix) -> None:
        super().__init__()
        self.config = config
        self.sasrec = SASRecModule(config)
        self.pred_list = None
        self.answer_list = None
        self.rec_avg_loss = 0.0
        self.rec_cur_loss = 0.0
        self.valid_matrix = valid_matrix
        self.test_matrix = test_matrix
        self.submission_matrix = submission_matrix
        self.training_step_outputs = []

        self.apply(self.sasrec.init_weights)

    def compute_loss(self, seq_output, target_pos, target_neg):
        # [batch seq_len hidden_size]
        pos_emb = self.sasrec.item_embeddings(target_pos)
        neg_emb = self.sasrec.item_embeddings(target_neg)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_output.view(-1, self.sasrec.hidden_size)  # [batch*seq_len hidden_size]

        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (target_pos > 0).view(target_pos.size(0) * self.sasrec.max_seq_length).float()  # [batch*seq_len]
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
        self.training_step_outputs.append({"rec_avg_loss": loss.detach()})
        self.rec_cur_loss = loss.detach()

        return loss

    def on_train_epoch_end(self):
        rec_avg_loss = torch.stack([x["rec_avg_loss"] for x in self.sasrec.training_step_outputs]).mean()
        self.log("rec_avg_loss", rec_avg_loss)
        self.log("rec_cur_loss", self.training_step_outputs[-1]["rec_avg_loss"])

        self.training_step_outputs.clear()

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

    def on_validation_epoch_end(self):
        metrics = self.get_full_sort_score(self.answer_list, self.pred_list)
        self.log("NDCG@10", metrics[3])

        self.pred_list = None
        self.answer_list = None

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

    def on_test_epoch_end(self):
        metrics = self.get_full_sort_score(self.answer_list, self.pred_list)
        self.log("NDCG@10", metrics[3])

        self.pred_list = None
        self.answer_list = None

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
        return self.sasrec.forward(input_ids)
