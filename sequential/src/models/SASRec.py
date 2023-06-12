import torch
import numpy as np
import lightning as L
from torch.optim import Adam
from scipy.sparse import csr_matrix
from src.config import Config
from src.utils import ndcg_k, recall_at_k
from src import modules


## sasrec for finetune
class SASRec(L.LightningModule):
    def __init__(self, config: Config, valid_matrix: csr_matrix, test_matrix: csr_matrix, submission_matrix: csr_matrix) -> None:
        super().__init__()
        self.config = config
        self.sasrec = modules.SASRec(config)
        self.pred_list = []
        self.answer_list = []
        self.rating_pred_list = []
        self.valid_matrix = valid_matrix
        self.test_matrix = test_matrix
        self.submission_matrix = submission_matrix
        self.training_step_outputs = []
        self.tr_result = []
        self.val_result = np.array([])

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

    def get_full_sort_score(self, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))

        return [recall[0], ndcg[0], recall[1], ndcg[1]]

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

        return loss

    def on_train_epoch_end(self):
        rec_avg_loss = torch.stack([x["rec_avg_loss"] for x in self.training_step_outputs]).mean()
        rec_cur_loss = self.training_step_outputs[-1]["rec_avg_loss"]
        self.log("rec_avg_loss", rec_avg_loss)
        self.log("rec_cur_loss", rec_cur_loss)
        self.tr_result.append({"rec_avg_loss": rec_avg_loss, "rec_cur_loss": rec_cur_loss})

        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        user_ids, input_ids, _, _, answers = batch
        seq_output = self(input_ids)
        seq_output = seq_output[:, -1, :]
        rating_pred = self.sasrec.predict_full(seq_output)

        rating_pred = rating_pred.cpu().data.numpy().copy()
        batch_user_index = user_ids.cpu().numpy()
        rating_pred[self.valid_matrix[batch_user_index].toarray() > 0] = 0

        ind = np.argpartition(rating_pred, -10)[:, -10:]

        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        self.pred_list.append(batch_pred_list)
        self.answer_list.append(answers.cpu().data.numpy())

    def on_validation_epoch_end(self):
        pred_list = np.concatenate(self.pred_list, axis=0)
        answer_list = np.concatenate(self.answer_list, axis=0)

        metrics = self.get_full_sort_score(answer_list, pred_list)
        self.log("Recall@10", metrics[2])
        self.val_result = np.append(self.val_result, metrics[2])

        self.pred_list.clear()
        self.answer_list.clear()

    def test_step(self, batch, batch_idx):
        user_ids, input_ids, _, _, answers = batch
        seq_output = self(input_ids)
        seq_output = seq_output[:, -1, :]
        rating_pred = self.sasrec.predict_full(seq_output)

        rating_pred = rating_pred.cpu().data.numpy().copy()
        batch_user_index = user_ids.cpu().numpy()
        rating_pred[self.test_matrix[batch_user_index].toarray() > 0] = 0

        ind = np.argpartition(rating_pred, -10)[:, -10:]

        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        self.pred_list.append(batch_pred_list)
        self.answer_list.append(answers.cpu().data.numpy())

    def on_test_epoch_end(self):
        pred_list = np.concatenate(self.pred_list, axis=0)
        answer_list = np.concatenate(self.answer_list, axis=0)

        metrics = self.get_full_sort_score(answer_list, pred_list)
        self.val_result = np.append(self.val_result, metrics[2])

        self.pred_list.clear()
        self.answer_list.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        user_ids, input_ids, _, _, answers = batch
        seq_output = self(input_ids)
        seq_output = seq_output[:, -1, :]

        rating_pred = self.sasrec.predict_full(seq_output)

        rating_pred = rating_pred.cpu().data.numpy().copy()
        batch_user_index = user_ids.cpu().numpy()
        rating_pred[self.submission_matrix[batch_user_index].toarray() > 0] = 0

        ind = np.argpartition(rating_pred, -10)[:, -10:]

        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        self.rating_pred_list.append(rating_pred)

        return batch_pred_list

    def on_predict_epoch_end(self) -> None:
        self.rating_pred_list = np.concatenate(self.rating_pred_list, axis=0)

    def forward(self, input_ids):
        return self.sasrec.forward(input_ids)

    def load_pretrained_module(self, path) -> None:
        self.sasrec.load(path)
