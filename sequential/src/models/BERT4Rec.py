import torch
import torch.nn as nn
import wandb
import numpy as np
import lightning as L
from torch.optim import Adam
from scipy.sparse import csr_matrix
from src.config import Config
from src.utils import ndcg_k, recall_at_k
from src import modules


## bert for finetune
class BERT4Rec(L.LightningModule):
    def __init__(self, config: Config, valid_matrix: csr_matrix, test_matrix: csr_matrix, submission_matrix: csr_matrix) -> None:
        super().__init__()
        self.config = config
        self.bert = modules.BERT4Rec(config)
        self.pred_list = []
        self.answer_list = []
        self.adam_beta1 = self.config.trainer.adam_beta1
        self.adam_beta2 = self.config.trainer.adam_beta2
        self.lr = self.config.trainer.lr
        self.weight_decay = self.config.trainer.weight_decay
        self.valid_matrix = valid_matrix
        self.test_matrix = test_matrix
        self.submission_matrix = submission_matrix
        self.training_step_outputs = []
        self.tr_result = []
        self.val_result = np.array([])
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.apply(self.bert.init_weights)

    def get_full_sort_score(self, labels, pred_list):
        recall, ndcg = [], []
        for k in [5, 10]:
            recall.append(recall_at_k(labels, pred_list, k))
            ndcg.append(ndcg_k(labels, pred_list, k))

        return [recall[0], ndcg[0], recall[1], ndcg[1]]

    def configure_optimizers(self):
        betas = (self.adam_beta1, self.adam_beta2)
        return Adam(
            self.parameters(),
            lr=self.lr,
            betas=betas,
            weight_decay=self.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        _, tokens, labels, _ = batch  # predict
        _, out = self(tokens)

        out = out.view(-1, out.size(-1))
        labels = labels.view(-1)

        loss = self.criterion(out, labels)
        self.training_step_outputs.append({"rec_avg_loss": loss.detach()})

        return loss

    def on_train_epoch_end(self):
        rec_avg_loss = torch.stack([x["rec_avg_loss"] for x in self.training_step_outputs]).mean()
        rec_cur_loss = self.training_step_outputs[-1]["rec_avg_loss"]

        self.log("rec_avg_loss", rec_avg_loss)
        self.log("rec_cur_loss", rec_cur_loss)

        self.tr_result.append({"rec_avg_loss": rec_avg_loss, "rec_cur_loss": rec_cur_loss})
        wandb.log({"rec_avg_loss": rec_avg_loss, "rec_cur_loss": rec_cur_loss})

        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        user_ids, tokens, _, answers = batch  # predict
        seq_output, _ = self(tokens)

        seq_output = seq_output[:, -1, :]

        rating_pred = self.bert.predict_full(seq_output)

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
        self.val_result = np.append(self.val_result, metrics[2])

        self.log("Recall@10", metrics[2])
        wandb.log({"valid_Recall@10": metrics[2]})

        self.pred_list.clear()
        self.answer_list.clear()

    def test_step(self, batch, batch_idx):
        user_ids, tokens, _, answers = batch  # predict
        seq_output, _ = self(tokens)

        seq_output = seq_output[:, -1, :]

        rating_pred = self.bert.predict_full(seq_output)

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

        self.log("Recall@10", metrics[2])
        wandb.log({"test_valid_Recall@10": metrics[2]})

        self.pred_list.clear()
        self.answer_list.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        user_ids, tokens, _, _ = batch  # predict
        seq_output, _ = self(tokens)
        seq_output = seq_output[:, -1, :]

        rating_pred = self.bert.predict_full(seq_output)

        rating_pred = rating_pred.cpu().data.numpy().copy()
        batch_user_index = user_ids.cpu().numpy()
        rating_pred[self.submission_matrix[batch_user_index].toarray() > 0] = 0

        return rating_pred

    def forward(self, tokens):
        return self.bert.forward(tokens)

    def load_pretrained_module(self, path) -> None:
        self.bert.load(path)
