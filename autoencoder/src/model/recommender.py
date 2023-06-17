import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning as L


class Recommender(L.LightningModule):
    def __init__(self, config, model: nn.Module):
        super().__init__()
        self.model = model
        self.config = config
        self.lr = config.model.lr
        self.wd = config.model.wd
        self.total_anneal_step = config.model.total_anneal_step
        self.anneal_cap = config.model.anneal_cap

        self.anneal = 0.0
        self.update_count = 0

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat, mu, logvar = self.model(x)

        self.anneal, self.update_count = self._step_anneal(self.total_anneal_step, self.anneal_cap, self.anneal, self.update_count)

        loss = self._compute_loss(x, x_hat, mu, logvar, self.anneal)
        self.log("train_loss", loss)
        self.log("anneal", self.anneal)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "test")

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        x = batch
        x_hat, _, _ = self.model(x)

        x_hat = self._remove_rated_item(x_hat, x)

        pred_topk = self._predict_topk(x_hat, 10)
        return pred_topk

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer

    def _compute_loss(self, x, x_hat, mu, logvar, anneal):
        BCE = -torch.mean(torch.sum(F.log_softmax(x_hat, 1) * x, -1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        loss = BCE + anneal * KLD
        return loss

    @torch.no_grad()
    def _shared_eval_step(self, batch, batch_idx, prefix):
        x, _ = batch
        x_hat, mu, logvar = self.model(x)

        loss = self._compute_loss(x, x_hat, mu, logvar, self.anneal)

        log_dict = {
            f"{prefix}_loss": loss,
        }

        self.log_dict(log_dict)
        return log_dict

    def _step_anneal(self, total_anneal_step, anneal_cap, anneal, update_count):
        if total_anneal_step > 0:
            anneal = min(anneal_cap, float(update_count / total_anneal_step))
        else:
            anneal = anneal_cap

        update_count += 1
        return anneal, update_count

    def _predict_topk(self, x_hat: np.ndarray, k: int) -> np.ndarray:
        ind = np.argpartition(x_hat, -k)[:, -k:]
        arr_ind = x_hat[np.arange(len(x_hat))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(x_hat)), ::-1]
        pred_topk = ind[np.arange(len(x_hat))[:, None], arr_ind_argsort]
        return pred_topk

    def _recall_at_k_batch(self, x_hat: torch.Tensor, x_target: torch.Tensor, k: int) -> np.ndarray:
        x_hat = x_hat.cpu().numpy()
        x_target = x_target.cpu().numpy()

        ind = np.argpartition(x_hat, -k)[:, -k:]
        x_hat_bool = np.zeros_like(x_hat, dtype=bool)
        x_hat_bool[np.arange(len(x_hat))[:, None], ind[:, :k]] = True

        x_target_bool = x_target > 0

        tmp = (np.logical_and(x_target_bool, x_hat_bool).sum(axis=1)).astype(np.float32)
        recall = tmp / np.minimum(k, x_target_bool.sum(axis=1))
        return recall

    def _remove_rated_item(self, x_hat: torch.Tensor, x: torch.Tensor) -> np.ndarray:
        x_hat_numpy = x_hat.cpu().numpy()
        x_numpy = x.cpu().numpy()
        x_hat_copy = x_hat_numpy.copy()
        x_copy = x_numpy.copy()

        for user_id in range(x_copy.shape[0]):
            idx_rated: list = np.asarray(np.where(x_copy[user_id] != 0))[0].tolist()

            x_hat_copy[user_id, idx_rated] = 0

        return x_hat_copy
