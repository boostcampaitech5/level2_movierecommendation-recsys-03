from typing import Optional, List, Dict
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

        self.shared_eval_step_outputs = []
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat, mu, logvar = self.model(x)

        self.anneal, self.update_count = self._step_anneal(self.total_anneal_step, self.anneal_cap, self.anneal, self.update_count)

        loss = self._compute_loss(x, x_hat, mu, logvar, self.anneal)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("anneal", self.anneal)

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self) -> None:
        return self._on_shared_eval_epoch_end("val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "test")

    def on_test_epoch_end(self) -> None:
        return self._on_shared_eval_epoch_end("test")

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        x = batch
        x_hat, _, _ = self.model(x)

        x = x.cpu().numpy()
        x_hat = x_hat.cpu().numpy()

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
        x, target = batch
        x_hat, mu, logvar = self.model(x)

        loss = self._compute_loss(x, x_hat, mu, logvar, self.anneal)

        x = x.cpu().numpy()
        target = target.cpu().numpy()
        x_hat = x_hat.cpu().numpy()

        x_hat = self._remove_rated_item(x_hat, x)
        score = self._recall_at_k_(target, x_hat, 10)

        output = {"loss": loss, "score": score}
        self.shared_eval_step_outputs.append(output)
        return output

    def _on_shared_eval_epoch_end(self, prefix: str):
        outputs: List[Dict[torch.Tensor, np.ndarray]] = self.shared_eval_step_outputs

        avg_loss = torch.stack([output["loss"] for output in outputs]).mean()
        avg_score = np.concatenate([output["score"] for output in outputs]).mean()

        log_dict = {
            f"{prefix}_loss": avg_loss,
            f"{prefix}_recall@10": avg_score,
        }

        self.log_dict(log_dict)
        self.shared_eval_step_outputs.clear()
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

    def _recall_at_k_(self, target: np.ndarray, x_hat: np.ndarray, k: int) -> np.ndarray:
        target = target > 0

        ind = np.argpartition(x_hat, -k)[:, -k:]
        preds = np.zeros_like(x_hat, dtype=bool)
        preds[np.arange(len(x_hat))[:, None], ind[:, :k]] = True

        tmp = np.logical_and(target, preds).sum(axis=1)
        recall = tmp / np.minimum(k, target.sum(axis=1))
        return recall

    def _remove_rated_item(self, x_hat: np.ndarray, x: np.ndarray) -> np.ndarray:
        x_hat_copy = x_hat.copy()
        x_copy = x.copy()

        for user_id in range(x_copy.shape[0]):
            idx_rated: list = np.asarray(np.where(x_copy[user_id] != 0))[0].tolist()

            x_hat_copy[user_id, idx_rated] = 0

        return x_hat_copy
