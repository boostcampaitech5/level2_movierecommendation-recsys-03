import torch
import torch.nn as nn
import wandb
import lightning as L
from torch.optim import Optimizer
from torch.optim import Adam
from src.config import Config
from src import modules


class S3Rec(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.mask_id = config.data.mask_id
        self.aap_weight = config.trainer.aap_weight
        self.mip_weight = config.trainer.mip_weight
        self.map_weight = config.trainer.map_weight
        self.sp_weight = config.trainer.sp_weight

        self.base_module = modules.SASRec(config)
        self.s3rec = modules.S3Rec(config, self.base_module)

        self.training_step_outputs = []

        # add unique dense layer for 4 losses respectively
        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.base_module.init_weights)

    def compute_loss(self, aap_score, mip_distance, map_score, sp_distance, attrs, masked_item_seq):
        ## AAP
        aap_loss = self.criterion(aap_score, attrs.view(-1, self.s3rec.attr_size).float())
        # only compute loss at non-masked position
        aap_mask = (masked_item_seq != self.mask_id).float() * (masked_item_seq != 0).float()
        aap_loss = torch.sum(aap_loss * aap_mask.flatten().unsqueeze(-1))

        ## MIP
        mip_loss = self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32))
        mip_mask = (masked_item_seq == self.mask_id).float()
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())

        ## MAP
        map_loss = self.criterion(map_score, attrs.view(-1, self.s3rec.attr_size).float())
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

        joint_loss = self.aap_weight * aap_loss + self.mip_weight * mip_loss + self.map_weight * map_loss + self.sp_weight * sp_loss

        batch_size = len(attrs)
        self.training_step_outputs.append(
            {
                "avg_aap_loss": aap_loss.detach() / batch_size,
                "avg_mip_loss": mip_loss.detach() / batch_size,
                "avg_map_loss": map_loss.detach() / batch_size,
                "avg_sp_loss": sp_loss.detach() / batch_size,
                "avg_joint_loss": joint_loss.detach() / batch_size,
            }
        )

        return joint_loss

    def on_train_epoch_end(self) -> None:
        avg_aap_loss = torch.stack([x["avg_aap_loss"] for x in self.training_step_outputs]).mean()
        avg_mip_loss = torch.stack([x["avg_mip_loss"] for x in self.training_step_outputs]).mean()
        avg_map_loss = torch.stack([x["avg_map_loss"] for x in self.training_step_outputs]).mean()
        avg_sp_loss = torch.stack([x["avg_sp_loss"] for x in self.training_step_outputs]).mean()
        avg_joint_loss = torch.stack([x["avg_joint_loss"] for x in self.training_step_outputs]).mean()

        self.log("avg_aap_loss", avg_aap_loss)
        self.log("avg_mip_loss", avg_mip_loss)
        self.log("avg_map_loss", avg_map_loss)
        self.log("avg_sp_loss", avg_sp_loss)
        self.log("avg_joint_loss", avg_joint_loss)

        wandb.log(
            {
                "avg_aap_loss": avg_aap_loss,
                "avg_mip_loss": avg_mip_loss,
                "avg_map_loss": avg_map_loss,
                "avg_sp_loss": avg_sp_loss,
                "avg_joint_loss": avg_joint_loss,
            }
        )

        self.training_step_outputs.clear()

    def save_pretrained_module(self, path) -> None:
        self.base_module.save(path)

    def forward(
        self,
        masked_item_seq,
        pos_items,
        neg_items,
        masked_segment_seq,
        pos_segment,
        neg_segment,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.s3rec.forward(masked_item_seq, pos_items, neg_items, masked_segment_seq, pos_segment, neg_segment)
