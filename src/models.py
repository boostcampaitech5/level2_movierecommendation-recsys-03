import torch
import torch.nn as nn
import lightning as pl
from torch.optim import Adam
from src.config import Config
from src.modules import Encoder, LayerNorm
from torch.optim import Optimizer


class BaseModule(pl.LightningModule):
    def __init__(self, config: Config):
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

    def make_sequence_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def compute_loss(self):
        """
        Not essential
        """
        pass

    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        pass

    def forward(self):
        pass


class S3Rec(BaseModule):
    def __init__(self, config: Config):
        super().__init__(config)
        self.training_step_outputs = []

    # AAP
    def associated_attr_prediction(self, seq_output, attr_embedding) -> torch.Tensor:
        """
        :param seq_output: [B L H]
        :param attr_embedding: [arribute_num H]
        :return: scores [B*L tag_num]
        """
        seq_output = self.aap_norm(seq_output)  # [B L H]
        seq_output = seq_output.view([-1, self.args.hidden_size, 1])  # [B*L H 1]
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
        seq_output = self.mip_norm(seq_output.view([-1, self.args.hidden_size]))  # [B*L H]
        target_item = target_item.view([-1, self.args.hidden_size])  # [B*L H]
        score = torch.mul(seq_output, target_item)  # [B*L H]
        return torch.sigmoid(torch.sum(score, -1))  # [B*L]

    # MAP
    def masked_attr_prediction(self, seq_output, attr_embedding) -> torch.Tensor:
        seq_output = self.map_norm(seq_output)  # [B L H]
        seq_output = seq_output.view([-1, self.args.hidden_size, 1])  # [B*L H 1]
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
        # Encode masked sequence
        seq_emb = self.make_sequence_embedding(masked_item_seq)
        seq_mask = (masked_item_seq == 0).float() * -1e8
        seq_mask = torch.unsqueeze(torch.unsqueeze(seq_mask, 1), 1)

        encoded_layers = self.item_encoder(seq_emb, seq_mask, output_all_encoded_layers=True)
        # [B L H]
        seq_output = encoded_layers[-1]

        attr_embeddings = self.attribute_embeddings.weight
        # AAP
        aap_score = self.associated_attr_prediction(seq_output, attr_embeddings)

        # MIP
        pos_item_embs = self.item_embeddings(pos_items)
        neg_item_embs = self.item_embeddings(neg_items)
        pos_score = self.masked_item_prediction(seq_output, pos_item_embs)
        neg_score = self.masked_item_prediction(seq_output, neg_item_embs)
        mip_distance = torch.sigmoid(pos_score - neg_score)

        # MAP
        map_score = self.masked_attribute_prediction(seq_output, attr_embeddings)

        # SP
        # segment context
        segment_context = self.add_position_embedding(masked_segment_seq)
        segment_mask = (masked_segment_seq == 0).float() * -1e8
        segment_mask = torch.unsqueeze(torch.unsqueeze(segment_mask, 1), 1)
        segment_encoded_layers = self.item_encoder(segment_context, segment_mask, output_all_encoded_layers=True)

        # take the last position hidden as the context
        segment_context = segment_encoded_layers[-1][:, -1, :]  # [B H]
        # pos_segment
        pos_segment_emb = self.add_position_embedding(pos_segment)
        pos_segment_mask = (pos_segment == 0).float() * -1e8
        pos_segment_mask = torch.unsqueeze(torch.unsqueeze(pos_segment_mask, 1), 1)
        pos_segment_encoded_layers = self.item_encoder(pos_segment_emb, pos_segment_mask, output_all_encoded_layers=True)
        pos_segment_emb = pos_segment_encoded_layers[-1][:, -1, :]

        # neg_segment
        neg_segment_emb = self.add_position_embedding(neg_segment)
        neg_segment_mask = (neg_segment == 0).float() * -1e8
        neg_segment_mask = torch.unsqueeze(torch.unsqueeze(neg_segment_mask, 1), 1)
        neg_segment_encoded_layers = self.item_encoder(neg_segment_emb, neg_segment_mask, output_all_encoded_layers=True)
        neg_segment_emb = neg_segment_encoded_layers[-1][:, -1, :]  # [B H]

        pos_segment_score = self.segment_prediction(segment_context, pos_segment_emb)
        neg_segment_score = self.segment_prediction(segment_context, neg_segment_emb)

        sp_distance = torch.sigmoid(pos_segment_score - neg_segment_score)

        return aap_score, mip_distance, map_score, sp_distance

    def compute_loss(self, aap_score, mip_distance, map_score, sp_distance, attrs, masked_item_seq):
        ## AAP
        aap_loss = self.criterion(aap_score, attrs.view(-1, self.args.attribute_size).float())
        # only compute loss at non-masked position
        aap_mask = (masked_item_seq != self.args.mask_id).float() * (masked_item_seq != 0).float()
        aap_loss = torch.sum(aap_loss * aap_mask.flatten().unsqueeze(-1))

        ## MIP
        mip_loss = self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32))
        mip_mask = (masked_item_seq == self.args.mask_id).float()
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())

        ## MAP
        map_loss = self.criterion(map_score, attrs.view(-1, self.args.attribute_size).float())
        map_mask = (masked_item_seq == self.args.mask_id).float()
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
        aap_score, mip_distance, map_score, sp_distance, attrs, masked_item_seq = self.forward(
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

        self.training_step_outputs.append({"aap_loss": aap_loss, "mip_loss": mip_loss, "map_loss": map_loss, "sp_loss": sp_loss})

        return joint_loss

    def on_train_epoch_end(self) -> None:
        avg_aap_loss = (torch.stack([x["aap_loss"] for x in self.training_step_outputs]).mean(),)
        avg_mip_loss = (torch.stack([x["mip_loss"] for x in self.training_step_outputs]).mean(),)
        avg_map_loss = (torch.stack([x["map_loss"] for x in self.training_step_outputs]).mean(),)
        avg_sp_loss = torch.stack([x["sp_loss"] for x in self.training_step_outputs]).mean()

        self.log("avg_aap_loss", avg_aap_loss)
        self.log("avg_mip_loss", avg_mip_loss)
        self.log("avg_map_loss", avg_map_loss)
        self.log("avg_sp_loss", avg_sp_loss)

        self.training_step_outputs.clear()
