import torch
import torch.nn as nn
import lightning as pl
from configs.config import config
from modules import Encoder, LayerNorm


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

    def add_position_embedding(self, sequence):
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
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_train_epoch_end(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError
