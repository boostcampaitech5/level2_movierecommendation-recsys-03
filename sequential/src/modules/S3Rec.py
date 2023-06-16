import torch
import torch.nn as nn
from src.config import Config
from typing import Mapping


class AttributePrediction(nn.Module):
    def __init__(self, attr_size: int, hidden_size: int):
        super().__init__()
        self.attr_size = attr_size
        self.hidden_size = hidden_size

        self.attr_embeddings = nn.Embedding(attr_size, hidden_size, padding_idx=0)
        self.aap_norm = nn.Linear(hidden_size, hidden_size)
        self.map_norm = nn.Linear(hidden_size, hidden_size)

    # AAP
    def associated_attr_prediction(self, seq_output, attr_embedding) -> torch.Tensor:
        """
        :param seq_output: [B L H]
        :param attr_embedding: [arribute_num H]
        :return: scores [B*L tag_num]
        """
        seq_output = self.aap_norm(seq_output)  # [B L H]
        seq_output = seq_output.view([-1, self.hidden_size, 1])  # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attr_embedding, seq_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L tag_num]

    # MAP
    def masked_attr_prediction(self, seq_output, attr_embedding) -> torch.Tensor:
        seq_output = self.map_norm(seq_output)  # [B L H]
        seq_output = seq_output.view([-1, self.hidden_size, 1])  # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attr_embedding, seq_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L tag_num]

    def forward(self, seq_output) -> torch.Tensor:
        attr_embeddings = self.attr_embeddings.weight

        # AAP
        aap_score = self.associated_attr_prediction(seq_output, attr_embeddings)
        # MAP
        map_score = self.masked_attr_prediction(seq_output, attr_embeddings)

        return aap_score, map_score


class S3Rec(nn.Module):
    def __init__(self, config: Config, base_module: nn.Module, attr_pred_modules: Mapping[str, AttributePrediction] | nn.ModuleDict):
        super().__init__()
        self.hidden_size = config.model.hidden_size

        self.base_module = base_module
        self.attr_pred_modules = attr_pred_modules

        self.mip_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.sp_norm = nn.Linear(self.hidden_size, self.hidden_size)

    # MIP sample neg items
    def masked_item_prediction(self, seq_output, target_item) -> torch.Tensor:
        """
        :param seq_output: [B L H]
        :param target_item: [B L H]
        :return: scores [B*L]
        """
        seq_output = self.mip_norm(seq_output.view([-1, self.hidden_size]))  # [B*L H]
        target_item = target_item.view([-1, self.hidden_size])  # [B*L H]
        score = torch.mul(seq_output, target_item)  # [B*L H]
        return torch.sigmoid(torch.sum(score, -1))  # [B*L]

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
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        # Encode masked seq
        seq_emb = self.base_module.make_seq_embedding(masked_item_seq)
        seq_mask = (masked_item_seq == 0).float() * -1e8
        seq_mask = torch.unsqueeze(torch.unsqueeze(seq_mask, 1), 1)

        encoded_layers = self.base_module.item_encoder(seq_emb, seq_mask, output_all_encoded_layers=True)
        # [B L H]
        seq_output = encoded_layers[-1]

        # AAP + MAP
        aap_scores, map_scores = {}, {}

        for attr_name, attr_pred_module in self.attr_pred_modules.items():
            aap_score, map_score = attr_pred_module.forward(seq_output)
            aap_scores[attr_name] = aap_score
            map_scores[attr_name] = map_score

        # MIP
        pos_item_embs = self.base_module.item_embeddings(pos_items)
        neg_item_embs = self.base_module.item_embeddings(neg_items)
        pos_score = self.masked_item_prediction(seq_output, pos_item_embs)
        neg_score = self.masked_item_prediction(seq_output, neg_item_embs)
        mip_distance = torch.sigmoid(pos_score - neg_score)

        # SP
        # segment context
        segment_context = self.base_module.make_seq_embedding(masked_segment_seq)
        segment_mask = (masked_segment_seq == 0).float() * -1e8
        segment_mask = torch.unsqueeze(torch.unsqueeze(segment_mask, 1), 1)
        segment_encoded_layers = self.base_module.item_encoder(segment_context, segment_mask, output_all_encoded_layers=True)

        # take the last position hidden as the context
        segment_context = segment_encoded_layers[-1][:, -1, :]  # [B H]
        # pos_segment
        pos_segment_emb = self.base_module.make_seq_embedding(pos_segment)
        pos_segment_mask = (pos_segment == 0).float() * -1e8
        pos_segment_mask = torch.unsqueeze(torch.unsqueeze(pos_segment_mask, 1), 1)
        pos_segment_encoded_layers = self.base_module.item_encoder(pos_segment_emb, pos_segment_mask, output_all_encoded_layers=True)
        pos_segment_emb = pos_segment_encoded_layers[-1][:, -1, :]

        # neg_segment
        neg_segment_emb = self.base_module.make_seq_embedding(neg_segment)
        neg_segment_mask = (neg_segment == 0).float() * -1e8
        neg_segment_mask = torch.unsqueeze(torch.unsqueeze(neg_segment_mask, 1), 1)
        neg_segment_encoded_layers = self.base_module.item_encoder(neg_segment_emb, neg_segment_mask, output_all_encoded_layers=True)
        neg_segment_emb = neg_segment_encoded_layers[-1][:, -1, :]  # [B H]

        pos_segment_score = self.segment_prediction(segment_context, pos_segment_emb)
        neg_segment_score = self.segment_prediction(segment_context, neg_segment_emb)

        sp_distance = torch.sigmoid(pos_segment_score - neg_segment_score)

        return aap_scores, map_scores, mip_distance, sp_distance
