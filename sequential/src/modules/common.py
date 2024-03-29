import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
    (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Embeddings(nn.Module):
    """Construct the embeddings from item, position."""

    def __init__(self, config: Config):
        super(Embeddings, self).__init__()

        self.item_size = config.data.item_size
        self.max_seq_length = config.data.max_seq_length
        self.hidden_size = config.model.hidden_size
        self.hidden_dropout_prob = config.model.hidden_dropout_prob

        self.item_embeddings = nn.Embedding(self.item_size, self.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = items_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, config: Config):
        super(SelfAttention, self).__init__()
        if config.model.hidden_size % config.model.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.model.hidden_size, config.model.num_attention_heads)
            )

        self.hidden_size = config.model.hidden_size
        self.hidden_dropout_prob = config.model.hidden_dropout_prob
        self.attention_probs_dropout_prob = config.model.attention_probs_dropout_prob

        self.num_attention_heads = config.model.num_attention_heads
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(self.attention_probs_dropout_prob)

        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(self.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        # attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # same attn_dist
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        # same output
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, config: Config):
        super(Intermediate, self).__init__()

        self.hidden_size = config.model.hidden_size
        self.hidden_act = config.model.hidden_act
        self.hidden_dropout_prob = config.model.hidden_dropout_prob

        self.dense_1 = nn.Linear(self.hidden_size, self.hidden_size * 4)
        if isinstance(self.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[self.hidden_act]
        else:
            self.intermediate_act_fn = self.hidden_act

        self.dense_2 = nn.Linear(self.hidden_size * 4, self.hidden_size)
        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):
    def __init__(self, config: Config):
        super(Layer, self).__init__()
        self.attention = SelfAttention(config)
        self.intermediate = Intermediate(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super(Encoder, self).__init__()
        layer = Layer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.model.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)

        return hidden_states


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config: Config):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_size = config.model.hidden_size
        self.dropout = nn.Dropout(config.model.hidden_dropout_prob)  # dropout rate

    def forward(self, Q, K, V, mask):
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.hidden_size)
        attn_score = attn_score.masked_fill(mask == 0, -1e9)  # 유사도가 0인 지점은 -infinity로 보내 softmax 결과가 0이 되도록 함
        attn_dist = self.dropout(F.softmax(attn_score, dim=-1))  # attention distribution
        output = torch.matmul(attn_dist, V)  # dim of output : batchSize x num_head x seqLen x hidden_units
        return output, attn_dist


class MultiHeadAttention(nn.Module):
    def __init__(self, config: Config):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = config.model.num_attention_heads
        self.hidden_size = config.model.hidden_size
        self.dropout_rate = config.model.hidden_dropout_prob
        # query, key, value, output 생성을 위해 Linear 모델 생성
        self.W_Q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_K = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_O = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.attention = ScaledDotProductAttention(config)  # scaled dot product attention module을 사용하여 attention 계산
        self.dropout = nn.Dropout(self.dropout_rate)  # dropout rate
        self.layerNorm = nn.LayerNorm(self.hidden_size, 1e-6)  # layer normalization

    def forward(self, enc, mask):
        residual = enc  # residual connection을 위해 residual 부분을 저장
        batch_size, seqlen = enc.size(0), enc.size(1)

        # Query, Key, Value를 (num_head)개의 Head로 나누어 각기 다른 Linear projection을 통과시킴
        Q = self.W_Q(enc).view(batch_size, seqlen, self.num_heads, self.hidden_size // self.num_heads)
        K = self.W_K(enc).view(batch_size, seqlen, self.num_heads, self.hidden_size // self.num_heads)
        V = self.W_V(enc).view(batch_size, seqlen, self.num_heads, self.hidden_size // self.num_heads)

        # Head별로 각기 다른 attention이 가능하도록 Transpose 후 각각 attention에 통과시킴
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        output, attn_dist = self.attention(Q, K, V, mask)

        # 다시 Transpose한 후 모든 head들의 attention 결과를 합칩니다.
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seqlen, -1)

        # Linear Projection, Dropout, Residual sum, and Layer Normalization
        output = self.layerNorm(self.dropout(self.W_O(output)) + residual)
        return output, attn_dist


class PositionwiseFeedForward(nn.Module):
    def __init__(self, config: Config):
        super(PositionwiseFeedForward, self).__init__()
        self.hidden_size = config.model.hidden_size
        self.dropout_rate = config.model.hidden_dropout_prob
        # SASRec과의 dimension 차이가 있습니다.
        self.W_1 = nn.Linear(self.hidden_size, 4 * self.hidden_size)
        self.W_2 = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layerNorm = nn.LayerNorm(self.hidden_size, 1e-6)  # layer normalization

    def forward(self, x):
        residual = x
        output = self.W_2(F.gelu(self.dropout(self.W_1(x))))  # activation: relu -> gelu
        output = self.layerNorm(self.dropout(output) + residual)
        return output


class BERT4RecBlock(nn.Module):
    def __init__(self, config: Config):
        super(BERT4RecBlock, self).__init__()
        self.num_heads = config.model.num_attention_heads
        self.hidden_size = config.model.hidden_size
        self.dropout_rate = config.model.hidden_dropout_prob
        self.attention = MultiHeadAttention(config)
        self.pointwise_feedforward = PositionwiseFeedForward(config)

    def forward(self, input_enc, mask):
        output_enc, attn_dist = self.attention(input_enc, mask)
        output_enc = self.pointwise_feedforward(output_enc)
        return output_enc, attn_dist


class BERT_Encoder(nn.Module):
    def __init__(self, config: Config):
        super(BERT_Encoder, self).__init__()
        self.blocks = nn.ModuleList([BERT4RecBlock(config) for _ in range(config.model.num_hidden_layers)])

    def forward(self, hidden_states, mask, output_all_encoded_layers=False):
        for block in self.blocks:
            hidden_states, attn_dist = block(hidden_states, mask)

        return hidden_states
