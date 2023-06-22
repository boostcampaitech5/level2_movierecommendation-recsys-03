import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act: torch.Tensor = ACT2FN[config.model.act]
        self.p_dims: list = config.model.p_dims  # [200, 600, 6807]
        self.q_dims: list = self.p_dims[::-1]  # [6807, 600, 200]

        # encoder
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]  # [6807, 600, 200*2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        # decoder
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        self.dropout = nn.Dropout(config.model.dropout)
        self.init_weights()

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar

    def encode(self, input):
        h = F.normalize(input)
        h = self.dropout(h)
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = self.act(h)
            else:
                mu = h[:, : self.q_dims[-1]]  # h [B, 1, 200] h[:,:,:200]
                logvar = h[:, self.q_dims[-1] :]  # h [B, 1, 200]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = self.act(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            nn.init.xavier_normal_(layer.weight.data)
            nn.init.normal_(layer.bias.data, 0.0, 0.001)

        for layer in self.p_layers:
            nn.init.xavier_normal_(layer.weight.data)
            nn.init.normal_(layer.bias.data, 0.0, 0.001)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"tanh": F.tanh, "gelu": gelu, "relu": F.relu, "swish": swish, "leakyrelu": F.leaky_relu}
