import torch
from torch import nn


class SimpleLSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, emb_size):
        super(SimpleLSTM, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.mlp = nn.Linear(hidden_dim, emb_size)

    def forward(self, x):
        _, (hn, cn) = self.lstm(x)
        # out = out[:, -1, :]
        # out = self.mlp(out)
        # print(hn.size)
        # print(hn.shape)
        last_layer_out = hn.permute(1, 0, 2)[:, -1, :].squeeze()
        output = self.mlp(last_layer_out)
        return output
