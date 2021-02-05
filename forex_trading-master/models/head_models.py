import torch
from torch import nn


class MLP_Classifier(nn.Module):
    def __init__(self, in_dim, n_class):
        super(MLP_Classifier, self).__init__()
        self.in_dim = in_dim
        self.n_class = n_class
        self.mlp = nn.Linear(in_dim, n_class)

    def forward(self, x):
        out = self.mlp(x)
        return out


class MLP_Regressor(nn.Module):
    def __init__(self, in_dim):
        super(MLP_Regressor, self).__init__()
        self.in_dim = in_dim
        self.n_class = 1
        self.mlp = nn.Linear(in_dim, 1)

    def forward(self, x):
        out = self.mlp(x)
        return out