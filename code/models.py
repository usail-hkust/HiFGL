from typing import List, Tuple, Dict, Any
import torch
from torch import nn
from torch.nn import functional as F


def get_module_dict(num_layers: int, feature_dim: int, hidden_dim: int, num_classes: int, dropout: float, *args, **kwargs):
    modules = {
        # 'encoder_raw': nn.Linear(feature_dim, hidden_dim),
        # 'encoder_fea': nn.Linear(hidden_dim * 2, hidden_dim),
        'aggregator_1': GCNAgg(feature_dim, hidden_dim, dropout=dropout, activation=True),
        'aggregator_2': GCNAgg(hidden_dim, num_classes, dropout=None, activation=False),
        # 'aggregator_1': GraphSageAgg(feature_dim, hidden_dim, dropout=None, activation=False),
        # 'aggregator_2': GraphSageAgg(hidden_dim, num_classes, dropout=None, activation=False),
        # 'classifier': nn.Sequential(
        #     nn.Linear(hidden_dim, 16),
        #     nn.Linear(16, num_classes)
        # ),
        # 'MLP': nn.Sequential(
        #     nn.Linear(feature_dim, hidden_dim),
        #     nn.Linear(hidden_dim, num_classes)
        # ),
        'loss_function': nn.CrossEntropyLoss()
    }
    for i in range(1, num_layers + 1):
        in_dim = feature_dim if i == 1 else hidden_dim
        out_dim = num_classes if i == num_layers else hidden_dim
        layer_dropout = None if i == num_layers else dropout
        layer_activation = False if i == num_layers else True
        modules[f'aggregator_{i}'] = GCNAgg(in_feature=in_dim, out_feature=out_dim, dropout=layer_dropout, activation=layer_activation)
    return nn.ModuleDict(modules)


class GCNAgg(nn.Module):

    def __init__(self, in_feature, out_feature, dropout: float = None, activation: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_feature, out_feature, bias=False)
        self.b = nn.Parameter(torch.Tensor(out_feature))
        nn.init.zeros_(self.b)
        self.dropout = dropout
        if self.dropout is not None:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.activation = activation
        if self.activation:
            self.activation_layer = nn.ReLU(inplace=True)

    def forward(self, x: List[torch.Tensor]):
        n = len(x)
        y = torch.stack(x, dim=0)
        y = y / ((n + 1) ** 0.5)
        y = self.linear(y)
        # y = torch.matmul(y, self.W)
        y = y.mean(dim=0)
        y = y + self.b
        if self.dropout:
            y = self.dropout_layer(y)
        if self.activation:
            y = self.activation_layer(y)
        return y.squeeze(0)


class GraphSageAgg(nn.Module):

    def __init__(self, in_feature, out_feature, dropout: float = None, activation: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_feature * 2, out_feature, bias=True)
        self.dropout = dropout
        if self.dropout is not None:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.activation = activation
        if self.activation:
            self.activation_layer = nn.ReLU(inplace=True)

    def forward(self, x: List[torch.Tensor]):
        h = x[0]
        y = torch.stack(x, dim=0)
        y = y.mean(dim=0)
        y = self.linear(torch.concat([h, y], dim=-1))
        y = F.softmax(y)
        if self.dropout:
            y = self.dropout_layer(y)
        if self.activation:
            y = self.activation_layer(y)
        return y.squeeze(0)
