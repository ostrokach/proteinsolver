import torch
import torch.nn as nn
from torch_geometric.nn.inits import reset
from torch_geometric.utils import scatter_


class EdgeConvMod(torch.nn.Module):
    def __init__(self, nn, aggr="max"):
        super().__init__()
        self.nn = nn
        self.aggr = aggr
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x, edge_index, edge_attr=None):
        """"""
        row, col = edge_index
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        # TODO: Try -x[col] instead of x[col] - x[row]
        if edge_attr is None:
            out = torch.cat([x[row], x[col]], dim=-1)
        else:
            out = torch.cat([x[row], x[col], edge_attr], dim=-1)
        out = self.nn(out)
        x = scatter_(self.aggr, out, row, dim_size=x.size(0))

        return x, out

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)


class EdgeConvBatch(nn.Module):
    def __init__(self, gnn, hidden_size, batch_norm=True, dropout=0.2):
        super().__init__()

        self.gnn = gnn

        x_post_modules = []
        edge_attr_post_modules = []

        if batch_norm is not None:
            x_post_modules.append(nn.LayerNorm(hidden_size))
            edge_attr_post_modules.append(nn.LayerNorm(hidden_size))

        if dropout:
            x_post_modules.append(nn.Dropout(dropout))
            edge_attr_post_modules.append(nn.Dropout(dropout))

        self.x_postprocess = nn.Sequential(*x_post_modules)
        self.edge_attr_postprocess = nn.Sequential(*edge_attr_post_modules)

    def forward(self, x, edge_index, edge_attr=None):
        x, edge_attr = self.gnn(x, edge_index, edge_attr)
        x = self.x_postprocess(x)
        edge_attr = self.edge_attr_postprocess(edge_attr)
        return x, edge_attr
