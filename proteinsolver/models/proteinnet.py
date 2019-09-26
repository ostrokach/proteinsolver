import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops

from proteinsolver.nn import EdgeConvBatch, EdgeConvMod


def get_graph_conv_layer(input_size, hidden_size, output_size):
    mlp = nn.Sequential(
        #
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )
    gnn = EdgeConvMod(nn=mlp, aggr="add")
    graph_conv = EdgeConvBatch(gnn, output_size, batch_norm=True, dropout=0.2)
    return graph_conv


class ProteinNet(nn.Module):
    def __init__(self, x_input_size, adj_input_size, hidden_size, output_size):
        super().__init__()

        self.embed_x = nn.Sequential(
            nn.Embedding(x_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        if adj_input_size:
            self.embed_adj = nn.Sequential(
                nn.Linear(adj_input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
            )
        else:
            self.embed_adj = None

        self.graph_conv_1 = get_graph_conv_layer(
            (2 + bool(adj_input_size)) * hidden_size, 2 * hidden_size, hidden_size
        )
        self.graph_conv_2 = get_graph_conv_layer(3 * hidden_size, 2 * hidden_size, hidden_size)
        self.graph_conv_3 = get_graph_conv_layer(3 * hidden_size, 2 * hidden_size, hidden_size)
        self.graph_conv_4 = get_graph_conv_layer(3 * hidden_size, 2 * hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index, edge_attr=None):

        x = self.embed_x(x)
        edge_index, _ = remove_self_loops(edge_index)
        edge_attr = self.embed_adj(edge_attr) if edge_attr is not None else None

        x_out, edge_attr_out = self.graph_conv_1(x, edge_index, edge_attr)
        x += x_out
        edge_attr = (edge_attr + edge_attr_out) if edge_attr is not None else edge_attr_out

        x = F.relu(x)
        edge_attr = F.relu(edge_attr)
        x_out, edge_attr_out = self.graph_conv_2(x, edge_index, edge_attr)
        x += x_out
        edge_attr += edge_attr_out

        x = F.relu(x)
        edge_attr = F.relu(edge_attr)
        x_out, edge_attr_out = self.graph_conv_3(x, edge_index, edge_attr)
        x += x_out
        edge_attr += edge_attr_out

        x = F.relu(x)
        edge_attr = F.relu(edge_attr)
        x_out, edge_attr_out = self.graph_conv_4(x, edge_index, edge_attr)
        x += x_out
        edge_attr += edge_attr_out

        x = self.linear_out(x)
        return x
