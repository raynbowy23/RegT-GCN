import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import ChebConv


class SpatialGCN(nn.Module):
    def __init__(self, node_features, periods, output_dim):
        super(SpatialGCN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.gcn = ChebConv(
            in_channels=node_features,
            out_channels=64,
            K = 2,
        )
        self.gcn2 = ChebConv(
            in_channels=64,
            out_channels=256,
            K = 2,
        )
        self.periods = periods
        # Equals single-shot prediction
        hidden_dim = 128
        self.output_dim = output_dim
        self.linear1 = nn.Linear(256, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.output_dim)
        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU()

    def forward(self, x, edge_index, edge_attr):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """ 

        H_accum = 0
        for period in range(self.periods):
            g = self.gcn(x[:, :, period], edge_index, edge_attr)
            g = self.relu(g)
            g = F.dropout(g, training=self.training)
            g = self.gcn2(g, edge_index, edge_attr)
            H_accum = H_accum + g

        h = self.relu(H_accum)
        h = self.linear1(H_accum)
        h = self.relu(h)
        h = self.linear2(h)
        return h, H_accum