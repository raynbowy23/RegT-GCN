import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv

from models.utils import TGCN

class TemporalGCN(nn.Module):
    def __init__(self, node_features, periods, output_dim):
        super(TemporalGCN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=256, 
                           periods=periods)
        # Equals single-shot prediction
        hidden_dim = 128
        self.output_dim = output_dim
        self.linear1 = nn.Linear(256, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index, edge_attr)
        out_hidden = h
        h = self.relu(h)
        h = self.linear1(h)
        h = self.relu(h)
        h = self.linear2(h)
        return h, out_hidden


class A3TGCN(nn.Module):
    r"""Based on `"A3T-GCN: Attention Temporal Graph Convolutional
    Network for Traffic Forecasting." <https://arxiv.org/abs/2006.11583>`_
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        periods: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True
    ):
        super(A3TGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            baseblock="gcn",
        )
        self.conv = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=2,
        )
        self.linear = torch.nn.Linear(64, self.out_channels)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))
        torch.nn.init.uniform_(self._attention)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        H_accum = 0
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        # h = H

        for period in range(self.periods):
            h = self.conv(X[:, :, period], edge_index, edge_weight)
            H_accum = H_accum + probs[period] * self._base_tgcn(
                X[:, :, period], edge_index, edge_weight, h)
        return H_accum
