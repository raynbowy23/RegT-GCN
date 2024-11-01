import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from models.utils import TGCN


class ConvStackedTemporalGCN(nn.Module):
    def __init__(self, node_features, periods, output_dim):
        super(ConvStackedTemporalGCN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = ConvStackedA3TGCN(in_channels=node_features, 
                           out_channels=512, 
                           periods=periods)
        # Equals single-shot prediction
        hidden_dim = 256
        self.linear1 = nn.Linear(512, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
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


class ConvStackedA3TGCN(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        periods: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True
    ):
        super(ConvStackedA3TGCN, self).__init__()

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
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        self.conv1 = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        self.conv2 = GCNConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        self.conv3 = GCNConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        self.conv4 = GCNConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        self.conv5 = GCNConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        self.linear = nn.Linear(512*5, 512)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._attention = nn.Parameter(torch.empty(self.periods, device=device))
        nn.init.uniform_(self._attention)


    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        H_accum = 0
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        for period in range(self.periods):
            h = self.conv1(X[:, :, period], edge_index, edge_weight)
            h = self.conv2(h, edge_index, edge_weight)
            h = self.conv3(h, edge_index, edge_weight)
            h = self.conv4(h, edge_index, edge_weight)
            h = self.conv5(h, edge_index, edge_weight)
            H_accum = H_accum + probs[period] * self._base_tgcn(
                X[:, :, period], edge_index, edge_weight, h
            )
        return H_accum