import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from models.utils import TGCN


class GraphSAGETemporalGCN(nn.Module):
    def __init__(self, node_features, num_nodes, periods, output_dim):
        super(GraphSAGETemporalGCN, self).__init__()
        self.tgnn = GraphSAGE(in_channels=node_features, 
                           out_channels=256, 
                           num_nodes=num_nodes,
                           periods=periods)
        self.output_dim = output_dim
        # Equals single-shot prediction
        hidden_dim = 128
        self.linear1 = nn.Linear(256, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.output_dim)
        self.relu = nn.ReLU()
        # self.norm = GraphNorm(in_channels=node_features)

    def forward(self, x, edge_index, edge_attr):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index, edge_attr)
        ## Track hidden features
        out_hidden = h
        h = self.relu(h)
        h = self.linear1(h)
        h = self.relu(h)
        h = self.linear2(h)
    
        return h, out_hidden


class GraphSAGE(nn.Module):
    r"""Based on `"Inductive Representation Learning on Large Graphs" (https://arxiv.org/abs/1706.02216)`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
        periods: int,
    ):
        super(GraphSAGE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.periods = periods
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            baseblock='graphsage',
        )

        self.conv = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
        )
 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._attention = nn.Parameter(torch.empty(self.periods, device=device))

        self._weight_att1 = nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(self.out_channels, 1), requires_grad=True))
        self._weight_att2 = nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(self.num_nodes, 1), requires_grad=True))
        self._bias_att1 = nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(1, 1), requires_grad=True))
        self._bias_att2 = nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(1, 1), requires_grad=True))
        nn.init.uniform_(self._attention)


    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        H_accum = 0
        probs = nn.functional.softmax(self._attention, dim=0)
        for period in range(self.periods):

            ## If using GRU
            H_accum = H_accum + probs[period] * self._base_tgcn(
                X[:, :, period], edge_index, H
            )
        return H_accum