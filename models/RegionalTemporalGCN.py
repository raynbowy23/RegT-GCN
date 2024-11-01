## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

from models.utils import TGCN

class RegionalTemporalGCN(nn.Module):
    def __init__(self, node_features, num_nodes, periods, output_dim):
        super(RegionalTemporalGCN, self).__init__()
        # Attention Temporal Graph Convolutional Cell + Regional concat feature
        self.tgnn = RegionalA3TGCN(in_channels=node_features, 
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

    def forward(self, x, edge_index, IAedge_index, KSedge_index, KYedge_index, OHedge_index, WIedge_index,
                IAedge_attr, KSedge_attr, KYedge_attr, OHedge_attr, WIedge_attr):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index, IAedge_index, KSedge_index, KYedge_index, OHedge_index, WIedge_index,
                      IAedge_attr, KSedge_attr, KYedge_attr, OHedge_attr, WIedge_attr)
        ## Track hidden features
        out_hidden = h
        h = self.relu(h)
        h = self.linear1(h)
        h = self.relu(h)
        h = self.linear2(h)
        return h, out_hidden


class RegionalA3TGCN(nn.Module):
    r"""Based on `"A3T-GCN: Attention Temporal Graph Convolutional
    Network for Traffic Forecasting." <https://arxiv.org/abs/2006.11583>`_
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
        periods: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True
    ):
        super(RegionalA3TGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=2)
        self.linear = torch.nn.Linear(self.out_channels*5, self.out_channels)
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))

        self._weight_att1 = torch.nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(self.out_channels, 1), requires_grad=True))
        self._weight_att2 = torch.nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(self.num_nodes, 1), requires_grad=True))
        self._bias_att1 = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(1, 1), requires_grad=True))
        self._bias_att2 = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(1, 1), requires_grad=True))
        torch.nn.init.uniform_(self._attention)
        

    def attention(self, x, period) -> torch.FloatTensor:
        '''
        Attention module from original code implementation
        '''
        input_x = x
        x = torch.matmul(torch.reshape(x, [-1, self.out_channels]), self._weight_att1) + self._bias_att1 # [num_nodes, 1]

        f = torch.matmul(torch.reshape(x, [-1, self.num_nodes]), self._weight_att2) + self._bias_att2 # [1, 1]
        g = torch.matmul(torch.reshape(x, [-1, self.num_nodes]), self._weight_att2) + self._bias_att2
        h = torch.matmul(torch.reshape(x, [-1, self.num_nodes]), self._weight_att2) + self._bias_att2

        f1 = f.squeeze(0).expand(self.periods)
        h1 = h.squeeze(0).expand(self.periods)
        g1 = g.squeeze(0).expand(self.periods)
        s = g1 * f1

        beta = torch.nn.functional.softmax(s, dim=-1)

        context = beta[period] * input_x

        return context, beta


    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_IA_index: torch.LongTensor,
        edge_KS_index: torch.LongTensor,
        edge_KY_index: torch.LongTensor,
        edge_OH_index: torch.LongTensor,
        edge_WI_index: torch.LongTensor,
        edge_IA_weight: torch.FloatTensor,
        edge_KS_weight: torch.FloatTensor,
        edge_KY_weight: torch.FloatTensor,
        edge_OH_weight: torch.FloatTensor,
        edge_WI_weight: torch.FloatTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        H_accum = 0
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        for period in range(self.periods):
            IAh = self.conv(X[:, :, period], edge_IA_index, edge_IA_weight)
            KSh = self.conv(X[:, :, period], edge_KS_index, edge_KS_weight)
            KYh = self.conv(X[:, :, period], edge_KY_index, edge_KY_weight)
            OHh = self.conv(X[:, :, period], edge_OH_index, edge_OH_weight)
            WIh = self.conv(X[:, :, period], edge_WI_index, edge_WI_weight)
            h = torch.concat((IAh, KSh, KYh, OHh, WIh), dim=1)
            h = self.linear(h)
            h = F.leaky_relu(h)

            ## If using GRU
            H_accum = H_accum + probs[period] * self._base_tgcn(
                X[:, :, period], edge_index, edge_weight, h
            )
        return H_accum