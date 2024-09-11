## Standard libraries

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Torch geometric
from torch_geometric.nn import ChebConv
from model_utils import A3TGCN, RegionalA3TGCN, ConvStackedA3TGCN, GConvLSTM, GraphSAGE, GAT
    

class CustomGConvLSTM(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        periods: int,
    ):
        super(CustomGConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self._setup_layers()

    def _setup_layers(self):
        self._base_gcvlstm = GConvLSTM(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K = 2, # Chebyshev filter size
        )

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        H_accum = 0
        for period in range(self.periods):
            H_accum = H_accum + self._base_gcvlstm(
                X[:, :, period], edge_index, edge_weight, H
            )[0]
        return H_accum


class CustomStackedLSTM(torch.nn.Module):
    """
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        periods (int): Number of time periods.
        improved (bool): Stronger self loops (default :obj:`False`).
        cached (bool): Caching the message weights (default :obj:`False`).
        add_self_loops (bool): Adding self-loops for smoothing (default :obj:`True`).
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
        super(CustomStackedLSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self._setup_layers()

    def _setup_layers(self):
        self.lstm = nn.LSTM(in_size=self.in_channels, hidden_size=self.out_channels)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        H_accum = 0
        for period in range(self.periods):
            H_accum = H_accum * self.lstm (
                X[:, :, period], H
            )
        return H_accum


class StackedGRU(torch.nn.Module):
    def __init__(self, in_channels, node_features, periods, output_dim):
        super(StackedGRU, self).__init__()
        hidden_dim = 256
        self.in_channels = in_channels
        self.output_dim = output_dim
        self.periods = periods
        self.node_features = node_features
        self.gru = torch.nn.GRU(input_size=in_channels, hidden_size=hidden_dim)
        self.gru2 = torch.nn.GRU(input_size=in_channels, hidden_size=hidden_dim)
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, self.output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        H_accum = 0
        out, h = self.gru(x)
        out = self.relu(out)
        out, h = self.gru2(x, h)
        h = self.linear1(out)
        h = self.relu(h)
        h = self.linear2(h)
        return h
        

class SpatialGCN(torch.nn.Module):
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
        self.linear1 = torch.nn.Linear(256, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, self.output_dim)
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


class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, output_dim):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=256, 
                           periods=periods)
        # Equals single-shot prediction
        hidden_dim = 128
        self.output_dim = output_dim
        self.linear1 = torch.nn.Linear(256, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, self.output_dim)
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


class ConvStackedTemporalGCN(torch.nn.Module):
    def __init__(self, node_features, periods, output_dim):
        super(ConvStackedTemporalGCN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = ConvStackedA3TGCN(in_channels=node_features, 
                           out_channels=512, 
                           periods=periods)
        # Equals single-shot prediction
        hidden_dim = 256
        self.linear1 = torch.nn.Linear(512, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)
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


class RegionalTemporalGCN(torch.nn.Module):
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
        self.linear1 = torch.nn.Linear(256, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, self.output_dim)
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


class TemporalGConvLSTM(torch.nn.Module):
    def __init__(self, node_features, periods, output_dim):
        super(TemporalGConvLSTM, self).__init__()
        self.gconvlstm = CustomGConvLSTM(in_channels=node_features, 
                           out_channels=256, periods=periods)

        self.output_dim = output_dim
        # Equals single-shot prediction
        hidden_dim = 128
        self.linear1 = nn.Linear(256, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        edge_weight = Graph edge weights
        """
        h = self.gconvlstm(x, edge_index, edge_attr)
        out_hidden = h
        h = self.relu(h)
        h = self.linear1(h)
        h = self.relu(h)
        h = self.linear2(h)
        return h, out_hidden


class RecurrentGCN(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(RecurrentGCN, self).__init__()
        self.lstm1 = GConvLSTM(input_size, hidden_size, 2)
        # GraphConv
        # LSTM with a few layers
        self.hidden_size = hidden_size

        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)

        self.out_linear = nn.Linear(hidden_size, out_size)

    def forward(self, input, Hidden_State, Cell_State):
        combined = torch.cat((input, Hidden_State), 1)
        f = torch.sigmoid(self.fl(combined))
        i = torch.sigmoid(self.il(combined))
        o = torch.sigmoid(self.ol(combined))
        C = torch.tanh(self.Cl(combined))
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * torch.tanh(Cell_State)

        return Hidden_State, Cell_State

    def loop(self, inputs, x, edge_index, edge_weight, time_window, device, hidden, cell):
        batch_size = inputs.size(0)
        input_size = inputs.size(1)
        hidden = Variable(torch.zeros(batch_size, self.hidden_size).to(device), requires_grad=True)
        cell = Variable(torch.zeros(batch_size, self.hidden_size).to(device), requires_grad=True)
        hidden, cell = self.forward(inputs, edge_index, edge_weight, hidden, cell)

        out = self.out_linear(hidden)
        return out, hidden, cell


class GraphSAGETemporalGCN(torch.nn.Module):
    def __init__(self, node_features, num_nodes, periods, output_dim):
        super(GraphSAGETemporalGCN, self).__init__()
        self.tgnn = GraphSAGE(in_channels=node_features, 
                           out_channels=256, 
                           num_nodes=num_nodes,
                           periods=periods)
        self.output_dim = output_dim
        # Equals single-shot prediction
        hidden_dim = 128
        self.linear1 = torch.nn.Linear(256, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, self.output_dim)
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


class GATTemporal(torch.nn.Module):
    def __init__(self, node_features, num_nodes, periods, output_dim):
        super(GATTemporal, self).__init__()
        self.gat = GAT(in_channels=node_features, 
                           out_channels=256, 
                           num_nodes=num_nodes,
                           periods=periods)
        self.output_dim = output_dim
        # Equals single-shot prediction
        hidden_dim = 128
        self.linear1 = torch.nn.Linear(256, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, self.output_dim)
        self.relu = nn.ReLU()
        # self.norm = GraphNorm(in_channels=node_features)

    def forward(self, x, edge_index, edge_attr):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.gat(x, edge_index, edge_attr)
        ## Track hidden features
        out_hidden = h
        h = self.relu(h)
        h = self.linear1(h)
        h = self.relu(h)
        h = self.linear2(h)
        return h, out_hidden