import torch.nn as nn


class StackedGRU(nn.Module):
    def __init__(self, in_channels, node_features, periods, output_dim):
        super(StackedGRU, self).__init__()
        hidden_dim = 256
        self.in_channels = in_channels
        self.output_dim = output_dim
        self.periods = periods
        self.node_features = node_features
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_dim)
        self.gru2 = nn.GRU(input_size=in_channels, hidden_size=hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

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