from typing import Optional, Any

import torch
from torch import Tensor
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)


def zeros(value: Any):
    constant(value, 0.)


def ones(tensor: Any):
    constant(tensor, 1.)


class GraphNorm(torch.nn.Module):
    r"""Based on 
    `"GraphNorm: A Principled Approach to Accelerating Graph Neural Network
    Training" <https://arxiv.org/abs/2009.03294>`_ paper

    """
    def __init__(self, in_channels: int, eps: float = 1e-5):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(in_channels))
        self.mean_scale = torch.nn.Parameter(torch.Tensor(in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)
        ones(self.mean_scale)


    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """"""
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        batch_size = int(batch.max()) + 1

        mean = scatter_mean(x, batch, dim=0, dim_size=batch_size)
        out = x - mean.index_select(0, batch) * self.mean_scale
        var = scatter_mean(out.pow(2), batch, dim=0, dim_size=batch_size)
        std = (var + self.eps).sqrt().index_select(0, batch)
        return self.weight * out / std + self.bias


    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels})'


class TGCN(torch.nn.Module):
    r"""Based on `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        baseblock: str = "gcn",
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
    ):
        super(TGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.baseblock = baseblock

        if self.baseblock == "gcn":
            self.BASEBLOCK = GCNConv
        elif self.baseblock == "gat":
            self.BASEBLOCK = GATConv
        elif self.baseblock == "graphsage":
            self.BASEBLOCK = SAGEConv
        else:
            raise NotImplementedError("Current baseblock %s is not supported." % (baseblock))

        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):

        if self.baseblock == "gcn":
            self.conv_z = self.BASEBLOCK(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                improved=self.improved,
                cached=self.cached,
                add_self_loops=self.add_self_loops,
            )
        else:
            self.conv_z = self.BASEBLOCK(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
            )


        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):

        if self.baseblock == "gcn":
            self.conv_r = self.BASEBLOCK(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                improved=self.improved,
                cached=self.cached,
                add_self_loops=self.add_self_loops,
            )
        else:
            self.conv_r = self.BASEBLOCK(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
            )

        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):

        if self.baseblock == "gcn":
            self.conv_h = self.BASEBLOCK(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                improved=self.improved,
                cached=self.cached,
                add_self_loops=self.add_self_loops,
            )
        else:
            self.conv_h = self.BASEBLOCK(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
            )
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis=1)
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], axis=1)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R], axis=1)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class TGCN_LSTM(torch.nn.Module):
    r"""Based on `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
    ):
        super(TGCN_LSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._create_parameters_and_layers()

    def _create_input_gate_parameters_and_layers(self):

        self.conv_i = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_i = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_forget_gate_parameters_and_layers(self):

        self.conv_f = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_f = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_cell_gate_parameters_and_layers(self):

        self.conv_g = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_g = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_output_gate_parameters_and_layers(self):

        self.conv_o = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_o = torch.nn.Linear(2 * self.out_channels, self.out_channels)
        
    def _create_parameters_and_layers(self):
        self._create_cell_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_input_gate_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_weight, H):
        I = torch.cat([self.conv_i(X, edge_index, edge_weight), H], axis=1)
        I = self.linear_i(I)
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H):
        F = torch.cat([self.conv_f(X, edge_index, edge_weight), H], axis=1)
        F = self.linear_f(F)
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_gate(self, X, edge_index, edge_weight, H):
        G = torch.cat([self.conv_g(X, edge_index, edge_weight), H], axis=1)
        G = self.linear_g(G)
        G = torch.tanh(G)
        return G

    def _calculate_output_gate(self, X, edge_index, edge_weight, H):
        O = torch.cat([self.conv_o(X, edge_index, edge_weight), H], axis=1)
        O = self.linear_o(O)
        O = torch.sigmoid(O)
        return O

    def _calculate_cell_state(self, F, C, I, G):
        C = F * C + I * G
        return C

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H)
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H)
        G = self._calculate_cell_gate(X, edge_index, edge_weight, H)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H)
        C = self._calculate_cell_state(F, C, I, G)
        H = self._calculate_hidden_state(O, C)
        return O, (H, C)