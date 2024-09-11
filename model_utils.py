from typing import Optional, Any

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GATConv
from torch_geometric.nn.inits import glorot


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


class A3TGCN(torch.nn.Module):
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


class RegionalA3TGCN(torch.nn.Module):
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


class ConvStackedA3TGCN(torch.nn.Module):

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
        self.linear = torch.nn.Linear(512*5, 512)
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


class GConvLSTM(torch.nn.Module):
    r"""Based on `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(GConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):

        self.conv_x_i = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_i = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):

        self.conv_x_f = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_f = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):

        self.conv_x_c = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_c = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):

        self.conv_x_o = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_o = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        I = self.conv_x_i(X, edge_index, edge_weight, lambda_max=lambda_max)
        I = I + self.conv_h_i(H, edge_index, edge_weight, lambda_max=lambda_max)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        F = self.conv_x_f(X, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + self.conv_h_f(H, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F, lambda_max):
        T = self.conv_x_c(X, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.conv_h_c(H, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        O = self.conv_x_o(X, edge_index, edge_weight, lambda_max=lambda_max)
        O = O + self.conv_h_o(H, edge_index, edge_weight, lambda_max=lambda_max)
        O = O + (self.w_c_o * C)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

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
        lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:

        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H, C, lambda_max)
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C, lambda_max)
        C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F, lambda_max)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H, C, lambda_max)
        H = self._calculate_hidden_state(O, C)
        return H, C


class GraphSAGE(torch.nn.Module):
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
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))

        self._weight_att1 = torch.nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(self.out_channels, 1), requires_grad=True))
        self._weight_att2 = torch.nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(self.num_nodes, 1), requires_grad=True))
        self._bias_att1 = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(1, 1), requires_grad=True))
        self._bias_att2 = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(1, 1), requires_grad=True))
        torch.nn.init.uniform_(self._attention)


    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        H_accum = 0
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        for period in range(self.periods):

            ## If using GRU
            H_accum = H_accum + probs[period] * self._base_tgcn(
                X[:, :, period], edge_index, H
            )
        return H_accum



class GAT(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
        periods: int,
    ):
        super(GAT, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.periods = periods
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            baseblock='gat',
        )
        self.relu = torch.nn.ReLU()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))

        torch.nn.init.uniform_(self._attention)


    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        H_accum = 0
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        for period in range(self.periods):
            ## If using GRU
            H_accum = H_accum + probs[period] * self._base_tgcn(
                X[:, :, period], edge_index, H
            )
        return H_accum
    