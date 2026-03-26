"""GNN model definition and training/evaluation helpers."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GraphConv, GINConv
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import scipy.sparse as sp

from src.GangPrediction.pattern_models import Pattern
from src.ml.metrics.metrics import average_precision_score
from src.GangPrediction.experiment_utils import (
    _capture_pattern_lineage,
    get_node_to_supernode_mapping,
)
from src.GangPrediction.embedding_diagnostics import (
    compute_embedding_and_basis_diagnostics,
)
from src.GangPrediction.utils.utils import *


def build_gnn_model(
    nfeat, nhid, nclass, num_layers, use_edge_weights=False, GNN_type="GIN"
):
    """Factory function to create a GNN model with the specified architecture."""
    if GNN_type == "GIN":
        return Build_GINConv(nfeat, nhid, nclass, num_layers, use_edge_weights)
    elif GNN_type == "DGCN":
        return Build_DGCN(nfeat, nhid, nclass, num_layers)
    elif GNN_type == "GCN":
        conv_layer = GCNConv
    elif GNN_type == "SAGE":
        conv_layer = SAGEConv
    elif GNN_type == "GAT":
        conv_layer = GATConv
    elif GNN_type == "GraphConv":
        conv_layer = GraphConv
    else:
        raise ValueError(f"Unsupported GNN type: {GNN_type}")

    convs = nn.ModuleList()
    convs.append(conv_layer(nfeat, nhid))  # First layer
    for _ in range(num_layers - 2):  # Hidden layers
        convs.append(conv_layer(nhid, nhid))
    convs.append(conv_layer(nhid, nclass))  # Output layer

    return convs


def Build_DGCN(nfeat, nhid, nclass, num_layers):
    """Factory function to create a DGCN model."""
    convs = nn.ModuleList()
    convs.append(DGCNConv(nfeat, nhid, num_layers=num_layers))  # First layer
    convs.append(nn.Linear(nhid, nclass))  # Output layer
    return convs


def Build_GINConv(nfeat, nhid, nclass, num_layers, use_edge_weights=False):
    """Factory function to create a GINConv layer with the given MLP."""
    if use_edge_weights:
        conv_layer = WeightedGINConv
    else:
        conv_layer = GINConv
    convs = nn.ModuleList()
    # GINConv requires an MLP as input
    convs.append(
        conv_layer(
            nn.Sequential(
                nn.Linear(nfeat, nhid),
                nn.ReLU(),
                nn.Linear(nhid, nhid),
            ),
        )
    )  # First layer
    for _ in range(num_layers - 2):  # Hidden layers
        convs.append(
            conv_layer(
                nn.Sequential(
                    nn.Linear(nhid, nhid),
                    nn.ReLU(),
                    nn.Linear(nhid, nhid),
                )
            )
        )
    convs.append(
        conv_layer(
            nn.Sequential(
                nn.Linear(nhid, nhid),
                nn.ReLU(),
                nn.Linear(nhid, nclass),
            ),
        )
    )  # Output layer

    return convs


class WeightedGINConv(MessagePassing):
    def __init__(self, nn_mlp, eps=0.0, train_eps=False):
        super().__init__(aggr="add")
        self.nn = nn_mlp
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.tensor([eps], dtype=torch.float))
        else:
            self.register_buffer("eps", torch.tensor([eps], dtype=torch.float))

    def forward(self, x, edge_index, edge_weight=None):
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = out + (1 + self.eps) * x
        return self.nn(out)

    def message(self, x_j, edge_weight):
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j


class DGCNConv(nn.Module):
    def __init__(self, n_in, n_out, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.nn = nn.Linear(n_in, n_out)

    def forward(self, x, edge_index, edge_weight=None):
        abar = calc_abar(
            edge_index,
            edge_weight=edge_weight,
            num_nodes=x.size(0),
            num_layers=self.num_layers,
        ).to(x.device)
        out = torch.sparse.mm(abar, x) + x  # Adding self-loop contribution
        return self.nn(out)


# ---------------------------------------------------------------------------
# IGNN (Implicit Graph Neural Network) — Gu et al., NeurIPS 2020
# ---------------------------------------------------------------------------


def _projection_norm_inf(A, kappa=0.99):
    """Project weight matrix so that each row's L1-norm <= kappa."""
    A_np = A.clone().detach().cpu().numpy()
    row_norms = np.abs(A_np).sum(axis=-1)
    for idx in np.where(row_norms > kappa)[0]:
        a_orig = A_np[idx, :]
        a_sign = np.sign(a_orig)
        a_abs = np.abs(a_orig)
        a_sorted = np.sort(a_abs)

        s = np.sum(a_sorted) - kappa
        l = float(len(a_sorted))
        for i in range(len(a_sorted)):
            if s / l > a_sorted[i]:
                s -= a_sorted[i]
                l -= 1
            else:
                break
        alpha = s / l
        A_np[idx, :] = a_sign * np.maximum(a_abs - alpha, 0)
    A.data.copy_(torch.tensor(A_np, dtype=A.dtype, device=A.device))
    return A


def _get_spectral_rad(adj_sparse, tol=1e-5):
    """Compute spectral radius of a sparse adjacency (torch sparse or scipy)."""
    if isinstance(adj_sparse, torch.Tensor):
        adj_sparse = adj_sparse.coalesce().cpu()
        A_scipy = sp.sparse.coo_matrix(
            (np.abs(adj_sparse.values().numpy()), adj_sparse.indices().numpy()),
            shape=adj_sparse.shape,
        )
    else:
        A_scipy = adj_sparse
    return (
        float(np.abs(sp.sparse.linalg.eigs(A_scipy, k=1, return_eigenvectors=False)[0]))
        + tol
    )


class _ImplicitFunction(Function):
    """Custom autograd for the IGNN fixed-point iteration."""

    @staticmethod
    def forward(ctx, W, X_0, A, B, fw_mitr=300, bw_mitr=300):
        X_0 = B if X_0 is None else X_0
        X, err, status, D = _ImplicitFunction._fixed_point(
            W, X_0, A, B, F.relu, mitr=fw_mitr, compute_dphi=True
        )
        ctx.save_for_backward(W, X, A, B, D, X_0, torch.tensor(bw_mitr))
        if status != "converged":
            pass  # silent — training may still progress
        return X

    @staticmethod
    def backward(ctx, *grad_outputs):
        W, X, A, B, D, X_0, bw_mitr = ctx.saved_tensors
        bw_mitr = int(bw_mitr.cpu().numpy())
        grad_x = grad_outputs[0]

        dphi = lambda Z: torch.mul(Z, D)
        grad_z, _, _, _ = _ImplicitFunction._fixed_point(
            W.T, X_0, A, grad_x, dphi, mitr=bw_mitr, transposed_A=True
        )

        grad_W = grad_z @ torch.spmm(A, X.T)
        grad_B = grad_z
        return grad_W, None, torch.zeros_like(A), grad_B, None, None

    @staticmethod
    def _fixed_point(
        W, X, A, B, phi, mitr=300, tol=3e-6, transposed_A=False, compute_dphi=False
    ):
        At = A if transposed_A else torch.transpose(A, 0, 1)
        err = 0.0
        status = "max itrs reached"
        for _ in range(mitr):
            X_ = W @ X
            support = torch.spmm(At, X_.T).T
            X_new = phi(support + B)
            err = torch.norm(X_new - X, float("inf"))
            if err < tol:
                status = "converged"
                break
            X = X_new

        dphi = None
        if compute_dphi:
            with torch.enable_grad():
                support = torch.spmm(At, (W @ X).T).T
                Z = support + B
                Z.requires_grad_(True)
                X_new = phi(Z)
                dphi = torch.autograd.grad(torch.sum(X_new), Z, only_inputs=True)[0]

        return X_new, err, status, dphi


class ImplicitGraphLayer(nn.Module):
    """Single IGNN implicit layer."""

    def __init__(self, in_features, out_features, num_node, kappa=0.99):
        super().__init__()
        self.p = in_features
        self.m = out_features
        self.n = num_node
        self.k = kappa

        self.W = nn.Parameter(torch.FloatTensor(self.m, self.m))
        self.Omega_1 = nn.Parameter(torch.FloatTensor(self.m, self.p))
        self.bias = nn.Parameter(torch.FloatTensor(self.m, 1))
        self._init_params()

    def _init_params(self):
        stdv = 1.0 / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.Omega_1.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X_0, A, U, A_rho=1.0, fw_mitr=300, bw_mitr=300):
        if self.k is not None and self.k > 0:
            _projection_norm_inf(self.W, kappa=self.k / A_rho)
        # B = Omega_1 @ U @ A^T  (input features projected through adjacency)
        support_1 = torch.spmm(torch.transpose(U, 0, 1), self.Omega_1.T).T
        support_1 = torch.spmm(torch.transpose(A, 0, 1), support_1.T).T
        b_Omega = support_1
        return _ImplicitFunction.apply(self.W, X_0, A, b_Omega, fw_mitr, bw_mitr)


class IGNN(nn.Module):
    """Implicit Graph Neural Network (Gu et al., NeurIPS 2020).

    Integrates with the existing GNN_model.py interface: forward(x, edge_index, edge_weight=None).
    Uses a fixed-point equilibrium equation to capture long-range dependencies.
    """

    def __init__(self, nfeat, nhid, nclass, num_node, dropout=0.5, kappa=0.9):
        super().__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.num_node = num_node

        self.ig1 = ImplicitGraphLayer(nfeat, nhid, num_node, kappa)
        self.X_0 = nn.Parameter(torch.zeros(nhid, num_node), requires_grad=False)
        self.V = nn.Linear(nhid, nclass, bias=False)

        # Cached adjacency
        self._adj_cache_id = None
        self._adj_sparse = None
        self._adj_rho = None

    def _build_adj(self, edge_index, edge_weight, num_nodes):
        """Build a normalised sparse adjacency from edge_index + optional weights."""
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        adj_scipy = sp.sparse.coo_matrix(
            (
                edge_weight.detach().cpu().numpy(),
                (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()),
            ),
            shape=(num_nodes, num_nodes),
        )
        # Symmetric normalisation  D^{-1/2} A D^{-1/2}
        rowsum = np.array(adj_scipy.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = sp.sparse.diags(d_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj_scipy @ D_inv_sqrt
        adj_norm = adj_norm.tocoo()

        indices = torch.from_numpy(
            np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64)
        )
        values = torch.from_numpy(adj_norm.data.astype(np.float32))
        adj_t = torch.sparse_coo_tensor(indices, values, torch.Size(adj_norm.shape))
        return adj_t.to(edge_index.device)

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)

        # Rebuild adjacency only when graph changes
        cache_id = id(edge_index)
        if self._adj_cache_id != cache_id:
            self._adj_sparse = self._build_adj(edge_index, edge_weight, num_nodes)
            self._adj_rho = _get_spectral_rad(self._adj_sparse)
            self._adj_cache_id = cache_id

            # Resize X_0 if graph size changed
            if self.X_0.shape[1] != num_nodes:
                self.X_0 = nn.Parameter(
                    torch.zeros(self.nhid, num_nodes, device=x.device),
                    requires_grad=False,
                )

        adj = self._adj_sparse
        # U = features transposed: (p, n) — features are (n, p)
        U = x.T.to_sparse() if not x.is_sparse else x.T

        out = self.ig1(self.X_0, adj, U, self._adj_rho).T  # (n, nhid)
        out = F.normalize(out, dim=-1)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.V(out)
        return out

    def get_embeddings(self, x, edge_index, edge_weight=None):
        """Return intermediate node embeddings (pre-classifier)."""
        num_nodes = x.size(0)
        cache_id = id(edge_index)
        if self._adj_cache_id != cache_id:
            self._adj_sparse = self._build_adj(edge_index, edge_weight, num_nodes)
            self._adj_rho = _get_spectral_rad(self._adj_sparse)
            self._adj_cache_id = cache_id
            if self.X_0.shape[1] != num_nodes:
                self.X_0 = nn.Parameter(
                    torch.zeros(self.nhid, num_nodes, device=x.device),
                    requires_grad=False,
                )

        adj = self._adj_sparse
        U = x.T.to_sparse() if not x.is_sparse else x.T
        out = self.ig1(self.X_0, adj, U, self._adj_rho).T
        return out

    def reset_parameters(self):
        self.ig1._init_params()
        nn.init.zeros_(self.X_0)
        self.V.reset_parameters()


class GCN(nn.Module):
    def __init__(
        self,
        nfeat,
        nhid,
        nclass,
        dropout=0.5,
        num_layers=4,
        use_edge_weights=False,
        GNN_type="GIN",
        num_node=None,
        kappa=0.9,
    ):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_edge_weights = use_edge_weights
        self._is_ignn = GNN_type == "IGNN"

        if self._is_ignn:
            # IGNN is a standalone model, not a stack of conv layers
            self._ignn = IGNN(
                nfeat,
                nhid,
                nclass,
                num_node=num_node or 1,  # will auto-resize on first forward
                dropout=dropout,
                kappa=kappa,
            )
        else:
            if num_layers < 2:
                raise ValueError("num_layers must be at least 2")
            self.convs = build_gnn_model(
                nfeat, nhid, nclass, num_layers, use_edge_weights, GNN_type
            )

    def _apply_conv(self, conv, x, edge_index, edge_weight=None):
        """Apply a conv layer while respecting layer-specific edge-weight support."""
        if isinstance(conv, nn.Linear):
            return conv(x)
        supports_edge_weight = isinstance(conv, (GCNConv, GraphConv, WeightedGINConv))
        if self.use_edge_weights and edge_weight is not None and supports_edge_weight:
            return conv(x, edge_index, edge_weight)
        return conv(x, edge_index)

    def forward(self, x, edge_index, edge_weight=None):
        """Compute logits for each node."""
        if self._is_ignn:
            return self._ignn(x, edge_index, edge_weight)
        for i, conv in enumerate(self.convs[:-1]):
            x = F.relu(self._apply_conv(conv, x, edge_index, edge_weight))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self._apply_conv(self.convs[-1], x, edge_index, edge_weight)
        return x

    def get_embeddings(self, x, edge_index, edge_weight=None):
        """Return intermediate node embeddings (pre-classifier)."""
        if self._is_ignn:
            return self._ignn.get_embeddings(x, edge_index, edge_weight)
        for conv in self.convs[:-2]:
            x = F.relu(self._apply_conv(conv, x, edge_index, edge_weight))
            # x = F.dropout(x, self.dropout, training=self.training)
        x = self._apply_conv(self.convs[-2], x, edge_index, edge_weight)
        return x

    def reset_parameters(self):
        if self._is_ignn:
            self._ignn.reset_parameters()
            return
        for conv in self.convs:
            conv.reset_parameters()
