"""Dual-head GNN: shared encoder with classification + coarsening basis heads.

This model extends the existing GCN architecture with two additional outputs:
1. A K-dimensional coarsening basis Z (replaces spectral B in variation edges)
2. A per-edge correction scalar (multiplicative residual on variation cost)

The shared encoder ensures classification signal informs the coarsening basis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from src.GangPrediction.GNN_model import GCN


class EdgeCorrectionMLP(nn.Module):
    """Small MLP that produces a bounded scalar correction per edge.

    Input per edge (i,j): [Z[i], Z[j], |Z[i]-Z[j]|, spectral_cost]
    Output: scalar ∈ (-1, 1) via tanh, later used as multiplicative residual.
    """

    def __init__(self, basis_dim: int, hidden_dim: int = 32):
        super().__init__()
        # Input: Z[i] (K) + Z[j] (K) + |Z[i]-Z[j]| (K) + spectral_cost (1) = 3K+1
        input_dim = 3 * basis_dim + 1
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, Z, edge_index, spectral_costs):
        """
        Args:
            Z: [N, K] node basis embeddings
            edge_index: [2, M] edges
            spectral_costs: [M] base variation costs per edge

        Returns:
            corrections: [M] raw correction scalars (pre-tanh)
        """
        src, tgt = edge_index[0], edge_index[1]
        z_src = Z[src]  # [M, K]
        z_tgt = Z[tgt]  # [M, K]
        z_diff = torch.abs(z_src - z_tgt)  # [M, K]
        cost_feat = spectral_costs.unsqueeze(1)  # [M, 1]

        # Normalize cost feature for numerical stability
        cost_feat = cost_feat / (cost_feat.max() + 1e-8)

        edge_feat = torch.cat([z_src, z_tgt, z_diff, cost_feat], dim=1)  # [M, 3K+1]
        return self.mlp(edge_feat).squeeze(1)  # [M]


class DualHeadGNN(nn.Module):
    """Shared-encoder GNN with classification, coarsening basis, and edge correction heads.

    Architecture:
        SharedEncoder (GCN from GNN_model.py) → H ∈ R^{N×nhid}
        Head 1 (classification): Linear(nhid, nclass) → logits
        Head 2 (coarsening basis): Linear(nhid, K) → Z
        Head 3 (edge correction): MLP([Z_i, Z_j, |Z_i-Z_j|, cost]) → scalar
    """

    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        basis_dim: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        GNN_type: str = "DGCN",
        use_edge_weights: bool = False,
        correction_hidden: int = 32,
        num_node: int = None,
    ):
        super().__init__()
        self.basis_dim = basis_dim
        self.nhid = nhid

        # Shared encoder: reuse existing GCN but with nhid output (not nclass)
        # We build an encoder that outputs nhid-dimensional embeddings
        self.encoder = GCN(
            nfeat=nfeat,
            nhid=nhid,
            nclass=nhid,  # encoder outputs nhid, not nclass
            dropout=dropout,
            num_layers=num_layers,
            GNN_type=GNN_type,
            use_edge_weights=use_edge_weights,
            num_node=num_node,
        )

        # Head 1: Classification
        self.classifier = nn.Linear(nhid, nclass)

        # Head 2: Coarsening basis
        self.basis_head = nn.Linear(nhid, basis_dim)

        # Head 3: Edge correction MLP
        self.edge_correction = EdgeCorrectionMLP(basis_dim, correction_hidden)

    def _encode(self, x, edge_index, edge_weight=None):
        """Get shared hidden representation H from encoder."""
        H = self.encoder(x, edge_index, edge_weight)
        return F.relu(H)

    def forward(self, x, edge_index, edge_weight=None):
        """Full forward: returns (logits, Z, None).

        Edge corrections are not computed here because they need spectral_costs
        which requires Z first. Use get_edge_corrections() separately.
        """
        H = self._encode(x, edge_index, edge_weight)
        logits = self.classifier(H)
        # Z = self.basis_head(H)
        return logits

    def get_logits(self, x, edge_index, edge_weight=None):
        """Classification logits only (for compatibility with existing code)."""
        H = self._encode(x, edge_index, edge_weight)
        return self.classifier(H)

    def get_basis(self, x, edge_index, edge_weight=None):
        """Coarsening basis Z only."""
        H = self._encode(x, edge_index, edge_weight)
        return self.basis_head(H)

    def get_embeddings(self, x, edge_index, edge_weight=None):
        """Return shared hidden embeddings H (for compatibility with existing code)."""
        return self._encode(x, edge_index, edge_weight)

    def get_edge_corrections(self, Z, edge_index, spectral_costs):
        """Compute edge corrections given basis Z and spectral costs.

        Args:
            Z: [N, K] basis from get_basis()
            edge_index: [2, M]
            spectral_costs: [M] base variation costs

        Returns:
            corrections: [M] raw correction values (apply tanh externally)
        """
        return self.edge_correction(Z, edge_index, spectral_costs)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()
        self.basis_head.reset_parameters()
        for layer in self.edge_correction.mlp:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class DualHeadWrapper(nn.Module):
    """Wrapper that makes DualHeadGNN compatible with existing training code.

    The existing training code calls model(x, edge_index, edge_weight) and expects
    logits. This wrapper stores Z as a side effect for later retrieval.
    """

    def __init__(self, dual_head: DualHeadGNN):
        super().__init__()
        self.dual_head = dual_head
        self._last_Z = None
        self._last_H = None

    def forward(self, x, edge_index, edge_weight=None):
        """Returns logits (compatible with existing code). Stores Z internally."""
        logits = self.dual_head(x, edge_index, edge_weight)
        # self._last_Z = Z
        return logits

    def get_embeddings(self, x, edge_index, edge_weight=None):
        """Return shared hidden embeddings (compatible with existing code)."""
        return self.dual_head.get_embeddings(x, edge_index, edge_weight)

    def get_basis(self, x, edge_index, edge_weight=None):
        """Get coarsening basis Z."""
        return self.dual_head.get_basis(x, edge_index, edge_weight)

    def get_edge_corrections(self, Z, edge_index, spectral_costs):
        """Get edge corrections."""
        return self.dual_head.get_edge_corrections(Z, edge_index, spectral_costs)

    @property
    def last_Z(self):
        return self._last_Z

    def reset_parameters(self):
        self.dual_head.reset_parameters()

    def parameters(self, recurse=True):
        return self.dual_head.parameters(recurse)

    def named_parameters(self, prefix="", recurse=True):
        return self.dual_head.named_parameters(prefix, recurse)

    def train(self, mode=True):
        super().train(mode)
        self.dual_head.train(mode)
        return self

    def eval(self):
        super().eval()
        self.dual_head.eval()
        return self
