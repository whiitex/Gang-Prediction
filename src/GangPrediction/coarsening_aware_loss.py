"""Loss and helpers for training across coarsened graph levels."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

# graph coarsening - Loukas 2020
from src.GangPrediction.coarsening_utils import *
from src.GangPrediction.graph_utils import *
from src.GangPrediction.experiment_utils import create_subspace
from src.GangPrediction.utils.utils import *


class SupernodeEmbeddingLoss(nn.Module):
    """
    Loss to enforce embedding structure for learned coarsening basis.

    This loss has two components:
    1. Intra-supernode consistency: Nodes in the same supernode (pattern) should
       have equal/similar embeddings
    2. Negative sampling: Prevent embedding collapse by pushing apart embeddings
       of nodes in different supernodes
    """

    def __init__(
        self,
        intra_weight: float = 1.0,
        negative_weight: float = 0.5,
        temperature: float = 0.1,
        n_negative_samples: int = 32,
    ):
        """
        Args:
            intra_weight: Weight for intra-supernode consistency loss
            negative_weight: Weight for negative sampling loss
            temperature: Temperature for contrastive loss (lower = sharper)
            n_negative_samples: Number of negative samples per supernode
        """
        super().__init__()
        self.intra_weight = intra_weight
        self.negative_weight = negative_weight
        self.temperature = temperature
        self.n_negative_samples = n_negative_samples

    def forward(
        self,
        embeddings: torch.Tensor,
        alert_patterns: Dict[str, List[int]] = None,
        normal_patterns: Dict[str, List[int]] = None,
    ) -> torch.Tensor:
        """
        Compute supernode embedding loss.

        Args:
            embeddings: [N, D] node embeddings from GNN
            alert_patterns: Dict mapping pattern_id -> list of node indices
            normal_patterns: Dict mapping pattern_id -> list of node indices

        Returns:
            Combined loss scalar
        """
        device = embeddings.device
        N = embeddings.shape[0]
        V = create_subspace(alert_patterns, normal_patterns, N, device)
        alert_nodes = set()
        for pattern in alert_patterns:
            alert_nodes.update(pattern.node_indices)
        normal_nodes = set()
        for pattern in normal_patterns:
            normal_nodes.update(pattern.node_indices)
        total_pattern_nodes = torch.tensor(
            sorted(alert_nodes.union(normal_nodes)),
            device=device,
            dtype=torch.long,
        )
        if total_pattern_nodes.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        logits = embeddings[total_pattern_nodes, :]
        targets = V[total_pattern_nodes, :].argmax(dim=1)
        loss = F.cross_entropy(logits, targets)
        # loss = F.nll_loss(
        #     F.log_softmax(embeddings[total_pattern_nodes, :], dim=1),
        #     V[total_pattern_nodes, :].argmax(dim=1),
        # )
        # loss = torch.mean(
        #     (embeddings[total_pattern_nodes] - V[total_pattern_nodes]) ** 2
        # )
        return loss

    # def forward(
    #     self,
    #     embeddings: torch.Tensor,
    #     alert_patterns: Dict[str, List[int]] = None,
    #     normal_patterns: Dict[str, List[int]] = None,
    # ) -> torch.Tensor:
    #     """
    #     Compute supernode embedding loss.

    #     Args:
    #         embeddings: [N, D] node embeddings from GNN
    #         alert_patterns: Dict mapping pattern_id -> list of node indices
    #         normal_patterns: Dict mapping pattern_id -> list of node indices

    #     Returns:
    #         Combined loss scalar
    #     """
    #     device = embeddings.device

    #     # Combine all patterns (both alert and normal patterns define supernodes)
    #     all_patterns = []
    #     if alert_patterns:
    #         all_patterns.extend(alert_patterns)
    #     if normal_patterns:
    #         all_patterns.extend(normal_patterns)

    #     if len(all_patterns) == 0:
    #         return torch.tensor(0.0, device=device, requires_grad=True)

    #     # Normalize embeddings for stability
    #     # embeddings_norm = F.normalize(embeddings, p=2, dim=1)

    #     intra_loss = torch.tensor(0.0, device=device)
    #     negative_loss = torch.tensor(0.0, device=device)
    #     n_valid_patterns = 0

    #     # Collect all pattern centroids for negative sampling
    #     pattern_centroids = []
    #     sum_nodes = 0
    #     for pattern in all_patterns:
    #         pattern_embeddings = embeddings[pattern.node_indices]

    #         # Compute centroid of the supernode
    #         centroid = pattern_embeddings.mean(dim=0, keepdim=True)
    #         pattern_centroids.append(centroid)

    #         # 1. Intra-supernode consistency loss
    #         # All nodes in the same supernode should have the same embedding
    #         if len(pattern.node_indices) >= 2:
    #             # Variance within supernode (want to minimize)
    #             # Using MSE from centroid
    #             intra_distances = ((pattern_embeddings - centroid) ** 2).sum(dim=1)
    #             intra_loss += intra_distances.mean()

    #         n_valid_patterns += 1
    #         sum_nodes += pattern.num_nodes

    #     # intra_loss = intra_loss / (sum_nodes + 1e-6)  # Average per node

    #     # 2. Negative sampling loss (prevent collapse)
    #     # Push apart centroids of different supernodes
    #     if len(pattern_centroids) >= 2:
    #         centroids = torch.cat(pattern_centroids, dim=0)  # [K, D]
    #         # normalized_centroids = F.normalize(
    #         #     centroids, p=2, dim=1
    #         # )  # Normalize for cosine similarity

    #         # Compute pairwise similarities between all centroids
    #         similarity_matrix = torch.mm(
    #             centroids,
    #             centroids.T,
    #             # normalized_centroids, normalized_centroids.T
    #         )  # [K, K]

    #         # Create mask to exclude self-similarity
    #         mask = torch.eye(len(pattern_centroids), device=device, dtype=torch.bool)

    #         # InfoNCE-style loss: maximize log probability of NOT being the same supernode
    #         # For each centroid, the negatives are all other centroids
    #         similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))

    #         # We want to MINIMIZE similarity between different supernodes
    #         # Use negative log of (1 - softmax) approximation
    #         # Simplified: push apart using hinge loss
    #         off_diag_similarities = similarity_matrix[~mask].view(
    #             len(pattern_centroids), -1
    #         )

    #         # Hinge loss: penalize if similarity > -margin (i.e., if too similar)
    #         margin = -0.5  # Want similarities to be < -0.5 (i.e., dissimilar)
    #         negative_loss = F.relu(off_diag_similarities - margin).mean()

    #     # Also add variance maximization to prevent collapse to a single point
    #     if len(pattern_centroids) >= 2:
    #         centroids = torch.cat(pattern_centroids, dim=0)
    #         centroid_variance = centroids.var(dim=0).mean()
    #         # We want to maximize variance (minimize negative variance)
    #         variance_loss = 1.0 / (centroid_variance + 1e-6)
    #         negative_loss = negative_loss + 0.0 * variance_loss
    #         # negative_loss = negative_loss + 0.1 * variance_loss

    #     # Normalize losses
    #     intra_loss = intra_loss / n_valid_patterns

    #     total_loss = (
    #         self.intra_weight * intra_loss + self.negative_weight * negative_loss
    #     )

    #     return total_loss


def l_orthonormalize(Z, L, eps=1e-6, remove_constant=True):
    """
    Z: [N, k]
    L: [N, N] symmetric PSD
    returns U with approximately U.T @ L @ U = I
    """
    N, k = Z.shape

    if remove_constant:
        Z = Z - Z.mean(dim=0, keepdim=True)

    G = Z.T @ torch.sparse.mm(L.T, L) @ Z
    G = G + eps * torch.eye(k, device=Z.device, dtype=Z.dtype)

    # G = R^T R
    R = torch.linalg.cholesky(G, upper=True)

    # U = Z @ R^{-1}
    # better than explicit inverse:
    U = torch.linalg.solve_triangular(R, Z.T, upper=True, left=False).T

    return U


def calc_B_from_embeddings(embeddings: torch.Tensor, K: int = None) -> torch.Tensor:
    """
    Compute an orthonormal basis B from learned node embeddings.

    The embeddings learned by the GNN (with SupernodeEmbeddingLoss) naturally
    encode the coarsening structure: nodes in the same supernode have similar
    embeddings. We extract an orthonormal basis from these embeddings.

    Args:
        embeddings: [N, D] node embeddings (should be normalized)
        K: Number of basis vectors to return. If None, use min(N, D)

    Returns:
        B: [N, K] orthonormal basis matrix
    """
    N, D = embeddings.shape

    if K is None:
        K = min(N, D)
    K = min(K, N, D)

    # Normalize embeddings
    # embeddings_norm = F.normalize(embeddings, p=2, dim=1)

    # SVD to get orthonormal basis
    # U @ S @ V^T = embeddings
    # U: [N, K] orthonormal columns (our B)
    # S: [K] singular values
    # V: [D, K] orthonormal columns
    # try:
    #     U, S, Vh = torch.linalg.svd(embeddings, full_matrices=False)

    #     # Take the top K left singular vectors as basis
    #     B = U[:, :K]

    #     # Weight by singular values (optional, for smoothness)
    #     # S_weights = S[:K] / (S[:K].max() + 1e-6)
    #     # B = B * S_weights.unsqueeze(0)

    # except Exception as e:
    #     # Fallback to QR decomposition if SVD fails
    # print(f"SVD failed ({e}), falling back to QR decomposition")
    Q, R = torch.linalg.qr(embeddings)
    B = Q[:, :K]

    return B


class CoarseningAwareLoss(nn.Module):
    def __init__(
        self,
        coarse_weight: float = 1,
        class_weights: torch.Tensor = None,
        alert_patterns: Dict[str, List[int]] = None,
        normal_patterns: Dict[str, List[int]] = None,
        pattern_margin: float = 1.0,
        intra_weight: float = 1.0,
        inter_weight: float = 1.0,
        # Supernode embedding loss parameters
        use_supernode_loss: bool = True,
        supernode_intra_weight: float = 100.0,
        supernode_negative_weight: float = 0,
    ):
        """
        Args:
          coarse_weight: weight for the pattern contrastive loss term.
          class_weights: tensor of shape [num_classes] with class weights for handling imbalance.
                        If None, no class weighting is applied.
          alert_patterns: Dict mapping pattern_id -> list of node indices for alert/malicious patterns.
          normal_patterns: Dict mapping pattern_id -> list of node indices for normal patterns.
          pattern_margin: Margin for contrastive loss (how far apart embeddings should be).
          intra_weight: Weight for intra-pattern cohesion loss (pulling nodes within patterns together).
          inter_weight: Weight for inter-pattern separation loss (pushing pattern nodes away from others).
          use_supernode_loss: Whether to use the supernode embedding loss for learned B.
          supernode_intra_weight: Weight for intra-supernode consistency.
          supernode_negative_weight: Weight for negative sampling to prevent collapse.
        """
        super().__init__()
        self.coarse_weight = coarse_weight
        self.class_weights = class_weights
        self.class_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.pattern_margin = pattern_margin
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight

        # Supernode embedding loss for learned B
        self.use_supernode_loss = use_supernode_loss
        self.supernode_loss = SupernodeEmbeddingLoss(
            intra_weight=supernode_intra_weight,
            negative_weight=supernode_negative_weight,
        )

        # Store patterns
        self.alert_patterns = alert_patterns or {}
        self.normal_patterns = normal_patterns or {}

        # Latest detached loss components for logging/plotting.
        self.latest_loss_components = {
            "loss_total": float("nan"),
            "loss_cls": float("nan"),
            "loss_supernode": float("nan"),
        }

        # Precompute pattern node sets for efficiency
        self._precompute_pattern_info()

    def _precompute_pattern_info(self):
        """Precompute pattern information for efficient loss computation."""
        # Combine all patterns
        all_patterns = []
        all_patterns.extend(self.alert_patterns)
        all_patterns.extend(self.normal_patterns)

        self.all_patterns = all_patterns

        # Get all pattern node indices as a set
        self.pattern_nodes_set = set()
        for pattern in all_patterns:
            self.pattern_nodes_set.update(pattern.node_indices)

    def update_class_weights(self, class_weights: torch.Tensor):
        """Update class weights (e.g., when labels change at coarser levels)."""
        self.class_weights = class_weights
        self.class_loss = nn.CrossEntropyLoss(weight=class_weights)

    def set_patterns(
        self,
        alert_patterns: Dict[str, List[int]] = None,
        normal_patterns: Dict[str, List[int]] = None,
    ):
        """Update patterns (e.g., after coarsening remaps node indices)."""
        self.alert_patterns = alert_patterns or {}
        self.normal_patterns = normal_patterns or {}
        self._precompute_pattern_info()

    def _compute_pattern_contrastive_loss(
        self, embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss to:
        1. Pull embeddings of nodes within the same pattern closer together
        2. Push embeddings of pattern nodes away from non-pattern nodes

        Uses a triplet-style loss with:
        - Anchor: centroid of each pattern
        - Positive: nodes within the same pattern
        - Negative: nodes outside the pattern
        """
        device = embeddings.device
        N = embeddings.shape[0]

        if len(self.all_patterns) == 0:
            return torch.tensor(0.0, device=device)

        # Normalize embeddings for cosine similarity
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)

        intra_loss = torch.tensor(0.0, device=device)
        inter_loss = torch.tensor(0.0, device=device)
        n_patterns = 0

        for pattern in self.all_patterns:
            node_indices = pattern.node_indices
            if len(node_indices) < 2:
                continue

            # Filter valid indices (in case of coarsening reducing node count)
            valid_indices = [idx for idx in node_indices if idx < N]
            if len(valid_indices) < 2:
                continue

            pattern_indices = torch.tensor(
                valid_indices, dtype=torch.long, device=device
            )
            pattern_embeddings = embeddings_norm[pattern_indices]

            # 1. Intra-pattern loss: minimize distance within pattern
            # Compute centroid of the pattern
            centroid = pattern_embeddings.mean(dim=0, keepdim=True)

            # Distance from each node to centroid (want to minimize)
            intra_distances = 1 - torch.sum(pattern_embeddings * centroid, dim=1)
            intra_loss += intra_distances.mean()

            # 2. Inter-pattern loss: maximize distance from non-pattern nodes
            # Sample negative nodes (nodes not in this pattern)
            pattern_set = set(valid_indices)
            non_pattern_indices = [i for i in range(N) if i not in pattern_set]

            if len(non_pattern_indices) > 0:
                # Sample a subset of negative nodes for efficiency
                n_neg_samples = min(len(non_pattern_indices), len(valid_indices) * 2)
                neg_sample_indices = torch.tensor(
                    non_pattern_indices[:n_neg_samples], dtype=torch.long, device=device
                )
                neg_embeddings = embeddings_norm[neg_sample_indices]

                # Distance from centroid to negative samples (want to maximize, so minimize negative)
                neg_distances = torch.sum(centroid * neg_embeddings, dim=1)

                # Hinge loss: push negatives beyond margin
                inter_loss += F.relu(self.pattern_margin + neg_distances - 1).mean()

            n_patterns += 1

        if n_patterns > 0:
            intra_loss = intra_loss / n_patterns
            inter_loss = inter_loss / n_patterns

        return self.intra_weight * intra_loss + self.inter_weight * inter_loss

    def forward(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        train_idx: torch.Tensor,
        embeddings: torch.Tensor = None,
        P: torch.Tensor = None,
        L: torch.Tensor = None,
        coarse_loss: bool = False,
        surrogate_epsilon: float = False,
    ):
        """
        output: [N, C] raw logits (CrossEntropyLoss expects raw logits, not softmax).
        embeddings: [N, D] raw features from model.get_embeddings().
        labels: [N] ground-truth class labels.
        train_idx: indices of nodes used for classification loss.
        coarse_loss: if True, compute pattern contrastive loss and supernode embedding loss.
        """

        # 1. Classification loss
        loss_cls = self.class_loss(output[train_idx], labels[train_idx])

        if not coarse_loss or embeddings is None:
            self.latest_loss_components = {
                "loss_total": float(loss_cls.detach().item()),
                "loss_cls": float(loss_cls.detach().item()),
                "loss_supernode": 0.0,
            }
            return loss_cls, 0, 0

        # 2. Pattern contrastive loss
        # loss_pattern = self._compute_pattern_contrastive_loss(embeddings)

        # 3. Supernode embedding loss (for learned B)
        if self.use_supernode_loss:
            loss_supernode = self.supernode_loss(
                embeddings,
                alert_patterns=self.alert_patterns,
                normal_patterns=self.normal_patterns,
            )
            # loss_supernode += torch.norm(
            #     embeddings.T @ L @ embeddings - torch.eye(embeddings.shape[1], device=P.device),
            #     p="fro",
            # )
        else:
            loss_supernode = torch.tensor(0.0, device=embeddings.device)

        if surrogate_epsilon:
            U = l_orthonormalize(embeddings, L)
            P_ = torch.eye(U.shape[0], device=U.device) - P
            epsillon_loss = torch.trace(U.T @ P_.T @ L @ P_ @ U) / U.shape[1]
        else:
            epsillon_loss = torch.tensor(0.0, device=embeddings.device)

        loss_total = (
            loss_cls
            # + 0 * loss_pattern
            + self.coarse_weight * loss_supernode
            + 1 * epsillon_loss
        )
        self.latest_loss_components = {
            "loss_total": float(loss_total.detach().item()),
            "loss_cls": float(loss_cls.detach().item()),
            "loss_supernode": float(loss_supernode.detach().item()),
        }
        return loss_total, loss_supernode, loss_cls
        # return loss_cls + self.coarse_weight * loss_pattern + loss_supernode
