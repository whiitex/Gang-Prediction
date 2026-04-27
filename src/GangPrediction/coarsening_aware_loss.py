"""Loss and helpers for training across coarsened graph levels."""

import torch
from torch.cuda import temperature
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

# graph coarsening - Loukas 2020
from src.GangPrediction.coarsening_utils import *
from src.GangPrediction.graph_utils import *
from src.GangPrediction.experiment_utils import create_subspace
from src.GangPrediction.utils.utils import *


class CoarseningAwareLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor = None,
        alert_patterns: Dict[str, List[int]] = None,
        normal_patterns: Dict[str, List[int]] = None,
        use_supernode_loss: bool = False,
        use_spectral_consistency_loss: bool = False,
        use_edge_correction_reg: bool = False,
        use_pattern_type_loss: bool = False,
        use_surrogate_epsilon: bool = False,
        coarse_weight: float = 1,
        pattern_type_weight: float = 1,
        spectral_consistency_weight: float = 1,
        edge_correction_weight: float = 1,
        supernode_intra_weight: float = 100.0,
        supernode_negative_weight: float = 0,
        epsilon_weight: float = 1,
        pattern_margin: float = 0.0,
        intra_weight: float = 1.0,
        inter_weight: float = 1.0,
        pull_weight: float = 1.0,
        push_weight: float = 1.0,
        **kwargs,
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
          pull_weight: Weight for pulling nodes within patterns together.
          push_weight: Weight for pushing nodes from different patterns apart.
          pattern_type_weight: Weight for the pattern type loss term.
          spectral_consistency_weight: Weight for the spectral consistency loss term.
          edge_correction_weight: Weight for the edge correction regularization term.
        """
        super().__init__()
        self.coarse_weight = coarse_weight
        self.class_weights = class_weights
        self.class_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.pattern_margin = pattern_margin
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight
        self.use_supernode_loss = use_supernode_loss
        self.use_pattern_type_loss = use_pattern_type_loss
        self.use_spectral_consistency_loss = use_spectral_consistency_loss
        self.use_edge_correction_reg = use_edge_correction_reg
        self.use_surrogate_epsilon = use_surrogate_epsilon
        self.pattern_type_weight = pattern_type_weight
        self.spectral_consistency_weight = spectral_consistency_weight
        self.edge_correction_weight = edge_correction_weight
        self.epsilon_weight = epsilon_weight
        # Supernode embedding loss for learned B
        self.supernode_loss = SupernodeEmbeddingLoss2(
            intra_weight=supernode_intra_weight,
            inter_weight=inter_weight,
            pattern_margin=pattern_margin,
        )
        # self.supernode_loss = SupernodeEmbeddingLoss(
        #     negative_weight=supernode_negative_weight,
        #     # temperature=temperature,
        # )

        self.pattern_type_criterion = PatternTypeLoss(
            pull_weight=pull_weight, push_weight=push_weight
        )
        self.spectral_consistency_criterion = SpectralConsistencyLoss()
        self.edge_correction_reg = EdgeCorrectionRegularizer()

        # Store patterns
        self.alert_patterns = alert_patterns or {}
        self.normal_patterns = normal_patterns or {}

        # Latest detached loss components for logging/plotting.
        self.latest_loss_components = {
            "loss_total": float("nan"),
            "loss_cls": float("nan"),
            "loss_supernode": float("nan"),
            "loss_pattern_type": float("nan"),
            "loss_spectral": float("nan"),
            "loss_correction": float("nan"),
            "loss_epsilon": float("nan"),
        }

    def forward(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        train_idx: torch.Tensor,
        embeddings: torch.Tensor = None,
        P: torch.Tensor = None,
        L: torch.Tensor = None,
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

        # 3. Supernode embedding loss (for learned B)
        if self.use_supernode_loss and embeddings is not None:
            loss_supernode = self.supernode_loss(
                embeddings,
                alert_patterns=self.alert_patterns,
                normal_patterns=self.normal_patterns,
            )
        else:
            loss_supernode = torch.tensor(0.0, device=device)

        if self.use_surrogate_epsilon:
            U = l_orthonormalize(embeddings, L)
            P_ = torch.eye(U.shape[0], device=U.device) - P
            epsillon_loss = torch.trace(U.T @ P_.T @ L @ P_ @ U) / U.shape[1]
        else:
            epsillon_loss = torch.tensor(0.0, device=device)

        if self.use_pattern_type_loss:
            loss_pattern_type = self.pattern_type_criterion(
                embeddings, self.alert_patterns, self.normal_patterns
            )
        else:
            loss_pattern_type = torch.tensor(0.0, device=device)

        if self.use_spectral_consistency_loss and L is not None:
            loss_spectral = self.spectral_consistency_criterion(embeddings, L)
        else:
            loss_spectral = torch.tensor(0.0, device=device)

        loss_total = (
            loss_cls
            + self.coarse_weight * loss_supernode
            + self.pattern_type_weight * loss_pattern_type
            + self.spectral_consistency_weight * loss_spectral
            + self.epsilon_weight * epsillon_loss
        )
        self.latest_loss_components = {
            "loss_total": float(loss_total.detach().item()),
            "loss_cls": float(loss_cls.detach().item()),
            "loss_supernode": float(loss_supernode.detach().item()),
            "loss_pattern_type": float(loss_pattern_type.detach().item()),
            "loss_spectral": float(loss_spectral.detach().item()),
            "loss_correction": float(loss_spectral.detach().item()),
            "loss_epsilon": float(epsillon_loss.detach().item()),
        }
        return loss_total
        # return loss_cls + self.coarse_weight * loss_pattern + loss_supernode


class PatternTypeLoss(nn.Module):
    """Type-level contrastive loss on the coarsening basis Z.

    Groups training patterns by pattern_type, computes type centroids in Z-space,
    and applies pull (within-type) + push (between-type) losses.
    This generalizes to unseen patterns that share type characteristics.
    """

    def __init__(self, pull_weight: float = 1.0, push_weight: float = 1.0):
        super().__init__()
        self.pull_weight = pull_weight
        self.push_weight = push_weight

    def forward(
        self,
        Z: torch.Tensor,
        alert_patterns: List,
        normal_patterns: List,
    ) -> torch.Tensor:
        """
        Args:
            Z: [N, K] coarsening basis embeddings
            alert_patterns: list of Pattern objects with .pattern_type and .node_indices
            normal_patterns: list of Pattern objects

        Returns:
            Scalar loss
        """
        device = Z.device
        N = Z.shape[0]

        # Group patterns by type
        type_to_nodes: Dict[str, List[int]] = {}
        all_patterns = list(alert_patterns or []) + list(normal_patterns or [])
        for p in all_patterns:
            ptype = p.pattern_type
            valid = [i for i in p.node_indices if i < N]
            if len(valid) == 0:
                continue
            if ptype not in type_to_nodes:
                type_to_nodes[ptype] = []
            type_to_nodes[ptype].extend(valid)

        if len(type_to_nodes) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Compute type centroids and pull loss
        centroids = []
        pull_loss = torch.tensor(0.0, device=device)
        n_types = 0

        for ptype, nodes in type_to_nodes.items():
            idx = torch.tensor(nodes, dtype=torch.long, device=device).unique()
            if len(idx) < 2:
                continue
            z_type = Z[idx]  # [n_type, K]
            centroid = z_type.mean(dim=0, keepdim=True)  # [1, K]
            centroids.append(centroid.squeeze(0))

            # Within-type pull: minimize variance around centroid
            pull_loss = pull_loss + ((z_type - centroid) ** 2).sum(dim=1).mean()
            n_types += 1

        if n_types == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        pull_loss = pull_loss / n_types

        # Between-type push: maximize distance between centroids
        push_loss = torch.tensor(0.0, device=device)
        if len(centroids) >= 2:
            C = torch.stack(centroids, dim=0)  # [T, K]
            # Pairwise squared distances
            dist_matrix = torch.cdist(C.unsqueeze(0), C.unsqueeze(0)).squeeze(0) ** 2
            # Mean of upper triangle (exclude diagonal)
            n_pairs = len(centroids) * (len(centroids) - 1) / 2
            if n_pairs > 0:
                mask = torch.triu(
                    torch.ones_like(dist_matrix, dtype=torch.bool), diagonal=1
                )
                push_loss = -dist_matrix[mask].mean()

        return self.pull_weight * pull_loss + self.push_weight * push_loss


class SpectralConsistencyLoss(nn.Module):
    """Regularizer: ||Z^T L Z - I_K||_F.

    Ensures the learned basis Z preserves the structural guarantees of
    variation edges by staying close to L-orthonormal.
    """

    def forward(self, Z: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z: [N, K] learned coarsening basis
            L: [N, N] graph Laplacian (sparse or dense)

        Returns:
            Scalar Frobenius norm of (Z^T L Z - I_K)
        """
        K = Z.shape[1]
        device = Z.device

        # Z^T L Z: [K, K]
        if L.is_sparse:
            LZ = torch.sparse.mm(L, Z)  # [N, K]
        else:
            LZ = L @ Z
        ZtLZ = Z.T @ LZ  # [K, K]

        I_K = torch.eye(K, device=device, dtype=Z.dtype)
        return torch.norm(ZtLZ - I_K, p="fro")


class EdgeCorrectionRegularizer(nn.Module):
    """L1 penalty on edge correction magnitudes to keep them small."""

    def forward(self, corrections: torch.Tensor) -> torch.Tensor:
        """
        Args:
            corrections: [M] raw correction scalars

        Returns:
            Mean absolute correction value
        """
        return corrections.abs().mean()


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


class SupernodeEmbeddingLoss2(nn.Module):
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
        inter_weight: float = 0.5,
        pattern_margin: float = 1.0,
    ):
        """
        Args:
            intra_weight: Weight for intra-supernode consistency loss
            inter_weight: Weight for inter-supernode consistency loss
            pattern_margin: Margin for inter-supernode contrastive loss
        """
        super().__init__()
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight
        self.pattern_margin = pattern_margin

    def forward(
        self,
        embeddings: torch.Tensor,
        alert_patterns: Dict[str, List[int]] = None,
        normal_patterns: Dict[str, List[int]] = None,
    ) -> torch.Tensor:
        all_patterns = alert_patterns + normal_patterns
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

        if len(all_patterns) == 0:
            return torch.tensor(0.0, device=device)

        # Normalize embeddings for cosine similarity
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)

        intra_loss = torch.tensor(0.0, device=device)
        inter_loss = torch.tensor(0.0, device=device)
        n_patterns = 0

        for pattern in all_patterns:
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
