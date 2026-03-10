"""Loss and helpers for training across coarsened graph levels."""

import torch

# import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Dict, List, Optional

# graph coarsening - Loukas 2020
from src.GangPrediction.coarsening_utils import *
from src.GangPrediction.graph_utils import *


class CoarseningAwareLoss(nn.Module):
    def __init__(
        self,
        coarse_weight: float = 1.0,
        class_weights: torch.Tensor = None,
        alert_patterns: Dict[str, List[int]] = None,
        normal_patterns: Dict[str, List[int]] = None,
        pattern_margin: float = 1.0,
        intra_weight: float = 1.0,
        inter_weight: float = 1.0,
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
        """
        super().__init__()
        self.coarse_weight = coarse_weight
        self.class_weights = class_weights
        self.class_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.pattern_margin = pattern_margin
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight

        # Store patterns
        self.alert_patterns = alert_patterns or {}
        self.normal_patterns = normal_patterns or {}

        # Precompute pattern node sets for efficiency
        self._precompute_pattern_info()

    def _precompute_pattern_info(self):
        """Precompute pattern information for efficient loss computation."""
        # Combine all patterns
        all_patterns = {}
        all_patterns.update(self.alert_patterns)
        all_patterns.update(self.normal_patterns)

        self.all_patterns = all_patterns

        # Get all pattern node indices as a set
        self.pattern_nodes_set = set()
        for nodes in all_patterns.values():
            self.pattern_nodes_set.update(nodes)

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

        for pattern_id, node_indices in self.all_patterns.items():
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
        coarse_loss: bool = False,
    ):
        """
        output: [N, C] raw logits (CrossEntropyLoss expects raw logits, not softmax).
        embeddings: [N, D] raw features from model.get_embeddings().
        labels: [N] ground-truth class labels.
        train_idx: indices of nodes used for classification loss.
        coarse_loss: if True, compute pattern contrastive loss.
        """

        # 1. Classification loss
        loss_cls = self.class_loss(output[train_idx], labels[train_idx])

        if not coarse_loss or embeddings is None:
            return loss_cls

        # 2. Pattern contrastive loss
        loss_pattern = self._compute_pattern_contrastive_loss(embeddings)

        return loss_cls + 0 * loss_pattern
        # return loss_cls + self.coarse_weight * loss_pattern


def apply_graph_coarsening(
    G: Data,
    # X=None,
    method="variation_neighborhoods",
    ratio=0.5,
    K=3,
    similarity_threshold=0.65,
    max_levels=1,
    log_info=False,
):
    """
    Output:
      - C: coarsening matrix (n, N)
      - Gc: coarsened graph (n, n)
      - Call: all coarsened graphs (levels, n_l, N)
      - Gall: all original graphs (levels, n_l, n_l)
    """

    available_methods = [
        "variation_neighborhoods",
        "variation_edges",
        "heavy_edge",
        "algebraic_JC",
        "kron",
    ]
    if method not in available_methods:
        raise ValueError(
            f"Unknown coarsening method: {method}. Method must be one of {available_methods}."
        )

    C, Gc, Call, Gall = coarsen(
        G,
        K=K,
        method=method,
        r=ratio,
        similarity_threshold=similarity_threshold,
        max_levels=max_levels,
    )

    if log_info:
        print(f"Coarsening: {method} ", end="")
        print(
            f"({G.num_nodes} n, {G.num_edges} e) -> ({Gc.num_nodes} n, {Gc.num_edges} e); "
        )

    return C, Gc, Call, Gall


def get_coarsened_edges_and_features(C: list, Gc: Data):
    """
    Coarsen edges and features based on the coarsening matrix C;
    the feature matrix (N x D) is made by summing up features from nodes that are coarsened into the same supernode.

    Output:
      - Gc: coarsened edges index (num_edges, 2)
      - features_coarsened: coarsened features (num_nodes_coarsened, num_features)
    """

    idx_map = {}  # map fine node -> coarse node
    for supernode, node in zip(*C.nonzero()):
        idx_map[node] = supernode

    # features coarsened
    features_coarsened = C @ Gc.x

    # edges coarsened
    edges_idx_coarsened = Gc.edge_index

    return edges_idx_coarsened, features_coarsened


def get_coarsened_labels(C: list, labels):
    """
    Coarsen edges and features based on the coarsening matrix C;
    the feature matrix (N x D) is made by summing up features from nodes that are coarsened into the same supernode.

    Output:
      - edges_idx_coarsened: coarsened edges index (num_edges, 2)
      - features_coarsened: coarsened features (num_nodes_coarsened, num_features)
      - labels_coarsened: coarsened labels (num_nodes_coarsened,)
    """

    # labels coarsened
    device = labels.device
    num_classes = len(torch.unique(labels))

    # Create one-hot encoding of labels
    labels_onehot = torch.zeros(len(labels), num_classes, device=device)
    labels_onehot.scatter_(1, labels.view(-1, 1), 1)

    # Compute label counts per supernode using matrix multiplication
    all_labels = C @ labels_onehot

    # Get the most common label for each supernode
    labels_coarsened = torch.argmax(all_labels, dim=1)

    return labels_coarsened
