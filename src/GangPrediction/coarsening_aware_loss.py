"""Loss and helpers for training across coarsened graph levels."""

import torch

# import numpy as np
import torch.nn as nn
from torch_geometric.data import Data

# graph coarsening - Loukas 2020
from src.GangPrediction.coarsening_utils import *
from src.GangPrediction.graph_utils import *


class CoarseningAwareLoss(nn.Module):
    def __init__(self, coarse_weight: float = 1, class_weights: torch.Tensor = None):
        """
        Args:
          coarse_weight: weight for the coarsening loss term.
          class_weights: tensor of shape [num_classes] with class weights for handling imbalance.
                        If None, no class weighting is applied.
        """
        super().__init__()
        self.coarse_weight = coarse_weight
        self.class_weights = class_weights
        self.class_loss = nn.CrossEntropyLoss(weight=class_weights)
        # self.class_loss = nn.NLLLoss()

    def update_class_weights(self, class_weights: torch.Tensor):
        """Update class weights (e.g., when labels change at coarser levels)."""
        self.class_weights = class_weights
        self.class_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        # coarsening_matrix: torch.Tensor,
        train_idx: torch.Tensor,
        embeddings: torch.Tensor = None,
        coarse_loss: bool = True,
    ):
        """
        output: [N, C] log-probabilities (log_softmax).
        embeddings: [N, D] raw features from model.get_embeddings().
        labels: [N] ground-truth class labels.
        coarsening_matrix: [Nc, N] coarsening matrix.
        train_idx: indices of coarsened nodes used for classification loss.
        """

        # 1. Classification loss
        loss_cls = self.class_loss(output[train_idx], labels[train_idx])
        if not coarse_loss:
            return loss_cls

        # 2. Embedding normalization
        # with torch.no_grad():
        #     supernodes = torch.zeros(N, dtype=torch.long, device=device)
        #     for i, j in zip(*coarsening_matrix.indices()):
        #         supernodes[j] = i

        loss_coarse = -torch.mean(torch.sum(embeddings**2, dim=1))
        N = embeddings.shape[0]

        # Adaptive negative sampling (scales with graph size, capped for efficiency)
        n_sample = min(max(int(N * 0.05), 100), 1000)

        # Sample distinct pairs for better negative sampling
        sampled_indices1 = torch.randint(0, N, (n_sample,), device=embeddings.device)
        sampled_indices2 = torch.randint(0, N, (n_sample,), device=embeddings.device)

        # Ensure we're sampling different nodes (avoid i==j)
        mask = sampled_indices1 != sampled_indices2
        if mask.sum() > 0:
            embeddings1 = embeddings[sampled_indices1[mask]]
            embeddings2 = embeddings[sampled_indices2[mask]]
            loss_coarse += torch.mean(torch.sum(embeddings1 * embeddings2, dim=1))
        # for _ in range(n_sample):
        # i, j =
        # emb_i = embeddings[i]
        # emb_j = embeddings[j]
        # sim = F.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0)).squeeze()

        # if supernodes[i] != supernodes[j]:
        # loss_coarse += sim
        # else:
        #     loss_coarse += sim
        # count += 1

        # loss_coarse = (
        #     loss_coarse / count if count > 0 else torch.tensor(0.0, device=device)
        # )
        return loss_cls + self.coarse_weight * loss_coarse


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
