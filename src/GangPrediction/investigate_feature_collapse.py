"""Diagnose why filtered-moment Fisher underperforms raw-feature LDA.

The per-pattern moment ``(M_j)_kl = v_j^T A^k (X X^T) A^l v_j`` traces over the
feature covariance ``X X^T``, so the scalar energy ``theta^T M_j theta`` is the
*total* energy of the filtered feature signature summed over all f feature
dimensions.  ``theta`` (a length-(K+1) spectral filter) cannot pick a
discriminative feature direction the way LDA can.  This script isolates the two
effects -- spectral filtering vs feature-direction collapse -- by comparing the
alert-vs-normal AUC of several representations on the same patterns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.GangPrediction.experiment_utils import load_and_preprocess_data
from src.GangPrediction.sgc_detection import (
    _roc_auc,
    feature_moment_matrices,
    fisher_theta,
    normalized_adjacency,
    pattern_indicator_matrix,
    propagation_stack,
    quadratic_scores,
)


def lda_auc(
    signatures: torch.Tensor, is_alert: torch.Tensor, ridge: float = 1e-3
) -> float:
    """AUC of the best linear (Fisher/LDA) projection of (m, d) signatures."""

    pos, neg = signatures[is_alert], signatures[~is_alert]
    if pos.shape[0] < 1 or neg.shape[0] < 1:
        return float("nan")
    gap = pos.mean(0) - neg.mean(0)
    pooled = (
        (pos - pos.mean(0)).T @ (pos - pos.mean(0))
        + (neg - neg.mean(0)).T @ (neg - neg.mean(0))
    ) / max(pos.shape[0] + neg.shape[0] - 2, 1)
    eye = torch.eye(pooled.shape[0], dtype=pooled.dtype)
    scale = float(torch.diagonal(pooled).mean().clamp_min(1e-12))
    w = torch.linalg.solve(pooled + ridge * scale * eye, gap)
    s = signatures @ w
    return _roc_auc(s[is_alert].tolist(), s[~is_alert].tolist())


def split_lda_auc(
    signatures: torch.Tensor, is_alert: torch.Tensor, seed: int = 0, ridge: float = 1e-3
) -> float:
    """Train LDA on half, score the held-out half (guards against overfit AUC)."""

    g = torch.Generator().manual_seed(seed)
    m = signatures.shape[0]
    perm = torch.randperm(m, generator=g)
    train, test = perm[: m // 2], perm[m // 2 :]
    pos = signatures[train][is_alert[train]]
    neg = signatures[train][~is_alert[train]]
    if pos.shape[0] < 2 or neg.shape[0] < 2:
        return float("nan")
    gap = pos.mean(0) - neg.mean(0)
    pooled = (
        (pos - pos.mean(0)).T @ (pos - pos.mean(0))
        + (neg - neg.mean(0)).T @ (neg - neg.mean(0))
    ) / max(pos.shape[0] + neg.shape[0] - 2, 1)
    eye = torch.eye(pooled.shape[0], dtype=pooled.dtype)
    scale = float(torch.diagonal(pooled).mean().clamp_min(1e-12))
    w = torch.linalg.solve(pooled + ridge * scale * eye, gap)
    s = signatures[test] @ w
    ta = is_alert[test]
    return _roc_auc(s[ta].tolist(), s[~ta].tolist())


def energy_auc(signatures: torch.Tensor, is_alert: torch.Tensor) -> float:
    """AUC of the squared-norm (energy) of each signature -- the scalar the
    pipeline collapses to.  Sums over feature dims, discarding direction."""

    e = signatures.square().sum(dim=1)
    return _roc_auc(e[is_alert].tolist(), e[~is_alert].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="tutorial_demo16")
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    root = Path.cwd() / "experiments" / args.experiment
    graph, at, nt, ate, nte = load_and_preprocess_data(
        data_dir=root / "config",
        patterns_dir=root,
        train_ratio=args.train_ratio,
        to_undirected=True,
        remove_overlaps=False,
        device=torch.device(args.device),
    )
    patterns = at + nt + ate + nte
    X = graph.x.to(torch.float64)
    n, f = X.shape
    is_alert = torch.tensor([p.label == "alert" for p in patterns])
    K = args.degree

    adj = normalized_adjacency(
        graph.edge_index, int(graph.num_nodes), getattr(graph, "edge_weight", None)
    ).to(torch.float64)
    V = pattern_indicator_matrix(patterns, n, dtype=torch.float64, device=adj.device)
    propagated = propagation_stack(adj, V, K)  # list of (n, m), k=0..K

    # Signature stack S[k] = X^T A^k V  -> (K+1, f, m).
    stack = torch.stack([X.T @ p for p in propagated], dim=0)  # (K+1, f, m)
    m = stack.shape[2]

    # Fitted Fisher theta on the feature moments (the pipeline's discriminator).
    moments = feature_moment_matrices(propagated, X)
    labels = [p.label for p in patterns]
    theta, sep = fisher_theta(moments, labels)
    filtered = torch.einsum("k,kfm->mf", theta, stack)  # (m, f) signature per pattern

    # Raw mean features over member nodes (no propagation), (m, f).
    raw = torch.stack(
        [X[torch.as_tensor(p.node_indices, dtype=torch.long)].mean(0) for p in patterns]
    )

    # Full joint (K+1)*f signature per pattern (richest linear representation).
    joint = stack.permute(2, 0, 1).reshape(m, (K + 1) * f)  # (m, (K+1)*f)

    print(
        f"experiment={args.experiment}  patterns={m} (alert={int(is_alert.sum())})  "
        f"features f={f}  degree K={K}"
    )
    print(f"fitted Fisher separation ratio (mean-moment S+/S-) = {sep:.4g}\n")

    print("alert-vs-normal AUC by representation (in-sample unless noted):")
    print(
        f"  [pipeline]  energy theta^T M_j theta (filtered, sum over f) : "
        f"{quadratic_auc(moments, theta, is_alert):.3f}"
    )
    print(
        f"  energy ||filtered signature||^2 (== above, check)           : "
        f"{energy_auc(filtered, is_alert):.3f}"
    )
    print(
        f"  LDA on filtered signature y_j in R^f (picks feature dir)     : "
        f"{lda_auc(filtered, is_alert):.3f}   "
        f"(split: {split_lda_auc(filtered, is_alert):.3f})"
    )
    print()
    print(
        f"  energy ||raw mean feature||^2 (no propagation, sum over f)   : "
        f"{energy_auc(raw, is_alert):.3f}"
    )
    print(
        f"  LDA on raw mean feature in R^f                               : "
        f"{lda_auc(raw, is_alert):.3f}   "
        f"(split: {split_lda_auc(raw, is_alert):.3f})"
    )
    print()
    print(
        f"  LDA on full joint (K+1)*f signature stack                    : "
        f"{lda_auc(joint, is_alert):.3f}   "
        f"(split: {split_lda_auc(joint, is_alert):.3f})"
    )
    print()

    # Per-power energy AUC: which spectral band carries the (collapsed) signal.
    print("per-power energy ||X^T A^k v_j||^2 AUC (each k, sum over f):")
    for k in range(K + 1):
        sig_k = stack[k].T  # (m, f)
        print(f"  k={k}: {energy_auc(sig_k, is_alert):.3f}", end="   ")
        if k % 3 == 2:
            print()
    print("\n")

    print("interpretation:")
    print("  - If LDA(filtered) >> energy(filtered), the loss is the scalar-energy")
    print("    collapse (trace over X X^T), not the graph filtering: theta cannot")
    print("    select the discriminative feature direction, only a spectral band.")
    print("  - If LDA(raw) >> energy(raw) too, the same collapse hurts even with no")
    print("    propagation, confirming it is a feature-direction problem.")


def quadratic_auc(
    moments: torch.Tensor, theta: torch.Tensor, is_alert: torch.Tensor
) -> float:
    e = quadratic_scores(moments, theta)
    return _roc_auc(e[is_alert].tolist(), e[~is_alert].tolist())


if __name__ == "__main__":
    main()
