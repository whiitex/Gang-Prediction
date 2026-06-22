"""Confirm whether node features carry signal correlated with the patterns.

The feature-aware Gram ``G(theta) = P^T g_theta(A_hat) X X^T g_theta(A_hat) P``
only helps over the isotropic ``Sigma_X = I`` case if the node features ``X``
actually differ between alert and normal patterns.  This script answers two
questions with explicit, falsifiable tests:

1.  *Machinery check (synthetic controls).*  On a synthetic graph with planted
    patterns we run the exact pipeline (``feature_moment_matrices`` ->
    ``fisher_theta`` -> ``quadratic_scores``) under two feature regimes:

      * negative control -- features independent of the patterns
        (expected alert-vs-normal AUC ~ 0.5);
      * positive control -- a feature signal injected on alert-pattern nodes
        (expected AUC >> 0.5).

    Passing both confirms the features enter the moments correctly: no signal
    in => no separation, signal in => separation.

2.  *Real-data permutation test.*  On a real experiment we measure the
    alert-vs-normal separation of the per-pattern feature signatures and compare
    it against a label-permutation null.  A separation far above the null
    (small empirical p-value) means the real node features are genuinely
    correlated with the patterns; a separation inside the null means they are
    not (so ``--feature-mode node`` cannot beat ``gaussian`` for a structural
    reason).

Run::

    python -m src.GangPrediction.simulate_feature_correlation \
        --experiment tutorial_demo16 --degree 8 --permutations 2000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from src.GangPrediction.experiment_utils import load_and_preprocess_data
from src.GangPrediction.sgc_detection import (
    _roc_auc,
    feature_moment_matrices,
    fisher_theta,
    normalized_adjacency,
    propagation_stack,
    quadratic_scores,
)
from src.GangPrediction.utils.utils import save_path


@dataclass
class _SimPattern:
    """Minimal pattern compatible with the SGC pipeline (id/nodes/label)."""

    pattern_id: int
    nodes: torch.Tensor
    label: str
    pattern_type: str = "sim"

    @property
    def id(self) -> int:
        return self.pattern_id

    @property
    def node_indices(self) -> torch.Tensor:
        return self.nodes


# --------------------------------------------------------------------------- #
# Generic feature-separation statistics (theta-free).
# --------------------------------------------------------------------------- #
def pattern_mean_features(
    features: torch.Tensor, patterns: Sequence[object]
) -> torch.Tensor:
    """Average the node features over each pattern's member nodes -> (m, f)."""

    rows = []
    for pattern in patterns:
        nodes = torch.as_tensor(pattern.node_indices, dtype=torch.long)
        rows.append(features[nodes].mean(dim=0))
    return torch.stack(rows, dim=0)


def hotelling_t2(
    signatures: torch.Tensor, is_alert: torch.Tensor, *, ridge: float = 1e-3
) -> float:
    """Two-sample Hotelling ``T^2`` separation of alert vs normal signatures.

    A single scalar measuring how far apart the two class-mean feature vectors
    are in pooled-covariance (Mahalanobis) units.  It needs no threshold and no
    fitting, so it is safe to drop into a label-permutation null.
    """

    pos = signatures[is_alert]
    neg = signatures[~is_alert]
    n_pos, n_neg = pos.shape[0], neg.shape[0]
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    mean_gap = pos.mean(dim=0) - neg.mean(dim=0)
    pooled = (
        (pos - pos.mean(dim=0)).T @ (pos - pos.mean(dim=0))
        + (neg - neg.mean(dim=0)).T @ (neg - neg.mean(dim=0))
    ) / max(n_pos + n_neg - 2, 1)
    eye = torch.eye(pooled.shape[0], dtype=pooled.dtype)
    scale = float(torch.diagonal(pooled).mean().clamp_min(1e-12))
    solved = torch.linalg.solve(pooled + ridge * scale * eye, mean_gap)
    factor = (n_pos * n_neg) / (n_pos + n_neg)
    return float(factor * (mean_gap @ solved))


def lda_separation_auc(
    signatures: torch.Tensor, is_alert: torch.Tensor, *, ridge: float = 1e-3
) -> float:
    """AUC of the best linear (Fisher/LDA) projection of the signatures."""

    pos = signatures[is_alert]
    neg = signatures[~is_alert]
    if pos.shape[0] < 1 or neg.shape[0] < 1:
        return float("nan")
    mean_gap = pos.mean(dim=0) - neg.mean(dim=0)
    pooled = (
        (pos - pos.mean(dim=0)).T @ (pos - pos.mean(dim=0))
        + (neg - neg.mean(dim=0)).T @ (neg - neg.mean(dim=0))
    ) / max(pos.shape[0] + neg.shape[0] - 2, 1)
    eye = torch.eye(pooled.shape[0], dtype=pooled.dtype)
    scale = float(torch.diagonal(pooled).mean().clamp_min(1e-12))
    direction = torch.linalg.solve(pooled + ridge * scale * eye, mean_gap)
    scores = signatures @ direction
    return _roc_auc(scores[is_alert].tolist(), scores[~is_alert].tolist())


def moment_separation_auc(
    adjacency: torch.Tensor,
    patterns: Sequence[object],
    features: torch.Tensor | None,
    *,
    degree: int,
) -> float:
    """Full-pipeline AUC: feature moments -> Fisher theta -> energy separation.

    This is the exact path used in training, so a high AUC here confirms the
    features flow correctly through ``feature_moment_matrices`` and the closed
    -form ``fisher_theta`` discriminator.
    """

    from src.GangPrediction.sgc_detection import pattern_indicator_matrix

    V = pattern_indicator_matrix(
        patterns, adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device
    )
    propagated = propagation_stack(adjacency, V, degree)
    if features is not None and features.dim() == 1:
        features = features.unsqueeze(1)
    moments = feature_moment_matrices(propagated, features)
    labels = [p.label for p in patterns]
    theta, _ = fisher_theta(moments, labels)
    energies = quadratic_scores(moments, theta)
    is_alert = torch.tensor([lbl == "alert" for lbl in labels])
    return _roc_auc(energies[is_alert].tolist(), energies[~is_alert].tolist())


# --------------------------------------------------------------------------- #
# Part 1 -- synthetic positive / negative controls.
# --------------------------------------------------------------------------- #
def _synthetic_graph(
    n_nodes: int, avg_degree: int, generator: torch.Generator
) -> torch.Tensor:
    """A random symmetric edge_index for an undirected Erdos-Renyi-ish graph."""

    n_edges = n_nodes * avg_degree // 2
    src = torch.randint(0, n_nodes, (n_edges,), generator=generator)
    dst = torch.randint(0, n_nodes, (n_edges,), generator=generator)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    edge_index = torch.stack((torch.cat((src, dst)), torch.cat((dst, src))), dim=0)
    return edge_index


def _synthetic_patterns(
    n_nodes: int,
    n_alert: int,
    n_normal: int,
    size: int,
    generator: torch.Generator,
) -> List[_SimPattern]:
    patterns: List[_SimPattern] = []
    pid = 0
    for label, count in (("alert", n_alert), ("normal", n_normal)):
        for _ in range(count):
            nodes = torch.randperm(n_nodes, generator=generator)[:size]
            patterns.append(_SimPattern(pid, nodes, label))
            pid += 1
    return patterns


def run_synthetic_controls(
    *,
    n_nodes: int = 1200,
    avg_degree: int = 8,
    n_alert: int = 60,
    n_normal: int = 60,
    pattern_size: int = 12,
    n_features: int = 16,
    signal: float = 1.5,
    degree: int = 6,
    seed: int = 0,
) -> dict:
    """Negative control (no signal) vs positive control (alert-correlated)."""

    generator = torch.Generator().manual_seed(seed)
    edge_index = _synthetic_graph(n_nodes, avg_degree, generator)
    adjacency = normalized_adjacency(edge_index, n_nodes, None).to(torch.float64)
    patterns = _synthetic_patterns(n_nodes, n_alert, n_normal, pattern_size, generator)
    is_alert = torch.tensor([p.label == "alert" for p in patterns])

    base = torch.randn(n_nodes, n_features, generator=generator, dtype=torch.float64)

    # Negative control: features independent of which patterns are alerts.
    neg_features = base.clone()

    # Positive control: add a fixed feature offset on every alert-pattern node.
    pos_features = base.clone()
    direction = torch.zeros(n_features, dtype=torch.float64)
    direction[: max(1, n_features // 4)] = 1.0
    direction = direction / direction.norm()
    alert_nodes = torch.unique(
        torch.cat([p.nodes for p in patterns if p.label == "alert"])
    )
    pos_features[alert_nodes] += signal * direction

    results = {}
    for name, feats in (("negative", neg_features), ("positive", pos_features)):
        results[name] = {
            "mean_feature_lda_auc": lda_separation_auc(
                pattern_mean_features(feats, patterns), is_alert
            ),
            "pipeline_moment_auc": moment_separation_auc(
                adjacency, patterns, feats, degree=degree
            ),
            "isotropic_moment_auc": moment_separation_auc(
                adjacency, patterns, None, degree=degree
            ),
        }
    return results


# --------------------------------------------------------------------------- #
# Part 2 -- real-data permutation test.
# --------------------------------------------------------------------------- #
def run_real_permutation_test(
    experiment: str,
    *,
    degree: int,
    permutations: int,
    train_ratio: float,
    seed: int,
    device: torch.device,
) -> dict:
    experiment_root = Path.cwd() / "experiments" / experiment
    graph, alert_train, normal_train, alert_test, normal_test = (
        load_and_preprocess_data(
            data_dir=experiment_root / "config",
            patterns_dir=experiment_root,
            train_ratio=train_ratio,
            to_undirected=True,
            remove_overlaps=False,
            device=device,
        )
    )
    patterns = alert_train + normal_train + alert_test + normal_test
    features = graph.x.to(torch.float64)
    is_alert = torch.tensor([p.label == "alert" for p in patterns])

    signatures = pattern_mean_features(features, patterns)
    observed_t2 = hotelling_t2(signatures, is_alert)
    observed_lda_auc = lda_separation_auc(signatures, is_alert)

    adjacency = normalized_adjacency(
        graph.edge_index, int(graph.num_nodes), getattr(graph, "edge_weight", None)
    ).to(torch.float64)
    observed_pipeline_auc = moment_separation_auc(
        adjacency, patterns, features, degree=degree
    )
    isotropic_pipeline_auc = moment_separation_auc(
        adjacency, patterns, None, degree=degree
    )

    # Label-permutation null for T^2 (does the *label* carry feature signal?).
    generator = torch.Generator().manual_seed(seed)
    n = is_alert.numel()
    n_pos = int(is_alert.sum())
    null = torch.empty(permutations, dtype=torch.float64)
    for i in range(permutations):
        perm = torch.randperm(n, generator=generator)
        shuffled = torch.zeros(n, dtype=torch.bool)
        shuffled[perm[:n_pos]] = True
        null[i] = hotelling_t2(signatures, shuffled)

    p_value = float((null >= observed_t2).double().mean().clamp_min(1.0 / permutations))

    # Node-level point-biserial correlation between membership and each feature.
    alert_nodes = torch.unique(
        torch.cat(
            [
                torch.as_tensor(p.node_indices, dtype=torch.long)
                for p in patterns
                if p.label == "alert"
            ]
        )
    )
    membership = torch.zeros(features.shape[0], dtype=torch.float64)
    membership[alert_nodes] = 1.0
    feat_centered = features - features.mean(dim=0, keepdim=True)
    mem_centered = membership - membership.mean()
    denom = feat_centered.norm(dim=0) * mem_centered.norm()
    point_biserial = (feat_centered * mem_centered.unsqueeze(1)).sum(
        dim=0
    ) / denom.clamp_min(1e-12)

    return {
        "n_patterns": n,
        "n_alert": n_pos,
        "n_normal": n - n_pos,
        "n_features": features.shape[1],
        "observed_t2": observed_t2,
        "null_mean_t2": float(null.mean()),
        "null_std_t2": float(null.std()),
        "p_value": p_value,
        "observed_lda_auc": observed_lda_auc,
        "observed_pipeline_auc": observed_pipeline_auc,
        "isotropic_pipeline_auc": isotropic_pipeline_auc,
        "null_t2": null,
        "point_biserial": point_biserial,
    }


def _save_plot(synthetic: dict, real: dict, *, experiment: str, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: synthetic controls.
    labels = ["negative\n(no signal)", "positive\n(alert signal)"]
    pipeline = [
        synthetic["negative"]["pipeline_moment_auc"],
        synthetic["positive"]["pipeline_moment_auc"],
    ]
    meanfeat = [
        synthetic["negative"]["mean_feature_lda_auc"],
        synthetic["positive"]["mean_feature_lda_auc"],
    ]
    x = torch.arange(2).tolist()
    axes[0].bar([i - 0.2 for i in x], pipeline, width=0.4, label="pipeline moments")
    axes[0].bar([i + 0.2 for i in x], meanfeat, width=0.4, label="mean-feature LDA")
    axes[0].axhline(0.5, color="grey", ls="--", lw=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("alert-vs-normal AUC")
    axes[0].set_title("Synthetic controls\n(0.5 = no feature signal)")
    axes[0].legend(fontsize=8)

    # Panel 2: real permutation null vs observed T^2.
    null = real["null_t2"].tolist()
    axes[1].hist(
        null,
        bins=40,
        color="tab:orange",
        alpha=0.7,
        density=True,
        label="label-permuted null",
    )
    axes[1].axvline(
        real["observed_t2"],
        color="tab:blue",
        lw=2,
        label=f"observed (p={real['p_value']:.3g})",
    )
    axes[1].set_xlabel(r"Hotelling $T^2$ (alert vs normal)")
    axes[1].set_ylabel("density")
    axes[1].set_title(f"{experiment}: real features vs null")
    axes[1].legend(fontsize=8)

    # Panel 3: top per-feature point-biserial correlations.
    pb = real["point_biserial"]
    order = torch.argsort(pb.abs(), descending=True)[:15]
    vals = pb[order].tolist()
    axes[2].barh(range(len(vals)), vals, color="tab:green")
    axes[2].axvline(0.0, color="grey", lw=1)
    axes[2].set_yticks(range(len(vals)))
    axes[2].set_yticklabels([f"feat {int(i)}" for i in order.tolist()], fontsize=8)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("point-biserial corr (alert membership)")
    axes[2].set_title("Node-level feature/pattern correlation")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="tutorial_demo16")
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--train-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("=" * 70)
    print("Part 1: synthetic controls (validate the feature plumbing)")
    print("=" * 70)
    synthetic = run_synthetic_controls(degree=args.degree, seed=args.seed)
    for name in ("negative", "positive"):
        r = synthetic[name]
        print(
            f"  {name:8s}  pipeline-moment AUC={r['pipeline_moment_auc']:.3f}  "
            f"mean-feature LDA AUC={r['mean_feature_lda_auc']:.3f}  "
            f"isotropic AUC={r['isotropic_moment_auc']:.3f}"
        )
    neg = synthetic["negative"]["pipeline_moment_auc"]
    pos = synthetic["positive"]["pipeline_moment_auc"]
    # In-sample Fisher slightly inflates the no-signal control, so judge the
    # plumbing by the gap the injected signal opens up, not an absolute floor.
    plumbing_ok = (pos - neg) > 0.2 and pos > 0.8
    print(
        f"  -> plumbing {'OK' if plumbing_ok else 'SUSPECT'}: injected signal "
        f"lifts AUC by {pos - neg:+.3f} over the no-signal control."
    )

    print("\n" + "=" * 70)
    print(f"Part 2: real data permutation test ({args.experiment})")
    print("=" * 70)
    real = run_real_permutation_test(
        args.experiment,
        degree=args.degree,
        permutations=args.permutations,
        train_ratio=args.train_ratio,
        seed=args.seed,
        device=torch.device(args.device),
    )
    print(
        f"  patterns: {real['n_patterns']} "
        f"(alert={real['n_alert']}, normal={real['n_normal']}), "
        f"features={real['n_features']}"
    )
    print(
        f"  observed Hotelling T^2 = {real['observed_t2']:.3f}  "
        f"(null mean={real['null_mean_t2']:.3f}, std={real['null_std_t2']:.3f})"
    )
    z = (real["observed_t2"] - real["null_mean_t2"]) / max(real["null_std_t2"], 1e-9)
    print(f"  permutation p-value = {real['p_value']:.4g}   (z = {z:.2f})")
    print(
        f"  mean-feature LDA AUC = {real['observed_lda_auc']:.3f}; "
        f"pipeline-moment AUC (node feats) = {real['observed_pipeline_auc']:.3f}  "
        f"vs isotropic = {real['isotropic_pipeline_auc']:.3f}"
    )
    verdict = (
        "features ARE correlated with the patterns"
        if real["p_value"] < 0.05
        else "no significant feature/pattern correlation"
    )
    print(f"  -> {verdict} (alpha=0.05).")

    out_dir = Path(args.output) if args.output else Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_out = out_dir / "feature_correlation.png"
    _save_plot(synthetic, real, experiment=args.experiment, output=plot_out)
    print(f"\nPlot: {plot_out}")


if __name__ == "__main__":
    main()
