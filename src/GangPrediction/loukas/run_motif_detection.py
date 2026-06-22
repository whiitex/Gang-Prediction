r"""Motif-detection comparison on AMLGenTex (tutorial_demo16).

Pipeline (all from-scratch building blocks live in this ``loukas`` package):

1. Load the AMLGenTex ``tutorial_demo16`` graph and its planted patterns.
2. Build TWO Loukas RSA coarsening targets on the same graph:
     * baseline  R = span(U_K)          -- the K lowest Laplacian eigenvectors
                                           (``loukas_coarsening.bottom_k_eigenvectors``).
     * proposed  R = span(g_theta*(A_hat) X)  -- the trainable-SGC subspace of
                                           Section 12 (``sgc_subspace.train_sgc_target``).
3. Run the from-scratch local-variation coarsening (``loukas_coarsen``) for both
   targets across a sweep of reduction ratios.
4. A pattern is "detected" when, after coarsening, a single super-node captures
   at least ``tau`` of its nodes (recall) AND is at least ``tau`` pure
   (precision) -- the same notion of detection used elsewhere in GangPrediction.
5. Report and plot the per-pattern-type detection rate for both targets.

Run from the repo root:

    python -m src.GangPrediction.loukas.run_motif_detection
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

# --- repo path setup (mirrors src/GangPrediction/main.py) ------------------
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from src.GangPrediction.experiment_utils import load_and_preprocess_data
from src.GangPrediction.loukas.loukas_coarsening import (
    bottom_k_eigenvectors,
    build_laplacian,
    loukas_coarsen,
)
from src.GangPrediction.loukas.sgc_subspace import train_sgc_target


# ---------------------------------------------------------------------------
# Graph / pattern preparation
# ---------------------------------------------------------------------------
def torch_graph_to_scipy(G):
    """Symmetric binary scipy adjacency from a PyG graph, isolated nodes dropped.

    Returns ``(W_active, full_to_active)`` where ``full_to_active[i]`` is the
    active index of original vertex ``i`` (or -1 if it was isolated).
    """
    ei = G.edge_index.detach().cpu().numpy()
    N = int(G.num_nodes)
    data = np.ones(ei.shape[1])
    W = sp.coo_matrix((data, (ei[0], ei[1])), shape=(N, N)).tocsr()
    W = W + W.T
    W.data[:] = 1.0
    W.setdiag(0.0)
    W.eliminate_zeros()

    deg = np.asarray(W.sum(axis=1)).ravel()
    active = np.nonzero(deg > 0)[0]
    full_to_active = np.full(N, -1, dtype=int)
    full_to_active[active] = np.arange(len(active))
    W_active = W[active][:, active]
    return W_active.tocsr(), full_to_active


def remap_patterns(patterns, full_to_active, min_size=2):
    """Map Pattern objects to active-graph node lists, grouped by type."""
    out = []
    for p in patterns:
        nodes = [full_to_active[int(i)] for i in p.node_indices]
        nodes = [i for i in nodes if i >= 0]
        if len(nodes) >= min_size:
            out.append((p.pattern_type, np.asarray(sorted(set(nodes)))))
    return out


# ---------------------------------------------------------------------------
# Detection metric
# ---------------------------------------------------------------------------
def detection_rate(node_to_super, patterns, n_super, tau=0.5):
    """Per-type detection rate and overall, given a node->super-node map.

    A pattern is detected iff its dominant super-node holds >= tau of the pattern
    (recall) and is >= tau pure (precision).
    """
    super_size = np.bincount(node_to_super, minlength=n_super)

    by_type = defaultdict(lambda: [0, 0])  # type -> [detected, total]
    recalls, precisions = [], []
    for ptype, nodes in patterns:
        supers = node_to_super[nodes]
        sids, counts = np.unique(supers, return_counts=True)
        best = counts.argmax()
        max_count = counts[best]
        dom = sids[best]
        recall = max_count / len(nodes)
        precision = max_count / super_size[dom]
        recalls.append(recall)
        precisions.append(precision)
        detected = (recall >= tau) and (precision >= tau)
        by_type[ptype][0] += int(detected)
        by_type[ptype][1] += 1

    rates = {t: d / n for t, (d, n) in by_type.items()}
    total_det = sum(d for d, _ in by_type.values())
    total_n = sum(n for _, n in by_type.values())
    overall = total_det / total_n if total_n else 0.0
    return {
        "rates": rates,
        "counts": dict(by_type),
        "overall": overall,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
    }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run(
    experiment="tutorial_demo16",
    subspace_dim=100,
    poly_degree=12,
    reductions=(0.3, 0.5, 0.6, 0.7, 0.8, 0.9),
    operating_reduction=0.7,
    tau=0.5,
    out_dir="results/loukas_motif_detection",
    seed=0,
):
    os.makedirs(out_dir, exist_ok=True)
    exp_root = project_root / "experiments" / experiment

    print("=" * 70)
    print(f"Loading AMLGenTex {experiment}")
    print("=" * 70)
    G, alert_train, normal_train, alert_test, normal_test = load_and_preprocess_data(
        data_dir=exp_root / "config",
        patterns_dir=exp_root,
        train_ratio=0.5,
        to_undirected=True,
        remove_overlaps=False,
        device="cpu",
    )

    W, f2a = torch_graph_to_scipy(G)
    N = W.shape[0]
    L, _ = build_laplacian(W)
    print(f"Active graph: N={N}  edges={int(W.nnz // 2)}")

    train_patterns = remap_patterns(alert_train + normal_train, f2a)
    test_patterns = remap_patterns(alert_test + normal_test, f2a)
    print(f"Train patterns: {len(train_patterns)}  Test patterns: {len(test_patterns)}")

    # ---- target 1: baseline top-K Laplacian eigenvectors --------------------
    print("\nBuilding baseline target  R = span(U_K) ...")
    B_eig = bottom_k_eigenvectors(L, subspace_dim)

    # ---- target 2: trainable-SGC subspace (Section 12) ----------------------
    print("Building SGC target       R = span(g_theta*(A_hat) X) ...")
    sgc = train_sgc_target(
        W,
        positive_node_sets=[nodes for _, nodes in train_patterns],
        K=poly_degree,
        width=subspace_dim,
        n_neg_per_pos=1,
        mode="difference",
        seed=seed,
    )
    B_sgc = sgc.basis
    theta = sgc.fit.theta
    print(
        f"  learned theta (deg {poly_degree}): "
        + np.array2string(theta, precision=3, suppress_small=True)
    )
    print(
        f"  C_SGC(K)={sgc.fit.energy_sgc:.4f}  C(theta*)={sgc.fit.energy_learned:.4f}  "
        f"discriminative gap lambda_max(dM)={sgc.fit.discriminative_gap:.4e}"
    )

    targets = {"top-K eigvecs": B_eig, "trainable-SGC": B_sgc}

    # ---- sweep reductions ---------------------------------------------------
    sweep = {name: [] for name in targets}
    all_types = sorted({t for t, _ in test_patterns})
    bar_rates = {name: {} for name in targets}

    print("\n" + "=" * 70)
    print("Coarsening + detection sweep")
    print("=" * 70)
    for r in reductions:
        for name, B in targets.items():
            res = loukas_coarsen(L, B, reduction=r, epsilon=np.inf)
            det = detection_rate(
                res.node_to_supernode, test_patterns, res.n_coarse, tau=tau
            )
            sweep[name].append(det["overall"])
            if abs(r - operating_reduction) < 1e-9:
                bar_rates[name] = det["rates"]
            print(
                f"  r={r:.2f}  {name:>16s}  n={res.n_coarse:5d}  "
                f"overall={det['overall']:.3f}  "
                f"recall={det['mean_recall']:.3f}  prec={det['mean_precision']:.3f}  "
                f"eps={res.epsilon:.2f}"
            )

    # ---- report -------------------------------------------------------------
    report_path = Path(out_dir) / "detection_report.txt"
    with open(report_path, "w") as fh:
        fh.write(f"AMLGenTex {experiment} — motif detection by coarsening target\n")
        fh.write(f"subspace_dim={subspace_dim} poly_degree={poly_degree} tau={tau}\n")
        fh.write(f"operating reduction r={operating_reduction}\n\n")
        fh.write(f"learned theta = {np.array2string(theta, precision=4)}\n\n")
        header = f"{'pattern_type':>16s} | " + " | ".join(
            f"{n:>14s}" for n in targets
        )
        fh.write(header + "\n")
        fh.write("-" * len(header) + "\n")
        for t in all_types:
            row = f"{t:>16s} | " + " | ".join(
                f"{bar_rates[n].get(t, 0.0):14.3f}" for n in targets
            )
            fh.write(row + "\n")
        fh.write("\nOverall detection rate vs reduction:\n")
        fh.write(f"{'reduction':>10s} | " + " | ".join(f"{n:>14s}" for n in targets) + "\n")
        for i, r in enumerate(reductions):
            fh.write(
                f"{r:10.2f} | " + " | ".join(f"{sweep[n][i]:14.3f}" for n in targets) + "\n"
            )
    print(f"\nReport written to {report_path}")

    # ---- plots --------------------------------------------------------------
    _plot_bars(bar_rates, all_types, targets, operating_reduction, out_dir)
    _plot_sweep(sweep, reductions, targets, out_dir)
    _plot_filter(theta, out_dir)
    print(f"Plots written to {out_dir}/")

    return {"sweep": sweep, "bar_rates": bar_rates, "theta": theta}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_bars(bar_rates, all_types, targets, r, out_dir):
    x = np.arange(len(all_types))
    width = 0.8 / len(targets)
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(all_types)), 5))
    for k, (name, rates) in enumerate(bar_rates.items()):
        vals = [rates.get(t, 0.0) for t in all_types]
        ax.bar(x + k * width, vals, width, label=name)
    ax.set_xticks(x + width * (len(targets) - 1) / 2)
    ax.set_xticklabels(all_types, rotation=30, ha="right")
    ax.set_ylabel("detection rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-pattern detection rate @ reduction r={r}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "detection_rate_by_type.png", dpi=150)
    plt.close(fig)


def _plot_sweep(sweep, reductions, targets, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in targets:
        ax.plot(reductions, sweep[name], "o-", label=name)
    ax.set_xlabel("reduction ratio  r = 1 - n/N")
    ax.set_ylabel("overall detection rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Detection rate vs coarsening aggressiveness")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "detection_rate_vs_reduction.png", dpi=150)
    plt.close(fig)


def _plot_filter(theta, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stem(np.arange(len(theta)), theta)
    ax.set_xlabel("polynomial degree k")
    ax.set_ylabel(r"$\theta_k$")
    ax.set_title(r"Learned SGC filter coefficients $g_\theta(\hat A)=\sum_k\theta_k\hat A^k$")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "learned_filter_coeffs.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
