"""Coarsening diagnostics: five self-contained experiments to understand
*why* spectral eigenvectors matter for gang detection and *why* fast epsilon
schedules outperform gradual ones.

All experiments are pure graph-structure analyses — no GNN training needed.

Experiments
-----------
1. spectral_fingerprint         – how much energy do pattern indicators carry at each eigenvector
2. merge_recall_precision_vs_K  – does using more eigenvectors (larger K) improve merge recall/precision?
3. plot_epsilon_schedules       – visualise every schedule power; when does each "unlock"?
4. epsilon_schedule_ablation    – run multilevel coarsening under each schedule; track merge recall/precision
5. supernode_entropy_analysis   – track label entropy inside super-nodes across levels & schedules
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

from GangPrediction.utils.utils import *
from GangPrediction.coarsening_utils import (
    calc_B,
    coarse_one_level,
    get_coarsening_matrix,
    graph_params,
    sparse_eye,
)
from src.GangPrediction.pattern_models import Pattern, create_pattern


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_graph_params(G: Data) -> Data:
    """Attach W, L, dw to a PyG graph if not already present."""
    if not hasattr(G, "L") or G.L is None:
        if not hasattr(G, "edge_weight") or G.edge_weight is None:
            G.edge_weight = torch.ones(G.edge_index.size(1), dtype=torch.float32)
        G.W, G.L, G.dw = graph_params(G)
    return G


def build_pattern_indicator(
    num_nodes: int,
    patterns: List[Pattern],
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """Return a dict {pattern_id: indicator_vector (length num_nodes)}.

    Each entry is a binary (or L2-normalised) indicator over the pattern nodes.
    """
    indicators: Dict[str, np.ndarray] = {}
    for p in patterns:
        v = np.zeros(num_nodes, dtype=np.float64)
        nodes = _to_numpy(p.node_indices).astype(int)
        v[nodes] = 1.0
        if normalize and v.sum() > 0:
            v = v / np.linalg.norm(v)
        indicators[str(p.pattern_id)] = v
    return indicators


def _clone_patterns(patterns: Optional[List[Pattern]]) -> List[Pattern]:
    """Create fresh pattern instances so diagnostics do not mutate caller state."""
    clones: List[Pattern] = []
    for pattern in patterns or []:
        clones.append(
            create_pattern(
                pattern_id=pattern.pattern_id,
                nodes=_to_numpy(pattern.node_indices).astype(int),
                pattern_type=pattern.pattern_type,
                label=pattern.label,
            )
        )
    return clones


def _pattern_pseudo_labels(
    num_nodes: int,
    alert_patterns: List[Pattern],
    normal_patterns: Optional[List[Pattern]],
    device: torch.device,
) -> torch.Tensor:
    """Build deterministic pseudo-labels so Pattern metrics can be reused."""
    pseudo_labels = torch.zeros((num_nodes, 2), dtype=torch.float32, device=device)
    for pattern in normal_patterns or []:
        pseudo_labels[_to_numpy(pattern.node_indices).astype(int), 0] = 1.0
    for pattern in alert_patterns or []:
        pseudo_labels[_to_numpy(pattern.node_indices).astype(int), 1] = 1.0
    return pseudo_labels


def _node_to_supernode_mapping(C: torch.Tensor) -> torch.Tensor:
    """Return the current super-node index for every original node."""
    return torch.argmax(C.to_dense(), dim=0)


def compute_merge_recall_precision(
    node_to_supernode: torch.Tensor,
    alert_patterns: List[Pattern],
    normal_patterns: Optional[List[Pattern]],
) -> Dict[str, float]:
    """Aggregate pattern-level merge recall and precision using Pattern helpers."""
    eval_alert_patterns = _clone_patterns(alert_patterns)
    eval_normal_patterns = _clone_patterns(normal_patterns)
    all_patterns = eval_alert_patterns + eval_normal_patterns
    if not all_patterns:
        nan_metrics = {"recall": float("nan"), "precision": float("nan")}
        return {
            "recall": nan_metrics["recall"],
            "precision": nan_metrics["precision"],
            "overall": dict(nan_metrics),
            "alert": dict(nan_metrics),
            "normal": dict(nan_metrics),
        }

    pseudo_labels = _pattern_pseudo_labels(
        num_nodes=node_to_supernode.numel(),
        alert_patterns=eval_alert_patterns,
        normal_patterns=eval_normal_patterns,
        device=node_to_supernode.device,
    )
    for pattern in all_patterns:
        pattern.capture_level(
            node_to_supernode=node_to_supernode,
            pseudo_labels=pseudo_labels,
        )

    overall = Pattern.average_metrics(all_patterns, metric_keys=("recall", "precision"))
    alert = Pattern.average_metrics(
        eval_alert_patterns, metric_keys=("recall", "precision")
    )
    normal = Pattern.average_metrics(
        eval_normal_patterns, metric_keys=("recall", "precision")
    )
    return {
        "recall": float(overall.get("recall", float("nan"))),
        "precision": float(overall.get("precision", float("nan"))),
        "overall": {
            "recall": float(overall.get("recall", float("nan"))),
            "precision": float(overall.get("precision", float("nan"))),
        },
        "alert": {
            "recall": float(alert.get("recall", float("nan"))),
            "precision": float(alert.get("precision", float("nan"))),
        },
        "normal": {
            "recall": float(normal.get("recall", float("nan"))),
            "precision": float(normal.get("precision", float("nan"))),
        },
    }


def _count_nontrivial_merges(C: torch.Tensor) -> int:
    """Count coarse nodes that merge more than one original/current node."""
    C_dense = C.to_dense()
    merge_count = 0
    for row in range(C_dense.size(0)):
        if torch.count_nonzero(C_dense[row] > 0).item() > 1:
            merge_count += 1
    return merge_count


def _plot_multilevel_merge_metric_trajectories(
    K_values: List[int],
    trajectories: Dict[int, Dict[str, List[float]]],
    pattern_label: str,
    save_dir: str,
    name_prefix: str,
) -> None:
    """Plot recall/precision vs K with one trajectory per coarsening level."""
    valid_levels = [
        level
        for level, metrics in trajectories.items()
        if np.isfinite(metrics["recall"]).any()
        or np.isfinite(metrics["precision"]).any()
    ]
    if not valid_levels:
        LOGGER.info(
            f"[merge_recall_precision_vs_K] no valid {pattern_label} trajectories — skipping"
        )
        return

    cmap = cm.get_cmap("viridis", len(valid_levels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for idx, level in enumerate(valid_levels):
        axes[0].plot(
            K_values,
            trajectories[level]["recall"],
            "o-",
            color=cmap(idx),
            lw=2,
            ms=5,
            label=f"level={level}",
        )
        axes[1].plot(
            K_values,
            trajectories[level]["precision"],
            "o-",
            color=cmap(idx),
            lw=2,
            ms=5,
            label=f"level={level}",
        )

    axes[0].set_xlabel("K (number of eigenvectors in basis)")
    axes[0].set_ylabel(f"{pattern_label.title()} merge recall")
    axes[0].set_title(f"{pattern_label.title()} merge recall vs K")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    axes[1].set_xlabel("K (number of eigenvectors in basis)")
    axes[1].set_ylabel(f"{pattern_label.title()} merge precision")
    axes[1].set_title(f"{pattern_label.title()} merge precision vs K")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.suptitle(
        f"{pattern_label.title()} merge quality trajectories across coarsening levels",
        y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(
        save_dir, f"{name_prefix}merge_recall_precision_vs_K_{pattern_label}.png"
    )
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    # LOGGER.info(f"  saved → {path}")


def _plot_multilevel_merge_count_trajectories(
    K_values: List[int],
    merge_counts: Dict[int, List[float]],
    save_dir: str,
    name_prefix: str,
) -> None:
    """Plot merge-count trajectories vs K with one line per level."""
    valid_levels = [
        level for level, values in merge_counts.items() if np.isfinite(values).any()
    ]
    if not valid_levels:
        return

    cmap = cm.get_cmap("plasma", len(valid_levels))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for idx, level in enumerate(valid_levels):
        ax.plot(
            K_values,
            merge_counts[level],
            "s-",
            color=cmap(idx),
            lw=2,
            ms=5,
            label=f"level={level}",
        )
    ax.set_xlabel("K (number of eigenvectors in basis)")
    ax.set_ylabel("Number of merges")
    ax.set_title("Merge count trajectories vs K")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, f"{name_prefix}merge_count_vs_K_multilevel.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    # LOGGER.info(f"  saved → {path}")


def compute_spectral_decomp(
    G: Data,
    K_max: int = 300,
    dense_threshold: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (eigenvalues, eigenvectors) of the graph Laplacian.

    For graphs with N ≤ *dense_threshold* the full dense decomposition is used.
    For larger graphs lobpcg is used (K_max smallest eigenvectors).

    Returns
    -------
    lk : ndarray, shape (K,)
    Uk : ndarray, shape (N, K)
    """
    G = _ensure_graph_params(G)
    N = G.num_nodes
    K = min(K_max, N)
    L = G.L

    if N <= dense_threshold:
        d, V = torch.linalg.eigh(L.to_dense())
        lk = _to_numpy(d[:K])
        Uk = _to_numpy(V[:, :K])
    else:
        # Use shift-invert via lobpcg to get smallest K eigenvectors.
        try:

            offset = float(2 * G.dw.max())
            T = offset * sparse_eye(N).to(L.device) - L
            X_init = torch.randn(N, K, device=L.device)
            lk_t, Uk_t = torch.lobpcg(T, k=K, X=X_init, largest=True, tol=1e-4)
            lk_t = torch.flip(offset - lk_t, [0])
            Uk_t = torch.flip(Uk_t, [1])
            lk = _to_numpy(lk_t)
            Uk = _to_numpy(Uk_t)
        except Exception as e:
            LOGGER.info(
                f"[spectral_decomp] lobpcg failed ({e}); falling back to dense for first {K} vectors"
            )
            d, V = torch.linalg.eigh(L.to_dense())
            lk = _to_numpy(d[:K])
            Uk = _to_numpy(V[:, :K])
    return lk, Uk


def build_synthetic_gang_graph() -> Tuple[Data, List[Pattern], List[Pattern]]:
    """Build a small synthetic graph with known gang motifs.

    Graph layout (≈60 nodes)
    ────────────────────────
    * Community A  (nodes  0-14): fan-in pattern  — many senders → one hub
    * Community B  (nodes 15-29): fan-out pattern — one hub → many receivers
    * Community C  (nodes 30-44): cycle pattern   — ring of 15 nodes
    * Noise / background (nodes 45-59): random Erdős–Rényi graph

    Returns
    -------
    G             : PyG Data object (undirected)
    alert_patterns: list[Pattern]  for communities A, B, C
    normal_patterns: list[Pattern] for the noise community
    """
    import random as _rnd

    _rnd.seed(0)
    np.random.seed(0)

    edges = []

    # Fan-in (community A): nodes 1..14 each send to node 0
    for i in range(1, 15):
        edges.append((i, 0))

    # Fan-out (community B): node 15 sends to 16..29
    for i in range(16, 30):
        edges.append((15, i))

    # Cycle (community C): 30 -> 31 -> ... -> 44 -> 30
    for i in range(30, 45):
        edges.append((i, (i - 30 + 1) % 15 + 30))

    # Background noise: sparse ER graph on nodes 45..59
    for u in range(45, 60):
        for v in range(u + 1, 60):
            if _rnd.random() < 0.08:
                edges.append((u, v))

    # A few cross-community edges so the graph is connected
    edges += [(14, 15), (29, 30), (44, 45)]

    src_list, dst_list = zip(*edges)
    # Build undirected edge list manually (add both directions, then deduplicate)
    both_src = list(src_list) + list(dst_list)
    both_dst = list(dst_list) + list(src_list)
    edge_pairs = list(set(zip(both_src, both_dst)))
    edge_pairs.sort()
    src_u, dst_u = zip(*edge_pairs)
    edge_index = torch.tensor([list(src_u), list(dst_u)], dtype=torch.long)

    N = 60
    x = torch.eye(N, dtype=torch.float32)
    y = torch.zeros(N, dtype=torch.long)
    y[:45] = 1  # communities A, B, C are "suspicious"

    G = Data(x=x, edge_index=edge_index, num_nodes=N, y=y)
    G.edge_weight = torch.ones(G.edge_index.size(1), dtype=torch.float32)
    G = _ensure_graph_params(G)

    from src.GangPrediction.pattern_models import create_pattern

    alert_patterns = [
        create_pattern("fan_in", np.arange(0, 15), "fan_in", label="alert"),
        create_pattern("fan_out", np.arange(15, 30), "fan_out", label="alert"),
        create_pattern("cycle", np.arange(30, 45), "cycle", label="alert"),
    ]
    normal_patterns = [
        create_pattern("noise", np.arange(45, 60), "noise", label="normal"),
    ]
    return G, alert_patterns, normal_patterns


# ---------------------------------------------------------------------------
# Experiment 1: Spectral Fingerprint of Patterns
# ---------------------------------------------------------------------------


def spectral_fingerprint(
    G: Data,
    alert_patterns: List[Pattern],
    normal_patterns: Optional[List[Pattern]],
    K_max: int = 200,
    save_dir: str = "results/coarsening_diagnostics",
    name_prefix: str = "",
    return_stats: bool = False,
    dense_threshold: int = 1000,
) -> Optional[Dict]:
    """Plot how much energy gang pattern indicators project onto each eigenvector.

    Plots saved
    -----------
    * {name_prefix}spectral_fingerprint_heatmap.png   — patterns × eigenvector rank heat map
    * {name_prefix}spectral_fingerprint_cumulative.png — cumulative energy vs K for gang vs random

    If *return_stats* is True, returns a dict with keys:
        energy, k50_alert, k90_alert, K
    """
    os.makedirs(save_dir, exist_ok=True)
    G = _ensure_graph_params(G)
    N = G.num_nodes
    K = min(K_max, N)

    LOGGER.info(
        f"[spectral_fingerprint] computing {K} eigenvectors for {N}-node graph …"
    )
    lk, Uk = compute_spectral_decomp(
        G, K_max=K, dense_threshold=dense_threshold
    )  # (N, K)

    # Build pattern indicators
    alerts = alert_patterns or []
    normals = normal_patterns or []
    all_patterns = list(alerts) + list(normals)
    indicators = build_pattern_indicator(N, all_patterns, normalize=True)

    if len(all_patterns) == 0:
        LOGGER.info("[spectral_fingerprint] no patterns — skipping")
        return

    # Energy matrix: (n_patterns, K) — enumerate positionally to avoid dup-ID issues
    n_pats = len(all_patterns)
    energy = np.zeros((n_pats, K))
    for i, p in enumerate(all_patterns):
        nodes = _to_numpy(p.node_indices).astype(int)
        v = np.zeros(N, dtype=np.float64)
        v[nodes] = 1.0
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        proj = Uk.T @ v  # (K,)
        energy[i] = proj**2

    # Random baseline: same-size random node-sets, averaged over 20 trials
    rng = np.random.default_rng(42)
    pattern_sizes = [int(p.num_nodes) for p in all_patterns]
    random_energy = np.zeros(K)
    n_trials = 20
    for sz in pattern_sizes:
        if sz < 1:
            continue
        for _ in range(n_trials):
            idx = rng.choice(N, size=sz, replace=False)
            v = np.zeros(N)
            v[idx] = 1.0
            norm = np.linalg.norm(v)
            if norm > 0:
                v /= norm
            random_energy += (Uk.T @ v) ** 2
    random_energy /= max(1, len(pattern_sizes) * n_trials)

    # ── Plot A: heat map ──────────────────────────────────────────────────
    MAX_DISPLAY_PATS = 80  # cap rows so the image stays readable
    if n_pats > MAX_DISPLAY_PATS:
        # Stratified sample: keep proportional alert/normal ratio
        n_alert_show = min(len(alerts), MAX_DISPLAY_PATS // 2)
        n_normal_show = min(len(normals), MAX_DISPLAY_PATS - n_alert_show)
        rng_disp = np.random.default_rng(0)
        alert_idx = rng_disp.choice(len(alerts), size=n_alert_show, replace=False)
        normal_idx = rng_disp.choice(len(normals), size=n_normal_show, replace=False)
        show_idx = np.concatenate([alert_idx, normal_idx + len(alerts)])
        show_idx = np.sort(show_idx)
        energy_show = energy[show_idx]
        pats_show = [all_patterns[i] for i in show_idx]
    else:
        energy_show = energy
        pats_show = all_patterns

    fig_h = max(4, len(pats_show) * 0.35 + 1)
    fig, ax = plt.subplots(figsize=(max(8, K // 10), fig_h))
    im = ax.imshow(energy_show, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_xlabel("Eigenvector rank (0 = lowest frequency)")
    ax.set_ylabel("Pattern")
    tick_step = max(1, len(pats_show) // 20)
    tick_positions = list(range(0, len(pats_show), tick_step))
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(
        [
            f"{'A' if pats_show[i].label == 'alert' else 'N'}-{pats_show[i].pattern_type}"
            for i in tick_positions
        ],
        fontsize=7,
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Projection energy")
    sample_note = (
        f" (showing {len(pats_show)}/{n_pats})" if n_pats > MAX_DISPLAY_PATS else ""
    )
    ax.set_title(f"Spectral fingerprint: pattern energy per eigenvector{sample_note}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{name_prefix}spectral_fingerprint_heatmap.png")
    plt.savefig(path, dpi=120)
    plt.close()
    # LOGGER.info(f"  saved → {path}")

    # ── Plot B: cumulative energy vs K ───────────────────────────────────
    alert_energy = energy[: len(alerts)].mean(axis=0) if alerts else None
    normal_energy = energy[len(alerts) :].mean(axis=0) if normals else None

    fig, ax = plt.subplots(figsize=(8, 4))
    ks = np.arange(1, K + 1)

    if alert_energy is not None:
        ax.plot(
            ks,
            # np.cumsum(alert_energy),
            alert_energy,
            label="Alert patterns (mean)",
            color="crimson",
            lw=2,
        )
    if normal_energy is not None:
        ax.plot(
            ks,
            # np.cumsum(normal_energy),
            normal_energy,
            label="Normal patterns (mean)",
            color="steelblue",
            lw=2,
        )
    ax.plot(
        ks,
        # np.cumsum(random_energy),
        random_energy,
        label="Random sets (baseline)",
        color="gray",
        lw=1.5,
        ls="--",
    )

    ax.set_xlabel("Number of eigenvectors K preserved")
    ax.set_ylabel("Projection energy")
    ax.set_title("How much gang signal lives in the K-th eigenvectors?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{name_prefix}spectral_fingerprint_energy.png")
    plt.savefig(path, dpi=120)
    plt.close()
    # LOGGER.info(f"  saved → {path}")

    # Summary stat
    k50_alert = (
        int(
            np.searchsorted(
                np.cumsum(alert_energy) / (np.sum(alert_energy) + 1e-12), 0.5
            )
        )
        + 1
        if alert_energy is not None
        else "N/A"
    )
    k90_alert = (
        int(
            np.searchsorted(
                np.cumsum(alert_energy) / (np.sum(alert_energy) + 1e-12), 0.9
            )
        )
        + 1
        if alert_energy is not None
        else "N/A"
    )
    LOGGER.info(
        f"  [summary] alert patterns: 50% energy in first {k50_alert} eigvecs, 90% in first {k90_alert}"
    )

    if return_stats:
        return {
            "energy": energy,
            "k50_alert": k50_alert,
            "k90_alert": k90_alert,
            "K": K,
        }


# ---------------------------------------------------------------------------
# Experiment 2: Merge Recall/Precision vs K
# ---------------------------------------------------------------------------


def _node_to_pattern_assignment(
    N: int,
    alert_patterns: List[Pattern],
    normal_patterns: Optional[List[Pattern]],
) -> np.ndarray:
    """Return array of length N: -1 for unlabeled, otherwise a unique pattern index."""
    assignment = np.full(N, -1, dtype=np.int32)
    pid = 0
    for p in alert_patterns or []:
        nodes = _to_numpy(p.node_indices).astype(int)
        assignment[nodes] = pid
        pid += 1
    for p in normal_patterns or []:
        nodes = _to_numpy(p.node_indices).astype(int)
        assignment[nodes] = pid
        pid += 1
    return assignment


def merge_recall_precision_vs_K(
    G: Data,
    alert_patterns: List[Pattern],
    normal_patterns: Optional[List[Pattern]],
    K_values: Optional[List[int]] = None,
    levels: int = 5,
    max_sigma: float = 1e6,
    save_dir: str = "results/coarsening_diagnostics",
    name_prefix: str = "",
) -> None:
    """For each K, run multiple coarsening levels and plot level trajectories.

    Plots saved
    -----------
    * {name_prefix}merge_recall_precision_vs_K_alert.png
    * {name_prefix}merge_recall_precision_vs_K_normal.png
    * {name_prefix}merge_count_vs_K_multilevel.png
    """
    os.makedirs(save_dir, exist_ok=True)
    G = _ensure_graph_params(G)
    N = G.num_nodes

    if K_values is None:
        K_values = [2, 5, 10, 20, 50, 100]
    # lobpcg requires k <= N/3; cap each K value to be safe
    K_max_safe = max(1, G.num_nodes // 4)
    K_values = [k for k in K_values if k <= K_max_safe]
    if not K_values:
        LOGGER.info(
            f"[merge_recall_precision_vs_K] no valid K values for {G.num_nodes}-node graph — skipping"
        )
        return

    trajectories = {
        group: {
            level: {"recall": [], "precision": []} for level in range(1, levels + 1)
        }
        for group in ("alert", "normal")
    }
    merge_counts = {level: [] for level in range(1, levels + 1)}
    LOGGER.info(
        "[merge_recall_precision_vs_K] running multilevel coarsening per K value …"
    )
    for K in K_values:
        B = calc_B(G, K)
        Gc = G
        C_cumulative = sparse_eye(N).to(G.L.device)
        epsilon_l = 0.0
        LOGGER.info(f"  K={K:4d}")
        for level in range(1, levels + 1):
            if Gc.num_nodes <= 2:
                for group in ("alert", "normal"):
                    trajectories[group][level]["recall"].append(float("nan"))
                    trajectories[group][level]["precision"].append(float("nan"))
                merge_counts[level].append(float("nan"))
                continue

            Gc_new, B_new, sigma_l, done = coarse_one_level(
                Gc,
                B=B,
                method="variation_edges",
                algorithm="greedy",
                # level=level,
                r_cur=1.0,
                max_sigma=max_sigma,
            )
            C_cumulative = torch.sparse.mm(Gc_new.C, C_cumulative)
            node_to_supernode = _node_to_supernode_mapping(C_cumulative)
            metrics = compute_merge_recall_precision(
                node_to_supernode=node_to_supernode,
                alert_patterns=alert_patterns,
                normal_patterns=normal_patterns,
            )
            merge_count = _count_nontrivial_merges(Gc_new.C)

            for group in ("alert", "normal"):
                trajectories[group][level]["recall"].append(metrics[group]["recall"])
                trajectories[group][level]["precision"].append(
                    metrics[group]["precision"]
                )
            merge_counts[level].append(float(merge_count))
            LOGGER.info(
                "    level=%2d  merges=%5d  alert(r=%.3f,p=%.3f)  normal(r=%.3f,p=%.3f)"
                % (
                    level,
                    merge_count,
                    metrics["alert"]["recall"],
                    metrics["alert"]["precision"],
                    metrics["normal"]["recall"],
                    metrics["normal"]["precision"],
                )
            )

            epsilon_l = (sigma_l + 1) * (epsilon_l + 1) - 1
            Gc, B = Gc_new, B_new

            if done:
                for future_level in range(level + 1, levels + 1):
                    for group in ("alert", "normal"):
                        trajectories[group][future_level]["recall"].append(float("nan"))
                        trajectories[group][future_level]["precision"].append(
                            float("nan")
                        )
                    merge_counts[future_level].append(float("nan"))
                break

        for level in range(1, levels + 1):
            while len(merge_counts[level]) < len(
                trajectories["alert"][level]["recall"]
            ):
                merge_counts[level].append(float("nan"))

    _plot_multilevel_merge_metric_trajectories(
        K_values=K_values,
        trajectories=trajectories["alert"],
        pattern_label="alert",
        save_dir=save_dir,
        name_prefix=name_prefix,
    )
    _plot_multilevel_merge_metric_trajectories(
        K_values=K_values,
        trajectories=trajectories["normal"],
        pattern_label="normal",
        save_dir=save_dir,
        name_prefix=name_prefix,
    )
    _plot_multilevel_merge_count_trajectories(
        K_values=K_values,
        merge_counts=merge_counts,
        save_dir=save_dir,
        name_prefix=name_prefix,
    )


# ---------------------------------------------------------------------------
# Experiment 3: Epsilon Schedule Visualisation
# ---------------------------------------------------------------------------


def plot_epsilon_schedules(
    levels: int = 10,
    max_epsilon: float = 50.0,
    powers: Optional[List[float]] = None,
    save_dir: str = "results/coarsening_diagnostics",
    name_prefix: str = "",
) -> None:
    """Visualise how different schedule powers distribute the epsilon budget.

    Plots saved
    -----------
    * {name_prefix}epsilon_schedule_curves.png   – eps_budget(level) for all powers
    * {name_prefix}epsilon_schedule_unlock.png   – bar chart of "80% unlock level"
    """
    os.makedirs(save_dir, exist_ok=True)
    if powers is None:
        powers = [0.04, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    ls = np.arange(1, levels + 1)
    cmap = cm.get_cmap("plasma", len(powers))

    # ── Plot A: schedule curves ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    unlock_levels = []
    for idx, p in enumerate(powers):
        budget = max_epsilon * (ls / levels) ** p
        ax.plot(ls, budget, "o-", color=cmap(idx), lw=2, ms=4, label=f"power={p}")
        # "unlock level": first level where budget ≥ 80% of max_epsilon
        reached = np.where(budget >= 0.8 * max_epsilon)[0]
        unlock_levels.append(int(reached[0] + 1) if len(reached) > 0 else levels + 1)
    ax.axhline(0.8 * max_epsilon, ls="--", color="gray", lw=1.2, label="80% of max_ε")
    ax.set_xlabel("Coarsening level")
    ax.set_ylabel("Epsilon budget available")
    ax.set_title("Epsilon budget schedule curves")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Plot B: unlock bar chart ──────────────────────────────────────────
    ax = axes[1]
    colors = [cmap(i) for i in range(len(powers))]
    bars = ax.bar(
        [str(p) for p in powers], unlock_levels, color=colors, edgecolor="k", lw=0.8
    )
    ax.axhline(levels, ls="--", color="gray", lw=1.2, label=f"Total levels={levels}")
    ax.set_xlabel("Schedule power")
    ax.set_ylabel("Level at which ε ≥ 80% of max_ε")
    ax.set_title('"Unlock level" per schedule')
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, ul in zip(bars, unlock_levels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(ul),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # ── Annotation panel ─────────────────────────────────────────────────
    plt.suptitle(
        f"Epsilon schedule comparison  (max_ε={max_epsilon}, {levels} levels)",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(save_dir, f"{name_prefix}epsilon_schedule_curves.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    # LOGGER.info(f"  saved → {path}")

    # Tabular summary
    LOGGER.info(f"  {'power':>8}  {'unlock level':>13}  {'% budget at level 1':>20}")
    for p, ul in zip(powers, unlock_levels):
        budget_l1 = max_epsilon * (1 / levels) ** p
        LOGGER.info(f"  {p:>8}  {ul:>13}  {100*budget_l1/max_epsilon:>19.1f}%")


# ---------------------------------------------------------------------------
# Experiment 4: Schedule Ablation — Coarsening Depth & Merge Recall/Precision
# ---------------------------------------------------------------------------


def epsilon_schedule_ablation(
    G: Data,
    alert_patterns: List[Pattern],
    normal_patterns: Optional[List[Pattern]],
    K: int = 100,
    powers: Optional[List[float]] = None,
    levels: int = 10,
    max_epsilon: float = 50.0,
    save_dir: str = "results/coarsening_diagnostics",
    name_prefix: str = "",
) -> None:
    """Run full multilevel coarsening under several schedule powers, recording per-level stats.

    Plots saved
    -----------
    * {name_prefix}schedule_ablation_node_count.png
    * {name_prefix}schedule_ablation_merge_recall_precision.png
    * {name_prefix}schedule_ablation_pattern_collapse.png
    """
    os.makedirs(save_dir, exist_ok=True)
    G = _ensure_graph_params(G)
    N = G.num_nodes
    assignment = _node_to_pattern_assignment(N, alert_patterns, normal_patterns)
    n_labeled = int((assignment >= 0).sum())

    if powers is None:
        powers = [0.04, 0.25, 1.0, 5.0]

    # Pre-compute B once for all runs
    B0 = calc_B(G, K)

    cmap = cm.get_cmap("plasma", len(powers))

    all_results: Dict[float, dict] = {}
    for p_idx, power in enumerate(powers):
        LOGGER.info(f"[schedule_ablation] power={power} running {levels} levels …")
        Gc = G  # start fresh from original
        B = B0.clone()
        C_cumulative = sparse_eye(N).to(G.L.device)
        epsilon_l = 0.0
        node_counts = [N]
        recall_per_level = []
        precision_per_level = []
        collapse_per_level = []  # fraction of labeled nodes that have been merged

        for level in range(1, levels + 1):
            level_progress = level / max(1, levels)
            if abs(power) < 1e-8:
                max_eps_in_level = max_epsilon
            else:
                max_eps_in_level = max_epsilon * (level_progress**power)
            max_sigma = max(0.0, (max_eps_in_level + 1) / (epsilon_l + 1) - 1)

            Gc_new, B_new, sigma_l, done = coarse_one_level(
                Gc,
                B=B,
                method="variation_edges",
                algorithm="greedy",
                # level=level,
                r_cur=1.0,
                max_sigma=max_sigma,
            )

            C_cumulative = torch.sparse.mm(Gc_new.C, C_cumulative)
            node_to_supernode = _node_to_supernode_mapping(C_cumulative)
            metrics = compute_merge_recall_precision(
                node_to_supernode=node_to_supernode,
                alert_patterns=alert_patterns,
                normal_patterns=normal_patterns,
            )
            recall_per_level.append(metrics["recall"])
            precision_per_level.append(metrics["precision"])

            # Pattern collapse: fraction of labeled original nodes now inside non-singleton super-nodes.
            node_to_supernode_np = _to_numpy(node_to_supernode).astype(int)
            supernode_sizes = np.bincount(
                node_to_supernode_np, minlength=Gc_new.num_nodes
            )
            labeled_nodes = node_to_supernode_np[assignment >= 0]
            collapse = (
                float(np.mean(supernode_sizes[labeled_nodes] > 1))
                if labeled_nodes.size > 0
                else float("nan")
            )
            collapse_per_level.append(collapse)

            node_counts.append(Gc_new.num_nodes)
            epsilon_l = (sigma_l + 1) * (epsilon_l + 1) - 1

            Gc, B = Gc_new, B_new

            if Gc.num_nodes <= 2:
                break

        all_results[power] = {
            "node_counts": node_counts,
            "recall": recall_per_level,
            "precision": precision_per_level,
            "collapse": collapse_per_level,
        }

    # ── Plot: node count ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, (p, res) in enumerate(all_results.items()):
        ax.plot(
            res["node_counts"], "o-", color=cmap(idx), lw=2, ms=4, label=f"power={p}"
        )
    ax.set_xlabel("Coarsening level")
    ax.set_ylabel("Number of nodes")
    ax.set_title("Node count vs coarsening level per schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{name_prefix}schedule_ablation_node_count.png")
    plt.savefig(path, dpi=120)
    plt.close()
    # LOGGER.info(f"  saved → {path}")

    # ── Plot: merge recall and precision ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for idx, (p, res) in enumerate(all_results.items()):
        ls_range = range(1, len(res["recall"]) + 1)
        axes[0].plot(
            ls_range,
            res["recall"],
            "o-",
            color=cmap(idx),
            lw=2,
            ms=4,
            label=f"power={p}",
        )
        axes[1].plot(
            ls_range,
            res["precision"],
            "o-",
            color=cmap(idx),
            lw=2,
            ms=4,
            label=f"power={p}",
        )
    axes[0].set_xlabel("Coarsening level")
    axes[0].set_ylabel("Merge recall")
    axes[0].set_title("Merge recall per level & schedule")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    axes[1].set_xlabel("Coarsening level")
    axes[1].set_ylabel("Merge precision")
    axes[1].set_title(
        "Merge precision per level & schedule\n(low power = more budget early)"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(
        save_dir, f"{name_prefix}schedule_ablation_merge_recall_precision.png"
    )
    plt.savefig(path, dpi=120)
    plt.close()
    # LOGGER.info(f"  saved → {path}")

    # ── Plot: pattern collapse rate ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, (p, res) in enumerate(all_results.items()):
        ls_range = range(1, len(res["collapse"]) + 1)
        ax.plot(
            ls_range,
            res["collapse"],
            "o-",
            color=cmap(idx),
            lw=2,
            ms=4,
            label=f"power={p}",
        )
    ax.set_xlabel("Coarsening level")
    ax.set_ylabel("Fraction of labeled nodes merged")
    ax.set_title("Pattern collapse rate per level & schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(
        save_dir, f"{name_prefix}schedule_ablation_pattern_collapse.png"
    )
    plt.savefig(path, dpi=120)
    plt.close()
    # LOGGER.info(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Experiment 5: Super-Node Label Entropy
# ---------------------------------------------------------------------------


def _label_entropy(soft_y: np.ndarray) -> np.ndarray:
    """Per-row entropy from soft label fractions (rows must sum to 1)."""
    eps = 1e-12
    p = soft_y / (soft_y.sum(axis=1, keepdims=True) + eps)
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def supernode_entropy_analysis(
    G: Data,
    K: int = 100,
    powers: Optional[List[float]] = None,
    levels: int = 10,
    max_epsilon: float = 50.0,
    save_dir: str = "results/coarsening_diagnostics",
    name_prefix: str = "",
) -> None:
    """At each coarsening level, compute label entropy of super-nodes and track it.

    Uses the soft_y tensor (one-hot aggregated by the coarsening matrix) that is
    built automatically inside `construct_G`.

    Plots saved
    -----------
    * {name_prefix}entropy_mean_per_level.png
    * {name_prefix}entropy_pure_fraction_per_level.png
    """
    os.makedirs(save_dir, exist_ok=True)
    G = _ensure_graph_params(G)
    N = G.num_nodes

    # Attach soft_y to original graph
    num_classes = int(G.y.max().item() + 1)
    soft_y = torch.zeros(N, num_classes, dtype=torch.float32)
    soft_y.scatter_(1, G.y.view(-1, 1), 1.0)
    G.soft_y = soft_y

    # Attach empty y_train so construct_G does not crash
    G.y_train = torch.zeros(N, num_classes, dtype=torch.float32)

    if powers is None:
        powers = [0.04, 0.25, 1.0, 5.0]

    B0 = calc_B(G, K)
    cmap = cm.get_cmap("plasma", len(powers))

    all_entropy: Dict[float, List[float]] = {}
    all_pure_frac: Dict[float, List[float]] = {}

    for power in powers:
        LOGGER.info(f"[supernode_entropy] power={power} …")
        Gc = G
        B = B0.clone()
        epsilon_l = 0.0
        mean_entropies = [float(np.mean(_label_entropy(_to_numpy(soft_y))))]
        pure_fracs = [float(np.mean(_label_entropy(_to_numpy(soft_y)) < 1e-9))]

        for level in range(1, levels + 1):
            level_progress = level / max(1, levels)
            if abs(power) < 1e-8:
                max_eps_in_level = max_epsilon
            else:
                max_eps_in_level = max_epsilon * (level_progress**power)
            max_sigma = max(0.0, (max_eps_in_level + 1) / (epsilon_l + 1) - 1)

            Gc_new, B_new, sigma_l, done = coarse_one_level(
                Gc,
                B=B,
                method="variation_edges",
                algorithm="greedy",
                # level=level,
                r_cur=1.0,
                max_sigma=max_sigma,
            )
            epsilon_l = (sigma_l + 1) * (epsilon_l + 1) - 1

            sy = _to_numpy(Gc_new.soft_y) if Gc_new.soft_y is not None else None
            if sy is None or sy.ndim < 2:
                break

            ent = _label_entropy(sy)
            mean_entropies.append(float(np.mean(ent)))
            pure_fracs.append(float(np.mean(ent < 1e-9)))

            Gc, B = Gc_new, B_new
            if Gc.num_nodes <= 2:
                break

        all_entropy[power] = mean_entropies
        all_pure_frac[power] = pure_fracs

    # ── Plot: mean entropy ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, (p, vals) in enumerate(all_entropy.items()):
        ax.plot(vals, "o-", color=cmap(idx), lw=2, ms=4, label=f"power={p}")
    ax.set_xlabel("Coarsening level (0 = original graph)")
    ax.set_ylabel("Mean super-node label entropy")
    ax.set_title(
        "Label entropy of super-nodes vs coarsening level & schedule\n"
        "(lower entropy = purer super-nodes = cleaner GNN training signal)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{name_prefix}entropy_mean_per_level.png")
    plt.savefig(path, dpi=120)
    plt.close()
    # LOGGER.info(f"  saved → {path}")

    # ── Plot: pure fraction ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, (p, vals) in enumerate(all_pure_frac.items()):
        ax.plot(vals, "o-", color=cmap(idx), lw=2, ms=4, label=f"power={p}")
    ax.set_xlabel("Coarsening level (0 = original graph)")
    ax.set_ylabel("Fraction of pure super-nodes (entropy ≈ 0)")
    ax.set_title("Fraction of homogeneous super-nodes per schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{name_prefix}entropy_pure_fraction_per_level.png")
    plt.savefig(path, dpi=120)
    plt.close()
    # LOGGER.info(f"  saved → {path}")
