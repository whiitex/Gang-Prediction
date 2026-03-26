"""Embedding and basis-matrix diagnostics for pattern separability."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

from src.GangPrediction.utils.utils import *


def _safe_to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _build_pattern_labels(
    num_nodes: int,
    alert_patterns: Optional[Iterable],
    normal_patterns: Optional[Iterable],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-node pattern-id, pattern-type, and pattern-subtype labels.

    pattern_id: -1 for unlabeled, otherwise contiguous id across alert+normal patterns.
    pattern_type: -1 for unlabeled, 1 for alert, 0 for normal.
    pattern_subtype: object array of str, "" for unlabeled, e.g. "fan_in", "cycle".
    """
    pattern_id = np.full(num_nodes, -1, dtype=np.int32)
    pattern_type = np.full(num_nodes, -1, dtype=np.int8)
    pattern_subtype = np.full(num_nodes, "", dtype=object)

    next_id = 0
    for p in alert_patterns or []:
        nodes = _safe_to_numpy(p.node_indices).astype(np.int64)
        if nodes.size == 0:
            continue
        pattern_id[nodes] = next_id
        pattern_type[nodes] = 1
        subtype = getattr(p, "pattern_type", "unknown")
        pattern_subtype[nodes] = subtype
        next_id += 1

    for p in normal_patterns or []:
        nodes = _safe_to_numpy(p.node_indices).astype(np.int64)
        if nodes.size == 0:
            continue
        pattern_id[nodes] = next_id
        pattern_type[nodes] = 0
        subtype = getattr(p, "pattern_type", "unknown")
        pattern_subtype[nodes] = subtype
        next_id += 1

    return pattern_id, pattern_type, pattern_subtype


def _sample_indices(indices: np.ndarray, max_points: int, seed: int = 42) -> np.ndarray:
    if indices.size <= max_points:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_points, replace=False))


def _pairwise_distance_stats(
    X: np.ndarray,
    pattern_id: np.ndarray,
    metric: str = "cosine",
) -> Dict[str, float | None]:
    """Compute mean intra/inter-pattern distance on labeled nodes."""
    labeled = np.where(pattern_id >= 0)[0]
    if labeled.size < 3:
        return {
            "intra_mean": None,
            "inter_mean": None,
            "ratio_intra_over_inter": None,
            "margin_inter_minus_intra": None,
        }

    D = pairwise_distances(X[labeled], metric=metric)
    ids = pattern_id[labeled]
    same = ids[:, None] == ids[None, :]
    upper = np.triu(np.ones_like(same, dtype=bool), k=1)

    intra_mask = same & upper
    inter_mask = (~same) & upper

    intra_vals = D[intra_mask]
    inter_vals = D[inter_mask]

    intra_mean = float(np.mean(intra_vals)) if intra_vals.size else None
    inter_mean = float(np.mean(inter_vals)) if inter_vals.size else None

    ratio = None
    margin = None
    if intra_mean is not None and inter_mean is not None and inter_mean > 1e-12:
        ratio = intra_mean / inter_mean
        margin = inter_mean - intra_mean

    return {
        "intra_mean": intra_mean,
        "inter_mean": inter_mean,
        "ratio_intra_over_inter": ratio,
        "margin_inter_minus_intra": margin,
    }


def _nn_pattern_retrieval_at_k(
    X: np.ndarray,
    pattern_id: np.ndarray,
    k: int = 10,
    metric: str = "cosine",
) -> Dict[str, float | None]:
    """Precision@k for retrieving same-pattern nodes among labeled nodes."""
    labeled = np.where(pattern_id >= 0)[0]
    if labeled.size < 3:
        return {
            "precision_at_k": None,
            "random_baseline": None,
            "lift_over_random": None,
        }

    Xl = X[labeled]
    ids = pattern_id[labeled]
    D = pairwise_distances(Xl, metric=metric)
    np.fill_diagonal(D, np.inf)

    k_eff = int(max(1, min(k, len(labeled) - 1)))
    nn_idx = np.argpartition(D, kth=k_eff - 1, axis=1)[:, :k_eff]

    nn_ids = ids[nn_idx]
    hits = (nn_ids == ids[:, None]).astype(np.float32)
    precision_at_k = float(hits.mean())

    _, counts = np.unique(ids, return_counts=True)
    # Probability a random non-self node belongs to same pattern.
    baseline = float(np.mean((counts - 1) / np.maximum(len(labeled) - 1, 1)))
    lift = precision_at_k / baseline if baseline > 1e-12 else None

    return {
        "precision_at_k": precision_at_k,
        "random_baseline": baseline,
        "lift_over_random": lift,
    }


def compute_space_diagnostics(
    rows: torch.Tensor | np.ndarray,
    pattern_id: np.ndarray,
    *,
    k: int = 10,
    metric: str = "cosine",
) -> Dict[str, float | None]:
    """Compute distance and retrieval diagnostics for a row-space matrix."""
    X = _safe_to_numpy(rows)
    distance_stats = _pairwise_distance_stats(X, pattern_id, metric=metric)
    retrieval_stats = _nn_pattern_retrieval_at_k(X, pattern_id, k=k, metric=metric)
    return {
        **distance_stats,
        **retrieval_stats,
    }


def compute_embedding_and_basis_diagnostics(
    embeddings: torch.Tensor | np.ndarray,
    *,
    alert_patterns: Optional[Iterable],
    normal_patterns: Optional[Iterable],
    basis_rows: Optional[torch.Tensor | np.ndarray] = None,
    metric: str = "cosine",
    k: int = 10,
) -> Dict:
    """Compute diagnostics for embedding rows and optional basis rows."""
    emb_np = _safe_to_numpy(embeddings)
    pattern_id, pattern_type, pattern_subtype = _build_pattern_labels(
        emb_np.shape[0], alert_patterns, normal_patterns
    )

    embedding_diag = compute_space_diagnostics(
        emb_np, pattern_id=pattern_id, k=k, metric=metric
    )

    basis_diag = None
    if basis_rows is not None:
        basis_np = _safe_to_numpy(basis_rows)
        if basis_np.shape[0] == emb_np.shape[0]:
            basis_diag = compute_space_diagnostics(
                basis_np, pattern_id=pattern_id, k=k, metric=metric
            )

    return {
        "embedding": embedding_diag,
        "basis": basis_diag,
        "pattern_node_count": int(np.sum(pattern_id >= 0)),
        "num_patterns": int(len(np.unique(pattern_id[pattern_id >= 0]))),
        "pattern_id": pattern_id,
        "pattern_type": pattern_type,
        "pattern_subtype": pattern_subtype,
    }


def plot_diagnostic_trends(results_history: List[Dict], save_dir: str) -> None:
    """Plot level-wise diagnostic trend curves saved in results history."""
    x = np.arange(1, len(results_history) + 1)

    def collect(path: Tuple[str, ...]) -> np.ndarray:
        vals: List[float] = []
        for level in results_history:
            cur = level.get("embedding_diagnostics", {})
            for key in path:
                if not isinstance(cur, dict):
                    cur = None
                    break
                cur = cur.get(key)
            vals.append(np.nan if cur is None else float(cur))
        return np.array(vals, dtype=float)

    emb_ratio = collect(("embedding", "ratio_intra_over_inter"))
    emb_p10 = collect(("embedding", "precision_at_k"))
    emb_random = collect(("embedding", "random_baseline"))
    emb_lift = collect(("embedding", "lift_over_random"))
    basis_ratio = collect(("basis", "ratio_intra_over_inter"))
    basis_p10 = collect(("basis", "precision_at_k"))
    basis_random = collect(("basis", "random_baseline"))
    basis_lift = collect(("basis", "lift_over_random"))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, emb_ratio, label="embedding intra/inter", color="tab:blue")
    ax.plot(x, basis_ratio, label="B-row intra/inter", color="tab:orange")
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Coarsening level")
    ax.set_ylabel("Distance ratio (lower is better)")
    ax.set_title("Pattern Separation Ratio Over Levels")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "embedding_basis_distance_ratio_trend.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, emb_p10, label="embedding P@10", color="tab:green")
    ax.plot(x, basis_p10, label="B-row P@10", color="tab:red")
    ax.plot(
        x,
        emb_random,
        label="embedding random baseline",
        color="tab:green",
        linestyle="--",
        alpha=0.7,
    )
    ax.plot(
        x,
        basis_random,
        label="B-row random baseline",
        color="tab:red",
        linestyle="--",
        alpha=0.7,
    )
    ax.set_xlabel("Coarsening level")
    ax.set_ylabel("Retrieval precision@10")
    ax.set_title("Same-Pattern Neighbor Retrieval Over Levels")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "embedding_basis_retrieval_trend.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, emb_lift, label="embedding lift over random", color="tab:green")
    ax.plot(x, basis_lift, label="B-row lift over random", color="tab:red")
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1, label="random = 1")
    ax.set_xlabel("Coarsening level")
    ax.set_ylabel("Lift over random")
    ax.set_title("Retrieval Lift Over Random Across Levels")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "embedding_basis_retrieval_lift_trend.png")
    plt.close(fig)


def _get_order_by_pattern(
    pattern_type: np.ndarray, pattern_id: np.ndarray
) -> np.ndarray:
    # Order: alert patterns first, then normal patterns, then unlabeled.
    return np.lexsort((pattern_id, pattern_type))


def plot_embedding_row_heatmaps(
    embeddings: torch.Tensor | np.ndarray,
    *,
    pattern_id: np.ndarray,
    pattern_type: np.ndarray,
    save_dir: str,
    max_nodes: int = 1000,
) -> None:
    """Create two row-focused heatmaps: embedding values and row closeness."""
    emb = _safe_to_numpy(embeddings)
    labeled = np.where(pattern_id >= 0)[0]
    if labeled.size < 3:
        return

    sampled = _sample_indices(labeled, max_points=max_nodes, seed=42)
    order = _get_order_by_pattern(pattern_type[sampled], pattern_id[sampled])
    idx = sampled[order]

    row_matrix = emb[idx]
    row_matrix = (row_matrix - row_matrix.mean(axis=0, keepdims=True)) / (
        row_matrix.std(axis=0, keepdims=True) + 1e-9
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(row_matrix, aspect="auto", cmap="coolwarm", interpolation="nearest")
    ax.set_title("Embedding Rows Ordered by Pattern Type")
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Node (ordered by pattern)")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "embedding_rows_by_pattern_type_heatmap.png")
    plt.close(fig)

    S = 1.0 - pairwise_distances(emb[idx], metric="cosine")
    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(
        S, aspect="auto", cmap="viridis", interpolation="nearest", vmin=-1, vmax=1
    )
    ax.set_title("Embedding Row Closeness (Cosine Similarity)")
    ax.set_xlabel("Node index (ordered)")
    ax.set_ylabel("Node index (ordered)")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "embedding_row_closeness_heatmap.png")
    plt.close(fig)


def plot_basis_row_closeness_heatmap(
    basis_rows: torch.Tensor | np.ndarray,
    *,
    pattern_id: np.ndarray,
    pattern_type: np.ndarray,
    save_dir: str,
    max_nodes: int = 1000,
) -> None:
    """Plot B-row cosine-similarity heatmap ordered by pattern grouping."""
    B = _safe_to_numpy(basis_rows)
    if B.ndim != 2 or B.shape[0] != len(pattern_id):
        return

    labeled = np.where(pattern_id >= 0)[0]
    if labeled.size < 3:
        return

    sampled = _sample_indices(labeled, max_points=max_nodes, seed=42)
    order = _get_order_by_pattern(pattern_type[sampled], pattern_id[sampled])
    idx = sampled[order]

    S = 1.0 - pairwise_distances(B[idx], metric="cosine")
    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(
        S, aspect="auto", cmap="magma", interpolation="nearest", vmin=-1, vmax=1
    )
    ax.set_title("B Matrix Row Closeness (Cosine Similarity)")
    ax.set_xlabel("Node index (ordered)")
    ax.set_ylabel("Node index (ordered)")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "b_rows_closeness_heatmap.png")
    plt.close(fig)


def plot_embedding_tsne(
    embeddings: torch.Tensor | np.ndarray,
    *,
    pattern_id: np.ndarray,
    pattern_type: np.ndarray,
    pattern_subtype: np.ndarray,
    save_dir: str,
    max_nodes: int = 2000,
) -> None:
    """t-SNE of embedding rows colored by pattern ID and marked by pattern type.

    Produces three plots:
    - Combined (different marker per pattern subtype)
    - Alert-only (different marker per subtype)
    - Normal-only (different marker per subtype)
    """
    # Marker map: each subtype gets a distinct shape
    SUBTYPE_MARKERS = {
        "fan_out": "v",            # triangle down
        "fan_in": "^",             # triangle up
        "cycle": "o",              # circle
        "bipartite": "s",          # square
        "stack": "D",              # diamond
        "random": "P",             # plus (filled)
        "scatter_gather": "*",     # star
        "gather_scatter": "X",     # x (filled)
        "single": ".",             # point
        "forward": ">",            # triangle right
        "mutual": "<",             # triangle left
        "periodical": "p",         # pentagon
    }
    DEFAULT_MARKER = "h"  # hexagon fallback

    emb = _safe_to_numpy(embeddings)
    labeled = np.where(pattern_id >= 0)[0]
    if labeled.size < 3:
        return

    idx = _sample_indices(labeled, max_points=max_nodes, seed=123)
    X = emb[idx]
    y_type = pattern_type[idx]
    y_pid = pattern_id[idx]
    y_sub = pattern_subtype[idx]

    perplexity = float(max(5, min(30, (len(idx) - 1) // 3)))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    Z = tsne.fit_transform(X)

    unique_patterns = np.unique(y_pid)
    num_patterns = len(unique_patterns)
    cmap_name = "tab20" if num_patterns <= 20 else "nipy_spectral"

    pid_to_color = {pid: i for i, pid in enumerate(unique_patterns.tolist())}
    color_values = np.array([pid_to_color[pid] for pid in y_pid], dtype=np.int32)

    normal_mask = y_type == 0
    alert_mask = y_type == 1

    def _scatter_by_subtype(ax, mask, Z, color_values, y_sub, cmap_name, size, alpha, edge_color, lw):
        """Plot each subtype with its own marker, return last scatter for colorbar."""
        subtypes_in_mask = np.unique(y_sub[mask])
        sc_ref = None
        for st in subtypes_in_mask:
            st_mask = mask & (y_sub == st)
            if not st_mask.any():
                continue
            mkr = SUBTYPE_MARKERS.get(st, DEFAULT_MARKER)
            sc_ref = ax.scatter(
                Z[st_mask, 0],
                Z[st_mask, 1],
                c=color_values[st_mask],
                cmap=cmap_name,
                s=size,
                alpha=alpha,
                marker=mkr,
                edgecolors=edge_color,
                linewidths=lw,
                label=st,
            )
        return sc_ref

    # --- Combined plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    sc_ref = None
    if normal_mask.any():
        sc_ref = _scatter_by_subtype(
            ax, normal_mask, Z, color_values, y_sub, cmap_name,
            size=18, alpha=0.75, edge_color="none", lw=0,
        )
    if alert_mask.any():
        sc_ref = _scatter_by_subtype(
            ax, alert_mask, Z, color_values, y_sub, cmap_name,
            size=24, alpha=0.85, edge_color="black", lw=0.2,
        )
    if sc_ref is not None:
        cbar = plt.colorbar(sc_ref, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Pattern ID (mapped)")
    ax.set_title("t-SNE of Node Embeddings (Color: Pattern ID, Marker: Subtype)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(title="pattern subtype", fontsize=8, title_fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "embedding_tsne_pattern_type.png")
    plt.close(fig)

    # --- Alert-only plot ---
    if alert_mask.any():
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = _scatter_by_subtype(
            ax, alert_mask, Z, color_values, y_sub, cmap_name,
            size=24, alpha=0.85, edge_color="black", lw=0.2,
        )
        if sc is not None:
            cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Pattern ID (mapped)")
        ax.set_title("t-SNE of Alert Pattern Embeddings")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(title="pattern subtype", fontsize=8, title_fontsize=9)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(Path(save_dir) / "embedding_tsne_alert_only.png")
        plt.close(fig)

    # --- Normal-only plot ---
    if normal_mask.any():
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = _scatter_by_subtype(
            ax, normal_mask, Z, color_values, y_sub, cmap_name,
            size=18, alpha=0.75, edge_color="none", lw=0,
        )
        if sc is not None:
            cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Pattern ID (mapped)")
        ax.set_title("t-SNE of Normal Pattern Embeddings")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(title="pattern subtype", fontsize=8, title_fontsize=9)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(Path(save_dir) / "embedding_tsne_normal_only.png")
        plt.close(fig)


def generate_diagnostic_plots_for_final_state(
    *,
    embeddings: torch.Tensor,
    alert_patterns: Optional[Iterable],
    normal_patterns: Optional[Iterable],
    save_dir: str,
    basis_rows: Optional[torch.Tensor | np.ndarray] = None,
) -> None:
    """Generate final-level t-SNE and row-based heatmaps."""
    diag = compute_embedding_and_basis_diagnostics(
        embeddings,
        alert_patterns=alert_patterns,
        normal_patterns=normal_patterns,
        basis_rows=basis_rows,
    )

    pattern_id = diag["pattern_id"]
    pattern_type = diag["pattern_type"]
    pattern_subtype = diag["pattern_subtype"]

    plot_embedding_tsne(
        embeddings,
        pattern_id=pattern_id,
        pattern_type=pattern_type,
        pattern_subtype=pattern_subtype,
        save_dir=save_dir,
    )
    plot_embedding_row_heatmaps(
        embeddings,
        pattern_id=pattern_id,
        pattern_type=pattern_type,
        save_dir=save_dir,
    )
    if basis_rows is not None:
        plot_basis_row_closeness_heatmap(
            basis_rows,
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            save_dir=save_dir,
        )
