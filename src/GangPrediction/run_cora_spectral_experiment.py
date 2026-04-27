"""run_cora_spectral_experiment.py
==================================
Test whether the narrow-band spectral energy concentration observed in the
AML / synthetic graphs is a general phenomenon by planting known motifs into
the Cora citation graph and computing spectral fingerprints.

Experiment sweep
----------------
* 5 pattern types : star, cycle, clique, path, bipartite
* 2 sizes         : 10, 15
* 2 repetitions   : 10, 30
* 2 strategies    : random (scattered), bfs (local neighbourhood)

Total configurations: 40.  Each gets its own heatmap + cumulative energy plot.
A final summary comparison plot collects k50 / k90 across all configs.

Usage (from repo root, with FedStruct conda env active):

    python src/GangPrediction/run_cora_spectral_experiment.py                  # CiteSeer (default)
    python src/GangPrediction/run_cora_spectral_experiment.py --dataset Cora
    python src/GangPrediction/run_cora_spectral_experiment.py --dataset PubMed
"""

from __future__ import annotations

import argparse
import copy
import itertools
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── path setup (mirrors run_coarsening_diagnostics.py) ───────────────────────
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected

from src.GangPrediction.coarsening_diagnostics import (
    spectral_fingerprint,
    compute_spectral_decomp,
    _ensure_graph_params,
)
from src.GangPrediction.pattern_models import create_pattern
from src.GangPrediction.utils.utils import graph_params, save_path, LOGGER

# ── experiment config ────────────────────────────────────────────────────────
PATTERN_TYPES = ["star", "cycle", "clique", "path", "bipartite"]
SIZES = [10, 15]
REPS = [10, 30]
STRATEGIES = ["random", "bfs"]
K_MAX = 500  # spectral decomposition budget
DENSE_THRESHOLD = 10000  # Cora has 2708 nodes — dense eigh is much faster than lobpcg
VALID_DATASETS = ["CiteSeer", "Cora", "PubMed"]


# ═══════════════════════════════════════════════════════════════════════════════
# Pattern edge generators
# ═══════════════════════════════════════════════════════════════════════════════


def star_edges(nodes: np.ndarray) -> List[Tuple[int, int]]:
    """Hub = nodes[0], spokes = nodes[1:]."""
    hub = nodes[0]
    return [(hub, n) for n in nodes[1:]]


def cycle_edges(nodes: np.ndarray) -> List[Tuple[int, int]]:
    """Ring: 0→1→2→…→k-1→0."""
    k = len(nodes)
    return [(nodes[i], nodes[(i + 1) % k]) for i in range(k)]


def clique_edges(nodes: np.ndarray) -> List[Tuple[int, int]]:
    """Complete subgraph."""
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            edges.append((nodes[i], nodes[j]))
    return edges


def path_edges(nodes: np.ndarray) -> List[Tuple[int, int]]:
    """Chain: 0-1-2-…-k-1."""
    return [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]


def bipartite_edges(nodes: np.ndarray) -> List[Tuple[int, int]]:
    """Two equal groups with all cross-edges."""
    mid = len(nodes) // 2
    left, right = nodes[:mid], nodes[mid:]
    return [(u, v) for u in left for v in right]


EDGE_GENERATORS = {
    "star": star_edges,
    "cycle": cycle_edges,
    "clique": clique_edges,
    "path": path_edges,
    "bipartite": bipartite_edges,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Node selection strategies
# ═══════════════════════════════════════════════════════════════════════════════


def select_random_nodes(
    N: int, size: int, used: set, rng: np.random.Generator
) -> Optional[np.ndarray]:
    """Pick *size* nodes uniformly at random, avoiding *used* nodes."""
    available = np.array(sorted(set(range(N)) - used))
    if len(available) < size:
        return None
    chosen = rng.choice(available, size=size, replace=False)
    return chosen


def _adj_list_from_edge_index(edge_index: torch.Tensor, N: int) -> Dict[int, List[int]]:
    """Build an adjacency list dict from a PyG edge_index (cached per call site)."""
    adj: Dict[int, List[int]] = defaultdict(list)
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[s].append(d)
    return adj


def select_bfs_nodes(
    edge_index: torch.Tensor,
    N: int,
    size: int,
    used: set,
    rng: np.random.Generator,
    adj: Optional[Dict[int, List[int]]] = None,
) -> Optional[np.ndarray]:
    """BFS from a random seed, collecting *size* neighbours.

    Avoids nodes in *used*.  Retries up to 50 seeds if BFS doesn't yield enough.
    """
    if adj is None:
        adj = _adj_list_from_edge_index(edge_index, N)
    available = list(set(range(N)) - used)
    if len(available) < size:
        return None

    for _ in range(50):
        seed = int(rng.choice(available))
        visited = [seed]
        visited_set = {seed}
        frontier = [seed]
        while len(visited) < size and frontier:
            next_frontier = []
            for node in frontier:
                for nb in adj.get(node, []):
                    if nb not in visited_set and nb not in used:
                        visited_set.add(nb)
                        visited.append(nb)
                        next_frontier.append(nb)
                        if len(visited) >= size:
                            break
                if len(visited) >= size:
                    break
            frontier = next_frontier
        if len(visited) >= size:
            return np.array(visited[:size])
    return None  # exhausted retries


# ═══════════════════════════════════════════════════════════════════════════════
# Graph planting
# ═══════════════════════════════════════════════════════════════════════════════


def plant_patterns_in_graph(
    G_original: Data,
    pattern_type: str,
    size: int,
    n_reps: int,
    strategy: str,
    seed: int = 42,
) -> Tuple[Data, List]:
    """Plant *n_reps* copies of *pattern_type* into a copy of the graph.

    Steps per repetition
    --------------------
    1. Select *size* nodes (random or BFS).
    2. Remove all existing edges among those nodes.
    3. Add the pattern-specific edges (undirected).

    Returns
    -------
    G_new : Data  — modified graph with patterns planted
    patterns : list[Pattern]  — Pattern objects for each planted instance
    """
    rng = np.random.default_rng(seed)
    N = G_original.num_nodes

    # Work with edge sets for efficient add/remove
    ei = G_original.edge_index
    src_list, dst_list = ei[0].tolist(), ei[1].tolist()
    edge_set = set(zip(src_list, dst_list))

    # Precompute adjacency for BFS
    adj = _adj_list_from_edge_index(ei, N) if strategy == "bfs" else None

    edge_gen = EDGE_GENERATORS[pattern_type]
    used_nodes: set = set()
    patterns = []

    for rep_idx in range(n_reps):
        # ── select nodes ────────────────────────────────────────────────
        if strategy == "random":
            nodes = select_random_nodes(N, size, used_nodes, rng)
        else:
            nodes = select_bfs_nodes(ei, N, size, used_nodes, rng, adj=adj)
        if nodes is None:
            LOGGER.warning(
                f"  [plant] could not find {size} unused nodes at rep {rep_idx} "
                f"({strategy}); stopping at {rep_idx} reps"
            )
            break

        used_nodes.update(nodes.tolist())

        # ── remove existing edges among selected nodes ──────────────────
        node_set = set(nodes.tolist())
        to_remove = {(u, v) for u, v in edge_set if u in node_set and v in node_set}
        edge_set -= to_remove

        # ── add pattern edges (both directions for undirected) ──────────
        new_edges = edge_gen(nodes)
        for u, v in new_edges:
            edge_set.add((u, v))
            edge_set.add((v, u))

        # ── create Pattern object ───────────────────────────────────────
        p = create_pattern(
            pattern_id=f"{pattern_type}_{rep_idx}",
            nodes=nodes,
            pattern_type=pattern_type,
            label="alert",
        )
        patterns.append(p)

    # ── rebuild PyG Data ────────────────────────────────────────────────
    if edge_set:
        all_src, all_dst = zip(*sorted(edge_set))
    else:
        all_src, all_dst = [], []
    new_edge_index = torch.tensor([list(all_src), list(all_dst)], dtype=torch.long)

    G_new = Data(
        x=G_original.x.clone(),
        edge_index=new_edge_index,
        num_nodes=N,
    )
    if hasattr(G_original, "y") and G_original.y is not None:
        G_new.y = G_original.y.clone()
    G_new.edge_weight = torch.ones(G_new.edge_index.size(1), dtype=torch.float32)
    G_new = _ensure_graph_params(G_new)

    return G_new, patterns


# ═══════════════════════════════════════════════════════════════════════════════
# Summary comparison plot
# ═══════════════════════════════════════════════════════════════════════════════


def plot_summary(results: List[Dict], save_dir: str) -> None:
    """Bar chart of k50 and k90 across all experiment configurations."""
    if not results:
        return

    labels = [r["label"] for r in results]
    k50s = [r["k50"] for r in results]
    k90s = [r["k90"] for r in results]

    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, n * 0.6), 6))
    bars1 = ax.bar(
        x - width / 2, k50s, width, label="k50 (50% energy)", color="crimson"
    )
    bars2 = ax.bar(
        x + width / 2, k90s, width, label="k90 (90% energy)", color="steelblue"
    )

    ax.set_ylabel("Number of eigenvectors needed")
    ax.set_title("Spectral energy concentration: k50 / k90 across configurations")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar in bars1:
        h = bar.get_height()
        if isinstance(h, (int, float)) and not isinstance(h, str):
            ax.annotate(
                f"{int(h)}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=6,
            )
    for bar in bars2:
        h = bar.get_height()
        if isinstance(h, (int, float)) and not isinstance(h, str):
            ax.annotate(
                f"{int(h)}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=6,
            )

    plt.tight_layout()
    path = os.path.join(save_dir, "summary_k50_k90.png")
    plt.savefig(path, dpi=150)
    plt.close()
    LOGGER.info(f"  Summary plot saved → {path}")


def plot_summary_heatmaps(results: List[Dict], save_dir: str) -> None:
    """Two heatmaps (k50, k90) with pattern_type × (size, reps, strategy)."""
    if not results:
        return

    # Build a structured table
    rows_by_pattern: Dict[str, Dict[str, Tuple]] = {}
    col_keys = []
    for r in results:
        pt = r["pattern_type"]
        col = f"s{r['size']}_r{r['reps']}_{r['strategy']}"
        if col not in col_keys:
            col_keys.append(col)
        rows_by_pattern.setdefault(pt, {})[col] = (r["k50"], r["k90"])

    pattern_types = list(rows_by_pattern.keys())
    n_rows = len(pattern_types)
    n_cols = len(col_keys)

    k50_mat = np.full((n_rows, n_cols), np.nan)
    k90_mat = np.full((n_rows, n_cols), np.nan)

    for i, pt in enumerate(pattern_types):
        for j, col in enumerate(col_keys):
            if col in rows_by_pattern[pt]:
                k50_val, k90_val = rows_by_pattern[pt][col]
                k50_mat[i, j] = k50_val if isinstance(k50_val, (int, float)) else np.nan
                k90_mat[i, j] = k90_val if isinstance(k90_val, (int, float)) else np.nan

    for mat, metric_name in [(k50_mat, "k50"), (k90_mat, "k90")]:
        fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.2), max(4, n_rows * 0.8)))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_keys, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(pattern_types, fontsize=9)
        ax.set_xlabel("Configuration (size_reps_strategy)")
        ax.set_ylabel("Pattern type")
        ax.set_title(f"Spectral concentration: {metric_name} (eigenvectors needed)")
        # Annotate cells
        for i in range(n_rows):
            for j in range(n_cols):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(
                        j,
                        i,
                        f"{int(val)}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="black" if val < np.nanmax(mat) * 0.7 else "white",
                    )
        plt.colorbar(im, ax=ax, label=f"# eigenvectors for {metric_name}")
        plt.tight_layout()
        path = os.path.join(save_dir, f"summary_{metric_name}_heatmap.png")
        plt.savefig(path, dpi=150)
        plt.close()
        LOGGER.info(f"  Summary heatmap saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spectral fingerprint experiment on Planetoid datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        choices=VALID_DATASETS,
        help="Planetoid dataset to use (default: Cora)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset

    SAVE_ROOT = os.path.join(save_path, f"{dataset_name.lower()}_spectral")
    os.makedirs(SAVE_ROOT, exist_ok=True)

    LOGGER.info("=" * 65)
    LOGGER.info(f"  {dataset_name} Spectral Fingerprint Experiment")
    LOGGER.info(f"  pattern types : {PATTERN_TYPES}")
    LOGGER.info(f"  sizes         : {SIZES}")
    LOGGER.info(f"  repetitions   : {REPS}")
    LOGGER.info(f"  strategies    : {STRATEGIES}")
    LOGGER.info(f"  K_max         : {K_MAX}")
    LOGGER.info(f"  output        : {SAVE_ROOT}")
    LOGGER.info("=" * 65)

    # ── Load dataset ─────────────────────────────────────────────────────
    LOGGER.info(f"\n[1] Loading {dataset_name} dataset …")
    dataset = Planetoid(root="data/Planetoid", name=dataset_name)
    G_data = dataset[0]
    G_data.edge_index = to_undirected(G_data.edge_index)
    G_data.edge_weight = torch.ones(G_data.edge_index.size(1), dtype=torch.float32)
    G_data = _ensure_graph_params(G_data)
    LOGGER.info(f"  {dataset_name}: {G_data.num_nodes} nodes, {G_data.num_edges} edges")

    # ── Baseline: spectral fingerprint of original graph (no planting) ──
    LOGGER.info(f"\n[2] Baseline spectral analysis of unmodified {dataset_name} …")
    baseline_dir = os.path.join(SAVE_ROOT, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    # Use class labels as rough "patterns" — one per class
    unique_classes = torch.unique(G_data.y).tolist()
    class_patterns = []
    for c in unique_classes:
        mask = (G_data.y == c).nonzero(as_tuple=True)[0].numpy()
        class_patterns.append(
            create_pattern(f"class_{c}", mask, "random", label="alert")
        )
    spectral_fingerprint(
        G_data,
        alert_patterns=class_patterns,
        normal_patterns=None,
        K_max=K_MAX,
        save_dir=baseline_dir,
        name_prefix="baseline_",
        dense_threshold=DENSE_THRESHOLD,
    )

    # ── Experiment loop ──────────────────────────────────────────────────
    LOGGER.info("\n[3] Running experiment sweep …")
    configs = list(itertools.product(STRATEGIES, PATTERN_TYPES, SIZES, REPS))
    LOGGER.info(f"  Total configurations: {len(configs)}")

    all_results: List[Dict] = []

    for idx, (strategy, ptype, size, reps) in enumerate(configs, 1):
        tag = f"{strategy}_{ptype}_s{size}_r{reps}"
        LOGGER.info(f"\n  [{idx}/{len(configs)}] {tag}")

        config_dir = os.path.join(SAVE_ROOT, strategy, f"{ptype}_s{size}_r{reps}")
        os.makedirs(config_dir, exist_ok=True)

        # Plant patterns
        G_planted, planted_patterns = plant_patterns_in_graph(
            G_data, ptype, size, reps, strategy, seed=42 + idx
        )
        n_planted = len(planted_patterns)
        LOGGER.info(
            f"    planted {n_planted}/{reps} {ptype} patterns (size={size}) "
            f"| graph: {G_planted.num_nodes} nodes, {G_planted.num_edges} edges"
        )

        if n_planted == 0:
            LOGGER.warning(f"    SKIPPED — no patterns planted")
            continue

        # Spectral fingerprint
        stats = spectral_fingerprint(
            G_planted,
            alert_patterns=planted_patterns,
            normal_patterns=None,
            K_max=K_MAX,
            save_dir=config_dir,
            name_prefix=f"{tag}_",
            return_stats=True,
            dense_threshold=DENSE_THRESHOLD,
        )

        k50 = stats["k50_alert"] if stats else "N/A"
        k90 = stats["k90_alert"] if stats else "N/A"
        LOGGER.info(f"    k50={k50}, k90={k90}")

        all_results.append(
            {
                "label": tag,
                "pattern_type": ptype,
                "size": size,
                "reps": reps,
                "strategy": strategy,
                "k50": k50 if isinstance(k50, (int, float)) else 0,
                "k90": k90 if isinstance(k90, (int, float)) else 0,
                "n_planted": n_planted,
            }
        )

    # ── Summary ──────────────────────────────────────────────────────────
    LOGGER.info("\n[4] Generating summary plots …")
    plot_summary(all_results, SAVE_ROOT)
    plot_summary_heatmaps(all_results, SAVE_ROOT)

    # Print summary table
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info(f"{'Config':<45}  {'k50':>5}  {'k90':>5}  {'planted':>7}")
    LOGGER.info("-" * 80)
    for r in all_results:
        LOGGER.info(
            f"{r['label']:<45}  {r['k50']:>5}  {r['k90']:>5}  {r['n_planted']:>7}"
        )
    LOGGER.info("=" * 80)

    n_plots = len(list(Path(SAVE_ROOT).rglob("*.png")))
    LOGGER.info(f"\nDone. {n_plots} PNG files saved under {SAVE_ROOT}")


if __name__ == "__main__":
    main()
