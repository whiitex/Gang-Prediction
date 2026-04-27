"""run_planted_detection.py
===========================
Plant known motif patterns into Planetoid citation graphs (Cora, CiteSeer)
using BFS-local placement, then run **edge_gangs** and **learning_vectors**
coarsening-based detection to see if the methods can recover the planted
patterns.

For each (dataset × pattern_type × size × reps) configuration the script:
  1. Plants patterns via BFS into a copy of the graph.
  2. Creates binary labels: 1 = pattern node, 0 = background.
  3. Splits planted patterns into train / test sets.
  4. Runs ``train_GNN_coarsening_aware_loss`` with both methods.
  5. Collects per-level detection metrics (recall, precision, detection_rate).
  6. Produces summary tables and comparison plots.

Usage (from repo root, with FedStruct conda env active):

    python src/GangPrediction/run_planted_detection.py                      # CiteSeer (default)
    python src/GangPrediction/run_planted_detection.py --dataset Cora
    python src/GangPrediction/run_planted_detection.py --dataset Cora --dataset CiteSeer
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected

from src.GangPrediction.utils.utils import *
from src.GangPrediction.experiment_utils import split_patterns
from src.GangPrediction.pattern_models import create_pattern, Pattern
from src.GangPrediction.coarsening_diagnostics import _ensure_graph_params
from src.GangPrediction.train_GNN_coarsening import train_GNN_coarsening_aware_loss

# ── experiment config ────────────────────────────────────────────────────────
PATTERN_TYPES = ["star", "cycle", "clique", "path", "bipartite"]
SIZES = [12]
REPS = [50]
METHODS = ["edge_gangs", "learning_vectors"]
VALID_DATASETS = ["Cora", "CiteSeer"]

# Coarsening / training hyper-parameters (matched to config.yaml defaults)
K = 250
MAX_LEVELS = 4
MAX_EPSILON = 0.1
ALPHA = 1.0
COMPRESSION_METHOD = "svd"
ESP = 0.01
NHID = 10
LR = 0.02
WD = 1e-6
DROPOUT = 0.1
NUM_LAYERS = 3
GNN_TYPE = "SAGE"
INITIAL_EPOCHS = 3
MIN_EPOCHS = 1
MAX_EPOCH_INTERVAL = 1
LOSS_WINDOW = 10
LOSS_THRESHOLD = 0.002
PRETRAIN_EPOCHS = 10  # needed for learning_vectors
TRAIN_RATIO = 0.1  # 10 % patterns for training, 90 % for evaluation


# ═══════════════════════════════════════════════════════════════════════════════
# Pattern edge generators  (reused from run_cora_spectral_experiment.py)
# ═══════════════════════════════════════════════════════════════════════════════


def star_edges(nodes: np.ndarray) -> List[Tuple[int, int]]:
    hub = nodes[0]
    return [(hub, n) for n in nodes[1:]]


def cycle_edges(nodes: np.ndarray) -> List[Tuple[int, int]]:
    k = len(nodes)
    return [(nodes[i], nodes[(i + 1) % k]) for i in range(k)]


def clique_edges(nodes: np.ndarray) -> List[Tuple[int, int]]:
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            edges.append((nodes[i], nodes[j]))
    return edges


def path_edges(nodes: np.ndarray) -> List[Tuple[int, int]]:
    return [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]


def bipartite_edges(nodes: np.ndarray) -> List[Tuple[int, int]]:
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
# Node selection (BFS only)
# ═══════════════════════════════════════════════════════════════════════════════


def _adj_list(edge_index: torch.Tensor, N: int) -> Dict[int, List[int]]:
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
    if adj is None:
        adj = _adj_list(edge_index, N)
    available = list(set(range(N)) - used)
    if len(available) < size:
        return None
    for _ in range(50):
        s = int(rng.choice(available))
        visited, visited_set, frontier = [s], {s}, [s]
        while len(visited) < size and frontier:
            nxt = []
            for node in frontier:
                for nb in adj.get(node, []):
                    if nb not in visited_set and nb not in used:
                        visited_set.add(nb)
                        visited.append(nb)
                        nxt.append(nb)
                        if len(visited) >= size:
                            break
                if len(visited) >= size:
                    break
            frontier = nxt
        if len(visited) >= size:
            return np.array(visited[:size])
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Graph planting (BFS strategy, binary labels)
# ═══════════════════════════════════════════════════════════════════════════════


def plant_patterns_bfs(
    G_original: Data,
    pattern_type: str,
    size: int,
    n_reps: int,
    seed_val: int = 42,
) -> Tuple[Data, List[Pattern], List[Pattern]]:
    """Plant patterns using BFS and create binary labels.

    Returns
    -------
    G_new : Data
        Modified graph with binary labels y (0 = normal, 1 = alert/pattern node).
    alert_patterns : list[Pattern]
        Planted alert patterns.
    normal_patterns : list[Pattern]
        One "normal" pseudo-pattern per connected component of non-pattern nodes
        (simplified: single pattern containing all non-pattern nodes).
    """
    rng = np.random.default_rng(seed_val)
    N = G_original.num_nodes

    ei = G_original.edge_index
    src_list, dst_list = ei[0].tolist(), ei[1].tolist()
    edge_set = set(zip(src_list, dst_list))
    adj = _adj_list(ei, N)
    edge_gen = EDGE_GENERATORS[pattern_type]

    used_nodes: set = set()
    alert_patterns: List[Pattern] = []

    for rep_idx in range(n_reps):
        nodes = select_bfs_nodes(ei, N, size, used_nodes, rng, adj=adj)
        if nodes is None:
            LOGGER.warning(
                f"  [plant] could not find {size} unused nodes at rep {rep_idx}; "
                f"stopping at {rep_idx} reps"
            )
            break

        used_nodes.update(nodes.tolist())

        # Remove existing intra-edges, add pattern edges
        node_set = set(nodes.tolist())
        to_remove = {(u, v) for u, v in edge_set if u in node_set and v in node_set}
        edge_set -= to_remove
        for u, v in edge_gen(nodes):
            edge_set.add((u, v))
            edge_set.add((v, u))

        p = create_pattern(
            pattern_id=f"{pattern_type}_{rep_idx}",
            nodes=nodes,
            pattern_type=pattern_type,
            label="alert",
        )
        alert_patterns.append(p)

    # Rebuild PyG Data
    if edge_set:
        all_src, all_dst = zip(*sorted(edge_set))
    else:
        all_src, all_dst = [], []
    new_edge_index = torch.tensor([list(all_src), list(all_dst)], dtype=torch.long)

    # Binary labels: 1 for pattern nodes, 0 for rest
    y = torch.zeros(N, dtype=torch.long)
    for p in alert_patterns:
        y[p.node_indices] = 1

    G_new = Data(
        x=G_original.x.clone(),
        edge_index=new_edge_index,
        num_nodes=N,
        y=y,
    )
    G_new.edge_weight = torch.ones(G_new.edge_index.size(1), dtype=torch.float32)
    G_new = _ensure_graph_params(G_new)

    # Create normal patterns: all background nodes as a single group
    bg_nodes = torch.where(y == 0)[0].numpy()
    # Split background into chunks of ~size nodes for normal patterns
    normal_patterns: List[Pattern] = []
    # chunk_size = max(size, 20)
    # for i in range(0, len(bg_nodes), chunk_size):
    #     chunk = bg_nodes[i : i + chunk_size]
    #     if len(chunk) >= 3:  # skip tiny fragments
    #         normal_patterns.append(
    #             create_pattern(
    #                 pattern_id=f"normal_{i // chunk_size}",
    #                 nodes=chunk,
    #                 pattern_type="random",
    #                 label="normal",
    #             )
    #         )

    # Set up train/val/test indices
    # Train on pattern nodes + a sample of background, test on the rest
    alert_node_set = torch.where(y == 1)[0]
    normal_node_set = torch.where(y == 0)[0]

    # Use 20% of all nodes for training
    n_train_alert = max(1, int(len(alert_node_set) * TRAIN_RATIO))
    n_train_normal = max(1, int(len(normal_node_set) * TRAIN_RATIO))

    perm_alert = torch.randperm(len(alert_node_set))
    perm_normal = torch.randperm(len(normal_node_set))

    train_idx = torch.cat(
        [
            alert_node_set[perm_alert[:n_train_alert]],
            normal_node_set[perm_normal[:n_train_normal]],
        ]
    )
    test_idx = torch.cat(
        [
            alert_node_set[perm_alert[n_train_alert:]],
            normal_node_set[perm_normal[n_train_normal:]],
        ]
    )

    G_new.train_idx = train_idx
    G_new.val_idx = test_idx
    G_new.test_idx = test_idx

    return G_new, alert_patterns, normal_patterns


# ═══════════════════════════════════════════════════════════════════════════════
# Run detection for one configuration
# ═══════════════════════════════════════════════════════════════════════════════


def run_detection(
    G: Data,
    alert_patterns: List[Pattern],
    normal_patterns: List[Pattern],
    method: str,
    tag: str,
) -> List[Dict]:
    """Run coarsening-based detection and return per-level results."""

    # Split patterns into train / test
    alert_train, alert_test = split_patterns(
        alert_patterns, train_ratio=TRAIN_RATIO, seed=seed
    )
    normal_train, normal_test = split_patterns(
        normal_patterns, train_ratio=TRAIN_RATIO, seed=seed
    )

    if len(alert_train) == 0:
        alert_train = alert_patterns[:1]
        alert_test = alert_patterns[1:] if len(alert_patterns) > 1 else alert_patterns
    if len(alert_test) == 0:
        alert_test = alert_patterns

    LOGGER.info(
        f"    [{method}] patterns: {len(alert_train)} train, {len(alert_test)} test alert | "
        f"{len(normal_train)} train, {len(normal_test)} test normal"
    )

    do_train = method == "learning_vectors"
    pretrain = PRETRAIN_EPOCHS if method == "learning_vectors" else 0

    try:
        Gall, Call, model, C_plus, results_history = train_GNN_coarsening_aware_loss(
            data=G,
            levels=MAX_LEVELS,
            method=method,
            alert_patterns=alert_test,
            normal_patterns=normal_test,
            alert_train_patterns=alert_train,
            normal_train_patterns=normal_train,
            K=K,
            alpha=ALPHA,
            max_epsilon=MAX_EPSILON,
            epsilon_schedule_power=ESP,
            train=do_train,
            lr=LR,
            wd=WD,
            nhid=NHID,
            dropout=DROPOUT,
            num_layers=NUM_LAYERS,
            GNN_type=GNN_TYPE,
            use_edge_weights=False,
            initial_epochs=INITIAL_EPOCHS,
            min_epochs=MIN_EPOCHS,
            max_epoch_interval=MAX_EPOCH_INTERVAL,
            loss_window=LOSS_WINDOW,
            loss_threshold=LOSS_THRESHOLD,
            pretrain_epochs=pretrain,
            compression_method=COMPRESSION_METHOD,
            alert_thresholds=(0.51, 0.51),
            normal_thresholds=(0.51, 0.51),
            use_supernode_loss=True,
        )
    except Exception as e:
        LOGGER.error(f"    [{method}] FAILED: {e}")
        return []

    return results_history


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════


def plot_detection_curves(
    all_results: List[Dict],
    save_dir: str,
) -> None:
    """Plot detection_rate and AUC vs coarsening level for each method × config."""
    if not all_results:
        return

    # Group by dataset
    by_dataset: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_results:
        by_dataset[r["dataset"]].append(r)

    for ds_name, ds_results in by_dataset.items():
        # ── detection rate vs level ──
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        for ax, method in zip(axes, METHODS):
            method_results = [r for r in ds_results if r["method"] == method]
            if not method_results:
                ax.set_title(f"{method} — no results")
                continue

            for r in method_results:
                levels = list(range(1, len(r["detection_rates"]) + 1))
                ax.plot(
                    levels, r["detection_rates"], marker=".", label=r["tag"], alpha=0.7
                )

            ax.set_xlabel("Coarsening level")
            ax.set_ylabel("Alert detection rate")
            ax.set_title(f"{ds_name} — {method} (Detection Rate)")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=6, ncol=2, loc="lower left")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(save_dir, f"{ds_name.lower()}_detection_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        LOGGER.info(f"  Detection curves saved → {path}")

        # ── AUC vs level ──
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        for ax, method in zip(axes, METHODS):
            method_results = [r for r in ds_results if r["method"] == method]
            if not method_results:
                ax.set_title(f"{method} — no results")
                continue

            for r in method_results:
                levels = list(range(1, len(r["aucs"]) + 1))
                ax.plot(levels, r["aucs"], marker=".", label=r["tag"], alpha=0.7)

            ax.set_xlabel("Coarsening level")
            ax.set_ylabel("Alert AUC")
            ax.set_title(f"{ds_name} — {method} (AUC)")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=6, ncol=2, loc="lower left")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(save_dir, f"{ds_name.lower()}_auc_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        LOGGER.info(f"  AUC curves saved → {path}")

        # ── final-level bar chart comparison (Detection Rate) ──
        fig, ax = plt.subplots(figsize=(max(10, len(ds_results) * 0.4), 6))

        tags = sorted(set(r["tag"] for r in ds_results))
        x = np.arange(len(tags))
        width = 0.35

        for i, method in enumerate(METHODS):
            rates = []
            for tag in tags:
                matching = [
                    r for r in ds_results if r["tag"] == tag and r["method"] == method
                ]
                if matching and matching[0]["detection_rates"]:
                    rates.append(matching[0]["detection_rates"][-1])
                else:
                    rates.append(0)
            offset = (i - 0.5) * width
            ax.bar(x + offset, rates, width, label=method)

        ax.set_ylabel("Alert detection rate (final level)")
        ax.set_title(f"{ds_name} — Detection Rate Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(tags, rotation=60, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        path = os.path.join(save_dir, f"{ds_name.lower()}_detection_bar.png")
        plt.savefig(path, dpi=150)
        plt.close()
        LOGGER.info(f"  Detection bar chart saved → {path}")

        # ── final-level bar chart comparison (AUC) ──
        fig, ax = plt.subplots(figsize=(max(10, len(ds_results) * 0.4), 6))

        for i, method in enumerate(METHODS):
            auc_vals = []
            for tag in tags:
                matching = [
                    r for r in ds_results if r["tag"] == tag and r["method"] == method
                ]
                if matching and matching[0]["aucs"]:
                    auc_vals.append(matching[0]["aucs"][-1])
                else:
                    auc_vals.append(0)
            offset = (i - 0.5) * width
            ax.bar(x + offset, auc_vals, width, label=method)

        ax.set_ylabel("Alert AUC (final level)")
        ax.set_title(f"{ds_name} — AUC Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(tags, rotation=60, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        path = os.path.join(save_dir, f"{ds_name.lower()}_auc_bar.png")
        plt.savefig(path, dpi=150)
        plt.close()
        LOGGER.info(f"  AUC bar chart saved → {path}")

    # ── summary heatmap per method ──
    for method in METHODS:
        for ds_name, ds_results in by_dataset.items():
            method_results = [r for r in ds_results if r["method"] == method]
            if not method_results:
                continue

            ptypes = sorted(set(r["pattern_type"] for r in method_results))
            configs = sorted(set(f"s{r['size']}_r{r['reps']}" for r in method_results))

            matrix = np.zeros((len(ptypes), len(configs)))
            for r in method_results:
                pi = ptypes.index(r["pattern_type"])
                ci = configs.index(f"s{r['size']}_r{r['reps']}")
                final_rate = r["detection_rates"][-1] if r["detection_rates"] else 0
                matrix[pi, ci] = final_rate

            fig, ax = plt.subplots(
                figsize=(max(8, len(configs) * 1.5), max(4, len(ptypes) * 0.8))
            )
            im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(configs, rotation=45, ha="right")
            ax.set_yticks(range(len(ptypes)))
            ax.set_yticklabels(ptypes)
            ax.set_xlabel("Configuration (size_reps)")
            ax.set_ylabel("Pattern type")
            ax.set_title(f"{ds_name} — {method} detection rate (final level)")

            for i in range(len(ptypes)):
                for j in range(len(configs)):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )

            fig.colorbar(im, ax=ax, label="Detection rate")
            plt.tight_layout()
            path = os.path.join(save_dir, f"{ds_name.lower()}_{method}_heatmap.png")
            plt.savefig(path, dpi=150)
            plt.close()
            LOGGER.info(f"  Heatmap saved → {path}")

            # ── AUC heatmap ──
            auc_matrix = np.zeros((len(ptypes), len(configs)))
            for r in method_results:
                pi = ptypes.index(r["pattern_type"])
                ci = configs.index(f"s{r['size']}_r{r['reps']}")
                final_auc = r["aucs"][-1] if r["aucs"] else 0
                auc_matrix[pi, ci] = final_auc

            fig, ax = plt.subplots(
                figsize=(max(8, len(configs) * 1.5), max(4, len(ptypes) * 0.8))
            )
            im = ax.imshow(auc_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(configs, rotation=45, ha="right")
            ax.set_yticks(range(len(ptypes)))
            ax.set_yticklabels(ptypes)
            ax.set_xlabel("Configuration (size_reps)")
            ax.set_ylabel("Pattern type")
            ax.set_title(f"{ds_name} — {method} AUC (final level)")

            for i in range(len(ptypes)):
                for j in range(len(configs)):
                    ax.text(
                        j,
                        i,
                        f"{auc_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )

            fig.colorbar(im, ax=ax, label="AUC")
            plt.tight_layout()
            path = os.path.join(save_dir, f"{ds_name.lower()}_{method}_auc_heatmap.png")
            plt.savefig(path, dpi=150)
            plt.close()
            LOGGER.info(f"  AUC heatmap saved → {path}")


def plot_metric_curves(all_results: List[Dict], save_dir: str) -> None:
    """Plot recall and precision vs level for each config."""
    by_dataset: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_results:
        by_dataset[r["dataset"]].append(r)

    for ds_name, ds_results in by_dataset.items():
        for method in METHODS:
            method_results = [r for r in ds_results if r["method"] == method]
            if not method_results:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            for r in method_results:
                levels = list(range(1, len(r["recalls"]) + 1))
                axes[0].plot(
                    levels, r["recalls"], marker=".", label=r["tag"], alpha=0.7
                )
                axes[1].plot(
                    levels, r["precisions"], marker=".", label=r["tag"], alpha=0.7
                )

            axes[0].set_title(f"{ds_name} — {method} Recall")
            axes[0].set_xlabel("Coarsening level")
            axes[0].set_ylabel("Recall")
            axes[0].set_ylim(-0.05, 1.05)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(fontsize=5, ncol=2)

            axes[1].set_title(f"{ds_name} — {method} Precision")
            axes[1].set_xlabel("Coarsening level")
            axes[1].set_ylabel("Precision")
            axes[1].set_ylim(-0.05, 1.05)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(fontsize=5, ncol=2)

            plt.tight_layout()
            path = os.path.join(
                save_dir, f"{ds_name.lower()}_{method}_recall_precision.png"
            )
            plt.savefig(path, dpi=150)
            plt.close()
            LOGGER.info(f"  Recall/Precision curves saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Planted pattern detection via coarsening on Planetoid datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["Cora"],
        choices=VALID_DATASETS,
        help="Planetoid dataset(s) to use (default: Cora)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = args.dataset

    SAVE_ROOT = os.path.join(save_path, "planted_detection")
    os.makedirs(SAVE_ROOT, exist_ok=True)

    LOGGER.info("=" * 65)
    LOGGER.info("  Planted Pattern Detection Experiment")
    LOGGER.info(f"  datasets      : {datasets}")
    LOGGER.info(f"  pattern types : {PATTERN_TYPES}")
    LOGGER.info(f"  sizes         : {SIZES}")
    LOGGER.info(f"  repetitions   : {REPS}")
    LOGGER.info(f"  methods       : {METHODS}")
    LOGGER.info(f"  K             : {K}")
    LOGGER.info(f"  levels        : {MAX_LEVELS}")
    LOGGER.info(f"  max_epsilon   : {MAX_EPSILON}")
    LOGGER.info(f"  output        : {SAVE_ROOT}")
    LOGGER.info("=" * 65)

    all_results: List[Dict] = []

    configs = list(itertools.product(datasets, PATTERN_TYPES, SIZES, REPS))
    total = len(configs) * len(METHODS)
    run_idx = 0

    for ds_name in datasets:
        LOGGER.info(f"\n{'=' * 65}")
        LOGGER.info(f"  Loading {ds_name} …")
        LOGGER.info(f"{'=' * 65}")
        dataset = Planetoid(root="data/Planetoid", name=ds_name)
        G_base = dataset[0]
        G_base.edge_index = to_undirected(G_base.edge_index)
        G_base.edge_weight = torch.ones(G_base.edge_index.size(1), dtype=torch.float32)
        G_base = _ensure_graph_params(G_base)
        LOGGER.info(f"  {ds_name}: {G_base.num_nodes} nodes, {G_base.num_edges} edges")

        for ptype, size, reps in itertools.product(PATTERN_TYPES, SIZES, REPS):
            tag = f"{ptype}_s{size}_r{reps}"
            LOGGER.info(f"\n{'─' * 50}")
            LOGGER.info(f"  [{ds_name}] Config: {tag}")
            LOGGER.info(f"{'─' * 50}")

            # Plant patterns
            G_planted, alert_patterns, normal_patterns = plant_patterns_bfs(
                G_base, ptype, size, reps, seed_val=42
            )
            LOGGER.info(
                f"  Planted {len(alert_patterns)}/{reps} {ptype} patterns (size={size}) | "
                f"y=1: {(G_planted.y == 1).sum().item()}, y=0: {(G_planted.y == 0).sum().item()} | "
                f"edges: {G_planted.num_edges}"
            )

            if len(alert_patterns) == 0:
                LOGGER.warning(f"  SKIPPED — no patterns planted")
                continue

            for method in METHODS:
                run_idx += 1
                LOGGER.info(f"\n  [{run_idx}/{total}] {ds_name} / {tag} / {method}")

                results_history = run_detection(
                    G_planted, alert_patterns, normal_patterns, method, tag
                )

                # Extract per-level metrics
                detection_rates = []
                recalls = []
                precisions = []
                aucs = []
                for res in results_history:
                    am = res.get("alert_metrics", {})
                    detection_rates.append(am.get("detection_rate", 0))
                    recalls.append(am.get("recall", 0))
                    precisions.append(am.get("precision", 0))
                    auc_dict = res.get("detection_auc", {})
                    aucs.append(auc_dict.get("alert_majority", 0))

                final_rate = detection_rates[-1] if detection_rates else 0
                final_recall = recalls[-1] if recalls else 0
                final_precision = precisions[-1] if precisions else 0
                final_auc = aucs[-1] if aucs else 0

                LOGGER.info(
                    f"    → final: det_rate={final_rate:.3f}, "
                    f"recall={final_recall:.3f}, precision={final_precision:.3f}, "
                    f"auc={final_auc:.3f}"
                )

                all_results.append(
                    {
                        "dataset": ds_name,
                        "pattern_type": ptype,
                        "size": size,
                        "reps": reps,
                        "method": method,
                        "tag": tag,
                        "detection_rates": detection_rates,
                        "recalls": recalls,
                        "precisions": precisions,
                        "aucs": aucs,
                        "final_detection_rate": final_rate,
                        "final_recall": final_recall,
                        "final_precision": final_precision,
                        "final_auc": final_auc,
                        "n_planted": len(alert_patterns),
                        "n_levels": len(results_history),
                    }
                )

    # ── Summary plots ────────────────────────────────────────────────────
    LOGGER.info("\n" + "=" * 65)
    LOGGER.info("  Generating summary plots …")
    LOGGER.info("=" * 65)

    plot_detection_curves(all_results, SAVE_ROOT)
    plot_metric_curves(all_results, SAVE_ROOT)

    # ── Summary table ────────────────────────────────────────────────────
    LOGGER.info("\n" + "=" * 100)
    LOGGER.info(
        f"{'Dataset':<10} {'Config':<25} {'Method':<20} "
        f"{'Det.Rate':>8} {'AUC':>8} {'Recall':>8} {'Prec':>8} {'Planted':>7}"
    )
    LOGGER.info("-" * 100)
    for r in all_results:
        LOGGER.info(
            f"{r['dataset']:<10} {r['tag']:<25} {r['method']:<20} "
            f"{r['final_detection_rate']:>8.3f} {r['final_auc']:>8.3f} "
            f"{r['final_recall']:>8.3f} "
            f"{r['final_precision']:>8.3f} {r['n_planted']:>7}"
        )
    LOGGER.info("=" * 100)

    # ── Save results as JSON ─────────────────────────────────────────────
    json_path = os.path.join(SAVE_ROOT, "results.json")
    # Convert numpy types for JSON serialization
    json_safe = []
    for r in all_results:
        row = {}
        for k, v in r.items():
            if isinstance(v, (np.integer, np.floating)):
                row[k] = v.item()
            elif (
                isinstance(v, list)
                and v
                and isinstance(v[0], (np.integer, np.floating))
            ):
                row[k] = [
                    x.item() if isinstance(x, (np.integer, np.floating)) else x
                    for x in v
                ]
            else:
                row[k] = v
        json_safe.append(row)
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2)
    LOGGER.info(f"\nResults JSON saved → {json_path}")

    n_plots = len(list(Path(SAVE_ROOT).rglob("*.png")))
    LOGGER.info(f"Done. {n_plots} PNG files saved under {SAVE_ROOT}")


if __name__ == "__main__":
    main()
