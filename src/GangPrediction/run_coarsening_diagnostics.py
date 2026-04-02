"""run_coarsening_diagnostics.py
==============================
Pure-structure diagnostic runner that addresses two key questions:

  1. WHY do low-frequency Laplacian eigenvectors matter for gang detection?
  2. WHY does a fast epsilon schedule (small power) outperform gradual coarsening?

No GNN training happens here — all experiments work on the graph structure
and the pattern label assignments only.

Usage (from repo root, with FedStruct conda env active):

        python scripts/run_coarsening_diagnostics.py

All plots are saved to:
    results/coarsening_diagnostics/aml/
    results/coarsening_diagnostics/synthetic/
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── path setup (mirrors main.py) ────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import torch

from src.utils.config_parser import load_main_config
from src.GangPrediction.experiment_utils import load_and_preprocess_data
from src.GangPrediction.coarsening_diagnostics import (
    build_synthetic_gang_graph,
    spectral_fingerprint,
    merge_recall_precision_vs_K,
    plot_epsilon_schedules,
    epsilon_schedule_ablation,
    supernode_entropy_analysis,
)
from src.GangPrediction.utils.utils import *

# ── load config ──────────────────────────────────────────────────────────────
CONFIG_PATH = project_root / "config.yaml"
CONFIG = load_main_config(CONFIG_PATH)

EXPERIMENT = CONFIG["experiment"]
MAX_LEVELS = int(CONFIG["max_levels"])
MAX_EPSILON = float(CONFIG["max_epsilon"])
TRAIN_CONFIG = CONFIG["train_config"]
K = int(TRAIN_CONFIG.get("K", 100))
ESP = float(TRAIN_CONFIG.get("epsilon_schedule_power", 0.04))

AML_SAVE_DIR = f"{save_path}/aml/"
SYNTHETIC_SAVE_DIR = f"{save_path}/synthetic/"
os.makedirs(AML_SAVE_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_SAVE_DIR, exist_ok=True)

# Schedule powers to compare across ablation experiments.
# We always include the currently configured power so the plots are relevant.
SCHEDULE_POWERS = sorted(set([0.04, 0.25, 1.0, 5.0, ESP]))

LOGGER.info("=" * 65)
LOGGER.info("  Coarsening Diagnostics Runner")
LOGGER.info(f"  experiment      : {EXPERIMENT}")
LOGGER.info(f"  K               : {K}")
LOGGER.info(f"  max_epsilon     : {MAX_EPSILON}")
LOGGER.info(f"  max_levels      : {MAX_LEVELS}")
LOGGER.info(f"  active power    : {ESP}")
# LOGGER.info(f"  output root     : {save_path}")
LOGGER.info(f"  AML plots       : {AML_SAVE_DIR}")
LOGGER.info(f"  synthetic plots : {SYNTHETIC_SAVE_DIR}")
LOGGER.info("=" * 65)

# ── Load AML data ─────────────────────────────────────────────────────────────
LOGGER.info("\n[1/6] Loading AML dataset …")
experiment_root = project_root / "experiments" / EXPERIMENT
G, alert_train, normal_train, alert_test, normal_test = load_and_preprocess_data(
    data_dir=experiment_root / "config",
    patterns_dir=experiment_root,
    train_ratio=CONFIG["pattern_split_config"].get("train_ratio", 0.2),
    to_undirected=True,
    remove_overlaps=CONFIG.get("remove_overlaps", False),
    device=torch.device("cpu"),
)
LOGGER.info(f"  Graph: {G.num_nodes} nodes, {G.num_edges} edges")
LOGGER.info(f"  Alert patterns  — train: {len(alert_train)}, test: {len(alert_test)}")
LOGGER.info(f"  Normal patterns — train: {len(normal_train)}, test: {len(normal_test)}")

# Combine all patterns (train + test) for structural experiments
all_alert = list(alert_train) + list(alert_test)
all_normal = list(normal_train) + list(normal_test)

# ── Build synthetic graph ─────────────────────────────────────────────────────
LOGGER.info("\n[2/6] Building synthetic gang graph …")
G_syn, alert_syn, normal_syn = build_synthetic_gang_graph()
LOGGER.info(f"  Synthetic graph: {G_syn.num_nodes} nodes, {G_syn.num_edges} edges")
LOGGER.info(f"  Synthetic alert patterns: {len(alert_syn)}, normal: {len(normal_syn)}")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Spectral Fingerprint of Patterns
# ─────────────────────────────────────────────────────────────────────────────
LOGGER.info("\n" + "=" * 65)
LOGGER.info("Experiment 1: Spectral Fingerprint of Patterns")
LOGGER.info("=" * 65)
LOGGER.info("AML graph …")
spectral_fingerprint(
    G,
    alert_patterns=all_alert,
    normal_patterns=all_normal,
    K_max=min(300, G.num_nodes),
    save_dir=str(AML_SAVE_DIR),
    name_prefix="",
)
LOGGER.info("Synthetic graph …")
spectral_fingerprint(
    G_syn,
    alert_patterns=alert_syn,
    normal_patterns=normal_syn,
    K_max=G_syn.num_nodes,
    save_dir=str(SYNTHETIC_SAVE_DIR),
    name_prefix="",
)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Merge Recall/Precision vs K Across Levels
# ─────────────────────────────────────────────────────────────────────────────
LOGGER.info("\n" + "=" * 65)
LOGGER.info("Experiment 2: Merge Recall/Precision vs K Across Levels")
LOGGER.info("=" * 65)
K_values_aml = [k for k in [2, 5, 10, 20, 50, 100, 200] if k <= G.num_nodes]
K_values_syn = [k for k in [2, 5, 10, 20, 40, 55] if k <= G_syn.num_nodes]

LOGGER.info("AML graph …")
merge_recall_precision_vs_K(
    G,
    alert_patterns=all_alert,
    normal_patterns=all_normal,
    K_values=K_values_aml,
    levels=MAX_LEVELS,
    max_sigma=1e6,
    save_dir=str(AML_SAVE_DIR),
    name_prefix="",
)
LOGGER.info("Synthetic graph …")
merge_recall_precision_vs_K(
    G_syn,
    alert_patterns=alert_syn,
    normal_patterns=normal_syn,
    K_values=K_values_syn,
    levels=MAX_LEVELS,
    max_sigma=1e6,
    save_dir=str(SYNTHETIC_SAVE_DIR),
    name_prefix="",
)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Epsilon Schedule Visualisation
# ─────────────────────────────────────────────────────────────────────────────
LOGGER.info("\n" + "=" * 65)
LOGGER.info("Experiment 3: Epsilon Schedule Visualisation")
LOGGER.info("=" * 65)
LOGGER.info("AML graph …")
plot_epsilon_schedules(
    levels=MAX_LEVELS,
    max_epsilon=MAX_EPSILON,
    powers=SCHEDULE_POWERS,
    save_dir=str(AML_SAVE_DIR),
    name_prefix="",
)
LOGGER.info("Synthetic graph …")
plot_epsilon_schedules(
    levels=MAX_LEVELS,
    max_epsilon=MAX_EPSILON,
    powers=SCHEDULE_POWERS,
    save_dir=str(SYNTHETIC_SAVE_DIR),
    name_prefix="",
)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4: Schedule Ablation — Merge Recall/Precision & Pattern Collapse
# ─────────────────────────────────────────────────────────────────────────────
LOGGER.info("\n" + "=" * 65)
LOGGER.info("Experiment 4: Schedule Ablation")
LOGGER.info("=" * 65)
LOGGER.info("AML graph …")
epsilon_schedule_ablation(
    G,
    alert_patterns=all_alert,
    normal_patterns=all_normal,
    K=K,
    powers=SCHEDULE_POWERS,
    levels=MAX_LEVELS,
    max_epsilon=MAX_EPSILON,
    save_dir=str(AML_SAVE_DIR),
    name_prefix="",
)
LOGGER.info("Synthetic graph …")
epsilon_schedule_ablation(
    G_syn,
    alert_patterns=alert_syn,
    normal_patterns=normal_syn,
    K=min(K, G_syn.num_nodes // 3 - 1),
    powers=SCHEDULE_POWERS,
    levels=MAX_LEVELS,
    max_epsilon=MAX_EPSILON,
    save_dir=str(SYNTHETIC_SAVE_DIR),
    name_prefix="",
)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 5: Super-Node Label Entropy
# ─────────────────────────────────────────────────────────────────────────────
LOGGER.info("\n" + "=" * 65)
LOGGER.info("Experiment 5: Super-Node Label Entropy")
LOGGER.info("=" * 65)
LOGGER.info("AML graph …")
supernode_entropy_analysis(
    G,
    K=K,
    powers=SCHEDULE_POWERS,
    levels=MAX_LEVELS,
    max_epsilon=MAX_EPSILON,
    save_dir=str(AML_SAVE_DIR),
    name_prefix="",
)
LOGGER.info("Synthetic graph …")
supernode_entropy_analysis(
    G_syn,
    K=min(K, G_syn.num_nodes // 3 - 1),
    powers=SCHEDULE_POWERS,
    levels=MAX_LEVELS,
    max_epsilon=MAX_EPSILON,
    save_dir=str(SYNTHETIC_SAVE_DIR),
    name_prefix="",
)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
LOGGER.info("\n" + "=" * 65)
LOGGER.info("All diagnostics complete.")
plots = sorted(Path(save_path).rglob("*.png"))
LOGGER.info(f"  {len(plots)} PNG files saved under {save_path}")
for dataset_dir in (AML_SAVE_DIR, SYNTHETIC_SAVE_DIR):
    dataset_plots = sorted(Path(dataset_dir).glob("*.png"))
    LOGGER.info(f"  {Path(dataset_dir).name}: {len(dataset_plots)} PNG files")
    for plot_path in dataset_plots:
        LOGGER.info(f"    {Path(plot_path).relative_to(save_path)}")
LOGGER.info("=" * 65)
