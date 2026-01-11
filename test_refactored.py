"""Experiment driver for coarsening-aware training on AMLGentex.

This is the main experiment runner that uses utility functions from:
- src.GangPrediction.experiment_utils: Data loading, pattern evaluation
- src.GangPrediction.plotting: Unified plotting functions
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from src.GangPrediction.utils.utils import LOGGER
from src.GangPrediction.GNN_model import evaluate_model
from src.GangPrediction.train_GNN_coarsening import (
    train_GNN_coarsening_aware_loss,
    train_GNN,
)
from src.GangPrediction.experiment_utils import (
    load_amlgentex_data,
    load_all_patterns,
    evaluate_pattern_detection,
    evaluate_at_level,
    get_node_to_supernode_mapping,
)
from src.GangPrediction.plotting import plot_all_results
from src.GangPrediction.utils.utils import *

import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# Configuration
# ============================================================================

EXPERIMENT = "tutorial_demo2"
METHOD = "variation_embedding"
MAX_LEVELS = 150
EPOCHS_PER_LEVEL = [2]  # Can simplify to [1] for quick tests
THRESHOLDS = [0.50, 0.75, 0.85, 0.95]  # Coarsening thresholds

# Training hyperparameters
TRAIN_CONFIG = {
    "K": 100,
    "nhid": 256,
    "lr": 0.01,
    "wd": 1e-4,
    "dropout": 0.1,
    "grad_clip": 1.0,
    "warmup_epochs": 3,
}

# Pattern detection thresholds
ALERT_THRESHOLDS = (0.75, 0.75)  # (majority, coarsening)
NORMAL_THRESHOLDS = (0.75, 0.75)
PROB_THRESHOLD = 0.3


# ============================================================================
# Main Experiment
# ============================================================================


def run_experiment():
    """Run the coarsening-aware GNN training experiment."""

    # Setup paths
    experiment_root = project_root / "experiments" / EXPERIMENT
    config_dir = experiment_root / "config"

    # Load data
    print("=" * 60)
    print("Loading AMLGentex Dataset")
    print("=" * 60)
    G, node_to_index = load_amlgentex_data(experiment_root, config_dir)

    # Load patterns
    alert_patterns, alert_types, normal_patterns, normal_types = load_all_patterns(
        experiment_root, node_to_index
    )

    # Train baseline model
    print("\n" + "=" * 60)
    print("Training Baseline GNN Model")
    print("=" * 60)
    model = train_GNN(G, epochs=100, lr=0.005)
    acc_test, prec_baseline, _, _ = evaluate_model(model, G, log_info=True)

    # Ground truth stats
    gt_labels = G.y
    n_suspicious_gt = (gt_labels == 1).sum().item()
    LOGGER.info(
        f"Ground truth: {n_suspicious_gt}/{len(gt_labels)} suspicious "
        f"({100*n_suspicious_gt/len(gt_labels):.2f}%)"
    )

    # Run experiments for each epoch setting
    for ep_per_lev in EPOCHS_PER_LEVEL:
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Epochs per Level: {ep_per_lev}")
        LOGGER.info(f"{'='*60}")

        all_data = []

        for threshold in THRESHOLDS:
            LOGGER.info(f"\nThreshold: {threshold*100:.0f}%")

            # Train coarsening-aware model
            Gall, Call, model, C_plus = train_GNN_coarsening_aware_loss(
                G,
                levels=MAX_LEVELS,
                epoch_per_level=ep_per_lev,
                method=METHOD,
                similarity_threshold=threshold,
                **TRAIN_CONFIG,
            )

            # Load saved results
            name = f"data_gnn_CoarseningAwareLoss_V2_th_{threshold*100:.0f}_epochs_{ep_per_lev}.npy"
            data = np.load(f"{save_path}/{name}", allow_pickle=True).item()

            LOGGER.info(
                f"Final: nodes={Gall[-1].num_nodes}, edges={Gall[-1].num_edges}, "
                f"coarse_acc={data['ycrs'][-1]:.4f}, fine_acc={data['yfine'][-1]:.4f}"
            )

            # Evaluate pattern detection at each level
            if alert_patterns or normal_patterns:
                data = evaluate_all_levels(
                    data, model, Gall, Call, gt_labels, alert_patterns, normal_patterns
                )
                np.save(f"{save_path}/{name}", data)

            all_data.append(data)

        # Train model on final coarsened graph for baseline comparison
        G_coarse = Gall[-1]
        model_coarse = train_GNN(G_coarse, epochs=100, lr=0.005)
        acc_coarse, _, _, _ = evaluate_model(model_coarse, G_coarse, log_info=False)

        # Generate all plots
        plot_all_results(
            data_list=all_data,
            thresholds=THRESHOLDS,
            save_dir=str(save_path),
            epochs_per_level=ep_per_lev,
            baseline_accuracy=acc_test,
            baseline_precision=prec_baseline,
            coarse_accuracy=acc_coarse,
            has_alert_patterns=bool(alert_patterns),
            has_normal_patterns=bool(normal_patterns),
            smooth=False,
        )

        LOGGER.info(f"Plots saved to {save_path}")


def evaluate_all_levels(
    data, model, Gall, Call, gt_labels, alert_patterns, normal_patterns
):
    """Evaluate pattern detection at all coarsening levels."""
    import torch.nn.functional as F

    pattern_detection_rates = []
    pattern_detection_rates_gt = []
    normal_pattern_detection_rates_gt = []

    for level_idx in range(len(Gall)):
        results = evaluate_at_level(
            model,
            Gall,
            Call,
            level_idx,
            gt_labels,
            alert_patterns,
            normal_patterns,
            alert_thresholds=ALERT_THRESHOLDS,
            normal_thresholds=NORMAL_THRESHOLDS,
            prob_threshold=PROB_THRESHOLD,
        )

        # Collect detection rates
        if alert_patterns:
            pattern_detection_rates.append(results.get("alert_model_rate", 0))
            pattern_detection_rates_gt.append(results.get("alert_gt_rate", 0))

        if normal_patterns:
            normal_pattern_detection_rates_gt.append(results.get("normal_gt_rate", 0))

        # Log at first and last level
        if level_idx == 0 or level_idx == len(Gall) - 1:
            log_level_results(level_idx, results, alert_patterns, normal_patterns)

    # Add to data dict
    if alert_patterns:
        data["pattern_detection_rates"] = pattern_detection_rates
        data["pattern_detection_rates_gt"] = pattern_detection_rates_gt
    if normal_patterns:
        data["normal_pattern_detection_rates_gt"] = normal_pattern_detection_rates_gt

    # Log final results
    if alert_patterns:
        LOGGER.info(f"Alert detection (model) final: {pattern_detection_rates[-1]:.4f}")
        LOGGER.info(f"Alert detection (GT) final: {pattern_detection_rates_gt[-1]:.4f}")
    if normal_patterns:
        LOGGER.info(
            f"Normal detection (GT) final: {normal_pattern_detection_rates_gt[-1]:.4f}"
        )

    return data


def log_level_results(level_idx, results, alert_patterns, normal_patterns):
    """Log pattern detection results for a specific level."""
    LOGGER.info(
        f"Level {level_idx}: {results['n_pred_suspicious']}/{results.get('total_nodes', 'N/A')} "
        f"pred suspicious, {results['n_prob_suspicious']} prob>{PROB_THRESHOLD} "
        f"(mean={results['mean_prob']:.3f}, max={results['max_prob']:.3f})"
    )

    if alert_patterns:
        n_alert = len(alert_patterns)
        n_detected = len(results.get("alert_gt_detected", []))
        n_model_detected = len(results.get("alert_model_detected", []))
        LOGGER.info(
            f"  Alert Model-based: {results.get('alert_model_rate', 0):.4f} ({n_model_detected}/{n_alert})"
        )
        LOGGER.info(
            f"  Alert GT-based: {results.get('alert_gt_rate', 0):.4f} ({n_detected}/{n_alert})"
        )

    if normal_patterns:
        n_normal = len(normal_patterns)
        n_detected = len(results.get("normal_gt_detected", []))
        LOGGER.info(
            f"  Normal GT-based: {results.get('normal_gt_rate', 0):.4f} ({n_detected}/{n_normal})"
        )


if __name__ == "__main__":
    run_experiment()
