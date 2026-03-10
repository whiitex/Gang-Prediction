"""Experiment driver for coarsening-aware training on AMLGentex.

This is the main experiment runner that uses utility functions from:
- src.GangPrediction.experiment_utils: Data loading, pattern evaluation
- src.GangPrediction.plotting: Unified plotting functions
"""

import os
import sys
from pathlib import Path
import numpy as np

from src.GangPrediction.gang_aware_subspace import get_gang_aware_basis
from src.GangPrediction.utils.plot_gif import (
    get_pattern_colors_and_positions,
    make_gif_with_patterns,
    save_pattern_graph_with_legend,
)

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from src.GangPrediction.coarsening_utils import calc_B_embedding, calc_B
from src.GangPrediction.utils.utils import LOGGER
from src.GangPrediction.GNN_model import evaluate_model
from src.GangPrediction.train_GNN_coarsening import (
    train_GNN_coarsening_aware_loss,
    train_GNN,
)
from src.GangPrediction.experiment_utils import (
    load_amlgentex_data,
    load_all_patterns,
    evaluate_at_level,
    split_patterns,
    get_pattern_node_indices,
)
from src.GangPrediction.plotting import plot_all_results
from src.GangPrediction.utils.utils import *

import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# Configuration
# ============================================================================

EXPERIMENT = "tutorial_demo2"
# METHOD = "variation_neighborhood"
# METHOD = "min_expected_gradient_loss"
METHOD = "variation_edges"
# METHOD = "variation_embedding"
MAX_LEVELS = 500  # Max coarsening levels
EPOCHS_PER_LEVEL = [5]  # Can simplify to [1] for quick tests
THRESHOLD = 0.0  # Coarsening threshold
MAX_EPSILONS = [float("inf")]  # Max coarsening epsilons
# Training hyperparameters
TRAIN_CONFIG = {
    "K": 50,
    "nhid": 16,
    "lr": 0.003,
    "wd": 1e-6,
    "dropout": 0.1,
    # "grad_clip": 1.0,
    # "warmup_epochs": 3,
    "initial_epochs": 5,
    "min_epochs": 1,
    "max_epoch_interval": 3,
    "loss_window": 10,
    "loss_threshold": 0.002,
}

# Pattern detection thresholds
ALERT_THRESHOLDS = (0.5, 0.5)  # (majority, coarsening)
NORMAL_THRESHOLDS = (0.5, 0.5)
PROB_THRESHOLD = 0.3

PLOT_GIFS = False

# Gang-aware subspace configuration
GANG_AWARE_CONFIG = {
    "enabled": True,  # Use gang-aware basis instead of spectral
    "alpha": 1.0,  # Smoothing strength (higher = smoother patterns)
    "method": "svd",  # "svd" for PCA-style, "lda" for Fisher LDA
}

# Pattern-based train/test split configuration
PATTERN_SPLIT_CONFIG = {
    "train_ratio": 0.5,  # Fraction of patterns used for training
    "seed": 42,  # Random seed for reproducibility
}


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
    G, node_to_index = load_amlgentex_data(config_dir)

    # Load patterns
    alert_patterns, alert_types, normal_patterns, normal_types = load_all_patterns(
        experiment_root, node_to_index
    )

    # Split patterns into train and test sets
    print("\n" + "=" * 60)
    print("Splitting Patterns into Train/Test Sets")
    print(f"  Train ratio: {PATTERN_SPLIT_CONFIG['train_ratio']}")
    print("=" * 60)

    alert_train, alert_train_types, alert_test, alert_test_types = split_patterns(
        alert_patterns,
        alert_types,
        train_ratio=PATTERN_SPLIT_CONFIG["train_ratio"],
        seed=PATTERN_SPLIT_CONFIG["seed"],
    )
    normal_train, normal_train_types, normal_test, normal_test_types = split_patterns(
        normal_patterns,
        normal_types,
        train_ratio=PATTERN_SPLIT_CONFIG["train_ratio"],
        seed=PATTERN_SPLIT_CONFIG["seed"],
    )

    print(f"  Alert patterns: {len(alert_train)} train, {len(alert_test)} test")
    print(f"  Normal patterns: {len(normal_train)} train, {len(normal_test)} test")

    # Get node indices from train patterns to use as training nodes
    alert_train_nodes = get_pattern_node_indices(alert_train)
    normal_train_nodes = get_pattern_node_indices(normal_train)
    alert_test_nodes = get_pattern_node_indices(alert_test)
    normal_test_nodes = get_pattern_node_indices(normal_test)

    # Combine train nodes from both alert and normal patterns
    train_nodes = torch.unique(torch.cat([alert_train_nodes, normal_train_nodes]))
    test_nodes = torch.unique(torch.cat([alert_test_nodes, normal_test_nodes]))

    # Update graph training masks to use pattern-based nodes
    print(f"\n  Training nodes from patterns: {len(train_nodes)}")
    print(f"  Test nodes from patterns: {len(test_nodes)}")

    # Override train/val/test indices with pattern-based split
    G.train_idx = train_nodes
    G.val_idx = test_nodes  # Use test patterns for validation
    G.test_idx = test_nodes  # Use test patterns for testing

    if PLOT_GIFS:
        colors0, pos0 = get_pattern_colors_and_positions(
            G, alert_patterns, normal_patterns, alert_types, normal_types
        )
        G.colors = torch.tensor(colors0, dtype=torch.float32, device=device)
        pos = [vals for vals in pos0.values()]
        G.pos = torch.tensor(pos, dtype=torch.float32, device=device)

    # Compute basis for coarsening using only TRAIN patterns
    if GANG_AWARE_CONFIG["enabled"]:
        # Use gang-aware basis that preserves pattern structures (TRAIN ONLY)
        print("\n" + "=" * 60)
        print("Building Gang-Aware Subspace (using TRAIN patterns only)")
        print(f"  Alpha (smoothing): {GANG_AWARE_CONFIG['alpha']}")
        print(f"  Method: {GANG_AWARE_CONFIG['method']}")
        print(
            f"  Train Patterns: {len(alert_train)} malicious, {len(normal_train)} normal"
        )
        print("=" * 60)
        Uk = get_gang_aware_basis(
            G,
            # alert_patterns=alert_patterns,  # Use only train patterns
            # normal_patterns=normal_patterns,  # Use only train patterns
            alert_patterns=alert_train,  # Use only train patterns
            normal_patterns=normal_train,  # Use only train patterns
            # K=TRAIN_CONFIG["K"],
            alpha=GANG_AWARE_CONFIG["alpha"],
            method=GANG_AWARE_CONFIG["method"],
        )
        B = calc_B(G, Uk.shape[1], U=Uk)  # Precompute eigenvectors
        # B = calc_B(G, TRAIN_CONFIG["K"], U=Uk)  # Precompute eigenvectors
        print(f"Gang-aware basis shape: {B.shape}")
    elif METHOD == "variation_embedding":
        Uk = calc_B_embedding(G, TRAIN_CONFIG["K"])
        B = calc_B(G, TRAIN_CONFIG["K"], U=Uk)
    else:
        B = calc_B(G, TRAIN_CONFIG["K"])
    # B2 = calc_B(G, TRAIN_CONFIG["K"])
    # B = torch.concat([B1, B2], dim=1)

    # Train baseline model
    print("\n" + "=" * 60)
    print("Training Baseline GNN Model")
    print("=" * 60)
    model = train_GNN(G, epochs=100, **TRAIN_CONFIG)
    acc_test, prec_baseline, _, _ = evaluate_model(model, G, log_info=True)

    # Ground truth stats
    gt_labels = G.y
    n_suspicious_gt = (gt_labels == 1).sum().item()
    LOGGER.info(
        f"Ground truth: {n_suspicious_gt}/{len(gt_labels)} suspicious "
        f"({100*n_suspicious_gt/len(gt_labels):.2f}%)"
    )

    all_data_train = []
    # all_data_inference = []
    # Run experiments for each epoch setting
    for idx, max_epsilon in enumerate(MAX_EPSILONS):
        LOGGER.info(f"\n{'='*60}")
        # LOGGER.info(f"Epochs per Level: {ep_per_lev}")
        LOGGER.info(f"{'='*60}")

        # for threshold in THRESHOLDS:
        LOGGER.info(f"\nThreshold: {THRESHOLD*100:.0f}%")
        LOGGER.info(f"max_epsilon: {max_epsilon:.4f}")

        # Train coarsening-aware model
        Gall_train, _, model, _ = train_GNN_coarsening_aware_loss(
            G,
            levels=MAX_LEVELS,
            # epoch_per_level=ep_per_lev,
            method=METHOD,
            similarity_threshold=THRESHOLD,
            max_epsilon=max_epsilon,
            B=B,
            train=True,
            # model=model,
            alert_thresholds=ALERT_THRESHOLDS,
            normal_thresholds=NORMAL_THRESHOLDS,
            prob_threshold=PROB_THRESHOLD,
            # Evaluation patterns (all patterns for evaluation)
            # alert_patterns=alert_patterns,
            # normal_patterns=normal_patterns,
            alert_patterns=alert_test,
            normal_patterns=normal_test,
            alert_types=alert_types,
            normal_types=normal_types,
            # Train patterns for contrastive loss
            alert_train_patterns=alert_train,
            normal_train_patterns=normal_train,
            **TRAIN_CONFIG,
        )

        # Load saved results
        name = f"data_gnn_CoarseningAwareLoss_V2_th_{THRESHOLD*100:.0f}_ep_{max_epsilon}_dynamic_train.npy"
        train_data = np.load(f"{save_path}/{name}", allow_pickle=True).item()
        all_data_train.append(train_data)

        # Gall_inference, _, _, _ = train_GNN_coarsening_aware_loss(
        #     G,
        #     levels=MAX_LEVELS,
        #     # epoch_per_level=ep_per_lev,
        #     method=METHOD,
        #     similarity_threshold=THRESHOLD,
        #     max_epsilon=max_epsilon,
        #     B=B,
        #     train=False,
        #     model=model,
        #     alert_thresholds=ALERT_THRESHOLDS,
        #     normal_thresholds=NORMAL_THRESHOLDS,
        #     prob_threshold=PROB_THRESHOLD,
        #     alert_patterns=alert_patterns,
        #     normal_patterns=normal_patterns,
        #     alert_types=alert_types,
        #     normal_types=normal_types,
        #     **TRAIN_CONFIG,
        # )

        # Load saved results
        # name = f"data_gnn_CoarseningAwareLoss_V2_th_{THRESHOLD*100:.0f}_ep_{max_epsilon}_dynamic_inference.npy"
        # inference_data = np.load(f"{save_path}/{name}", allow_pickle=True).item()

        LOGGER.info(
            f"Final: nodes={Gall_train[-1].num_nodes}, edges={Gall_train[-1].num_edges}, "
            f"coarse_acc={train_data['ycrs'][-1]:.4f}, fine_acc={train_data['yfine'][-1]:.4f}"
        )

        # all_data_inference.append(inference_data)

        if PLOT_GIFS:
            # Create GIF with pattern-based coloring and clustering
            make_gif_with_patterns(
                Gall_train,
                gif_path=f"{save_path}max_eps_{max_epsilon:.4f}_patterns.gif",
                frame_duration=60,
            )

            # Save static images of initial and final graphs with legends
            save_pattern_graph_with_legend(
                Gall_train[0],
                save_path=f"{save_path}initial_graph_patterns.png",
                title="Initial Graph with Pattern Coloring",
            )
            save_pattern_graph_with_legend(
                Gall_train[-1],
                save_path=f"{save_path}final_graph_patterns.png",
                title=f"Final Coarsened Graph ({Gall_train[-1].num_nodes} nodes)",
            )

    # Train model on final coarsened graph for baseline comparison
    G_coarse = Gall_train[-1]
    model_coarse = train_GNN(G_coarse, epochs=100, lr=0.005)
    acc_coarse, _, _, _ = evaluate_model(model_coarse, G_coarse, log_info=False)

    # Generate all plots
    plot_all_results(
        data_list=all_data_train,
        # threshold=THRESHOLD,
        save_dir=str(save_path),
        # epochs_per_level=ep_per_lev,
        max_epsilons=MAX_EPSILONS,
        baseline_accuracy=acc_test,
        baseline_precision=prec_baseline,
        coarse_accuracy=acc_coarse,
        has_alert_patterns=bool(alert_patterns),
        has_normal_patterns=bool(normal_patterns),
        smooth=False,
        alert_coarse_th=ALERT_THRESHOLDS[1],
        alert_majority_th=ALERT_THRESHOLDS[0],
        normal_coarse_th=NORMAL_THRESHOLDS[1],
        normal_majority_th=NORMAL_THRESHOLDS[0],
        name_prefix="_train",
    )
    # Generate all plots
    # plot_all_results(
    #     data_list=all_data_inference,
    #     # threshold=THRESHOLD,
    #     save_dir=str(save_path),
    #     # epochs_per_level=ep_per_lev,
    #     max_epsilons=MAX_EPSILONS,
    #     baseline_accuracy=acc_test,
    #     baseline_precision=prec_baseline,
    #     coarse_accuracy=acc_coarse,
    #     has_alert_patterns=bool(alert_patterns),
    #     has_normal_patterns=bool(normal_patterns),
    #     smooth=False,
    #     alert_coarse_th=ALERT_THRESHOLDS[1],
    #     alert_majority_th=ALERT_THRESHOLDS[0],
    #     normal_coarse_th=NORMAL_THRESHOLDS[1],
    #     normal_majority_th=NORMAL_THRESHOLDS[0],
    #     name_prefix="_inference",
    # )

    LOGGER.info(f"Plots saved to {save_path}")


def evaluate_all_levels(
    data,
    model,
    Gall,
    Call,
    gt_labels,
    alert_patterns,
    normal_patterns,
    alert_types=None,
    normal_types=None,
):
    """Evaluate pattern detection at all coarsening levels."""
    import torch.nn.functional as F

    alert_types = alert_types or {}
    normal_types = normal_types or {}

    pattern_detection_rates = []
    pattern_detection_rates_gt = []
    normal_pattern_detection_rates_gt = []

    # Per-type detection tracking
    alert_gt_type_rates = []
    alert_model_type_rates = []
    normal_gt_type_rates = []

    for level_idx in range(len(Gall)):
        results = evaluate_at_level(
            model,
            Gall[level_idx],
            Call[level_idx],
            gt_labels,
            alert_patterns,
            normal_patterns,
            alert_types=alert_types,
            normal_types=normal_types,
            alert_thresholds=ALERT_THRESHOLDS,
            normal_thresholds=NORMAL_THRESHOLDS,
            prob_threshold=PROB_THRESHOLD,
        )

        # Collect detection rates
        if alert_patterns:
            pattern_detection_rates.append(results.get("alert_model_rate", 0))
            pattern_detection_rates_gt.append(results.get("alert_gt_rate", 0))
            # Per-type rates
            alert_gt_type_rates.append(results.get("alert_gt_type_rates", {}))
            alert_model_type_rates.append(results.get("alert_model_type_rates", {}))

        if normal_patterns:
            normal_pattern_detection_rates_gt.append(results.get("normal_gt_rate", 0))
            normal_gt_type_rates.append(results.get("normal_gt_type_rates", {}))

        # Log at first and last level
        if level_idx == 0 or level_idx == len(Gall) - 1:
            log_level_results(
                level_idx,
                results,
                alert_patterns,
                normal_patterns,
                alert_types,
                normal_types,
            )

    # Add to data dict
    if alert_patterns:
        data["pattern_detection_rates"] = pattern_detection_rates
        data["pattern_detection_rates_gt"] = pattern_detection_rates_gt
        data["alert_gt_type_rates"] = alert_gt_type_rates
        data["alert_model_type_rates"] = alert_model_type_rates
    if normal_patterns:
        data["normal_pattern_detection_rates_gt"] = normal_pattern_detection_rates_gt
        data["normal_gt_type_rates"] = normal_gt_type_rates

    # Log final results
    if alert_patterns:
        LOGGER.info(f"Alert detection (model) final: {pattern_detection_rates[-1]:.4f}")
        LOGGER.info(f"Alert detection (GT) final: {pattern_detection_rates_gt[-1]:.4f}")
    if normal_patterns:
        LOGGER.info(
            f"Normal detection (GT) final: {normal_pattern_detection_rates_gt[-1]:.4f}"
        )

    return data


def log_level_results(
    level_idx,
    results,
    alert_patterns,
    normal_patterns,
    alert_types=None,
    normal_types=None,
):
    """Log pattern detection results for a specific level."""
    alert_types = alert_types or {}
    normal_types = normal_types or {}

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
