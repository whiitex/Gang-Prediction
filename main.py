"""Experiment driver for coarsening-aware training on AMLGentex.

This is the main experiment runner that uses utility functions from:
- src.GangPrediction.experiment_utils: Data loading, pattern evaluation
- src.GangPrediction.plotting: Unified plotting functions
"""

import os
import sys
from pathlib import Path

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

EXPERIMENT = "tutorial_demo5"
# METHOD = "variation_neighborhood"
# METHOD = "min_expected_gradient_loss"
# METHOD = "edge_gangs"
METHOD = "learning_subspace"
# METHOD = "variation_edges"
# METHOD = "variation_embedding"
MAX_LEVELS = 1000  # Max coarsening levels
EPOCHS_PER_LEVEL = [5]  # Can simplify to [1] for quick tests
THRESHOLD = 0.0  # Coarsening threshold
MAX_EPSILON = 10  # Max coarsening epsilons
# MAX_EPSILON = float("inf")  # Max coarsening epsilons
# Training hyperparameters
TRAIN_CONFIG = {
    "K": 100,
    "nhid": 8,
    "lr": 0.001,
    "wd": 1e-6,
    "dropout": 0.2,
    # "grad_clip": 1.0,
    # "warmup_epochs": 3,
    "initial_epochs": 3,
    "min_epochs": 1,
    "max_epoch_interval": 3,
    "loss_window": 10,
    "loss_threshold": 0.002,
    "alpha": 1.0,  # Smoothing strength (higher = smoother patterns)
    "compression_method": "svd",  # "svd" for PCA-style, "lda" for Fisher LDA
}

# Pattern detection thresholds
ALERT_THRESHOLDS = (0.5, 0.5)  # (majority, coarsening)
NORMAL_THRESHOLDS = (0.5, 0.5)
PROB_THRESHOLD = 0.3

PLOT_GIFS = False

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
    alert_patterns, normal_patterns = load_all_patterns(experiment_root, node_to_index)

    # Split patterns into train and test sets
    print("\n" + "=" * 60)
    print("Splitting Patterns into Train/Test Sets")
    print(f"  Train ratio: {PATTERN_SPLIT_CONFIG['train_ratio']}")
    print("=" * 60)

    alert_train, alert_test = split_patterns(
        alert_patterns,
        train_ratio=PATTERN_SPLIT_CONFIG["train_ratio"],
        seed=PATTERN_SPLIT_CONFIG["seed"],
    )
    normal_train, normal_test = split_patterns(
        normal_patterns,
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
            G, alert_patterns, normal_patterns
        )
        G.colors = torch.tensor(colors0, dtype=torch.float32, device=device)
        pos = [vals for vals in pos0.values()]
        G.pos = torch.tensor(pos, dtype=torch.float32, device=device)

    # Train baseline model
    print("\n" + "=" * 60)
    print("Training Baseline GNN Model")
    print("=" * 60)
    model = train_GNN(G, epochs=100, **TRAIN_CONFIG)
    baseline_results = evaluate_model(model, G)
    acc_test = baseline_results["accuracy_test"]
    prec_baseline = baseline_results["precision_test"]

    # Ground truth stats
    gt_labels = G.y
    n_suspicious_gt = (gt_labels == 1).sum().item()
    LOGGER.info(
        f"Ground truth: {n_suspicious_gt}/{len(gt_labels)} suspicious "
        f"({100*n_suspicious_gt/len(gt_labels):.2f}%)"
    )

    # for threshold in THRESHOLDS:
    LOGGER.info(f"\nThreshold: {THRESHOLD*100:.0f}%")
    LOGGER.info(f"max_epsilon: {MAX_EPSILON:.4f}")

    # Train coarsening-aware model
    Gall_train, _, model, _, results_history = train_GNN_coarsening_aware_loss(
        G,
        levels=MAX_LEVELS,
        # epoch_per_level=ep_per_lev,
        method=METHOD,
        similarity_threshold=THRESHOLD,
        max_epsilon=MAX_EPSILON,
        train=True,
        # model=model,
        alert_thresholds=ALERT_THRESHOLDS,
        normal_thresholds=NORMAL_THRESHOLDS,
        # alert_patterns=alert_patterns,
        # normal_patterns=normal_patterns,
        alert_patterns=alert_test,
        normal_patterns=normal_test,
        alert_train_patterns=alert_train,
        normal_train_patterns=normal_train,
        **TRAIN_CONFIG,
    )

    # Load saved results
    LOGGER.info(
        f"Final: nodes={Gall_train[-1].num_nodes}, edges={Gall_train[-1].num_edges}, "
        f"coarse_acc={results_history[-1]['accuracy_test']:.4f}, fine_acc={results_history[-1]['accuracy_fine']:.4f}"
    )

    # all_data_inference.append(inference_data)

    if PLOT_GIFS:
        # Create GIF with pattern-based coloring and clustering
        make_gif_with_patterns(
            Gall_train,
            gif_path=f"{save_path}max_eps_{MAX_EPSILON:.4f}_patterns.gif",
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
    coarse_results = evaluate_model(model_coarse, G_coarse)
    acc_coarse = coarse_results["accuracy_test"]

    # Generate all plots
    plot_all_results(
        data_list=results_history,
        # threshold=THRESHOLD,
        save_dir=str(save_path),
        # epochs_per_level=ep_per_lev,
        max_epsilon=MAX_EPSILON,
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
        name_prefix="",
    )

    LOGGER.info(f"Plots saved to {save_path}")


if __name__ == "__main__":
    run_experiment()
