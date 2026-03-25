import os
import shutil
import sys
from pathlib import Path


# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

from src.GangPrediction.GNN_model import evaluate_model
from src.GangPrediction.train_GNN_coarsening import (
    train_GNN_coarsening_aware_loss,
    train_GNN,
)
from src.GangPrediction.experiment_utils import (
    create_subspace,
    load_and_preprocess_data,
)
from src.GangPrediction.plotting import plot_all_results
from src.GangPrediction.embedding_diagnostics import (
    plot_diagnostic_trends,
    generate_diagnostic_plots_for_final_state,
)
from src.GangPrediction.utils.plot_gif import (
    get_pattern_colors_and_positions,
    make_gif_with_patterns,
    save_pattern_graph_with_legend,
)
from src.utils.config_parser import load_main_config
from src.GangPrediction.utils.utils import *


# ============================================================================
# Configuration
# ============================================================================

CONFIG_PATH = project_root / "config.yaml"
CONFIG = load_main_config(CONFIG_PATH)

EXPERIMENT = CONFIG["experiment"]
METHOD = CONFIG["method"]
MAX_LEVELS = int(CONFIG["max_levels"])  # Max coarsening levels
THRESHOLD = float(CONFIG["threshold"])  # Coarsening threshold
MAX_EPSILON = float(CONFIG["max_epsilon"])  # Max coarsening epsilon
TRAIN_CONFIG = CONFIG["train_config"]
ALERT_THRESHOLDS = CONFIG["alert_thresholds"]  # (majority, coarsening)
NORMAL_THRESHOLDS = CONFIG["normal_thresholds"]
PLOT_GIFS = bool(CONFIG["plot_gifs"])
PATTERN_SPLIT_CONFIG = CONFIG["pattern_split_config"]


# ============================================================================
# Main Experiment
# ============================================================================
def run_experiment():
    """Run the coarsening-aware GNN training experiment."""
    # Keep an exact copy of the run configuration next to generated artifacts.
    shutil.copy2(CONFIG_PATH, Path(save_path) / CONFIG_PATH.name)

    # Setup paths
    experiment_root = project_root / "experiments" / EXPERIMENT

    # Load data
    print("=" * 60)
    print("Loading AMLGentex Dataset")
    print("=" * 60)
    G, alert_train, normal_train, alert_test, normal_test = load_and_preprocess_data(
        data_dir=experiment_root / "config",
        patterns_dir=experiment_root,
        train_ratio=PATTERN_SPLIT_CONFIG.get("train_ratio", 0.5),
        to_undirected="variation" in METHOD.lower(),
        device=device,
    )

    if PLOT_GIFS:
        colors0, pos0 = get_pattern_colors_and_positions(G, alert_train, normal_train)
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
    n_suspicious_gt = (G.y == 1).sum().item()
    LOGGER.info(
        f"Ground truth: {n_suspicious_gt}/{len(G.y)} suspicious "
        f"({100*n_suspicious_gt/len(G.y):.2f}%)"
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
        has_alert_patterns=bool(alert_train),
        has_normal_patterns=bool(normal_train),
        smooth=False,
        alert_coarse_th=ALERT_THRESHOLDS[1],
        alert_majority_th=ALERT_THRESHOLDS[0],
        normal_coarse_th=NORMAL_THRESHOLDS[1],
        normal_majority_th=NORMAL_THRESHOLDS[0],
        name_prefix="",
    )

    plot_diagnostic_trends(results_history=results_history, save_dir=str(save_path))

    model.eval()
    with torch.no_grad():
        final_embeddings = model.get_embeddings(
            G.x,
            G.edge_index,
            G.edge_weight if hasattr(G, "edge_weight") else None,
        )
        # N = G.num_nodes
        # final_embeddings = create_subspace(alert_test, normal_test, N, device)
    learned_basis = getattr(model, "latest_learned_basis_rows", None)
    generate_diagnostic_plots_for_final_state(
        embeddings=final_embeddings,
        alert_patterns=alert_test,
        normal_patterns=normal_test,
        save_dir=str(save_path),
        basis_rows=learned_basis,
    )

    LOGGER.info(f"Plots saved to {save_path}")


if __name__ == "__main__":
    run_experiment()
