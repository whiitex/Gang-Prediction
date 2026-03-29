import os
import sys
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from src.GangPrediction.training import evaluate_model
from src.GangPrediction.train_GNN_coarsening import (
    train_GNN_coarsening_aware_loss,
    train_GNN,
)
from src.GangPrediction.experiment_utils import (
    create_subspace,
    load_and_preprocess_data,
)
from src.GangPrediction.plotting import (
    plot_all_results,
    plot_training_loss_components,
    plot_detection_rate_vs_threshold,
)
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
        remove_overlaps=CONFIG.get("remove_overlaps", True),
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
    model = train_GNN(
        G,
        epochs=100,
        lr=0.005,
        nhid=32,
        use_class_weights=True,
        num_layers=3,
        GNN_type="SAGE",
    )
    baseline_results = evaluate_model(model, G)
    acc_test = baseline_results["accuracy_test"]
    prec_baseline = baseline_results["precision_test"]

    # Save baseline model
    baseline_model_path = Path(save_path) / "models" / "baseline_model.pt"
    baseline_model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), str(baseline_model_path))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "SAGE",
            "model_kwargs": {
                "nfeat": G.num_features,
                "nhid": TRAIN_CONFIG.get("nhid", 128),
                "nclass": len(np.unique(G.y.numpy())),
                "dropout": TRAIN_CONFIG.get("dropout", 0.1),
                "num_layers": TRAIN_CONFIG.get("num_layers", 2),
                "GNN_type": TRAIN_CONFIG.get("GNN_type", "SAGE"),
                "use_edge_weights": TRAIN_CONFIG.get("use_edge_weights", False),
            },
            "results": baseline_results,
        },
        str(baseline_model_path.with_suffix(".pth")),
    )
    LOGGER.info(f"Baseline model saved to {baseline_model_path}")

    # Ground truth stats
    n_suspicious_gt = (G.y == 1).sum().item()
    LOGGER.info(
        f"Ground truth: {n_suspicious_gt}/{len(G.y)} suspicious "
        f"({100*n_suspicious_gt/len(G.y):.2f}%)"
    )

    # for threshold in THRESHOLDS:
    LOGGER.info(f"max_epsilon: {MAX_EPSILON:.4f}")

    Gall_train, _, model, _, results_history = train_GNN_coarsening_aware_loss(
        G,
        levels=MAX_LEVELS,
        method=METHOD,
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

    EXPERIMENT_test = "tutorial_demo14"
    experiment_root_test = project_root / "experiments" / EXPERIMENT_test
    # Train coarsening-aware model
    G_test, _, _, alert_test2, normal_test2 = load_and_preprocess_data(
        data_dir=experiment_root_test / "config",
        patterns_dir=experiment_root_test,
        train_ratio=0,
        to_undirected="variation" in METHOD.lower(),
        remove_overlaps=CONFIG.get("remove_overlaps", True),
        device=device,
    )

    Gall_test, _, _, _, results_history_test = train_GNN_coarsening_aware_loss(
        G_test,
        levels=5,
        method=METHOD,
        max_epsilon=MAX_EPSILON,
        train=False,
        model=model,
        alert_thresholds=ALERT_THRESHOLDS,
        normal_thresholds=NORMAL_THRESHOLDS,
        alert_patterns=alert_test2,
        normal_patterns=normal_test2,
        epsilon_schedule_power=0.0,
        use_label_for_coarsening=PATTERN_SPLIT_CONFIG.get(
            "use_label_for_coarsening", False
        ),
    )

    # Load saved results
    LOGGER.info(
        f"Final: nodes={Gall_test[-1].num_nodes}, edges={Gall_test[-1].num_edges}, "
        f"coarse_acc={results_history_test[-1]['accuracy_test']:.4f}, fine_acc={results_history_test[-1]['accuracy_fine']:.4f}"
    )

    # Save coarsening-aware model
    coarse_model_path = Path(save_path) / "models" / "coarsening_aware_model.pt"
    coarse_model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), str(coarse_model_path))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "GCN",
            "model_kwargs": {
                "nfeat": G.num_features,
                "nhid": TRAIN_CONFIG.get("nhid", 128),
                "nclass": len(np.unique(G.y.numpy())),
                "dropout": TRAIN_CONFIG.get("dropout", 0.1),
                "num_layers": TRAIN_CONFIG.get("num_layers", 2),
                "GNN_type": TRAIN_CONFIG.get("GNN_type", "GAT"),
                "use_edge_weights": TRAIN_CONFIG.get("use_edge_weights", False),
            },
            "method": METHOD,
            "max_epsilon": MAX_EPSILON,
            "results_history": results_history,
            "results_history_test": results_history_test,
        },
        str(coarse_model_path.with_suffix(".pth")),
    )
    LOGGER.info(f"Coarsening-aware model saved to {coarse_model_path}")

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

    # Save coarse model
    coarse_gnn_path = Path(save_path) / "models" / "coarse_gnn_model.pt"
    coarse_gnn_path.parent.mkdir(exist_ok=True)
    torch.save(model_coarse.state_dict(), str(coarse_gnn_path))
    torch.save(
        {
            "model_state_dict": model_coarse.state_dict(),
            "model_class": "GCN",
            "model_kwargs": {
                "nfeat": G_coarse.num_features,
                "nhid": TRAIN_CONFIG.get("nhid", 128),
                "nclass": len(np.unique(G_coarse.y.numpy())),
                "dropout": TRAIN_CONFIG.get("dropout", 0.1),
                "num_layers": TRAIN_CONFIG.get("num_layers", 2),
                "GNN_type": TRAIN_CONFIG.get("GNN_type", "GAT"),
                "use_edge_weights": TRAIN_CONFIG.get("use_edge_weights", False),
            },
            "results": coarse_results,
        },
        str(coarse_gnn_path.with_suffix(".pth")),
    )
    LOGGER.info(f"Coarse GNN model saved to {coarse_gnn_path}")

    # Generate all plots
    train_path = f"{save_path}train_results/"
    os.makedirs(train_path, exist_ok=True)
    plot_all_results(
        data_list=results_history,
        # threshold=THRESHOLD,
        save_dir=str(train_path),
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
    plot_training_loss_components(
        data_list=results_history,
        save_dir=str(train_path),
        max_epsilon=MAX_EPSILON,
        name_prefix="",
    )
    test_path = f"{save_path}test_results/"
    os.makedirs(test_path, exist_ok=True)
    plot_all_results(
        data_list=results_history_test,
        # threshold=THRESHOLD,
        save_dir=str(test_path),
        # epochs_per_level=ep_per_lev,
        max_epsilon=5 * MAX_EPSILON,
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

    plot_diagnostic_trends(results_history=results_history, save_dir=str(train_path))

    model.eval()
    with torch.no_grad():
        final_embeddings = model.get_embeddings(
            G.x,
            G.edge_index,
            G.edge_weight if hasattr(G, "edge_weight") else None,
        )
        # N = G.num_nodes
        # final_embeddings = create_subspace(alert_test, normal_test, N, device)
    generate_diagnostic_plots_for_final_state(
        embeddings=final_embeddings,
        alert_patterns=alert_train,
        normal_patterns=normal_train,
        save_dir=str(train_path),
    )

    plot_detection_rate_vs_threshold(
        alert_patterns=alert_test,
        normal_patterns=normal_test,
        save_dir=str(train_path),
    )

    LOGGER.info(f"Plots saved to {save_path}")


if __name__ == "__main__":
    run_experiment()
