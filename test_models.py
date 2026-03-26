"""
Test script to load saved models and run inference on test data.
This script demonstrates how to load and use the trained models for testing.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import pickle
import json

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

from src.GangPrediction.GNN_model import GCN, evaluate_model
from src.GangPrediction.experiment_utils import load_and_preprocess_data
from src.utils.config_parser import load_main_config


# ============================================================================
# Configuration
# ============================================================================
CONFIG_PATH = project_root / "config.yaml"
CONFIG = load_main_config(CONFIG_PATH)

EXPERIMENT = CONFIG["experiment"]
PATTERN_SPLIT_CONFIG = CONFIG["pattern_split_config"]
ALERT_THRESHOLDS = CONFIG["alert_thresholds"]
NORMAL_THRESHOLDS = CONFIG["normal_thresholds"]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================================
# Helper Functions
# ============================================================================
def load_model_from_checkpoint(checkpoint_path, model_class=GCN):
    """
    Load a model from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pth checkpoint file
        model_class: The model class to instantiate

    Returns:
        model: The loaded model on the appropriate device
        checkpoint_data: Dictionary containing model metadata and results
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model kwargs
    if "model_kwargs" in checkpoint:
        model_kwargs = checkpoint["model_kwargs"]
        model = model_class(**model_kwargs).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        checkpoint_data = {
            "model_class": checkpoint.get("model_class", "GCN"),
            "model_kwargs": model_kwargs,
            "results": checkpoint.get("results", {}),
            "method": checkpoint.get("method", None),
            "max_epsilon": checkpoint.get("max_epsilon", None),
        }
    else:
        raise ValueError("Invalid checkpoint format. Expected 'model_kwargs' key.")

    return model, checkpoint_data


def test_baseline_model(model, graph_data, model_info):
    """
    Test the baseline model on the full-resolution graph.

    Args:
        model: The trained model
        graph_data: The full-resolution graph data
        model_info: Dictionary with model metadata
    """
    print("\n" + "=" * 80)
    print("TESTING BASELINE MODEL")
    print("=" * 80)

    model.eval()
    with torch.no_grad():
        results = evaluate_model(model, graph_data)

    print(f"\nModel Configuration:")
    print(f"  - Architecture: {model_info['model_class']}")
    print(f"  - Input features: {model_info['model_kwargs']['nfeat']}")
    print(f"  - Hidden dimension: {model_info['model_kwargs']['nhid']}")
    print(f"  - Output classes: {model_info['model_kwargs']['nclass']}")
    print(f"  - Number of layers: {model_info['model_kwargs']['num_layers']}")
    print(f"  - Dropout: {model_info['model_kwargs']['dropout']}")

    print(f"\nTest Results:")
    print(f"  - Test Accuracy: {results.get('accuracy_test', 'N/A'):.4f}")
    print(f"  - Precision: {results.get('precision_test', 'N/A'):.4f}")
    print(f"  - Recall: {results.get('recall_test', 'N/A'):.4f}")

    # Print stored results if available
    if model_info["results"]:
        print(f"\nStored Results from Training:")
        for key, value in model_info["results"].items():
            if isinstance(value, (int, float)):
                print(
                    f"  - {key}: {value:.4f}"
                    if isinstance(value, float)
                    else f"  - {key}: {value}"
                )

    return results


def test_coarsening_aware_model(model, graph_data, model_info):
    """
    Test the coarsening-aware model.

    Args:
        model: The trained model
        graph_data: The graph data
        model_info: Dictionary with model metadata
    """
    print("\n" + "=" * 80)
    print("TESTING COARSENING-AWARE MODEL")
    print("=" * 80)

    print(f"\nModel Configuration:")
    print(f"  - Method: {model_info.get('method', 'N/A')}")
    print(f"  - Max Epsilon: {model_info.get('max_epsilon', 'N/A'):.4f}")
    print(f"  - Architecture: {model_info['model_class']}")
    print(f"  - Number of layers: {model_info['model_kwargs']['num_layers']}")

    model.eval()
    with torch.no_grad():
        results = evaluate_model(model, graph_data)

    print(f"\nTest Results:")
    print(f"  - Test Accuracy: {results.get('accuracy_test', 'N/A'):.4f}")
    print(f"  - Precision: {results.get('precision_test', 'N/A'):.4f}")
    print(f"  - Recall: {results.get('recall_test', 'N/A'):.4f}")

    return results


def run_inference(model, graph_data):
    """
    Run inference on the model and return predictions and embeddings.

    Args:
        model: The trained model
        graph_data: The graph data

    Returns:
        predictions: Model predictions (logits)
        predicted_classes: Predicted class labels
        embeddings: Node embeddings
    """
    model.eval()
    with torch.no_grad():
        # Get raw predictions (logits)
        predictions = model(
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_weight if hasattr(graph_data, "edge_weight") else None,
        )

        # Get predicted classes
        predicted_classes = predictions.argmax(dim=1)

        # Get embeddings
        embeddings = model.get_embeddings(
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_weight if hasattr(graph_data, "edge_weight") else None,
        )

    return predictions, predicted_classes, embeddings


def load_dataset(experiment_name):
    """
    Load the dataset for testing.

    Args:
        experiment_name: Name of the experiment

    Returns:
        G: Full graph
        alert_train, normal_train, alert_test, normal_test: Pattern data
    """
    print("=" * 80)
    print("Loading Dataset")
    print("=" * 80)

    experiment_root = project_root / "experiments" / experiment_name

    G, alert_train, normal_train, alert_test, normal_test = load_and_preprocess_data(
        data_dir=experiment_root / "config",
        patterns_dir=experiment_root,
        train_ratio=PATTERN_SPLIT_CONFIG.get("train_ratio", 0.5),
        to_undirected="variation" in CONFIG.get("method", "").lower(),
        device=device,
    )

    print(f"\nDataset Information:")
    print(f"  - Number of nodes: {G.num_nodes}")
    print(f"  - Number of edges: {G.num_edges}")
    print(f"  - Number of features: {G.num_features}")
    print(f"  - Number of classes: {len(np.unique(G.y.cpu().numpy()))}")
    if hasattr(G, "train_idx"):
        print(f"  - Training nodes: {len(G.train_idx)}")

    return G, alert_train, normal_train, alert_test, normal_test


def main():
    """Main test function."""
    print("\n" + "=" * 80)
    print("MODEL TESTING FRAMEWORK")
    print("=" * 80)

    # Load dataset
    G, alert_train, normal_train, alert_test, normal_test = load_dataset(EXPERIMENT)

    # Find the latest results directory with saved models
    results_dir = project_root / "results"
    if not results_dir.exists():
        print(f"\nERROR: Results directory not found at {results_dir}")
        return

    # Get the most recent results directory
    result_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    if not result_dirs:
        print(f"\nERROR: No result directories found in {results_dir}")
        return

    latest_result_dir = result_dirs[-1]
    models_dir = latest_result_dir / "models"

    if not models_dir.exists():
        print(f"\nERROR: Models directory not found at {models_dir}")
        print("Please run main.py first to train and save models.")
        return

    print(f"\nUsing results directory: {latest_result_dir}")
    print(f"Models directory: {models_dir}")

    # Load and test baseline model
    baseline_model_path = models_dir / "baseline_model.pth"
    if baseline_model_path.exists():
        print(f"\nLoading baseline model from {baseline_model_path}")
        baseline_model, baseline_info = load_model_from_checkpoint(baseline_model_path)
        baseline_results = test_baseline_model(baseline_model, G, baseline_info)

        # Run inference
        print("\nRunning inference with baseline model...")
        preds, pred_classes, embeddings = run_inference(baseline_model, G)
        print(f"  - Predictions shape: {preds.shape}")
        print(f"  - Predicted classes shape: {pred_classes.shape}")
        print(f"  - Embeddings shape: {embeddings.shape}")
        print(f"  - Number of predicted class 1: {(pred_classes == 1).sum().item()}")
    else:
        print(f"\nWARNING: Baseline model not found at {baseline_model_path}")

    # Load and test coarsening-aware model
    coarse_model_path = models_dir / "coarsening_aware_model.pth"
    if coarse_model_path.exists():
        print(f"\nLoading coarsening-aware model from {coarse_model_path}")
        coarse_model, coarse_info = load_model_from_checkpoint(coarse_model_path)
        coarse_results = test_coarsening_aware_model(coarse_model, G, coarse_info)

        # Run inference
        print("\nRunning inference with coarsening-aware model...")
        preds, pred_classes, embeddings = run_inference(coarse_model, G)
        print(f"  - Predictions shape: {preds.shape}")
        print(f"  - Predicted classes shape: {pred_classes.shape}")
        print(f"  - Embeddings shape: {embeddings.shape}")
        print(f"  - Number of predicted class 1: {(pred_classes == 1).sum().item()}")
    else:
        print(f"\nWARNING: Coarsening-aware model not found at {coarse_model_path}")

    # Load and test coarse GNN model
    coarse_gnn_path = models_dir / "coarse_gnn_model.pth"
    if coarse_gnn_path.exists():
        print(f"\nLoading coarse GNN model from {coarse_gnn_path}")
        coarse_gnn_model, coarse_gnn_info = load_model_from_checkpoint(coarse_gnn_path)

        print("\n" + "=" * 80)
        print("TESTING COARSE GNN MODEL")
        print("=" * 80)
        print(f"\nModel Configuration:")
        print(f"  - Architecture: {coarse_gnn_info['model_class']}")
        print(f"  - Input features: {coarse_gnn_info['model_kwargs']['nfeat']}")
        print(f"  - Number of layers: {coarse_gnn_info['model_kwargs']['num_layers']}")

        if coarse_gnn_info["results"]:
            print(f"\nStored Results from Training:")
            for key, value in coarse_gnn_info["results"].items():
                if isinstance(value, (int, float)):
                    print(
                        f"  - {key}: {value:.4f}"
                        if isinstance(value, float)
                        else f"  - {key}: {value}"
                    )
    else:
        print(f"\nWARNING: Coarse GNN model not found at {coarse_gnn_path}")

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print(f"\nModels successfully loaded and tested from: {models_dir}")
    print("\nYou can now use these models for inference on new data.")
    print(
        "Refer to the run_inference() function to see how to get predictions and embeddings."
    )


if __name__ == "__main__":
    main()
