"""
Quick Start Guide: Using Saved Models
======================================

This guide demonstrates how to use the saved models for testing and inference.
"""

import torch
import sys
from pathlib import Path

# Setup paths (adjust if needed)
sys.path.append(str(Path.cwd() / "src"))

from src.GangPrediction.GNN_model import GCN
from src.GangPrediction.experiment_utils import load_and_preprocess_data


# ============================================================================
# EXAMPLE 1: Load a Saved Model
# ============================================================================
def example_load_model():
    """Example: Load a saved model from checkpoint."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path to your saved model checkpoint
    checkpoint_path = (
        Path("results") / "20260326_012345" / "models" / "baseline_model.pth"
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model configuration
    model_kwargs = checkpoint["model_kwargs"]

    # Recreate model
    model = GCN(**model_kwargs).to(device)

    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set to evaluation mode
    model.eval()

    print("Model loaded successfully!")
    print(f"Model configuration: {model_kwargs}")

    return model


# ============================================================================
# EXAMPLE 2: Run Inference on Graph Data
# ============================================================================
def example_inference():
    """Example: Run inference on graph data."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint_path = (
        Path("results") / "TIMESTAMP_HERE" / "models" / "baseline_model.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = GCN(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load dataset
    from src.utils.config_parser import load_main_config

    config = load_main_config(Path("config.yaml"))
    experiment = config["experiment"]

    G, _, _, _, _ = load_and_preprocess_data(
        data_dir=Path("experiments") / experiment / "config",
        patterns_dir=Path("experiments") / experiment,
        train_ratio=0.5,
        to_undirected=True,
        device=device,
    )

    # Run inference
    with torch.no_grad():
        # Get predictions (logits)
        logits = model(
            G.x,
            G.edge_index,
            G.edge_weight if hasattr(G, "edge_weight") else None,
        )

        # Get predicted classes
        predicted_classes = logits.argmax(dim=1)

        # Get probabilities
        probabilities = torch.softmax(logits, dim=1)

        # Get embeddings
        embeddings = model.get_embeddings(
            G.x,
            G.edge_index,
            G.edge_weight if hasattr(G, "edge_weight") else None,
        )

    print(f"Predictions shape: {logits.shape}")
    print(f"Predicted classes: {predicted_classes.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Embeddings shape: {embeddings.shape}")

    # Example: Get predictions for first 10 nodes
    print(f"\nFirst 10 node predictions:")
    print(f"  Class predictions: {predicted_classes[:10]}")
    print(f"  Confidence: {probabilities[range(10), predicted_classes[:10]]}")

    return logits, predicted_classes, embeddings


# ============================================================================
# EXAMPLE 3: Get Model Metadata
# ============================================================================
def example_metadata():
    """Example: Access saved model metadata and results."""

    checkpoint_path = (
        Path("results") / "TIMESTAMP_HERE" / "models" / "baseline_model.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print("Model Metadata:")
    print(f"  Model class: {checkpoint['model_class']}")
    print(f"  Model configuration: {checkpoint['model_kwargs']}")

    if "results" in checkpoint:
        print(f"\nTraining Results:")
        for key, value in checkpoint["results"].items():
            print(f"  {key}: {value}")

    return checkpoint


# ============================================================================
# EXAMPLE 4: Compare Multiple Models
# ============================================================================
def example_compare_models():
    """Example: Compare predictions from different models."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path("results") / "TIMESTAMP_HERE" / "models"

    models_to_test = {
        "baseline": results_dir / "baseline_model.pth",
        "coarsening_aware": results_dir / "coarsening_aware_model.pth",
        "coarse_gnn": results_dir / "coarse_gnn_model.pth",
    }

    # Load all models
    models = {}
    for name, path in models_to_test.items():
        if path.exists():
            checkpoint = torch.load(path, map_location=device)
            model = GCN(**checkpoint["model_kwargs"]).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models[name] = model
            print(f"Loaded {name} model")

    print(f"\nSuccessfully loaded {len(models)} models")

    return models


# ============================================================================
# EXAMPLE 5: Save Model Predictions
# ============================================================================
def example_save_predictions():
    """Example: Save model predictions to file for later analysis."""

    import json
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and run inference (as in example_inference)
    checkpoint_path = (
        Path("results") / "TIMESTAMP_HERE" / "models" / "baseline_model.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = GCN(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Prepare data directory
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)

    print(f"Predictions will be saved to: {output_dir}")

    # You can save:
    # - Predicted classes
    # - Probabilities for each class
    # - Node embeddings
    # - Evaluation metrics

    return output_dir


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
if __name__ == "__main__":
    print(__doc__)

    print("\n" + "=" * 80)
    print("QUICK-START EXAMPLES")
    print("=" * 80)

    print("\n1. To load a model:")
    print("   model = example_load_model()")

    print("\n2. To run inference:")
    print("   logits, predictions, embeddings = example_inference()")

    print("\n3. To access model metadata:")
    print("   metadata = example_metadata()")

    print("\n4. To compare multiple models:")
    print("   models = example_compare_models()")

    print("\n5. To save predictions:")
    print("   output_dir = example_save_predictions()")

    print("\n" + "=" * 80)
    print("\nNote: Replace 'TIMESTAMP_HERE' with the actual result directory timestamp")
    print("      (e.g., '20260326_012345')")
    print("\nRun test_models.py for automated testing:")
    print("   python test_models.py")
