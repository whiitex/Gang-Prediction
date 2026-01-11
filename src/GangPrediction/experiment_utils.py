"""Utility functions for pattern loading, evaluation, and experiment setup."""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric.data
import torch_geometric.transforms
from sklearn.preprocessing import MinMaxScaler

from src.GangPrediction.utils.utils import graph_params


def load_patterns_from_file(file_path: str, node_to_index: dict, min_nodes: int = 2):
    """
    Load patterns from CSV or parquet file and group accounts by pattern ID.

    Args:
        file_path: Path to CSV or parquet file with columns [modelID, accountID, type]
        node_to_index: Dict mapping account IDs to node indices
        min_nodes: Minimum number of mapped nodes required for a valid pattern

    Returns:
        patterns: dict mapping pattern_id -> list of node indices
        pattern_types: dict mapping pattern_id -> pattern type
    """
    file_path = Path(file_path)

    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    patterns = defaultdict(list)
    pattern_types = {}

    for _, row in df.iterrows():
        model_id = row["modelID"]
        account_id = row["accountID"]
        pattern_type = row["type"]

        if account_id in node_to_index:
            patterns[model_id].append(node_to_index[account_id])
            pattern_types[model_id] = pattern_type

    # Filter patterns with insufficient mapped nodes
    valid_patterns = {k: v for k, v in patterns.items() if len(v) >= min_nodes}
    valid_pattern_types = {k: pattern_types[k] for k in valid_patterns.keys()}

    return valid_patterns, valid_pattern_types


def get_node_to_supernode_mapping(Call):
    """
    Get mapping from original nodes to super nodes after coarsening.

    Args:
        Call: List containing cumulative coarsening matrix

    Returns:
        mapping: Tensor where mapping[i] = super node index for original node i
    """
    if len(Call) == 0:
        return None

    C_total = Call[-1]
    C_dense = C_total.to_dense()
    return torch.argmax(C_dense, dim=0)


def evaluate_pattern_detection(
    labels: torch.Tensor,
    patterns: dict,
    node_to_supernode: torch.Tensor = None,
    target_label: int = 1,
    majority_threshold: float = 0.5,
    coarsening_threshold: float = 0.5,
    use_probs: bool = False,
    probs: torch.Tensor = None,
    prob_threshold: float = 0.3,
):
    """
    Unified pattern detection evaluation for both alert (suspicious) and normal patterns.

    A pattern is correctly detected if:
    1. More than majority_threshold of accounts have the target label
    2. More than coarsening_threshold of target accounts are coarsened together

    Args:
        labels: Tensor of labels (predictions or ground truth)
        patterns: Dict mapping pattern_id -> list of node indices
        node_to_supernode: Optional mapping from nodes to super nodes
        target_label: Label to detect (1 for suspicious/alert, 0 for normal)
        majority_threshold: Fraction of nodes that must have target label
        coarsening_threshold: Fraction of target nodes that must be in same super node
        use_probs: If True, use probability threshold instead of labels
        probs: Probability tensor (required if use_probs=True)
        prob_threshold: Probability threshold for detection (if use_probs=True)

    Returns:
        detection_rate: Fraction of patterns correctly detected
        detected_patterns: List of detected pattern IDs
        pattern_details: Dict with details for each pattern
    """
    detected_patterns = []
    pattern_details = {}

    for pattern_id, node_indices in patterns.items():
        node_indices_tensor = torch.tensor(node_indices, dtype=torch.long)

        # Determine which nodes match target
        if use_probs and probs is not None:
            pattern_probs = probs[node_indices_tensor]
            if target_label == 1:
                target_mask = pattern_probs > prob_threshold
            else:
                target_mask = pattern_probs <= (1 - prob_threshold)
            n_target = target_mask.sum().item()
        else:
            pattern_labels = labels[node_indices_tensor]
            n_target = (pattern_labels == target_label).sum().item()
            target_mask = pattern_labels == target_label

        n_total = len(node_indices)
        target_ratio = n_target / n_total if n_total > 0 else 0

        # Condition 1: More than majority_threshold have target label
        condition1_met = target_ratio > majority_threshold

        # Condition 2: Most target nodes coarsened together
        condition2_met = True
        coarsening_ratio = 1.0

        if node_to_supernode is not None and n_target > 0:
            target_indices = node_indices_tensor[target_mask]
            super_nodes = node_to_supernode[target_indices]
            _, counts = torch.unique(super_nodes, return_counts=True)
            max_count = counts.max().item()
            coarsening_ratio = max_count / n_target
            condition2_met = coarsening_ratio > coarsening_threshold

        is_detected = condition1_met and condition2_met

        if is_detected:
            detected_patterns.append(pattern_id)

        pattern_details[pattern_id] = {
            "n_nodes": n_total,
            "n_target": n_target,
            "target_ratio": target_ratio,
            "coarsening_ratio": coarsening_ratio,
            "condition1_met": condition1_met,
            "condition2_met": condition2_met,
            "detected": is_detected,
        }

    detection_rate = len(detected_patterns) / len(patterns) if len(patterns) > 0 else 0
    return detection_rate, detected_patterns, pattern_details


def load_amlgentex_data(experiment_root: Path, config_dir: Path):
    """
    Load and preprocess AMLGentex dataset.

    Args:
        experiment_root: Path to experiment directory
        config_dir: Path to config directory

    Returns:
        G: PyTorch Geometric Data object
        node_to_index: Dict mapping account IDs to node indices
    """
    from src.feature_engineering import DataPreprocessor
    from src.utils.config import load_preprocessing_config

    preproc_config = load_preprocessing_config(str(config_dir / "preprocessing.yaml"))

    print("Preprocessing configuration:")
    print(f"  Raw data: {preproc_config['raw_data_file']}")
    print(f"  Output dir: {preproc_config['preprocessed_data_dir']}")
    print("\nPreprocessing transactions...")
    print("Generating temporal features with rolling windows...\n")

    preprocessor = DataPreprocessor(preproc_config)
    datasets = preprocessor(preproc_config["raw_data_file"])

    nodes_df = datasets["trainset_nodes"]
    edges_df = datasets["trainset_edges"]

    # Create node ID to index mapping
    node_to_index = {
        account_id: idx for idx, account_id in enumerate(nodes_df["account"])
    }

    # Get train/val/test indices
    train_idx = torch.tensor(
        [node_to_index[acc] for acc in nodes_df[nodes_df["train_mask"]]["account"]],
        dtype=torch.long,
    )
    val_idx = torch.tensor(
        [node_to_index[acc] for acc in nodes_df[nodes_df["val_mask"]]["account"]],
        dtype=torch.long,
    )
    test_idx = torch.tensor(
        [node_to_index[acc] for acc in nodes_df[nodes_df["test_mask"]]["account"]],
        dtype=torch.long,
    )

    # Prepare features and labels
    nodes_df = nodes_df.drop(columns=["bank"])
    X = (
        nodes_df.drop(
            columns=["account", "train_mask", "val_mask", "test_mask", "is_sar"]
        )
        .to_numpy()
        .astype(np.float32)
    )
    y = nodes_df["is_sar"].to_numpy().astype(np.int64)

    # Map edges
    edges_df["src_idx"] = edges_df["src"].map(node_to_index)
    edges_df["dst_idx"] = edges_df["dst"].map(node_to_index)
    edges_df = edges_df.dropna(subset=["src_idx", "dst_idx"])
    edges = edges_df[["src_idx", "dst_idx"]].to_numpy().astype(np.int64)
    edges_index = torch.tensor(edges.T, dtype=torch.long)

    # Normalize features
    scaler = MinMaxScaler().fit(X)
    X_normalized = torch.tensor(scaler.transform(X), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)

    # Create PyG Data object
    G = torch_geometric.data.Data(x=X_normalized, edge_index=edges_index, y=y)
    G = torch_geometric.transforms.ToUndirected()(G)
    G.edge_weight = torch.ones(G.edge_index.size(1))
    G.train_idx = train_idx
    G.val_idx = val_idx
    G.test_idx = test_idx
    G.W, G.L, G.dw = graph_params(G)

    return G, node_to_index


def load_all_patterns(experiment_root: Path, node_to_index: dict):
    """
    Load alert and normal patterns from experiment directory.

    Args:
        experiment_root: Path to experiment directory
        node_to_index: Dict mapping account IDs to node indices

    Returns:
        alert_patterns: Dict of alert patterns
        alert_types: Dict of alert pattern types
        normal_patterns: Dict of normal patterns
        normal_types: Dict of normal pattern types
    """
    alert_patterns, alert_types = {}, {}
    normal_patterns, normal_types = {}, {}

    # Try parquet first, then CSV for alerts
    alert_parquet = experiment_root / "spatial" / "alert_models.parquet"
    alert_csv = experiment_root / "spatial" / "alert_models.csv"

    if alert_parquet.exists():
        alert_patterns, alert_types = load_patterns_from_file(
            str(alert_parquet), node_to_index
        )
        print(f"\nLoaded {len(alert_patterns)} alert patterns from parquet")
    elif alert_csv.exists():
        alert_patterns, alert_types = load_patterns_from_file(
            str(alert_csv), node_to_index
        )
        print(f"\nLoaded {len(alert_patterns)} alert patterns from CSV")
    else:
        print("\nNo alert patterns file found")

    if alert_types:
        print(f"Alert pattern types: {set(alert_types.values())}")

    # Load normal patterns
    normal_csv = experiment_root / "spatial" / "normal_models.csv"
    if normal_csv.exists():
        normal_patterns, normal_types = load_patterns_from_file(
            str(normal_csv), node_to_index
        )
        print(f"\nLoaded {len(normal_patterns)} normal patterns")
        print(f"Normal pattern types: {set(normal_types.values())}")
    else:
        print("\nNo normal patterns file found")

    return alert_patterns, alert_types, normal_patterns, normal_types


def evaluate_at_level(
    model,
    Gall,
    Call,
    level_idx: int,
    gt_labels: torch.Tensor,
    alert_patterns: dict,
    normal_patterns: dict,
    alert_thresholds: tuple = (0.75, 0.75),
    normal_thresholds: tuple = (0.5, 0.5),
    prob_threshold: float = 0.3,
):
    """
    Evaluate pattern detection at a specific coarsening level.

    Args:
        model: Trained GNN model
        Gall: List of graphs at each coarsening level
        Call: List of cumulative coarsening matrices
        level_idx: Index of current level
        gt_labels: Ground truth labels
        alert_patterns: Dict of alert patterns
        normal_patterns: Dict of normal patterns
        alert_thresholds: (majority_threshold, coarsening_threshold) for alerts
        normal_thresholds: (majority_threshold, coarsening_threshold) for normals
        prob_threshold: Probability threshold for model-based detection

    Returns:
        Dict with detection rates and details
    """
    import torch.nn.functional as F

    Gc = Gall[level_idx]
    model.eval()

    with torch.no_grad():
        S_mp = Gc.S_mp if hasattr(Gc, "S_mp") else Gc.W
        logits = model(Gc.x, S_mp)

        # Project to fine graph if coarsened
        if level_idx > 0:
            C_cur = Call[level_idx - 1]
            C_dense = C_cur.to_dense().t()
            row_sums = torch.clamp(C_dense.sum(dim=1, keepdim=True), min=1e-8)
            C_plus_normalized = C_dense / row_sums
            fine_logits = C_plus_normalized @ logits
        else:
            fine_logits = logits

        probs = F.softmax(fine_logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        suspicious_probs = probs[:, 1]

    # Get coarsening mapping
    node_to_supernode = (
        get_node_to_supernode_mapping([Call[level_idx - 1]]) if level_idx > 0 else None
    )

    results = {
        "predictions": predictions,
        "suspicious_probs": suspicious_probs,
        "n_pred_suspicious": (predictions == 1).sum().item(),
        "n_prob_suspicious": (suspicious_probs > prob_threshold).sum().item(),
        "mean_prob": suspicious_probs.mean().item(),
        "max_prob": suspicious_probs.max().item(),
    }

    # Alert pattern detection (model-based)
    if alert_patterns:
        rate, detected, details = evaluate_pattern_detection(
            predictions,
            alert_patterns,
            node_to_supernode,
            target_label=1,
            majority_threshold=alert_thresholds[0],
            coarsening_threshold=alert_thresholds[1],
            use_probs=True,
            probs=suspicious_probs,
            prob_threshold=prob_threshold,
        )
        results["alert_model_rate"] = rate
        results["alert_model_detected"] = detected
        results["alert_model_details"] = details

        # Alert pattern detection (GT-based)
        rate_gt, detected_gt, details_gt = evaluate_pattern_detection(
            gt_labels,
            alert_patterns,
            node_to_supernode,
            target_label=1,
            majority_threshold=alert_thresholds[0],
            coarsening_threshold=alert_thresholds[1],
        )
        results["alert_gt_rate"] = rate_gt
        results["alert_gt_detected"] = detected_gt
        results["alert_gt_details"] = details_gt

    # Normal pattern detection (GT-based only)
    if normal_patterns:
        rate_gt, detected_gt, details_gt = evaluate_pattern_detection(
            gt_labels,
            normal_patterns,
            node_to_supernode,
            target_label=0,
            majority_threshold=normal_thresholds[0],
            coarsening_threshold=normal_thresholds[1],
        )
        results["normal_gt_rate"] = rate_gt
        results["normal_gt_detected"] = detected_gt
        results["normal_gt_details"] = details_gt

    return results
