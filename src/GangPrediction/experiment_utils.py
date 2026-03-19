"""Utility functions for pattern loading, evaluation, and experiment setup."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch_geometric.data
import torch_geometric.transforms
from sklearn.preprocessing import MinMaxScaler

from src.GangPrediction.pattern_models import Pattern, create_pattern
from src.GangPrediction.utils.utils import graph_params


def _capture_pattern_lineage(
    patterns: List[Pattern],
    node_to_supernode: Optional[torch.Tensor],
    pseudo_labels: Optional[torch.Tensor],
) -> None:
    """Capture per-level super-node and pseudo-label lineage for pattern nodes."""

    for pattern in patterns:
        pattern.capture_level(
            node_to_supernode=node_to_supernode,
            pseudo_labels=pseudo_labels,
        )


def load_patterns_from_file(
    file_path: str, node_to_index: dict, label: str = "unknown"
) -> List[Pattern]:
    """
    Load patterns from CSV or parquet file and group accounts by pattern ID.

    Args:
        file_path: Path to CSV or parquet file with columns [modelID, accountID, type]
        node_to_index: Dict mapping account IDs to node indices
        min_nodes: Minimum number of mapped nodes required for a valid pattern

    Returns:
        patterns: dict mapping pattern_id -> list of node indices OR List[Pattern]
        pattern_types: dict mapping pattern_id -> pattern type
    """
    file_path = Path(file_path)
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    tmp = df.loc[
        df["accountID"].isin(node_to_index), ["modelID", "accountID", "type"]
    ].assign(account_idx=lambda d: d["accountID"].map(node_to_index))

    pattern_dicts = tmp.groupby("modelID")["account_idx"].agg(list).to_dict()
    pattern_types = tmp.groupby("modelID")["type"].first().to_dict()

    patterns = []
    for pattern_id, node_indices in pattern_dicts.items():
        pattern = create_pattern(
            pattern_id=pattern_id,
            nodes=node_indices,
            pattern_type=pattern_types.get(pattern_id, "unknown"),
            label=label,
        )
        patterns.append(pattern)

    return patterns


def get_node_to_supernode_mapping(C):
    """
    Get mapping from original nodes to super nodes after coarsening.

    Args:
        Call: List containing cumulative coarsening matrix

    Returns:
        mapping: Tensor where mapping[i] = super node index for original node i
    """
    C_dense = C.to_dense()
    return torch.argmax(C_dense, dim=0)


def load_amlgentex_data(config_dir: Path):
    """
    Load and preprocess AMLGentex dataset.

    Args:
        config_dir: Path to config directory
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
    # train_idx = torch.tensor(
    #     [node_to_index[acc] for acc in nodes_df[nodes_df["train_mask"]]["account"]],
    #     dtype=torch.long,
    # )
    # val_idx = torch.tensor(
    #     [node_to_index[acc] for acc in nodes_df[nodes_df["val_mask"]]["account"]],
    #     dtype=torch.long,
    # )
    # test_idx = torch.tensor(
    #     [node_to_index[acc] for acc in nodes_df[nodes_df["test_mask"]]["account"]],
    #     dtype=torch.long,
    # )

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
    # G = torch_geometric.transforms.ToUndirected()(G)
    G.edge_weight = torch.ones(G.edge_index.size(1))
    # G.train_idx = train_idx
    # G.val_idx = val_idx
    # G.test_idx = test_idx
    G.W, G.L, G.dw = graph_params(G)

    return G, node_to_index


def split_patterns(
    patterns: dict,
    train_ratio: float = 0.7,
    seed: int = 42,
):
    """
    Split patterns into train and test sets.

    Args:
        patterns: Dict mapping pattern_id -> list of node indices
        pattern_types: Dict mapping pattern_id -> pattern type
        train_ratio: Fraction of patterns to use for training
        seed: Random seed for reproducibility

    Returns:
        train_patterns: Dict of training patterns
        train_types: Dict of training pattern types
        test_patterns: Dict of test patterns
        test_types: Dict of test pattern types
    """
    import random

    random.seed(seed)

    random.shuffle(patterns)

    n_train = int(len(patterns) * train_ratio)
    train_patterns = patterns[:n_train]
    test_patterns = patterns[n_train:]

    return train_patterns, test_patterns


def get_pattern_node_indices(patterns: dict) -> torch.Tensor:
    """
    Get all unique node indices from patterns.

    Args:
        patterns: Dict mapping pattern_id -> list of node indices

    Returns:
        Tensor of unique node indices from all patterns
    """
    all_nodes = set()

    for pattern in patterns:
        all_nodes.update(pattern.node_indices)

    return torch.tensor(list(all_nodes), dtype=torch.long)


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
    alert_patterns = {}
    normal_patterns = {}

    # Try parquet first, then CSV for alerts
    alert_csv = experiment_root / "spatial" / "alert_models.csv"

    if alert_csv.exists():
        alert_patterns = load_patterns_from_file(
            str(alert_csv), node_to_index, label="alert"
        )
        print(f"\nLoaded {len(alert_patterns)} alert patterns from CSV")
    else:
        print("\nNo alert patterns file found")

    # Load normal patterns
    normal_csv = experiment_root / "spatial" / "normal_models.csv"
    if normal_csv.exists():
        normal_patterns = load_patterns_from_file(
            str(normal_csv), node_to_index, label="normal"
        )
        print(f"\nLoaded {len(normal_patterns)} normal patterns")
    else:
        print("\nNo normal patterns file found")

    return alert_patterns, normal_patterns
