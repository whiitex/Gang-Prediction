"""Experiment driver for coarsening-aware training on AMLGentex."""

import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch_geometric.data
import torch_geometric.transforms
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import uniform_filter1d

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from src.GangPrediction.experiment_utils import get_node_to_supernode_mapping
from src.GangPrediction.graph_utils import *
from src.GangPrediction.utils.utils import *
from src.GangPrediction.coarsening_utils import *
from src.GangPrediction.GNN_model import evaluate_model
from src.GangPrediction.train_GNN_coarsening import (
    train_GNN_coarsening_aware_loss,
    train_GNN,
)

import warnings

warnings.filterwarnings("ignore")


def smooth_curve(y, window_size=5):
    """Apply moving average smoothing to a curve."""
    if len(y) < window_size:
        return y
    return uniform_filter1d(y, size=window_size, mode="nearest")


def load_patterns_from_csv(csv_file, node_to_index):
    """
    Load patterns from CSV file and group accounts by pattern ID.

    Returns:
        patterns: dict mapping pattern_id -> list of node indices
        pattern_types: dict mapping pattern_id -> pattern type (fan_out, fan_in, etc.)
    """
    df = pd.read_csv(csv_file)

    patterns = defaultdict(list)
    pattern_types = {}

    for _, row in df.iterrows():
        model_id = row["modelID"]
        account_id = row["accountID"]
        pattern_type = row["type"]

        # Map account ID to node index
        if account_id in node_to_index:
            patterns[model_id].append(node_to_index[account_id])
            pattern_types[model_id] = pattern_type

    # Filter out patterns with less than 2 mapped nodes
    valid_patterns = {k: v for k, v in patterns.items() if len(v) >= 2}
    # valid_pattern_types = {k: pattern_types[k] for k in valid_patterns.keys()}

    return valid_patterns, pattern_types


def load_alert_patterns(alert_file, node_to_index):
    """
    Load alert patterns and group accounts by pattern ID.

    Returns:
        patterns: dict mapping pattern_id -> list of node indices
        pattern_types: dict mapping pattern_id -> pattern type (fan_out, fan_in, etc.)
    """
    alert_df = pd.read_parquet(alert_file)

    patterns = defaultdict(list)
    pattern_types = {}

    for _, row in alert_df.iterrows():
        model_id = row["modelID"]
        account_id = row["accountID"]
        pattern_type = row["type"]

        # Map account ID to node index
        if account_id in node_to_index:
            patterns[model_id].append(node_to_index[account_id])
            pattern_types[model_id] = pattern_type

    # Filter out patterns with less than 2 mapped nodes
    valid_patterns = {k: v for k, v in patterns.items() if len(v) >= 2}
    valid_pattern_types = {k: pattern_types[k] for k in valid_patterns.keys()}

    return valid_patterns, valid_pattern_types


def evaluate_normal_pattern_detection_gt(
    gt_labels,
    patterns,
    node_to_supernode=None,
    majority_threshold=0.5,
    coarsening_threshold=0.5,
):
    """
    Evaluate how many NORMAL patterns are correctly detected using ground truth labels
    and coarsening information only.

    A normal pattern is correctly detected if:
    1. More than majority_threshold of accounts in the pattern are actually normal (GT label=0)
    2. More than coarsening_threshold of those normal accounts are coarsened to the same super node

    This measures the quality of coarsening w.r.t. preserving normal pattern structure.

    Args:
        gt_labels: tensor of ground truth labels (0 or 1) for each node
        patterns: dict mapping pattern_id -> list of node indices
        node_to_supernode: mapping from nodes to super nodes (None if no coarsening)
        majority_threshold: fraction of nodes that must be truly normal
        coarsening_threshold: fraction of normal nodes that must be in same super node

    Returns:
        detection_rate: fraction of patterns where normal accounts are coarsened together
        detected_patterns: list of pattern IDs that were detected
        pattern_details: dict with details for each pattern
    """
    detected_patterns = []
    pattern_details = {}

    for pattern_id, node_indices in patterns.items():
        node_indices_tensor = torch.tensor(node_indices, dtype=torch.long)
        pattern_gt = gt_labels[node_indices_tensor]

        # Use ground truth to identify truly NORMAL nodes (label=0)
        n_normal = (pattern_gt == 0).sum().item()
        normal_mask = pattern_gt == 0

        n_total = len(node_indices)
        normal_ratio = n_normal / n_total

        # Condition 1: More than majority_threshold are truly normal (GT)
        condition1_met = normal_ratio > majority_threshold

        # Condition 2: Most normal nodes coarsened together
        condition2_met = True
        coarsening_ratio = 1.0

        if node_to_supernode is not None and n_normal > 0:
            # Get super node assignments for normal nodes
            normal_indices = node_indices_tensor[normal_mask]
            super_nodes = node_to_supernode[normal_indices]

            # Count most common super node
            unique_super_nodes, counts = torch.unique(super_nodes, return_counts=True)
            max_count = counts.max().item()
            coarsening_ratio = max_count / n_normal

            condition2_met = coarsening_ratio > coarsening_threshold

        is_detected = condition1_met and condition2_met

        if is_detected:
            detected_patterns.append(pattern_id)

        pattern_details[pattern_id] = {
            "n_nodes": n_total,
            "n_normal_gt": n_normal,
            "normal_ratio": normal_ratio,
            "coarsening_ratio": coarsening_ratio,
            "condition1_met": condition1_met,
            "condition2_met": condition2_met,
            "detected": is_detected,
        }

    detection_rate = len(detected_patterns) / len(patterns) if len(patterns) > 0 else 0

    return detection_rate, detected_patterns, pattern_details


def evaluate_pattern_detection_gt(
    gt_labels,
    patterns,
    node_to_supernode=None,
    majority_threshold=0.5,
    coarsening_threshold=0.5,
):
    """
    Evaluate how many patterns are correctly detected using ground truth labels
    and coarsening information only.

    A pattern is correctly detected if:
    1. More than majority_threshold of accounts in the pattern are actually suspicious (GT)
    2. More than coarsening_threshold of those suspicious accounts are coarsened to the same super node

    This measures the quality of coarsening w.r.t. preserving pattern structure.

    Args:
        gt_labels: tensor of ground truth labels (0 or 1) for each node
        patterns: dict mapping pattern_id -> list of node indices
        node_to_supernode: mapping from nodes to super nodes (None if no coarsening)
        majority_threshold: fraction of nodes that must be truly suspicious
        coarsening_threshold: fraction of suspicious nodes that must be in same super node

    Returns:
        detection_rate: fraction of patterns where suspicious accounts are coarsened together
        detected_patterns: list of pattern IDs that were detected
        pattern_details: dict with details for each pattern
    """
    detected_patterns = []
    pattern_details = {}

    for pattern_id, node_indices in patterns.items():
        node_indices_tensor = torch.tensor(node_indices, dtype=torch.long)
        pattern_gt = gt_labels[node_indices_tensor]

        # Use ground truth to identify truly suspicious nodes
        n_suspicious = (pattern_gt == 1).sum().item()
        suspicious_mask = pattern_gt == 1

        n_total = len(node_indices)
        suspicious_ratio = n_suspicious / n_total

        # Condition 1: More than majority_threshold are truly suspicious (GT)
        condition1_met = suspicious_ratio > majority_threshold

        # Condition 2: Most suspicious nodes coarsened together
        condition2_met = True
        coarsening_ratio = 1.0

        if node_to_supernode is not None and n_suspicious > 0:
            # Get super node assignments for suspicious nodes
            suspicious_indices = node_indices_tensor[suspicious_mask]
            super_nodes = node_to_supernode[suspicious_indices]

            # Count most common super node
            unique_super_nodes, counts = torch.unique(super_nodes, return_counts=True)
            max_count = counts.max().item()
            coarsening_ratio = max_count / n_suspicious

            condition2_met = coarsening_ratio > coarsening_threshold

        is_detected = condition1_met and condition2_met

        if is_detected:
            detected_patterns.append(pattern_id)

        pattern_details[pattern_id] = {
            "n_nodes": n_total,
            "n_suspicious_gt": n_suspicious,
            "suspicious_ratio": suspicious_ratio,
            "coarsening_ratio": coarsening_ratio,
            "condition1_met": condition1_met,
            "condition2_met": condition2_met,
            "detected": is_detected,
        }

    detection_rate = len(detected_patterns) / len(patterns) if len(patterns) > 0 else 0

    return detection_rate, detected_patterns, pattern_details


def evaluate_pattern_detection(
    predictions,
    patterns,
    node_to_supernode=None,
    majority_threshold=0.5,
    coarsening_threshold=0.5,
    suspicious_probs=None,
    prob_threshold=0.3,
):
    """
    Evaluate how many patterns are correctly detected.

    A pattern is correctly detected if:
    1. More than majority_threshold of accounts are predicted as suspicious
       (or have suspicious probability above prob_threshold if suspicious_probs provided)
    2. More than coarsening_threshold of suspicious accounts are in the same super node

    Args:
        predictions: tensor of predicted labels (0 or 1) for each node
        patterns: dict mapping pattern_id -> list of node indices
        node_to_supernode: mapping from nodes to super nodes (None if no coarsening)
        majority_threshold: fraction of nodes that must be predicted suspicious
        coarsening_threshold: fraction of suspicious nodes that must be in same super node
        suspicious_probs: optional tensor of probabilities for being suspicious (class 1)
        prob_threshold: probability threshold to consider a node "suspicious" when using probs

    Returns:
        detection_rate: fraction of patterns correctly detected
        detected_patterns: list of pattern IDs that were detected
        pattern_details: dict with details for each pattern
    """
    detected_patterns = []
    pattern_details = {}

    for pattern_id, node_indices in patterns.items():
        node_indices_tensor = torch.tensor(node_indices, dtype=torch.long)
        pattern_predictions = predictions[node_indices_tensor]

        # Use probability-based suspicion if hard predictions don't work
        if suspicious_probs is not None:
            pattern_probs = suspicious_probs[node_indices_tensor]
            n_suspicious = (pattern_probs > prob_threshold).sum().item()
            suspicious_mask = pattern_probs > prob_threshold
        else:
            n_suspicious = (pattern_predictions == 1).sum().item()
            suspicious_mask = pattern_predictions == 1

        n_total = len(node_indices)
        suspicious_ratio = n_suspicious / n_total

        # Condition 1: More than majority_threshold predicted suspicious
        condition1_met = suspicious_ratio > majority_threshold

        # Condition 2: Most suspicious nodes coarsened together
        condition2_met = True
        coarsening_ratio = 1.0

        if node_to_supernode is not None and n_suspicious > 0:
            # Get super node assignments for suspicious nodes
            suspicious_indices = node_indices_tensor[suspicious_mask]
            super_nodes = node_to_supernode[suspicious_indices]

            # Count most common super node
            unique_super_nodes, counts = torch.unique(super_nodes, return_counts=True)
            max_count = counts.max().item()
            coarsening_ratio = max_count / n_suspicious

            condition2_met = coarsening_ratio > coarsening_threshold

        is_detected = condition1_met and condition2_met

        if is_detected:
            detected_patterns.append(pattern_id)

        pattern_details[pattern_id] = {
            "n_nodes": n_total,
            "n_suspicious_pred": n_suspicious,
            "suspicious_ratio": suspicious_ratio,
            "coarsening_ratio": coarsening_ratio,
            "condition1_met": condition1_met,
            "condition2_met": condition2_met,
            "detected": is_detected,
        }

    detection_rate = len(detected_patterns) / len(patterns) if len(patterns) > 0 else 0

    return detection_rate, detected_patterns, pattern_details


# # Load AMLGentex dataset
# experiment_root = "experiments/tutorial_demo"
# preprocessed_dir = os.path.join(experiment_root, "preprocessed", "centralized")

from src.feature_engineering import DataPreprocessor
from src.utils.config import load_preprocessing_config

# Add project root to path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

# Set experiment name - this is the ONLY thing you need to configure!
EXPERIMENT = "tutorial_demo2"
experiment_root = project_root / "experiments" / EXPERIMENT

# Copy the preprocessing.yaml from template
template_dir = project_root / "experiments" / "template_experiment" / "config"

# Create experiment directories
config_dir = experiment_root / "config"
# os.makedirs(config_dir, exist_ok=True)


# Load config with auto-discovered paths
preproc_config = load_preprocessing_config(str(config_dir / "preprocessing.yaml"))

print("Preprocessing configuration:")
print(f"  Raw data: {preproc_config['raw_data_file']}")
print(f"  Output dir: {preproc_config['preprocessed_data_dir']}")
print()

print("Preprocessing transactions...")
print("Generating temporal features with rolling windows...\n")

preprocessor = DataPreprocessor(preproc_config)
datasets = preprocessor(preproc_config["raw_data_file"])

# Load node and edge data
nodes_df = datasets["trainset_nodes"]
edges_df = datasets["trainset_edges"]

# Create node ID to index mapping (map account IDs to 0, 1, 2, ..., N-1)
node_to_index = {account_id: idx for idx, account_id in enumerate(nodes_df["account"])}

# Get train/val/test indices AFTER creating the mapping
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

# Prepare node features and labels
nodes_df = nodes_df.drop(columns=["bank"])
X = (
    nodes_df.drop(columns=["account", "train_mask", "val_mask", "test_mask", "is_sar"])
    .to_numpy()
    .astype(np.float32)
)
y = nodes_df["is_sar"].to_numpy().astype(np.int64)

# Map edges from account IDs to indices
edges_df["src_idx"] = edges_df["src"].map(node_to_index)
edges_df["dst_idx"] = edges_df["dst"].map(node_to_index)

# Remove any edges with unmapped nodes (NaN values)
edges_df = edges_df.dropna(subset=["src_idx", "dst_idx"])
edges = edges_df[["src_idx", "dst_idx"]].to_numpy().astype(np.int64)
edges_index = torch.tensor(edges.T, dtype=torch.long)

# Normalize features
scaler = MinMaxScaler().fit(X)
X_normalized = scaler.transform(X)
X_normalized = torch.tensor(X_normalized, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)


# Create PyTorch Geometric Data object
G = torch_geometric.data.Data(x=X_normalized, edge_index=edges_index, y=y)
G = torch_geometric.transforms.ToUndirected()(G)
G.edge_weight = torch.ones(G.edge_index.size(1))

# Create train/val/test split
num_nodes = len(G.y)
G.train_idx = train_idx
G.val_idx = val_idx
G.test_idx = test_idx
G.W, G.L, G.dw = graph_params(G)

# Load alert patterns for evaluation
alert_file = experiment_root / "spatial" / "alert_models.parquet"
if alert_file.exists():
    patterns, pattern_types = load_alert_patterns(str(alert_file), node_to_index)
    print(f"\nLoaded {len(patterns)} alert patterns")
    print(f"Pattern types: {set(pattern_types.values())}")
else:
    # Try CSV format
    alert_file_csv = experiment_root / "spatial" / "alert_models.csv"
    if alert_file_csv.exists():
        patterns, pattern_types = load_patterns_from_csv(
            str(alert_file_csv), node_to_index
        )
        print(f"\nLoaded {len(patterns)} alert patterns from CSV")
        print(f"Pattern types: {set(pattern_types.values())}")
    else:
        patterns = {}
        pattern_types = {}
        print("\nNo alert patterns file found")

# Load normal patterns for evaluation
normal_file = experiment_root / "spatial" / "normal_models.csv"
if normal_file.exists():
    normal_patterns, normal_pattern_types = load_patterns_from_csv(
        str(normal_file), node_to_index
    )
    print(f"\nLoaded {len(normal_patterns)} normal patterns")
    print(f"Normal pattern types: {set(normal_pattern_types.values())}")
else:
    normal_patterns = {}
    normal_pattern_types = {}
    print("\nNo normal patterns file found")

method = "variation_embedding"
# method = "variation_edges"
# method = "variation_neighborhoods"
epochs_per_lev = [1, 2, 5, 10]
thresholds = [0.50, 0.75, 0.85, 0.95]
# epochs_per_lev = [10]
# thresholds = [0.50]

colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]

model = train_GNN(G, epochs=100, lr=0.005)
acc_test, prec, _, _ = evaluate_model(model, G, log_info=True)

max_level = 150
for ep_per_lev in epochs_per_lev:
    LOGGER.info(f"Epochs per Level: {ep_per_lev}")
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.hlines(
        y=acc_test,
        xmin=0,
        xmax=max_level,
        color="black",
        linestyles="--",
        label="orig",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.set_title("Coarse vs Fine Accuracy over Iterations")
    plt.tight_layout()
    fig2, ax2 = plt.subplots(figsize=(16, 9))
    ax2.hlines(
        y=acc_test,
        xmin=0,
        xmax=G.num_nodes,
        color="black",
        linestyles="--",
        label="orig",
    )
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Coarse vs Fine Accuracy over Iterations")
    plt.tight_layout()
    fig3, ax3 = plt.subplots(figsize=(16, 9))
    ax3.hlines(
        y=prec,
        xmin=0,
        xmax=G.num_nodes,
        color="black",
        linestyles="--",
        label="orig",
    )
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("precision")
    ax3.set_title("Coarse vs precision over Iterations")
    plt.tight_layout()
    for idx, threshold in enumerate(thresholds):
        Gall, Call, model, C_plus = train_GNN_coarsening_aware_loss(
            G,
            levels=max_level,
            K=100,
            nhid=256,
            lr=0.01,
            wd=1e-4,
            epoch_per_level=ep_per_lev,
            method=method,
            similarity_threshold=threshold,
            dropout=0.1,
            grad_clip=1.0,  # Enable gradient clipping
            warmup_epochs=3,  # Add warmup epochs
        )
        name = f"data_gnn_CoarseningAwareLoss_V2_th_{threshold*100:.0f}_epochs_{ep_per_lev}.npy"
        data = np.load(f"{save_path}{name}", allow_pickle=True).item()
        LOGGER.info(f"epochs: {ep_per_lev}, threshold: {threshold}")
        LOGGER.info(
            f"nodes: {Gall[-1].num_nodes}, edges: {Gall[-1].num_edges}, coarse accuracy: {data['ycrs'][-1]:.4f}, fine accuracy: {data['yfine'][-1]:.4f}"
        )

        # Evaluate pattern detection at each coarsening level
        if len(patterns) > 0:
            import torch.nn.functional as F

            pattern_detection_rates = []
            pattern_detection_rates_gt = []  # Using ground truth labels
            normal_pattern_detection_rates_gt = []  # For normal patterns

            # Debug: check label distribution in ground truth
            gt_labels = G.y
            n_suspicious_gt = (gt_labels == 1).sum().item()
            LOGGER.info(
                f"Ground truth: {n_suspicious_gt}/{len(gt_labels)} suspicious ({100*n_suspicious_gt/len(gt_labels):.2f}%)"
            )

            for level_idx in range(len(Gall)):
                # Get predictions using the model on the coarse graph
                Gc = Gall[level_idx]
                model.eval()
                with torch.no_grad():
                    S_mp = Gc.S_mp if hasattr(Gc, "S_mp") else Gc.W
                    logits = model(Gc.x, S_mp)

                    # Project coarse predictions to fine graph
                    # Call[level_idx-1] is the cumulative coarsening matrix from original to this level
                    if level_idx > 0:
                        # Call[level_idx-1] has shape (n_coarse, n_original)
                        # We need C_plus which has shape (n_original, n_coarse)
                        # C_plus = C.t() normalized
                        C_cur = Call[level_idx - 1]  # (n_coarse, n_original)
                        # Normalize transpose to get C_plus
                        C_dense = C_cur.to_dense().t()  # (n_original, n_coarse)
                        # Normalize rows
                        row_sums = C_dense.sum(dim=1, keepdim=True)
                        row_sums = torch.clamp(row_sums, min=1e-8)
                        C_plus_normalized = C_dense / row_sums
                        fine_logits = C_plus_normalized @ logits
                    else:
                        fine_logits = logits

                    probs = F.softmax(fine_logits, dim=1)
                    predictions = torch.argmax(probs, dim=1)
                    suspicious_probs = probs[
                        :, 1
                    ]  # Probability of class 1 (suspicious)

                # Get coarsening mapping up to this level
                # Call[level_idx-1] maps original nodes to supernodes at this level
                if level_idx > 0:
                    node_to_supernode = get_node_to_supernode_mapping(
                        Call[level_idx - 1]
                    )
                else:
                    node_to_supernode = None

                # Evaluate pattern detection using probability-based threshold (0.3)
                # since the model has class imbalance issues
                detection_rate, detected_ids, details = evaluate_pattern_detection(
                    predictions,
                    patterns,
                    node_to_supernode,
                    majority_threshold=0.75,
                    coarsening_threshold=0.75,
                    suspicious_probs=suspicious_probs,
                    prob_threshold=0.3,
                )
                pattern_detection_rates.append(detection_rate)

                # Also evaluate using ground truth (measures coarsening quality)
                detection_rate_gt, detected_ids_gt, details_gt = (
                    evaluate_pattern_detection_gt(
                        gt_labels,
                        patterns,
                        node_to_supernode,
                        majority_threshold=0.75,
                        coarsening_threshold=0.75,
                    )
                )
                pattern_detection_rates_gt.append(detection_rate_gt)

                # Evaluate normal pattern detection at this level
                if len(normal_patterns) > 0:
                    (
                        normal_detection_rate_gt,
                        normal_detected_ids_gt,
                        normal_details_gt,
                    ) = evaluate_normal_pattern_detection_gt(
                        gt_labels,
                        normal_patterns,
                        node_to_supernode,
                        majority_threshold=0.5,
                        coarsening_threshold=0.5,
                    )
                    if not hasattr(data, "normal_pattern_detection_rates_gt"):
                        if "normal_pattern_detection_rates_gt" not in dir():
                            normal_pattern_detection_rates_gt = []
                    normal_pattern_detection_rates_gt.append(normal_detection_rate_gt)

                # Debug output at first and last level
                if level_idx == 0 or level_idx == len(Gall) - 1:
                    n_pred_suspicious = (predictions == 1).sum().item()
                    n_prob_suspicious = (suspicious_probs > 0.3).sum().item()
                    mean_prob = suspicious_probs.mean().item()
                    max_prob = suspicious_probs.max().item()
                    LOGGER.info(
                        f"Level {level_idx}: {n_pred_suspicious}/{len(predictions)} pred suspicious, {n_prob_suspicious} prob>0.3 (mean={mean_prob:.3f}, max={max_prob:.3f})"
                    )
                    LOGGER.info(
                        f"  Alert Model-based detection: {detection_rate:.4f} ({len(detected_ids)}/{len(patterns)} patterns)"
                    )
                    LOGGER.info(
                        f"  Alert GT-based coarsening quality: {detection_rate_gt:.4f} ({len(detected_ids_gt)}/{len(patterns)} patterns)"
                    )
                    if len(normal_patterns) > 0:
                        LOGGER.info(
                            f"  Normal GT-based coarsening quality: {normal_detection_rate_gt:.4f} ({len(normal_detected_ids_gt)}/{len(normal_patterns)} patterns)"
                        )

                    # Show undetected patterns
                    undetected_gt = [
                        pid for pid in patterns.keys() if pid not in detected_ids_gt
                    ]
                    if len(undetected_gt) > 0 and len(undetected_gt) <= 5:
                        for pid in undetected_gt:
                            d = details_gt[pid]
                            LOGGER.info(
                                f"  UNDETECTED Pattern {pid} (GT): {d['n_suspicious_gt']}/{d['n_nodes']} truly suspicious ({d['suspicious_ratio']:.2f}), coarsened={d['coarsening_ratio']:.2f}"
                            )

                    # Check first few patterns with GT
                    sample_patterns = list(patterns.keys())[:3]
                    for pid in sample_patterns:
                        if pid in details_gt:
                            d = details_gt[pid]
                            LOGGER.info(
                                f"  Pattern {pid} (GT): {d['n_suspicious_gt']}/{d['n_nodes']} truly suspicious, coarsened={d['coarsening_ratio']:.2f}, detected={d['detected']}"
                            )

            # Add pattern detection to the saved data
            data["pattern_detection_rates"] = pattern_detection_rates
            data["pattern_detection_rates_gt"] = pattern_detection_rates_gt
            if len(normal_patterns) > 0:
                data["normal_pattern_detection_rates_gt"] = (
                    normal_pattern_detection_rates_gt
                )
            np.save(f"{save_path}{name}", data)

            LOGGER.info(
                f"Alert pattern detection (model) at final: {pattern_detection_rates[-1]:.4f}"
            )
            LOGGER.info(
                f"Alert pattern detection (GT coarsening quality) at final: {pattern_detection_rates_gt[-1]:.4f}"
            )
            if len(normal_patterns) > 0:
                LOGGER.info(
                    f"Normal pattern detection (GT coarsening quality) at final: {normal_pattern_detection_rates_gt[-1]:.4f}"
                )

        name = f"data_gnn_CoarseningAwareLoss_V2_th_{threshold*100:.0f}_epochs_{ep_per_lev}.npy"
        data = np.load(f"{save_path}/{name}", allow_pickle=True).item()
        width = 2 if idx == len(thresholds) - 1 else 1

        # Apply smoothing to reduce jumpiness in plots
        # smooth_window = 5
        # ycrs_smooth = smooth_curve(np.array(data["ycrs"]), smooth_window)
        # yfine_smooth = smooth_curve(np.array(data["yfine"]), smooth_window)
        # prec_l_smooth = smooth_curve(np.array(data["prec_l"]), smooth_window)
        # prec_fine_smooth = smooth_curve(np.array(data["prec_fine"]), smooth_window)

        ax.plot(
            # ycrs_smooth,
            data["ycrs"],
            color=colors[idx],
            linewidth=width,
            marker="*",
            markersize=2,
            label=f"Coarse {threshold*100:.0f}%",
        )
        ax.plot(
            # yfine_smooth,
            data["yfine"],
            linestyle=":",
            marker="o",
            markersize=2,
            color=colors[idx],
            linewidth=width,
            label=f"Fine {threshold*100:.0f}%",
        )
        ax.legend()
        fig.savefig(f"{save_path}/iterative_custom_loss_accuracy_{ep_per_lev}.png")

        name = f"data_gnn_CoarseningAwareLoss_V2_th_{threshold*100:.0f}_epochs_{ep_per_lev}.npy"
        data = np.load(f"{save_path}/{name}", allow_pickle=True).item()
        width = 2 if idx == len(thresholds) - 1 else 1
        ax2.plot(
            np.array(data["num_nodes_coarse"]),
            # ycrs_smooth,
            data["ycrs"],
            color=colors[idx],
            linewidth=width,
            marker="*",
            markersize=2,
            label=f"Coarse {threshold*100:.0f}%",
        )
        ax2.plot(
            np.array(data["num_nodes_coarse"]),
            # yfine_smooth,
            data["yfine"],
            linestyle=":",
            marker="o",
            markersize=2,
            color=colors[idx],
            linewidth=width,
            label=f"Fine {threshold*100:.0f}%",
        )
        ax2.legend()
        fig2.savefig(
            f"{save_path}/iterative_custom_loss_accuracy_{ep_per_lev}_with_nodes.png"
        )

        # name = f"data_gnn_CoarseningAwareLoss_V2_th_{threshold*100:.0f}_epochs_{ep_per_lev}.npy"
        # data = np.load(f"{save_path}/{name}", allow_pickle=True).item()
        # width = 2 if idx == len(thresholds) - 1 else 1
        ax3.plot(
            np.array(data["num_nodes_coarse"]),
            # prec_l_smooth,
            data["prec_l"],
            color=colors[idx],
            linewidth=width,
            marker="*",
            markersize=2,
            label=f"Coarse {threshold*100:.0f}%",
        )
        ax3.plot(
            np.array(data["num_nodes_coarse"]),
            # prec_fine_smooth,
            data["prec_fine"],
            linestyle=":",
            marker="o",
            markersize=2,
            color=colors[idx],
            linewidth=width,
            label=f"Fine {threshold*100:.0f}%",
        )
        ax3.legend()
        fig3.savefig(
            f"{save_path}/iterative_custom_loss_precission_{ep_per_lev}_with_nodes.png"
        )

    G_coarse = Gall[-1]
    model = train_GNN(G_coarse, epochs=100, lr=0.005)
    acc_coarse, prec_coarse, _, _ = evaluate_model(model, G_coarse, log_info=False)
    ax.hlines(
        y=acc_coarse,
        xmin=0,
        xmax=max_level,
        color="gray",
        linestyles="--",
        label="coarse",
    )
    ax.legend()
    fig.savefig(f"{save_path}/iterative_custom_loss_accuracy_{ep_per_lev}.png")
    ax2.hlines(
        y=acc_coarse,
        xmin=0,
        xmax=G.num_nodes,
        color="gray",
        linestyles="--",
        label="coarse",
    )
    ax2.legend()
    fig2.savefig(
        f"{save_path}/iterative_custom_loss_accuracy_{ep_per_lev}_with_nodes.png"
    )

    # Plot pattern detection rates if available
    if len(patterns) > 0:
        fig5, ax5 = plt.subplots(1, 1, figsize=(12, 6))
        for idx, threshold in enumerate(thresholds):
            name = f"data_gnn_CoarseningAwareLoss_V2_th_{threshold*100:.0f}_epochs_{ep_per_lev}.npy"
            data = np.load(f"{save_path}/{name}", allow_pickle=True).item()
            if "pattern_detection_rates_gt" in data:
                width = 2 if idx == len(thresholds) - 1 else 1
                # Plot GT-based pattern detection (coarsening quality)
                ax5.plot(
                    data["pattern_detection_rates_gt"],
                    color=colors[idx],
                    linewidth=width,
                    marker="o",
                    markersize=3,
                    label=f"Alert GT (th={threshold*100:.0f}%)",
                )
            if "pattern_detection_rates" in data:
                width = 2 if idx == len(thresholds) - 1 else 1
                ax5.plot(
                    data["pattern_detection_rates"],
                    color=colors[idx],
                    linewidth=width,
                    linestyle="--",
                    marker="x",
                    markersize=3,
                    label=f"Alert Model (th={threshold*100:.0f}%)",
                )
        ax5.set_xlabel("Coarsening Level")
        ax5.set_ylabel("Pattern Detection Rate")
        ax5.set_title(
            "Alert Pattern Detection Rate vs Coarsening Level\n(>75% suspicious + >75% coarsened together)"
        )
        ax5.set_ylim(0, 1.05)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        fig5.savefig(f"{save_path}/alert_pattern_detection_rates_{ep_per_lev}.png")
        LOGGER.info(
            f"Alert pattern detection plot saved to {save_path}/alert_pattern_detection_rates_{ep_per_lev}.png"
        )

    # Plot normal pattern detection rates if available
    if len(normal_patterns) > 0:
        fig4, ax4 = plt.subplots(1, 1, figsize=(12, 6))
        for idx, threshold in enumerate(thresholds):
            name = f"data_gnn_CoarseningAwareLoss_V2_th_{threshold*100:.0f}_epochs_{ep_per_lev}.npy"
            data = np.load(f"{save_path}/{name}", allow_pickle=True).item()
            if "normal_pattern_detection_rates_gt" in data:
                width = 2 if idx == len(thresholds) - 1 else 1
                # Plot GT-based normal pattern detection (coarsening quality)
                ax4.plot(
                    data["normal_pattern_detection_rates_gt"],
                    color=colors[idx],
                    linewidth=width,
                    marker="s",
                    markersize=3,
                    label=f"Normal GT (th={threshold*100:.0f}%)",
                )
        ax4.set_xlabel("Coarsening Level")
        ax4.set_ylabel("Pattern Detection Rate")
        ax4.set_title(
            "Normal Pattern Detection Rate vs Coarsening Level\n(>50% normal + >50% coarsened together)"
        )
        ax4.set_ylim(0, 1.05)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        fig4.savefig(f"{save_path}/normal_pattern_detection_rates_{ep_per_lev}.png")
        LOGGER.info(
            f"Normal pattern detection plot saved to {save_path}/normal_pattern_detection_rates_{ep_per_lev}.png"
        )


plt.show()
