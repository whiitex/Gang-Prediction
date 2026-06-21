"""Loader for the Elliptic++ Actors (Wallet Addresses) dataset.

Reads the raw CSV files from *data_dir*, filters wallets to those active in
the requested time-step window, builds a PyG Data graph, and derives alert /
normal patterns from connected components of illicit / licit wallets.

Expected CSV files in *data_dir*
---------------------------------
wallets_features.csv  — columns: address, Time step, <55 feature columns>
                        Each wallet appears once per time step it is active.
wallets_classes.csv   — columns: address, class  (1=illicit, 2=licit, 3=unknown)
AddrAddr_edgelist.csv — columns: input_address, output_address
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sp_connected_components
import torch
import torch_geometric.data
import torch_geometric.transforms
from sklearn.preprocessing import MinMaxScaler

from src.GangPrediction.pattern_models import Pattern, create_pattern
from src.GangPrediction.experiment_utils import split_patterns, get_pattern_node_indices
from src.GangPrediction.utils.utils import graph_params


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def load_elliptic_actors_data(
    data_dir: Path,
    day_start: int = 27,
    day_end: int = 35,
    to_undirected: bool = True,
    train_ratio: float = 0.5,
    min_pattern_size: int = 2,
    max_pattern_size: Optional[int] = None,
    seed: Optional[int] = 42,
    device: Optional[torch.device] = None,
) -> Tuple[
    torch_geometric.data.Data,
    List[Pattern],
    List[Pattern],
    List[Pattern],
    List[Pattern],
]:
    """Load Elliptic++ Actors dataset filtered to wallets active in *day_start*–*day_end*.

    Parameters
    ----------
    data_dir        : directory containing wallets_features.csv,
                      wallets_classes.csv, AddrAddr_edgelist.csv
    day_start       : first time step to include (inclusive)
    day_end         : last  time step to include (inclusive)
    to_undirected   : symmetrize the graph
    train_ratio     : fraction of patterns used for training
    min_pattern_size: minimum number of nodes for a component to become a pattern
    max_pattern_size: if set, skip components larger than this
    seed            : random seed for pattern split
    device          : torch device

    Returns
    -------
    G                : PyG Data with x, edge_index, edge_attr, y, W, L, dw,
                       train_idx, val_idx, test_idx
    alert_train, normal_train, alert_test, normal_test : List[Pattern]
    """
    if device is None:
        device = torch.device("cpu")

    data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # 1. Load & filter raw CSV files
    # ------------------------------------------------------------------
    print("  Reading wallets_features.csv …")
    features_df = pd.read_csv(
        data_dir / "wallets_features.csv",
        dtype={"address": str, "Time step": "int32"},
        low_memory=False,
    )

    print("  Reading wallets_classes.csv …")
    classes_df = pd.read_csv(
        data_dir / "wallets_classes.csv",
        dtype={"address": str},
        low_memory=False,
    )

    print("  Reading AddrAddr_edgelist.csv …")
    edgelist_df = pd.read_csv(
        data_dir / "AddrAddr_edgelist.csv",
        dtype={"input_address": str, "output_address": str},
        low_memory=False,
    )

    # Filter features to wallets active in the requested time window
    features_df = features_df[
        features_df["Time step"].between(day_start, day_end)
    ].copy()

    print(
        f"  Time steps {day_start}–{day_end}: {len(features_df):,} rows "
        f"({features_df['address'].nunique():,} unique wallets, "
        f"{features_df['Time step'].nunique()} distinct steps)"
    )

    # Each wallet appears once per time step with (identical) global properties.
    # Deduplicate to get one feature row per wallet.
    features_df = features_df.drop_duplicates(subset=["address"], keep="last").copy()

    # ------------------------------------------------------------------
    # 2. Merge with class labels
    # ------------------------------------------------------------------
    nodes_df = features_df.merge(classes_df, on="address", how="left")
    nodes_df["class"] = nodes_df["class"].fillna(3).astype(int)

    # Re-index nodes contiguously 0 … N-1
    node_to_index: dict = {
        addr: idx for idx, addr in enumerate(nodes_df["address"].values)
    }
    N = len(nodes_df)
    print(f"  Nodes in subgraph: {N:,}")

    # ------------------------------------------------------------------
    # 3. Node features  (drop metadata columns, fill NaN, normalize)
    # ------------------------------------------------------------------
    meta_cols = ["address", "Time step", "class"]
    feature_cols = [c for c in nodes_df.columns if c not in meta_cols]

    X = nodes_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Labels  (class 1 → 1 suspicious, class 2/3 → 0)
    # ------------------------------------------------------------------
    y = (nodes_df["class"].values == 1).astype(np.int64)

    print(
        f"  Illicit (class=1): {(y == 1).sum():,}  |  "
        f"Licit  (class=2): {(nodes_df['class'] == 2).sum():,}  |  "
        f"Unknown (class=3): {(nodes_df['class'] == 3).sum():,}"
    )

    # ------------------------------------------------------------------
    # 5. Edges — keep only edges within the filtered wallet set
    # ------------------------------------------------------------------
    node_set = set(node_to_index.keys())
    edge_mask = edgelist_df["input_address"].isin(node_set) & edgelist_df[
        "output_address"
    ].isin(node_set)
    edges_filtered = edgelist_df[edge_mask].copy()

    src_idx = (
        edges_filtered["input_address"].map(node_to_index).to_numpy(dtype=np.int64)
    )
    dst_idx = (
        edges_filtered["output_address"].map(node_to_index).to_numpy(dtype=np.int64)
    )
    edges = np.stack([src_idx, dst_idx], axis=1)  # (M, 2)

    print(f"  Edges in subgraph: {len(edges):,}")

    # ------------------------------------------------------------------
    # 6. Build PyG Data
    # ------------------------------------------------------------------
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.int64, device=device)
    edge_index = torch.tensor(edges.T, dtype=torch.long, device=device)

    G = torch_geometric.data.Data(x=X_t, edge_index=edge_index, y=y_t)
    if to_undirected:
        G = torch_geometric.transforms.ToUndirected()(G)
    G.edge_weight = torch.ones(G.edge_index.size(1), dtype=torch.float32, device=device)
    G.W, G.L, G.dw = graph_params(G)

    # ------------------------------------------------------------------
    # 7. Build alert and normal patterns from connected components
    # ------------------------------------------------------------------
    classes_array = nodes_df["class"].values  # raw 1/2/3 per node index

    alert_patterns = _components_to_patterns(
        node_indices=np.where(classes_array == 1)[0],
        edges=edges,
        label="alert",
        min_size=min_pattern_size,
        max_size=max_pattern_size,
    )
    normal_patterns = _components_to_patterns(
        node_indices=np.where(classes_array == 2)[0],
        edges=edges,
        label="normal",
        min_size=min_pattern_size,
        max_size=max_pattern_size,
    )

    print(f"\n  Alert patterns (connected illicit components): {len(alert_patterns)}")
    print(f"  Normal patterns (connected licit components): {len(normal_patterns)}")

    # ------------------------------------------------------------------
    # 8. Train / test split
    # ------------------------------------------------------------------
    alert_train, alert_test = split_patterns(
        alert_patterns, train_ratio=train_ratio, seed=seed
    )
    normal_train, normal_test = split_patterns(
        normal_patterns, train_ratio=train_ratio, seed=seed
    )

    print(f"\n  Alert  — train: {len(alert_train)}, test: {len(alert_test)}")
    print(f"  Normal — train: {len(normal_train)}, test: {len(normal_test)}")

    # ------------------------------------------------------------------
    # 9. Derive train_idx / val_idx / test_idx from pattern membership
    # ------------------------------------------------------------------
    alert_train_nodes = get_pattern_node_indices(alert_train)
    normal_train_nodes = get_pattern_node_indices(normal_train)
    train_nodes = (
        torch.unique(torch.cat([alert_train_nodes, normal_train_nodes]))
        if (len(alert_train_nodes) + len(normal_train_nodes) > 0)
        else torch.tensor([], dtype=torch.long)
    )

    all_nodes = torch.arange(N, dtype=torch.long)
    test_nodes = all_nodes[~torch.isin(all_nodes, train_nodes)]

    print(
        f"\n  Training nodes: {len(train_nodes):,}  |  Test/val nodes: {len(test_nodes):,}"
    )

    G.train_idx = train_nodes
    G.val_idx = test_nodes
    G.test_idx = test_nodes

    return G, alert_train, normal_train, alert_test, normal_test


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _components_to_patterns(
    node_indices: np.ndarray,
    edges: np.ndarray,
    label: str,
    min_size: int = 2,
    max_size: Optional[int] = None,
) -> List[Pattern]:
    """Build one Pattern per connected component in the induced subgraph.

    Parameters
    ----------
    node_indices : 1-D array of node indices (in the 0-based remapped space)
    edges        : (M, 2) array of all edges in the subgraph
    label        : "alert" or "normal"
    min_size     : skip components smaller than this
    max_size     : skip components larger than this (None = no cap)
    """
    if len(node_indices) == 0:
        return []

    # Boolean lookup array — O(N) memory, vectorised membership test
    num_nodes_total = (
        max(int(edges.max()) + 1, int(node_indices.max()) + 1)
        if len(edges) > 0
        else int(node_indices.max()) + 1
    )
    in_set = np.zeros(num_nodes_total, dtype=bool)
    in_set[node_indices] = True

    # Vectorised edge filtering (no Python loop)
    if len(edges) > 0:
        mask = in_set[edges[:, 0]] & in_set[edges[:, 1]]
        induced_edges = edges[mask]
    else:
        induced_edges = np.empty((0, 2), dtype=np.int64)

    # Compact re-indexing: original node id → 0 … n-1
    n = len(node_indices)
    compact = np.full(num_nodes_total, -1, dtype=np.int64)
    compact[node_indices] = np.arange(n)

    # Build scipy sparse adjacency and run connected_components in C
    if len(induced_edges) > 0:
        src_c = compact[induced_edges[:, 0]]
        dst_c = compact[induced_edges[:, 1]]
        vals = np.ones(len(src_c), dtype=np.int8)
        adj = csr_matrix((vals, (src_c, dst_c)), shape=(n, n))
    else:
        adj = csr_matrix((n, n), dtype=np.int8)

    n_comps, comp_labels = sp_connected_components(
        adj, directed=False, return_labels=True
    )

    patterns: List[Pattern] = []
    for comp_id in range(n_comps):
        comp_mask = comp_labels == comp_id
        comp_size = int(comp_mask.sum())
        if comp_size < min_size:
            continue
        if max_size is not None and comp_size > max_size:
            continue
        nodes_arr = node_indices[comp_mask]
        pattern = create_pattern(
            pattern_id=f"{label}_{comp_id}",
            nodes=nodes_arr,
            pattern_type="unknown",
            label=label,
        )
        patterns.append(pattern)

    return patterns
