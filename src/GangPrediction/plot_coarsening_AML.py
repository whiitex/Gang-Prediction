import os
import math
import sys
import warnings
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from src.GangPrediction.experiment_utils import (
    get_node_to_supernode_mapping,
    load_all_patterns,
    load_amlgentex_data,
)
from src.GangPrediction.gang_aware_subspace import get_gang_aware_basis
from src.GangPrediction.coarsening_utils import calc_B, coarse_one_level, coarsen_vector
from src.GangPrediction.graph_utils import *
from src.GangPrediction.utils.utils import *
from src.GangPrediction.utils.plot_gif import *
from torch_geometric.utils import to_undirected


def evaluate_pattern_detection(
    patterns: dict,
    pattern_types: dict = None,
    node_to_supernode: torch.Tensor = None,
    coarsening_threshold: float = 0.5,
):
    """
    Unified pattern detection evaluation for both alert (suspicious) and normal patterns.

    A pattern is correctly detected if:
    1. More than majority_threshold of accounts have the target label
    2. More than coarsening_threshold of target accounts are coarsened together

    Args:
        labels: Tensor of labels (predictions or ground truth)
        patterns: Dict mapping pattern_id -> list of node indices
        pattern_types: Dict mapping pattern_id -> pattern type string
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
        type_detection_rates: Dict mapping pattern_type -> detection rate
    """
    pattern_types = pattern_types or {}
    detected_patterns = []
    detected_patterns1 = []
    detected_patterns2 = []
    pattern_details = {}

    # Track detection by pattern type
    type_counts = defaultdict(
        lambda: {"total": 0, "detected": 0, "detected1": 0, "detected2": 0}
    )

    for pattern_id, node_indices in patterns.items():
        node_indices_tensor = torch.tensor(node_indices, dtype=torch.long)

        n_total = len(node_indices)

        # Condition 2: Most target nodes coarsened together
        condition2_met = True
        condition3_met = True
        coarsening_ratio = 1.0
        coarsening_ratio2 = 1.0

        if node_to_supernode is not None and n_total > 0:
            # Condition 2: Coarsened node should contain mostly the target nodes
            super_nodes = node_to_supernode[node_indices_tensor]
            super_nodes_unique, counts = torch.unique(super_nodes, return_counts=True)
            max_count = counts.max().item()
            coarsening_ratio = max_count / n_total
            condition2_met = coarsening_ratio > coarsening_threshold

            super_node = super_nodes_unique[counts.argmax().item()]
            max_count2 = (node_to_supernode == super_node).sum().item()
            coarsening_ratio2 = max_count / max_count2
            condition3_met = coarsening_ratio2 > coarsening_threshold

        is_detected = condition2_met and condition3_met
        is_detected1 = condition2_met
        is_detected2 = condition3_met

        if is_detected:
            detected_patterns.append(pattern_id)
        if is_detected1:
            detected_patterns1.append(pattern_id)
        if is_detected2:
            detected_patterns2.append(pattern_id)

        pattern_details[pattern_id] = {
            "n_nodes": n_total,
            "coarsening_ratio": coarsening_ratio,
            "condition2_met": condition2_met,
            "condition3_met": condition3_met,
            "detected1": is_detected1,
            "detected2": is_detected2,
            "detected": is_detected,
        }

        # Track per-type detection
        ptype = pattern_types.get(pattern_id, "unknown")
        type_counts[ptype]["total"] += 1
        if is_detected:
            type_counts[ptype]["detected"] += 1
        if is_detected1:
            type_counts[ptype]["detected1"] += 1
        if is_detected2:
            type_counts[ptype]["detected2"] += 1

    detection_rate = len(detected_patterns) / len(patterns) if len(patterns) > 0 else 0
    detection_rate1 = (
        len(detected_patterns1) / len(patterns) if len(patterns) > 0 else 0
    )
    detection_rate2 = (
        len(detected_patterns2) / len(patterns) if len(patterns) > 0 else 0
    )

    # Calculate per-type detection rates
    type_detection_rates = {}
    for ptype, counts in type_counts.items():
        if counts["total"] > 0:
            type_detection_rates[ptype] = {
                "rate": counts["detected"] / counts["total"],
                "rate1": counts["detected1"] / counts["total"],
                "rate2": counts["detected2"] / counts["total"],
                "detected": counts["detected"],
                "total": counts["total"],
            }

    return (
        detection_rate,
        detection_rate1,
        detection_rate2,
        detected_patterns,
        detected_patterns1,
        detected_patterns2,
        pattern_details,
        type_detection_rates,
    )


def evaluate(
    C,
    alert_patterns: dict,
    normal_patterns: dict,
    alert_thresholds: tuple = (0.75, 0.75),
    normal_thresholds: tuple = (0.5, 0.5),
):
    results = {}
    # Get coarsening mapping
    node_to_supernode = get_node_to_supernode_mapping(C)

    # Alert pattern detection (model-based)
    if alert_patterns:
        rate, rate1, rate2, detected, detected1, detected2, details, type_rates = (
            evaluate_pattern_detection(
                alert_patterns,
                node_to_supernode=node_to_supernode,
                coarsening_threshold=alert_thresholds[1],
            )
        )
        results["alert_model_rate"] = rate
        results["alert_model_rate1"] = rate1
        results["alert_model_rate2"] = rate2
        results["alert_model_detected"] = detected
        results["alert_model_detected1"] = detected1
        results["alert_model_detected2"] = detected2
        results["alert_model_details"] = details
        results["alert_model_type_rates"] = type_rates

    # Normal pattern detection (GT-based only)
    if normal_patterns:
        (
            rate_gt,
            rate1_gt,
            rate2_gt,
            detected_gt,
            detected1_gt,
            detected2_gt,
            details_gt,
            type_rates_gt,
        ) = evaluate_pattern_detection(
            normal_patterns,
            node_to_supernode=node_to_supernode,
            coarsening_threshold=normal_thresholds[1],
        )
        results["normal_gt_rate"] = rate_gt
        results["normal_gt_rate1"] = rate1_gt
        results["normal_gt_rate2"] = rate2_gt
        results["normal_gt_detected"] = detected_gt
        results["normal_gt_detected1"] = detected1_gt
        results["normal_gt_detected2"] = detected2_gt
        results["normal_gt_details"] = details_gt
        results["normal_gt_type_rates"] = type_rates_gt

    return results


def build_gang_picture_graph(num_node_features: int = 1):
    """
    Builds the graph from the picture + adds 2D node positions (data.pos).

    Node indexing
    -------------
    Left circled (alert) diamond (4-cycle):
      0=left, 1=top, 2=right, 3=bottom

    Center (normal) subgraph (hexagon + one diagonal):
      4=left connector, 5=upper-left, 6=top junction,
      7=right, 8=bottom-right, 9=bottom-left

    Top circled (alert) pentagon (5-cycle):
      10=bottom (connects to 6), 11=right, 12=top-right, 13=top, 14=top-left

    Bottom circled (alert) triangle (3-cycle):
      15=top (connects to 8), 16=left, 17=right
    """

    # ----- Edges (undirected) -----
    edges = []

    # Left alert diamond: 0-1-2-3-0
    edges += [(0, 1), (1, 2), (2, 3), (3, 0)]

    # Connect left diamond to center
    edges += [(2, 4)]

    # Center normal subgraph: 4-5-6-7-8-9-4 (hexagon)
    edges += [(4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 4)]

    # Extra diagonal in the center (roughly matches the drawing)
    edges += [(6, 8)]

    # Top alert pentagon: 10-11-12-13-14-10
    edges += [(10, 11), (11, 12), (12, 13), (13, 14), (14, 10)]
    # Connect top pentagon to center junction
    edges += [(6, 10)]

    # Bottom alert triangle: 15-16-17-15
    edges += [(15, 16), (16, 17), (17, 15)]
    # Connect bottom triangle to center bottom-right
    edges += [(8, 15)]

    # Make undirected edge_index (add both directions)
    edge_index = (
        torch.tensor(
            [(u, v) for (u, v) in edges] + [(v, u) for (u, v) in edges],
            dtype=torch.long,
        )
        .t()
        .contiguous()
    )

    n = 18
    x = torch.zeros((n, num_node_features), dtype=torch.float)

    # ----- Node positions (rough layout similar to the photo) -----
    # Shape: [num_nodes, 2]  (x, y)
    pos = torch.tensor(
        [
            [-3.0, 0.0],  # 0  left diamond (left)
            [-2.0, 1.0],  # 1  left diamond (top)
            [-1.0, 0.0],  # 2  left diamond (right, connects to 4)
            [-2.0, -1.0],  # 3  left diamond (bottom)
            [0.0, 0.0],  # 4  center (left connector)
            [1.0, 1.0],  # 5  center (upper-left)
            [2.0, 1.5],  # 6  center (top junction, connects to 10)
            [3.0, 0.7],  # 7  center (right)
            [2.6, -0.7],  # 8  center (bottom-right, connects to 15)
            [1.0, -1.0],  # 9  center (bottom-left)
            [2.8, 2.2],  # 10 top pentagon (bottom, connects to 6)
            [4.0, 2.2],  # 11 top pentagon (right)
            [4.6, 3.4],  # 12 top pentagon (top-right)
            [3.5, 4.2],  # 13 top pentagon (top)
            [2.4, 3.4],  # 14 top pentagon (top-left)
            [2.6, -1.7],  # 15 bottom triangle (top, connects to 8)
            [1.9, -2.8],  # 16 bottom triangle (left)
            [4.0, -2.8],  # 17 bottom triangle (right)
        ],
        dtype=torch.float,
    )

    # ----- Patterns (as requested) -----
    alert_patterns = {
        "left_diamond": [12, 13, 14],  # left circled diamond
        "top_pentagon": [2, 3, 4, 5, 9],  # top circled pentagon
        "bottom_triangle": [6, 10, 11],  # bottom circled triangle
        "bottom_triangle2": [7, 8, 15, 16, 17],  # bottom circled triangle
    }
    normal_pattern = {"center": [0, 1]}  # center big subgraph
    # alert_patterns = {
    #     "left_diamond": [0, 1, 2, 3],  # left circled diamond
    #     "top_pentagon": [10, 11, 12, 13, 14],  # top circled pentagon
    #     "bottom_triangle": [15, 16, 17],  # bottom circled triangle
    # }
    # normal_pattern = {"center": [4, 5, 6, 7, 8, 9]}  # center big subgraph

    # Optional: node labels (1=alert node, 0=normal node)
    y = torch.zeros(n, dtype=torch.long)
    for pat in alert_patterns.values():
        y[torch.tensor(pat, dtype=torch.long)] = 1

    G = Data(x=x, edge_index=edge_index, y=y, pos=pos)

    G.W, G.L, G.dw = graph_params(G)
    G.edge_weight = torch.ones(G.edge_index.size(1), device=G.edge_index.device)
    L_dense = G.L.to_dense().cpu().numpy()
    _, V = np.linalg.eigh(L_dense)
    colors = V[:, :3]
    colors = (colors - np.mean(colors, axis=0)) / (np.std(colors, axis=0) + 1e-8)
    colors = (colors - colors.min(axis=0)) / (
        colors.max(axis=0) - colors.min(axis=0) + 1e-8
    )
    G.colors = torch.tensor(colors, device=device)
    return G, alert_patterns, normal_pattern


# Example:
# data, alert_pats, normal_pat = build_gang_picture_graph(num_node_features=8)
# print(data.pos)  # positions are in data.pos


# Example usage:
# data, alert_pats, normal_pat = build_gang_picture_graph(num_node_features=8)
# print(data)
# print("alert patterns:", alert_pats)
# print("normal pattern:", normal_pat)


def create_custom_graph():
    # ---------- Nodes ----------
    num_nodes = 22

    # ---------- Edges ----------
    edges = [
        # ring
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 0),
        # leaves
        (1, 8),
        (1, 9),
        (2, 10),
        (3, 11),
        (3, 12),
        (4, 13),
        (5, 14),
        (5, 15),
        (6, 16),
        (7, 17),
        (7, 18),
        # stem
        (0, 19),
        (19, 20),
        (20, 21),
    ]

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_index = to_undirected(edge_index)

    # ---------- Positions ----------
    pos = torch.zeros((num_nodes, 2))

    # circle (8 nodes)
    R = 1.0
    for i in range(8):
        angle = 2 * math.pi * i / 8 - math.pi / 2
        pos[i] = torch.tensor([R * math.cos(angle), R * math.sin(angle)])

    # leaves (small offsets)
    pos[8] = pos[1] + torch.tensor([0.3, -0.2])
    pos[9] = pos[1] + torch.tensor([0.3, 0.2])
    pos[10] = pos[2] + torch.tensor([0.4, 0.0])
    pos[11] = pos[3] + torch.tensor([0.3, 0.2])
    pos[12] = pos[3] + torch.tensor([0.2, 0.4])
    pos[13] = pos[4] + torch.tensor([0.0, 0.4])
    pos[14] = pos[5] + torch.tensor([-0.3, 0.2])
    pos[15] = pos[5] + torch.tensor([-0.2, 0.4])
    pos[16] = pos[6] + torch.tensor([-0.4, 0.0])
    pos[17] = pos[7] + torch.tensor([-0.3, -0.2])
    pos[18] = pos[7] + torch.tensor([-0.2, -0.4])

    # stem (downwards)
    pos[19] = pos[0] + torch.tensor([0.0, -0.6])
    pos[20] = pos[0] + torch.tensor([0.0, -1.2])
    pos[21] = pos[0] + torch.tensor([0.0, -1.8])

    # ---------- Node features ----------
    x = torch.ones((num_nodes, 1), dtype=torch.float)

    G = Data(x=x, edge_index=edge_index, pos=pos)
    # colors by eigenvectors
    G.W, G.L, G.dw = graph_params(G)
    G.edge_weight = torch.ones(G.edge_index.size(1), device=G.edge_index.device)
    L_dense = G.L.to_dense().cpu().numpy()
    _, V = np.linalg.eigh(L_dense)
    colors = V[:, :3]
    colors = (colors - np.mean(colors, axis=0)) / (np.std(colors, axis=0) + 1e-8)
    colors = (colors - colors.min(axis=0)) / (
        colors.max(axis=0) - colors.min(axis=0) + 1e-8
    )
    G.colors = torch.tensor(colors, device=device)

    return G


def get_graph(DATA):
    if DATA == 0:  # create a Cora dataset
        dataset = Planetoid(root="data/Planetoid", name="Cora")
        G = dataset[0].to(device)
        G.edge_index = to_undirected(G.edge_index)
        G.edge_weight = torch.ones(G.edge_index.size(1), device=G.edge_index.device)

        # colors by eigenvectors
        G.W, G.L, G.dw = graph_params(G)
        L_dense = G.L.to_dense().cpu().numpy()
        _, V = np.linalg.eigh(L_dense)

    elif DATA == 1:  # create a KarateClub dataset
        dataset = nx.karate_club_graph()
        club_labels = [dataset.nodes[n]["club"] for n in dataset.nodes]
        club_to_idx = {club: idx for idx, club in enumerate(sorted(set(club_labels)))}
        club_labels = torch.tensor(
            [club_to_idx[club] for club in club_labels], dtype=torch.long, device=device
        )
        x = torch.eye(dataset.number_of_nodes(), dtype=torch.float, device=device)
        edge_index = (
            torch.tensor(list(dataset.edges), dtype=torch.long, device=device)
            .t()
            .contiguous()
        )
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        G = Data(
            x=x,
            edge_index=edge_index,
            y=club_labels,
            num_nodes=dataset.number_of_nodes(),
            edge_weight=torch.ones(edge_index.size(1), device=edge_index.device),
            embeddings=None,
        ).to(device)

        # position by spring layout
        GG = to_networkx(G, to_undirected=True)
        pos = nx.spring_layout(GG)
        pos = [vals for vals in pos.values()]
        G.pos = torch.tensor(pos, dtype=torch.float32, device=device)

        # colors by eigenvectors
        G.W, G.L, G.dw = graph_params(G)
        L_dense = G.L.to_dense().cpu().numpy()
        _, V = np.linalg.eigh(L_dense)
        colors = V[:, :3]
        colors = (colors - np.mean(colors, axis=0)) / (np.std(colors, axis=0) + 1e-8)
        colors = (colors - colors.min(axis=0)) / (
            colors.max(axis=0) - colors.min(axis=0) + 1e-8
        )
        G.colors = torch.tensor(colors, device=device)

    return G


def coarsen(
    G,
    levels=10,
    method="variation_embedding",
    algorithm="greedy",
    max_epsilon=float("inf"),
    similarity_threshold=0.0,
    K=4,
    B=None,
    alert_patterns=None,
    normal_patterns=None,
):
    N = G.num_nodes

    # Initialize coarsening matrices and a layout for visualization/debugging.
    C = sparse_eye(N)

    # iC = None

    Call, Gall = [], []
    Gall.append(G)
    # Call.append(C)

    bar = tqdm(total=levels)
    epsilon_l, epsilons = 0, [0]
    max_eps_in_level = 0.0
    Gc = G.clone()
    alert_rates = []
    normal_rates = []

    for level in range(1, levels + 1):
        ratio = np.log(level ** (4 / 3)) / 100 + 0.02
        # ratio = 1
        # get embeddings from the GNN

        max_eps_in_level += max_epsilon / levels
        max_sigma = (max_epsilon + 1) / (epsilon_l + 1) - 1

        Gc, B, sigma_l, done_flag = coarse_one_level(
            Gc,
            B,
            K=K,
            method=method,
            algorithm=algorithm,
            similarity_threshold=similarity_threshold,
            level=level,
            r_cur=ratio,
            max_sigma=max_sigma,
        )
        C = torch.sparse.mm(Gc.C, C)

        Gall.append(Gc)
        # iCs.append(iC)

        epsilon_l = (sigma_l + 1) * (epsilon_l + 1) - 1
        epsilons.append(epsilon_l)
        if epsilon_l >= max_epsilon:
            print(
                f"Reached max epsilon {max_epsilon} at level {level} with epsilon_l {epsilon_l:.2f}."
            )
            break

        results = evaluate(C, alert_patterns, normal_patterns)
        alert_rates.append(results.get("alert_model_rate", 0))
        normal_rates.append(results.get("normal_gt_rate", 0))

        bar.set_postfix_str(
            f"{epsilon_l:.6f}/{max_epsilon:0.3f}, nodes: {Gc.num_nodes} | "
            f"alert_rate: {results.get('alert_model_rate', 0):.4f} | "
            f"normal_rate: {results.get('normal_gt_rate', 0):.4f}"
        )
        bar.update(1)

    bar.close()
    return C, Gc, Call, Gall, epsilons, alert_rates, normal_rates


def coarsen_and_plot(
    G,
    DATA,
    levels=50,
    # method="variation_embedding",
    method="variation_edges",
    K=10,
    alert_patterns=None,
    normal_patterns=None,
    # save_path="results/",
):
    if DATA == 0:
        max_eps = np.logspace(3, 7, 15, dtype=float)
    elif DATA == 1:
        # max_eps = np.logspace(-3, 2, 20, dtype=float)
        max_eps = [100.01]  # large enough to avoid early stopping

    # if method == "variation_embedding":
    #     B = calc_B_embedding(G, K)
    # else:
    #     B = calc_B(G, K)

    alert_nodes = [alert_pattern for alert_pattern in alert_patterns.values()]
    normal_nodes = [normal_pattern for normal_pattern in normal_patterns.values()]
    remaining_nodes = list(
        set(range(G.num_nodes))
        - set([n for pattern in alert_nodes + normal_nodes for n in pattern])
    )
    normal_patterns["remaining"] = remaining_nodes

    Uk = get_gang_aware_basis(
        G,
        alert_patterns=alert_patterns,
        normal_patterns=normal_patterns,
        K=K,
        alpha=1,
        method="svd",
    )
    B = calc_B(G, K=Uk.shape[1], U=Uk)  # Precompute eigenvectors
    # B = calc_B(G, K=K)  # Precompute eigenvectors

    for eps in max_eps:
        _, _, Call, Gall, epsilons, alert_rates, normal_rates = coarsen(
            G,
            K=K,
            # r=0.999,  # coarsening ratio set to max
            method=method,
            max_epsilon=eps,
            # similarity_threshold=0.5,
            # max_levels=int(10 + mxcost // 100),
            levels=levels,
            B=B,
            alert_patterns=alert_patterns,
            normal_patterns=normal_patterns,
        )

        nodes = [G.num_nodes for G in Gall]

        plt.figure(figsize=(6, 4))
        plt.plot(nodes, epsilons, label="Upper bound")
        plt.axhline(
            y=eps,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=f"max_eps = {eps:.2f}",
        )
        plt.xlabel("Number of nodes")
        plt.ylabel("Cost")
        plt.legend()
        plt.title(f"Cost vs Level (max_eps={eps:.2f})")
        plt.tight_layout()
        plt.savefig(f"{save_path}cost_vs_level_max_eps_{eps:.2f}.png")
        # plt.show()

        plt.figure(figsize=(6, 4))
        plt.plot(alert_rates, label="Alert pattern detection rate")
        plt.plot(normal_rates, label="Normal pattern detection rate")
        plt.xlabel("Number of nodes")
        plt.ylabel("Pattern Detection Rate")
        plt.legend()
        plt.title(f"Pattern Detection Rate vs Level (max_eps={eps:.2f})")
        plt.tight_layout()
        plt.savefig(f"{save_path}pattern_detection_vs_level_max_eps_{eps:.2f}.png")

        # plot_Gall_structural_only(Gall, Call) # visualize coarsening steps
        if DATA == 1:
            make_gif(
                Gall,
                gif_path=f"{save_path}max_eps_{eps:.2f}.gif",
                frame_duration=50,
                reset_pos=False,
            )


if __name__ == "__main__":
    """
    Use DATA = 0 for Cora dataset
    Use DATA = 1 for KarateClub dataset
    """

    DATA = 1

    # G = get_graph(DATA)
    # G = create_custom_graph()
    # G, alert_patterns, normal_patterns = build_gang_picture_graph(num_node_features=8)

    # Setup paths
    EXPERIMENT = "tutorial_demo8"
    experiment_root = project_root / "experiments" / EXPERIMENT
    config_dir = experiment_root / "config"
    G, node_to_index = load_amlgentex_data(config_dir)

    # Load patterns
    alert_patterns, alert_types, normal_patterns, normal_types = load_all_patterns(
        experiment_root, node_to_index
    )

    colors0, pos0 = get_pattern_colors_and_positions(
        G, alert_patterns, normal_patterns, alert_types, normal_types
    )
    G.colors = torch.tensor(colors0, dtype=torch.float32, device=device)
    pos = [vals for vals in pos0.values()]
    G.pos = torch.tensor(pos, dtype=torch.float32, device=device)

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(4, 6))
    # for i, j in G.edge_index.t():
    #     plt.plot([G.pos[i][0], G.pos[j][0]], [G.pos[i][1], G.pos[j][1]], "k-")

    # plt.scatter(G.pos[:, 0], G.pos[:, 1], s=50)
    # plt.axis("equal")
    # plt.axis("off")
    # plt.savefig(f"{save_path}custom_graph.png", bbox_inches="tight")
    # plt.show()

    coarsen_and_plot(
        G, DATA, alert_patterns=alert_patterns, normal_patterns=normal_patterns
    )
