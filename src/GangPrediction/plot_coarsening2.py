import os
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
import warnings

from tqdm import tqdm


warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import math
from src.GangPrediction.gang_aware_subspace import get_gang_aware_basis
from src.GangPrediction.coarsening_utils import calc_B, coarse_one_level, coarsen_vector
from src.GangPrediction.graph_utils import *
from src.GangPrediction.utils.utils import *
from src.GangPrediction.utils.plot_gif import *
from src.GangPrediction.coarsening_utils import calc_B_embedding, coarsen
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

import torch
from torch_geometric.data import Data


import torch
from torch_geometric.data import Data


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
        "left_diamond": [0, 1, 2, 3],  # left circled diamond
        "top_pentagon": [10, 11, 12, 13, 14],  # top circled pentagon
        "bottom_triangle": [15, 16, 17],  # bottom circled triangle
    }
    normal_pattern = {"center": [4, 5, 6, 7, 8, 9]}  # center big subgraph
    # ----- Patterns (as requested) -----
    # alert_patterns = {
    #     "left_diamond": [12, 13, 14],  # left circled diamond
    #     "top_pentagon": [2, 3, 4, 5, 9],  # top circled pentagon
    #     "bottom_triangle": [6, 10, 11],  # bottom circled triangle
    #     "bottom_triangle2": [7, 8, 15, 16, 17],  # bottom circled triangle
    # }
    # normal_pattern = {"center": [0, 1]}  # center big subgraph

    # R = [[0,0,0, ..., 1,1,1,0,0,0],[0, 1,1,1,1,0,0,0,1,0],[],[]]
    # 0,1,0,0
    # GNN (G)-> H = [h_v forall v in G.V]  # node embeddings from a GNN

    R = [
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
    ]

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

        bar.set_postfix({"epsilon_l": f"{epsilon_l:.2f}"})
        bar.update(1)

    bar.close()
    return C, Gc, Call, Gall, epsilons



def coarsen_and_plot(
    G,
    DATA,
    levels=16,
    # method="variation_embedding",
    method="variation_edges",
    K=5,
    alert_patterns=None,
    normal_patterns=None,
    # save_path="results/",
):
    if DATA == 0:
        max_eps = np.logspace(3, 7, 15, dtype=float)
    elif DATA == 1:
        # max_eps = np.logspace(-3, 2, 20, dtype=float)
        max_eps = [0.01]  # large enough to avoid early stopping

    # if method == "variation_embedding":
    #     B = calc_B_embedding(G, K)
    # else:
    #     B = calc_B(G, K)

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
        _, _, Call, Gall, epsilons = coarsen(
            G,
            K=K,
            # r=0.999,  # coarsening ratio set to max
            method=method,
            max_epsilon=eps,
            # similarity_threshold=0.5,
            # max_levels=int(10 + mxcost // 100),
            levels=levels,
            B=B,
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
    G, alert_patterns, normal_pattern = build_gang_picture_graph(num_node_features=8)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 6))
    for i, j in G.edge_index.t():
        plt.plot([G.pos[i][0], G.pos[j][0]], [G.pos[i][1], G.pos[j][1]], "k-")

    plt.scatter(G.pos[:, 0], G.pos[:, 1], s=50)
    plt.axis("equal")
    plt.axis("off")
    plt.savefig(f"{save_path}custom_graph.png", bbox_inches="tight")
    # plt.show()

    coarsen_and_plot(
        G, DATA, alert_patterns=alert_patterns, normal_patterns=normal_pattern
    )
