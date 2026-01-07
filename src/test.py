"""Experiment driver for coarsening-aware training on Cora."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from graph_utils import *
from utils.utils import *
from coarsening_utils import *
from GNN_model import evaluate_model
from train_GNN_coarsening import train_GNN_coarsening_aware_loss, train_GNN

import warnings

warnings.filterwarnings("ignore")

# Create a Planetoid dataset
dataset = Planetoid(root="data/Planetoid", name="Cora")
G = dataset[0]  # Get the first graph object
G.edge_index = to_undirected(G.edge_index)
G.edge_weight = torch.ones(G.edge_index.size(1), device=G.edge_index.device)
train_idx, val_idx, test_idx = create_train_val_test_split(
    G.num_nodes, train_ratio=0.2, val_ratio=0.1
)
G.train_idx = train_idx
G.val_idx = val_idx
G.test_idx = test_idx
G.W, G.L, G.dw = graph_params(G)

method = "variation_embedding"
# method = "variation_edges"
# method = "variation_neighborhoods"
# epochs_per_lev = [1]
# thresholds = [0.50]
epochs_per_lev = [1, 2, 5, 10]
thresholds = [0.50, 0.75, 0.85, 0.95]

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
acc_test, _, _ = evaluate_model(model, G, log_info=True)

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
    for idx, threshold in enumerate(thresholds):
        Gall, Call = train_GNN_coarsening_aware_loss(
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
        )
        name = f"data_gnn_CoarseningAwareLoss_V2_th_{threshold*100:.0f}_epochs_{ep_per_lev}.npy"
        data = np.load(f"{save_path}{name}", allow_pickle=True).item()
        LOGGER.info(f"epochs: {ep_per_lev}, threshold: {threshold}")
        LOGGER.info(
            f"nodes: {Gall[-1].num_nodes}, edges: {Gall[-1].num_edges}, coarse accuracy: {data['ycrs'][-1]:.4f}, fine accuracy: {data['yfine'][-1]:.4f}"
        )

        name = f"data_gnn_CoarseningAwareLoss_V2_th_{threshold*100:.0f}_epochs_{ep_per_lev}.npy"
        data = np.load(f"{save_path}/{name}", allow_pickle=True).item()
        width = 2 if idx == len(thresholds) - 1 else 1
        ax.plot(
            data["ycrs"],
            color=colors[idx],
            linewidth=width,
            marker="*",
            markersize=2,
            label=f"Coarse {threshold*100:.0f}%",
        )
        ax.plot(
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
            data["ycrs"],
            color=colors[idx],
            linewidth=width,
            marker="*",
            markersize=2,
            label=f"Coarse {threshold*100:.0f}%",
        )
        ax2.plot(
            np.array(data["num_nodes_coarse"]),
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

    G_coarse = Gall[-1]
    model = train_GNN(G_coarse, epochs=100, lr=0.005)
    acc_coarse, _, _ = evaluate_model(model, G_coarse, log_info=False)
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


plt.show()
