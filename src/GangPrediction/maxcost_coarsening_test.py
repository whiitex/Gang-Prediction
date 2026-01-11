import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from graph_utils import *
from utils.utils import *
from utils.plot_gif import *
from coarsening_utils import coarsen


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


def coarsen_and_plot(G, DATA):

    if DATA == 0:
        max_eps = np.logspace(3, 7, 15, dtype=float)
    elif DATA == 1:
        max_eps = np.logspace(-3, 2, 20, dtype=float)

    for eps in max_eps:
        _, _, Call, Gall, epsilons = coarsen(
            G,
            K=10,
            r=0.999,  # coarsening ratio set to max
            method="variation_edges",
            max_epsilon=eps,
            # max_levels=int(10 + mxcost // 100),
            max_levels=20,
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
                Call,
                gif_path=f"{save_path}max_eps_{eps:.2f}.gif",
                frame_duration=100,
            )


if __name__ == "__main__":
    """
    Use DATA = 0 for Cora dataset
    Use DATA = 1 for KarateClub dataset
    """

    DATA = 1

    G = get_graph(DATA)
    coarsen_and_plot(G, DATA)
