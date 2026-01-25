import os
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict

import io
from PIL import Image

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from src.GangPrediction.graph_utils import *
from src.GangPrediction.utils.utils import *

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pattern_colors_and_positions(
    data: Data,
    alert_patterns: dict = None,
    normal_patterns: dict = None,
    alert_types: dict = None,
    normal_types: dict = None,
):
    """
    Compute node colors based on pattern participation and positions that
    cluster nodes in the same patterns together.

    Args:
        data: PyTorch Geometric Data object
        alert_patterns: dict mapping pattern_id -> list of node indices (suspicious patterns)
        normal_patterns: dict mapping pattern_id -> list of node indices (normal patterns)
        alert_types: dict mapping pattern_id -> pattern type string
        normal_types: dict mapping pattern_id -> pattern type string

    Returns:
        colors: numpy array of shape (N, 3) with RGB colors for each node
        pos: dict mapping node index -> (x, y) position
    """
    N = data.num_nodes
    alert_patterns = alert_patterns or {}
    normal_patterns = normal_patterns or {}
    alert_types = alert_types or {}
    normal_types = normal_types or {}

    # Define color scheme for pattern types
    # Alert patterns (suspicious) - warm colors
    alert_type_colors = {
        "fan_out": np.array([0.9, 0.2, 0.2]),  # red
        "fan_in": np.array([0.9, 0.5, 0.1]),  # orange
        "cycle": np.array([0.8, 0.1, 0.5]),  # magenta
        "bipartite": np.array([0.6, 0.1, 0.8]),  # purple
        "stack": np.array([0.9, 0.8, 0.1]),  # yellow
        "scatter_gather": np.array([1.0, 0.4, 0.4]),  # light red
        "gather_scatter": np.array([0.8, 0.3, 0.0]),  # dark orange
    }

    # Normal patterns - cool colors
    normal_type_colors = {
        "single": np.array([0.2, 0.6, 0.9]),  # blue
        "fan_out": np.array([0.2, 0.8, 0.6]),  # teal
        "fan_in": np.array([0.3, 0.9, 0.4]),  # green
        "forward": np.array([0.4, 0.4, 0.9]),  # indigo
        "mutual": np.array([0.1, 0.7, 0.7]),  # cyan
        "periodical": np.array([0.5, 0.8, 0.2]),  # lime
    }

    # Track which patterns each node belongs to
    node_alert_patterns = defaultdict(list)  # node_idx -> list of (pattern_id, type)
    node_normal_patterns = defaultdict(list)

    for pid, nodes in alert_patterns.items():
        ptype = alert_types.get(pid, "unknown")
        for node in nodes:
            if node < N:
                node_alert_patterns[node].append((pid, ptype))

    for pid, nodes in normal_patterns.items():
        ptype = normal_types.get(pid, "unknown")
        for node in nodes:
            if node < N:
                node_normal_patterns[node].append((pid, ptype))

    # Assign colors based on pattern membership
    # Priority: alert patterns > normal patterns > default (gray)
    colors = np.ones((N, 3)) * 0.7  # default gray

    for node in range(N):
        if node in node_alert_patterns:
            # Use the color of the first alert pattern type
            ptype = node_alert_patterns[node][0][1]
            colors[node] = alert_type_colors.get(ptype, np.array([0.9, 0.2, 0.2]))
        elif node in node_normal_patterns:
            # Use the color of the first normal pattern type
            ptype = node_normal_patterns[node][0][1]
            colors[node] = normal_type_colors.get(ptype, np.array([0.2, 0.6, 0.9]))

    # Create position layout that clusters pattern members
    GG = to_networkx(data, to_undirected=True)

    # Add virtual edges between nodes in the same pattern for layout purposes
    pattern_graph = nx.Graph()
    pattern_graph.add_nodes_from(range(N))

    # Add original edges
    for u, v in GG.edges():
        pattern_graph.add_edge(u, v, weight=1.0)

    # Add strong "spring" connections within patterns
    all_patterns = {}
    all_patterns.update({f"alert_{k}": v for k, v in alert_patterns.items()})
    all_patterns.update({f"normal_{k}": v for k, v in normal_patterns.items()})

    for pid, nodes in all_patterns.items():
        valid_nodes = [n for n in nodes if n < N]
        # Connect all pairs within a pattern with strong attraction
        for i, n1 in enumerate(valid_nodes):
            for n2 in valid_nodes[i + 1 :]:
                if pattern_graph.has_edge(n1, n2):
                    # Increase weight of existing edge
                    pattern_graph[n1][n2]["weight"] += 3.0
                else:
                    pattern_graph.add_edge(n1, n2, weight=3.0)

    # Use spring layout with the weighted graph
    if hasattr(data, "pos") and data.pos is not None:
        init_pos = {i: data.pos[i].cpu().numpy() for i in range(N)}
        pos = nx.spring_layout(
            pattern_graph,
            pos=init_pos,
            weight="weight",
            iterations=10,
            k=1.5 / np.sqrt(N),
        )
    else:
        pos = nx.spring_layout(
            pattern_graph, weight="weight", iterations=10, k=1.5 / np.sqrt(N)
        )

    return colors, pos


def plot_structural_with_patterns(
    data: Data,
    C=None,
    ax=None,
    xlim=None,
    ylim=None,
):
    """
    Plot the graph structure with nodes colored by pattern participation.

    Args:
        data: PyTorch Geometric Data object
        C: Optional coarsening matrix (nodes being merged get dashed borders)
        ax: Matplotlib axis
        reset_pos: Whether to recompute positions
        xlim, ylim: Axis limits
        alert_patterns: dict mapping pattern_id -> list of node indices
        normal_patterns: dict mapping pattern_id -> list of node indices
        alert_types: dict mapping pattern_id -> pattern type string
        normal_types: dict mapping pattern_id -> pattern type string
        node_to_supernode: Mapping from original nodes to current super nodes
    """
    N = int(data.num_nodes)
    # alert_patterns = alert_patterns or {}
    # normal_patterns = normal_patterns or {}

    # # Map patterns to current graph level if node_to_supernode is provided
    # if node_to_supernode is not None:
    #     mapped_alert = {}
    #     for pid, nodes in alert_patterns.items():
    #         supernodes = set()
    #         for n in nodes:
    #             if n < len(node_to_supernode):
    #                 sn = node_to_supernode[n].item()
    #                 if sn < N:
    #                     supernodes.add(sn)
    #         if supernodes:
    #             mapped_alert[pid] = list(supernodes)
    #     alert_patterns = mapped_alert

    #     mapped_normal = {}
    #     for pid, nodes in normal_patterns.items():
    #         supernodes = set()
    #         for n in nodes:
    #             if n < len(node_to_supernode):
    #                 sn = node_to_supernode[n].item()
    #                 if sn < N:
    #                     supernodes.add(sn)
    #         if supernodes:
    #             mapped_normal[pid] = list(supernodes)
    #     normal_patterns = mapped_normal

    # Compute colors and positions based on patterns
    # if (
    #     reset_pos
    #     or not hasattr(data, "pattern_colors")
    #     or not hasattr(data, "pattern_pos")
    # ):
    #     # colors, pos_dict = get_pattern_colors_and_positions(
    #     #     data, alert_patterns, normal_patterns, alert_types, normal_types
    #     # )
    #     # data.colors = torch.tensor(colors, dtype=torch.float32, device=device)
    #     # pos_array = np.array([pos_dict[i] for i in range(N)])
    #     # data.pattern_pos = torch.tensor(pos_array, dtype=torch.float32, device=device)
    # else:
    #     colors = data.colors.cpu().numpy()

    # Get edges
    ei = data.edge_index.cpu().numpy()
    edge_set = set()
    for i in range(ei.shape[1]):
        u = int(ei[0, i])
        v = int(ei[1, i])
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        edge_set.add(key)
    edges = list(edge_set)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Define border colors for merged nodes
    border_colors = ["white"] * N
    border_widths = [0.5] * N

    if C is not None:
        merged_nodes = []
        rows, cols = C.indices()
        for r in range(C.size(0)):
            mask = rows == r
            col_indices = cols[mask].tolist()
            if len(col_indices) > 1:
                merged_nodes.append(col_indices)

        merge_cmap = [
            "black",
            "darkblue",
            "darkgreen",
            "darkred",
            "darkorange",
            "darkmagenta",
            "darkcyan",
            "saddlebrown",
            "navy",
            "olive",
        ]

        for g_idx, merged_group in enumerate(merged_nodes):
            for node in merged_group:
                if node < N:
                    border_colors[node] = merge_cmap[g_idx % len(merge_cmap)]
                    border_widths[node] = 2.0

    # Plot edges first
    for u, v in edges:
        xu, yu = data.pos[u].cpu().numpy()
        xv, yv = data.pos[v].cpu().numpy()
        ax.plot([xu, xv], [yu, yv], color="#aaaaaa", alpha=0.4, linewidth=0.6, zorder=1)

    # Plot nodes
    colors_np = data.colors.cpu().numpy()
    for i in range(N):
        xy = data.pos[i].cpu().numpy()
        ax.scatter(
            xy[0],
            xy[1],
            s=100,
            color=tuple(colors_np[i]),
            edgecolors=border_colors[i],
            linewidths=border_widths[i],
            zorder=2,
            linestyle="--" if border_widths[i] > 1 else "-",
        )

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_aspect("equal")
    ax.set_axis_off()

    return ax


def make_gif_with_patterns(
    Gall,
    gif_path="animation_patterns.gif",
    frame_duration=60,
):
    """
    Make a gif animation of the coarsening process with pattern-based coloring.

    Args:
        Gall: List of PyTorch Geometric Data objects at each coarsening level
        alert_patterns: dict mapping pattern_id -> list of original node indices
        normal_patterns: dict mapping pattern_id -> list of original node indices
        alert_types: dict mapping pattern_id -> pattern type string
        normal_types: dict mapping pattern_id -> pattern type string
        reset_pos: Whether to recompute positions at each level
        gif_path: Output path for the gif
        frame_duration: Duration per frame in ms
    """

    frames = []

    # Compute initial layout and limits from first graph
    G0 = Gall[0].to("cpu")

    # Get limits from initial layout
    xlim = (G0.pos[:, 0].min() - 0.1, G0.pos[:, 0].max() + 0.1)
    ylim = (G0.pos[:, 1].min() - 0.1, G0.pos[:, 1].max() + 0.1)

    # Track cumulative coarsening for pattern mapping
    for i in range(len(Gall)):
        G = Gall[i].to("cpu")
        C = Gall[i + 1].C.coalesce() if i < len(Gall) - 1 else None

        fig, ax = plt.subplots(figsize=(8, 8))

        plot_structural_with_patterns(
            G,
            C,
            ax=ax,
            xlim=xlim,
            ylim=ylim,
        )

        ax.set_title(
            f"Level {i}: {G.num_nodes} nodes, {G.num_edges} edges", fontsize=12
        )
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close(fig)

        buf.seek(0)
        frames.append(Image.open(buf))

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=frame_duration * len(Gall),
        loop=0,
    )
    print(f"Saved pattern-colored GIF to {gif_path}")


def create_pattern_legend(alert_types: dict = None, normal_types: dict = None, ax=None):
    """Create a legend showing pattern type colors."""
    from matplotlib.patches import Patch

    alert_type_colors = {
        "fan_out": "#E63333",
        "fan_in": "#E6801A",
        "cycle": "#CC1A80",
        "bipartite": "#991ACC",
        "stack": "#E6CC1A",
        "scatter_gather": "#FF6666",
        "gather_scatter": "#CC4D00",
    }
    normal_type_colors = {
        "single": "#3399E6",
        "fan_out": "#33CC99",
        "fan_in": "#4DE666",
        "forward": "#6666E6",
        "mutual": "#1AB3B3",
        "periodical": "#80CC33",
    }

    legend_elements = []

    if alert_types:
        unique_alert_types = set(alert_types.values())
        for ptype in sorted(unique_alert_types):
            color = alert_type_colors.get(ptype, "#E63333")
            legend_elements.append(
                Patch(facecolor=color, edgecolor="black", label=f"Alert: {ptype}")
            )

    if normal_types:
        unique_normal_types = set(normal_types.values())
        for ptype in sorted(unique_normal_types):
            color = normal_type_colors.get(ptype, "#3399E6")
            legend_elements.append(
                Patch(facecolor=color, edgecolor="black", label=f"Normal: {ptype}")
            )

    legend_elements.append(
        Patch(facecolor="#B3B3B3", edgecolor="black", label="No pattern")
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 4))
        ax.axis("off")

    ax.legend(handles=legend_elements, loc="center", frameon=True, fontsize=9)
    return ax


def save_pattern_graph_with_legend(
    data: Data,
    alert_types: dict = None,
    normal_types: dict = None,
    save_path: str = "graph_patterns.png",
    title: str = None,
):
    """
    Save a static figure of the graph with pattern-based coloring and a legend.

    Args:
        data: PyTorch Geometric Data object
        alert_types: dict mapping pattern_id -> pattern type string
        normal_types: dict mapping pattern_id -> pattern type string
        save_path: Path to save the figure
        title: Optional title for the figure
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(1, 2, width_ratios=[4, 1], figure=fig)

    ax_graph = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])

    # Plot graph
    plot_structural_with_patterns(
        data.to("cpu"),
        ax=ax_graph,
    )

    if title:
        ax_graph.set_title(title, fontsize=14)
    else:
        ax_graph.set_title(
            f"Graph: {data.num_nodes} nodes, {data.num_edges} edges", fontsize=14
        )

    # Create legend
    create_pattern_legend(alert_types, normal_types, ax=ax_legend)
    ax_legend.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved pattern graph with legend to {save_path}")


def plot_structural_only(
    data: Data, C=None, ax=None, reset_pos=False, xlim=None, ylim=None
):
    """
    Plot the graph structure only, without node labels.
    If C is provided, it is a coarsening matrix, and nodes that are merged
    will be highlighted with a border color.
    """

    N = int(data.num_nodes)

    if reset_pos or not hasattr(data, "pos"):
        GG = to_networkx(data, to_undirected=True)
        pos = pos = data.pos.numpy()
        pos = {i: pos[i] for i in range(pos.shape[0])}
        pos = nx.spring_layout(GG, pos=pos)
        pos = [vals for vals in pos.values()]
        data.pos = torch.tensor(pos, dtype=torch.float32, device=device)

    if not hasattr(data, "colors"):
        L_dense = data.L.to_dense().cpu().numpy()
        _, V = np.linalg.eigh(L_dense)
        colors = V[:, :3]
        colors = (colors - np.mean(colors, axis=0)) / (np.std(colors, axis=0) + 1e-8)
        colors = (colors - colors.min(axis=0)) / (
            colors.max(axis=0) - colors.min(axis=0) + 1e-8
        )
        colors = colors.clip(0, 0.9)
        data.colors = torch.tensor(colors, device=device)

    ei = data.edge_index.cpu().numpy()
    edge_set = set()
    for i in range(ei.shape[1]):
        u = int(ei[0, i])
        v = int(ei[1, i])
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        edge_set.add(key)
    edges = list(edge_set)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # define border colors for merged nodes
    from matplotlib import cm

    border_colors = ["white" for _ in range(N)]

    if C is not None:
        merged_nodes = []
        rows, cols = C.indices()
        for r in range(C.size(0)):
            mask = rows == r
            col_indices = cols[mask].tolist()
            if len(col_indices) > 1:
                merged_nodes.append(col_indices)

        cmap = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "brown",
            "magenta",
            "cyan",
            "lime",
            "black",
        ]

        # print(f"{N=}, {C.shape=}, {len(merged_nodes)=}, {len(border_colors)=}, {len(colors)=}, {len(group_colors)=}")

        for g_idx, merged_group in enumerate(merged_nodes):
            for node in merged_group:
                border_colors[node] = cmap[g_idx % len(cmap)]

    # plot nodes
    for i in range(N):
        xy = data.pos[i].cpu().numpy()
        if border_colors[i] == "white":
            # print(f'{data.colors[i]=}')
            ax.scatter(
                xy[0],
                xy[1],
                s=120,
                color=tuple(data.colors[i].cpu().numpy()),
                edgecolors=border_colors[i],
                zorder=2,
            )
        else:
            ax.scatter(
                xy[0],
                xy[1],
                s=120,
                color=tuple(data.colors[i].cpu().numpy()),
                edgecolors=border_colors[i],
                linewidths=1.2,
                zorder=2,
                linestyle="--",
            )
        # Add node ID label
        ax.text(
            xy[0],
            xy[1],
            str(i),
            fontsize=8,
            ha="center",
            va="center",
            color="black",
            fontweight="bold",
            zorder=3,
        )

    # plot edges
    for u, v in edges:
        xu, yu = data.pos[u].cpu().numpy()
        xv, yv = data.pos[v].cpu().numpy()
        ax.plot([xu, xv], [yu, yv], color="#777777", alpha=0.6, linewidth=0.8, zorder=1)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    # ax.set_ylim(-1.1, 1.1)
    # ax.set_xlim(-.1, 1.1)
    # ax.set_ylim(-.1, 1.1)
    ax.axhline(0, color="#999999", linewidth=0.8, alpha=0.6, linestyle="--")
    ax.axvline(0, color="#999999", linewidth=0.8, alpha=0.6, linestyle="--")
    ax.set_aspect("equal")
    ax.set_axis_on()


def plot_Gall_structural_only(Gall, iCs=[], ncols=3, reset_pos=False):
    """
    Plot all graphs (in Gall) in a grid layout.
    Note: when ax has only 1 row, its shape is (ncols,), otw it is (nrows, ncols).
    """
    levels = len(Gall) - 1
    nrows = (levels + ncols) // ncols
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

    if nrows == 1:
        ax = [ax]  # make 2D for consistency
    for i in range(len(Gall)):
        G = Gall[i].to("cpu")
        C = iCs[i].coalesce() if i < len(iCs) else None
        plot_structural_only(G, C, ax=ax[i // ncols][i % ncols], reset_pos=reset_pos)
    plt.show()


def make_gif(Gall, reset_pos=False, gif_path="animation.gif", frame_duration=60):
    """
    Make a gif animation of the coarsening process.
    """

    xlim = Gall[0].pos[:, 0].min().item() - 0.1, Gall[0].pos[:, 0].max().item() + 0.1
    ylim = Gall[0].pos[:, 1].min().item() - 0.1, Gall[0].pos[:, 1].max().item() + 0.1
    frames = []
    for i in range(len(Gall)):
        G = Gall[i].to("cpu")
        C = Gall[i + 1].C.coalesce() if i < len(Gall) - 1 else None

        fig, ax = plt.subplots(figsize=(6, 6))
        plot_structural_only(G, C, ax=ax, reset_pos=reset_pos, xlim=xlim, ylim=ylim)
        plt.tight_layout()
        plt.savefig(f"{save_path}temp_frame{i}.png", format="png")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)

        buf.seek(0)
        frames.append(Image.open(buf))

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=frame_duration * len(Gall),
        loop=0,
    )
