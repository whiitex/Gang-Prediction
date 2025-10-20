import os
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import io
from PIL import Image

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from graph_utils import *

import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_structural_only(data: Data, C=None, ax=None, reset_pos=False):
    '''
    Plot the graph structure only, without node labels.
    If C is provided, it is a coarsening matrix, and nodes that are merged
    will be highlighted with a border color.
    '''

    N = int(data.num_nodes)

    if reset_pos or not hasattr(data, 'pos'):
        GG = to_networkx(data, to_undirected=True)
        pos = nx.spring_layout(GG)
        pos = [vals for vals in pos.values()]
        data.pos = torch.tensor(pos, dtype=torch.float32, device=device)

    if not hasattr(data, 'colors'):
        L_dense = data.L.to_dense().cpu().numpy()
        _, V = np.linalg.eigh(L_dense)
        colors = V[:,:3]
        colors = (colors - np.mean(colors, axis=0)) / (np.std(colors, axis=0) + 1e-8)
        colors = (colors - colors.min(axis=0)) / (colors.max(axis=0) - colors.min(axis=0) + 1e-8)
        colors = colors.clip(0, .9)
        data.colors = torch.tensor(colors, device=device)


    ei = data.edge_index.cpu().numpy()
    edge_set = set()
    for i in range(ei.shape[1]):
        u = int(ei[0, i]); v = int(ei[1, i])
        if u == v: continue
        key = (min(u, v), max(u, v))
        edge_set.add(key)
    edges = list(edge_set)

    if ax is None: fig, ax = plt.subplots(figsize=(8, 5))

    # define border colors for merged nodes
    from matplotlib import cm

    border_colors = ['white' for _ in range(N)]

    if C is not None:
        merged_nodes = []
        rows, cols = C.indices()
        for r in range(C.size(0)):
            mask = rows == r
            col_indices = cols[mask].tolist()
            if len(col_indices) > 1:
                merged_nodes.append(col_indices)

        cmap = [
            "red", "blue", "green", "orange", "purple", "brown",
            "magenta", "cyan", "lime", "black"
        ]

        # print(f"{N=}, {C.shape=}, {len(merged_nodes)=}, {len(border_colors)=}, {len(colors)=}, {len(group_colors)=}")

        for g_idx, merged_group in enumerate(merged_nodes):
            for node in merged_group:
                border_colors[node] = cmap[g_idx % len(cmap)]
    
    # plot nodes
    for i in range(N):
        xy = data.pos[i].cpu().numpy()
        if border_colors[i] == 'white':
            # print(f'{data.colors[i]=}')
            ax.scatter(xy[0], xy[1], s=120, color=tuple(data.colors[i].cpu().numpy()), edgecolors=border_colors[i], zorder=2)
        else:
            ax.scatter(xy[0], xy[1], s=120, color=tuple(data.colors[i].cpu().numpy()), edgecolors=border_colors[i], linewidths=1.2, zorder=2, linestyle='--')

    # plot edges
    for (u, v) in edges:
        xu, yu = data.pos[u].cpu().numpy()
        xv, yv = data.pos[v].cpu().numpy()
        ax.plot([xu, xv], [yu, yv], color="#777777", alpha=0.6, linewidth=0.8, zorder=1)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    # ax.set_xlim(-.1, 1.1)
    # ax.set_ylim(-.1, 1.1)
    ax.axhline(0, color="#999999", linewidth=0.8, alpha=0.6, linestyle='--')
    ax.axvline(0, color="#999999", linewidth=0.8, alpha=0.6, linestyle='--')
    ax.set_aspect('equal')
    ax.set_axis_on()


def plot_Gall_structural_only(Gall, iCs=[], ncols=3, reset_pos=False):
    '''
    Plot all graphs (in Gall) in a grid layout.
    Note: when ax has only 1 row, its shape is (ncols,), otw it is (nrows, ncols).
    '''
    levels = len(Gall) - 1
    nrows = (levels + ncols) // ncols
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

    if nrows == 1: ax = [ax] # make 2D for consistency
    for i in range(len(Gall)):
        G = Gall[i].to('cpu')
        C = iCs[i].coalesce() if i < len(iCs) else None
        plot_structural_only(G, C, ax=ax[i // ncols][i % ncols], reset_pos=reset_pos)
    plt.show()


def make_gif(Gall, iCs=[], reset_pos=False, gif_path="animation.gif", frame_duration=60):
    '''
    Make a gif animation of the coarsening process.
    '''
    frames = []
    for i in range(len(Gall)):
        G = Gall[i].to('cpu')
        C = iCs[i].coalesce() if i < len(iCs) else None

        fig, ax = plt.subplots(figsize=(6,6))
        plot_structural_only(G, C, ax=ax, reset_pos=reset_pos)
        plt.tight_layout()

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