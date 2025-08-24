from datetime import datetime
import os
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import (
    to_undirected,
    remove_self_loops,
    add_self_loops,
    scatter,
)
from pygsp import graphs
import scipy as sp
from logger import getLOGGER

now = datetime.now().strftime("%Y%m%d_%H%M%S")

result_path = "results/"
save_path = f"{result_path}{now}/"

os.makedirs(save_path, exist_ok=True)
LOGGER = getLOGGER(
    name=f"{now}_Cora",
    log_on_file=True,
    save_path=save_path,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(1)
np.random.seed(1)


def create_pyg_data(features, edges_idx, labels) -> Data:
    x = torch.FloatTensor(features.astype(np.float32))
    y = torch.LongTensor(labels)
    edge_index = torch.LongTensor(edges_idx.T)  # expects (2, num_edges)
    edge_index = to_undirected(edge_index)

    return Data(x=x, edge_index=edge_index, y=y)


def create_pygsp_graph(data: Data) -> graphs.Graph:
    # adjacency matrix
    adj_matrix = sp.sparse.coo_matrix(
        (np.ones(len(data.edge_index)), (data.edge_index[0], data.edge_index[1])),
        shape=(data.num_nodes, data.num_nodes),
    )
    # adj_matrix = adj_matrix + adj_matrix.T
    # adj_matrix.data = np.ones(len(adj_matrix.data))

    # pygsp graph
    G = graphs.Graph(adj_matrix)
    G.compute_laplacian()
    G.set_coordinates()

    return G


def create_train_val_test_split(num_nodes, train_ratio=0.6, val_ratio=0.2):
    indices = torch.randperm(num_nodes)
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    return train_idx, val_idx, test_idx


def degree(edge_index, num_nodes, edge_weights=None):
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    if edge_weights is not None:
        for i, j, w in zip(edge_index[0], edge_index[1], edge_weights):
            deg[i] += w
            deg[j] += w
    else:
        for i, j in zip(edge_index[0], edge_index[1]):
            deg[i] += 1
            deg[j] += 1
    return deg


def sparse_eye(size):
    indices = torch.arange(size).repeat(2, 1)
    values = torch.ones(size)
    C = torch.sparse_coo_tensor(indices, values, (size, size))
    return C


def graph_params(G: Data):
    num_nodes = G.num_nodes
    edge_index, edge_weight = remove_self_loops(G.edge_index, G.edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1), dtype=torch.float32, device=edge_index.device
        )
    W = torch.sparse_coo_tensor(
        G.edge_index,
        G.edge_weight,
        size=(num_nodes, num_nodes),
        device=edge_index.device,
    )

    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce="sum")

    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_weight = torch.cat([-edge_weight, deg], dim=0)

    L = torch.sparse_coo_tensor(
        edge_index, edge_weight, size=(num_nodes, num_nodes), device=edge_index.device
    )

    return W, L, deg


def create_P(indices, n, device="cpu"):
    n_t = len(indices)
    P_indices = torch.stack(
        [
            indices,  # Column indices
            torch.arange(n_t, device=device),  # Row indices
        ]
    )
    P_values = torch.ones(n_t, device=device)
    P = torch.sparse_coo_tensor(P_indices, P_values, (n, n_t), device=device)
    return P


import torch


def _spmm(A, B):
    """Sparse x dense matrix multiply that works for COO/CSR/BSR."""
    if B.dim() == 1:  # (n,) -> (n,1)
        return (A @ B.unsqueeze(1)).squeeze(1)
    return A @ B


def _cg_multi(A_mv, B, tol=1e-6, maxiter=None, M=None):
    """
    Solve A X = B for multiple RHS (columns of B) with (preconditioned) CG.
    A_mv: callable(V) -> A @ V
    B: [m, d] dense
    M: optional preconditioner callable(V) ≈ A^{-1} V (e.g., Jacobi)
    """
    m, d = B.shape
    X = torch.zeros_like(B)
    R = B - A_mv(X)
    Z = M(R) if M is not None else R
    P = Z.clone()

    # Per-column inner products
    def col_dot(U, V):  # returns [d]
        return (U * V).sum(dim=0)

    rz_old = col_dot(R, Z)  # [d]
    normB = B.norm(dim=0)  # [d]
    active = normB > 0
    if maxiter is None:
        maxiter = 2 * m  # simple cap

    for _ in range(maxiter):
        AP = A_mv(P)  # [m, d]
        denom = col_dot(P, AP).clamp_min(1e-30)  # [d]
        alpha = torch.zeros_like(denom)
        alpha[active] = rz_old[active] / denom[active]
        X = X + P * alpha.unsqueeze(0)
        R = R - AP * alpha.unsqueeze(0)

        # Convergence (per column)
        done = R.norm(dim=0) <= tol * normB
        new_active = active & (~done)
        if not new_active.any():
            break
        active = new_active

        Z = M(R) if M is not None else R
        rz_new = col_dot(R, Z)
        beta = torch.zeros_like(rz_new)
        nz = rz_old.abs() > 0
        beta[nz] = rz_new[nz] / rz_old[nz]
        P = Z + P * beta.unsqueeze(0)
        rz_old = rz_new
    return X


def min_norm_lstsq_sparse_multi(
    C, Y, lam=0.0, tol=1e-6, maxiter=None, precondition=True
):
    """
    Solve min_X ||Y - C X||_F^2 with sparse C (m<n) via dual CG on (C C^T + lam I) Z = Y.
    Returns the (approx.) minimum-norm solution X = C^T Z (ridge if lam>0).

    C:   sparse tensor [m, n] (COO/CSR/BSR)
    Y:   dense tensor  [m, d]
    lam: Tikhonov damping (>=0) for stability
    """
    assert C.layout in (
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_bsr,
    ), "C must be sparse"
    m, n = C.shape
    assert Y.shape[0] == m

    Ct = C.transpose(0, 1)

    def A_mv(V):  # V: [m, d]
        out = _spmm(C, _spmm(Ct, V))
        return out if lam == 0.0 else out + lam * V

    # Jacobi preconditioner M ≈ (C C^T + lam I)^{-1}
    if precondition:
        Cc = (
            C.coalesce()
            if C.layout == torch.sparse_coo
            else C.to_sparse_coo().coalesce()
        )
        rows = Cc.indices()[0]
        diag = torch.zeros(m, dtype=Cc.dtype, device=Cc.device)
        diag.scatter_add_(0, rows, Cc.values() ** 2)
        if lam != 0.0:
            diag = diag + lam
        diag = diag.clamp_min(1e-12)
        M = lambda V: V / diag.unsqueeze(1)  # row-wise scaling
    else:
        M = None

    # Solve for Z, then recover X
    Z = _cg_multi(A_mv, Y, tol=tol, maxiter=maxiter, M=M)  # [m, d]
    X = _spmm(Ct, Z)  # [n, d]
    return X
