r"""Loukas restricted-spectral-approximation (RSA) coarsening — from scratch.

Faithful, self-contained re-implementation of the *local variation* coarsening
algorithm of

    A. Loukas, "Graph Reduction with Spectral and Cut Guarantees",
    JMLR 20 (2019) 1-42.   (paper "18-680")

Only numpy / scipy.sparse are used so the code can be checked line-by-line
against the paper.  The pieces implemented here are:

* Laplacian-consistent coarsening matrices  P_l  (Definition 5, Proposition 7).
* Reduction / lifting  x_c = P x,  x_tilde = P^+ x_c,  L_c = P^{-T} L P^+
  (Scheme 1 of Section 2.1; Proposition 6 "easy inversion").
* The edge-based *local variation cost* of a contraction set (Eq. 6, Section 4.3).
* ``Algorithm 2`` — single-level coarsening by local variation.
* ``Algorithm 1`` — multi-level greedy coarsening driven by the (R, eps) bound
  eps <= prod_l (1 + sigma_l) - 1   (Proposition 17).

The *target subspace* ``R`` is supplied as an arbitrary basis matrix ``B0``
(N x k).  Two natural choices are used by the experiments:

* ``B0 = U_K`` — the K lowest-frequency Laplacian eigenvectors  ->  the classic
  unsupervised "top-K eigenvector" target.
* ``B0 = g_theta(A_hat) X`` — the trainable-SGC target of ``sgc_subspace`` (the
  algorithm of Section 12 of the companion note).

Because the per-level cost only depends on ``span(B0)`` (it is computed from the
L-orthonormalisation ``A = B (B^T L B)^{-1/2}``, which is invariant to a change of
basis), any basis of R may be passed in directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import torch


# ---------------------------------------------------------------------------
# Graph algebra
# ---------------------------------------------------------------------------
def build_laplacian(W: sp.spmatrix):
    """Combinatorial Laplacian ``L = D - W`` and weighted degree vector."""
    W = sp.csr_matrix(W)
    deg = np.asarray(W.sum(axis=1)).ravel()
    L = (sp.diags(deg) - W).tocsr()
    return L, deg


def coarsening_matrix(partition: Sequence[Sequence[int]], N: int):
    r"""Laplacian-consistent coarsening matrix ``P`` (Definition 5 / Prop. 7).

    ``partition`` is a list of contraction sets that together cover ``0..N-1``.
    Following Proposition 7,  ``P[r, i] = 1 / |C_r|`` for ``i in C_r`` and the
    pseudo-inverse (lift) ``P^+[i, r] = 1`` for ``i in C_r`` (equal-valued
    non-zeros -> the reduction stays a combinatorial Laplacian).
    """
    rows, cols, vals = [], [], []
    lift_rows, lift_cols = [], []
    for r, cset in enumerate(partition):
        inv = 1.0 / len(cset)
        for i in cset:
            rows.append(r)
            cols.append(int(i))
            vals.append(inv)
            lift_rows.append(int(i))
            lift_cols.append(r)
    n = len(partition)
    P = sp.csr_matrix((vals, (rows, cols)), shape=(n, N))
    Pinv = sp.csr_matrix(  # P^+  (membership / lift matrix)
        (np.ones(len(lift_rows)), (lift_rows, lift_cols)), shape=(N, n)
    )
    return P, Pinv


def reduce_laplacian(L: sp.spmatrix, Pinv: sp.spmatrix):
    r"""``L_c = (P^+)^T L P^+`` — Laplacian-consistent reduction (Property 8).

    With the equal-valued lift of :func:`coarsening_matrix` this contracts each
    set to a single vertex and sums the cut weights between sets, so ``L_c`` is
    again a combinatorial Laplacian.
    """
    Lc = (Pinv.T @ L @ Pinv).tocsr()
    return Lc


# ---------------------------------------------------------------------------
# Local variation cost (Section 4.2 - 4.3)
# ---------------------------------------------------------------------------
def _l_orthonormalize(B: np.ndarray, L: sp.spmatrix) -> np.ndarray:
    r"""Return ``A = B (B^T L B)^{-1/2}`` so that ``A^T L A = I`` and
    ``span(A) = span(B)``.

    This is the matrix ``A_{l-1}`` of the paper (Section 4.1): for the eigenspace
    target ``B = U_K`` it reduces to ``A = U_K Lambda_K^{-1/2}`` (the remark after
    Proposition 17).  The local-variation cost below only depends on ``span(B)``
    through this object.
    """
    M = B.T @ (L @ B)
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    inv_sqrt = np.zeros_like(w)
    nz = w > 1e-12
    inv_sqrt[nz] = w[nz] ** -0.5
    return B @ (V @ np.diag(inv_sqrt) @ V.T)


def _edge_costs(edge_index: np.ndarray, deg: np.ndarray, A: np.ndarray):
    r"""Edge-based local variation cost for every candidate edge.

    For a two-vertex contraction set ``C = {i, j}`` the cost (Eq. 6 with
    ``|C| - 1 = 1``) evaluates in closed form to

        cost(i, j) = || Pi_C^perp A ||^2_{L_C}
                   = 1/4 (deg_i + deg_j)^2 || A_i - A_j ||^4 .
    """
    i, j = edge_index[0], edge_index[1]
    diff = A[i] - A[j]
    diff_sq = np.einsum("ek,ek->e", diff, diff)
    deg_sum = deg[i] + deg[j]
    return 0.25 * (deg_sum**2) * (diff_sq**2)


# ---------------------------------------------------------------------------
# Algorithm 2 — single level coarsening by local variation (edge family)
# ---------------------------------------------------------------------------
def coarsen_one_level(
    L: sp.spmatrix,
    deg: np.ndarray,
    B: np.ndarray,
    n_target: int,
    sigma_max: float,
):
    """One level of edge-based local-variation coarsening.

    Returns ``(partition, P, Pinv, sigma_l)`` where ``partition`` covers every
    current vertex (unmatched vertices become singletons) and ``sigma_l`` is the
    realised level variation cost ``sqrt(sum_C cost(C))``.
    """
    N = L.shape[0]
    A = _l_orthonormalize(B, L)

    # candidate family F_l = one set per (undirected) edge of the current graph.
    Lu = sp.triu(L, k=1).tocoo()
    edge_index = np.vstack([Lu.row, Lu.col])
    if edge_index.shape[1] == 0:
        partition = [[v] for v in range(N)]
        P, Pinv = coarsening_matrix(partition, N)
        return partition, P, Pinv, 0.0

    costs = _edge_costs(edge_index, deg, A)
    order = np.argsort(costs, kind="stable")  # increasing cost (Algorithm 2:4)

    marked = np.zeros(N, dtype=bool)
    matched: List[List[int]] = []
    n_cur = N
    sigma_sq = 0.0
    sigma_max_sq = sigma_max**2

    for e in order:
        if n_cur <= n_target:
            break
        i = int(edge_index[0, e])
        j = int(edge_index[1, e])
        c = float(costs[e])
        # stop once the level error budget would be exceeded (Algorithm 2:7).
        if sigma_sq + c > sigma_max_sq:
            break
        if marked[i] or marked[j]:
            continue
        marked[i] = marked[j] = True
        matched.append([i, j])
        sigma_sq += c
        n_cur -= 1

    partition = list(matched)
    partition.extend([v] for v in np.nonzero(~marked)[0])

    P, Pinv = coarsening_matrix(partition, N)
    return partition, P, Pinv, float(np.sqrt(sigma_sq))


# ---------------------------------------------------------------------------
# Algorithm 1 — multi-level coarsening
# ---------------------------------------------------------------------------
@dataclass
class CoarseningResult:
    """Output of :func:`loukas_coarsen`."""

    node_to_supernode: np.ndarray  # (N_orig,) original vertex -> final super-node
    n_coarse: int
    n_original: int
    epsilon: float  # cumulative (R, eps) guarantee (Proposition 17)
    Lc: sp.spmatrix  # final coarse Laplacian
    sigmas: List[float] = field(default_factory=list)
    sizes: List[int] = field(default_factory=list)  # |V_l| per level

    @property
    def reduction(self) -> float:
        return 1.0 - self.n_coarse / self.n_original


def loukas_coarsen(
    L: sp.spmatrix,
    B0: np.ndarray,
    reduction: float = 0.5,
    epsilon: float = np.inf,
    max_levels: int = 30,
    min_reduce_per_level: int = 1,
) -> CoarseningResult:
    r"""Multi-level local-variation coarsening (Algorithm 1).

    Parameters
    ----------
    L : (N, N) sparse combinatorial Laplacian.
    B0 : (N, k) basis of the target subspace ``R``.
    reduction : desired ``r = 1 - n/N`` (target coarse size ``n = (1-r) N``).
    epsilon : restricted-spectral-approximation budget ``eps'``.  The realised
        guarantee satisfies ``eps <= prod_l (1 + sigma_l) - 1 <= epsilon``.
    max_levels : safety cap on the number of levels.

    Returns
    -------
    CoarseningResult with the original-vertex -> super-node map and diagnostics.
    """
    N0 = L.shape[0]
    n_target = max(1, int(round((1.0 - reduction) * N0)))

    L_cur = sp.csr_matrix(L)
    _, deg = build_laplacian(_adj_from_laplacian(L_cur))
    B = np.asarray(B0, dtype=float)

    # global map original vertex -> current super-node (identity at start).
    node_to_super = np.arange(N0)

    eps_cur = 0.0
    sigmas: List[float] = []
    sizes = [N0]

    for _ in range(max_levels):
        N_cur = L_cur.shape[0]
        if N_cur <= n_target:
            break
        if eps_cur >= epsilon:
            break

        # per-level budget (Algorithm 1, line 5).
        sigma_max = (1.0 + epsilon) / (1.0 + eps_cur) - 1.0

        partition, P, Pinv, sigma_l = coarsen_one_level(
            L_cur, deg, B, n_target, sigma_max
        )

        n_new = len(partition)
        if N_cur - n_new < min_reduce_per_level:
            break  # no useful contraction possible within the budget.

        # advance the global map: current super-node -> new super-node.
        cur_to_new = np.empty(N_cur, dtype=int)
        for r, cset in enumerate(partition):
            for v in cset:
                cur_to_new[v] = r
        node_to_super = cur_to_new[node_to_super]

        # reduce graph and project the basis (B_l = P_l B_{l-1}).
        L_cur = reduce_laplacian(L_cur, Pinv)
        _, deg = build_laplacian(_adj_from_laplacian(L_cur))
        B = P @ B

        eps_cur = (1.0 + eps_cur) * (1.0 + sigma_l) - 1.0
        sigmas.append(sigma_l)
        sizes.append(L_cur.shape[0])

    # relabel super-nodes to a dense 0..n-1 range.
    _, node_to_super = np.unique(node_to_super, return_inverse=True)

    return CoarseningResult(
        node_to_supernode=node_to_super,
        n_coarse=int(node_to_super.max() + 1) if N0 else 0,
        n_original=N0,
        epsilon=float(eps_cur),
        Lc=L_cur,
        sigmas=sigmas,
        sizes=sizes,
    )


def _adj_from_laplacian(L: sp.spmatrix) -> sp.spmatrix:
    """Recover the (non-negative) adjacency ``W`` from a Laplacian ``L``."""
    L = sp.csr_matrix(L)
    W = -L.copy()
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W


# ---------------------------------------------------------------------------
# Eigenvector target (baseline: span(U_K))
# ---------------------------------------------------------------------------
def bottom_k_eigenvectors(L: sp.spmatrix, K: int) -> np.ndarray:
    r"""The ``K`` lowest-frequency Laplacian eigenvectors ``U_K`` (orthonormal).

    This is the classic *unsupervised* coarsening target ``R = span(U_K)`` used
    as the baseline against which the trainable-SGC subspace is compared.  Uses
    shift-invert ARPACK on the lightly-regularised ``L`` to handle the (large)
    null space of a disconnected transaction graph robustly.
    """
    from scipy.sparse.linalg import eigsh

    N = L.shape[0]
    K = int(min(K, N - 1))
    Lreg = (L + 1e-8 * sp.eye(N)).tocsc()
    try:
        _, U = eigsh(Lreg, k=K, sigma=0.0, which="LM")
    except Exception:
        _, U = eigsh(Lreg, k=K, which="SM")
    return np.asarray(U)


def dominant_eigenvectors(L: torch.Tensor, K: int) -> torch.Tensor:
    r"""The dominant eigenvector of the Laplacian (orthonormal).

    This is a simple heuristic target that can be used instead of the trainable-SGC
    subspace when only a single dimension is desired.  It is the closest 1D subspace
    to the top-K eigenvector target, and can be computed efficiently by power
    iteration on the adjacency matrix.
    """

    lk, Uk = torch.lobpcg(L, k=K, largest=False, tol=1e-6)

    return lk, Uk
