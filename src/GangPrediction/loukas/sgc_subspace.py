r"""Trainable-SGC coarsening subspace — from scratch.

Implementation of Sections 7-12 of the companion note

    "Spectral Energy of Planted Gangs: Top-K Eigenvectors, SGC, and Their
     Dependence on Motif Size and Repetition"   (paper "Graph_Coarsening").

The note replaces the eigendecomposition target ``R = span(U_K)`` by an
eigendecomposition-free target obtained from a *learnable polynomial filter* of
the symmetric normalised adjacency

    A_hat = D_tilde^{-1/2} (A + I) D_tilde^{-1/2}.

Key objects implemented here (with the equation numbers of the note):

* ``normalized_adjacency``         -- A_hat                                  (Eq. 2)
* ``indicator`` / ``normalized_conductance`` -- v_S = 1_S/sqrt(s), phi_N    (Def. 1 / Thm 2)
* ``moment_matrix``                -- Hankel moment matrix M, M[k,l]=m_hat_{k+l} (Def. 6)
*                                     by 2K sparse mat-vecs, NO eigendecomposition (Sec. 12.3)
* ``retained_energy``              -- C(theta) = theta^T M theta             (Thm. 3)
* ``fit_filter``                   -- closed-form theta* = top eigvec of Delta M (Eq. 34 / Thm. 5)
* ``build_subspace``               -- Z_theta = g_theta(A_hat) X            (Eq. 26, step 5)

The whole "Training procedure for better motif detection" (Section 12) is the
function :func:`train_sgc_target`, which ties the steps together and returns the
basis ``Z_theta*`` to hand to Loukas RSA coarsening.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Step 1 — build A_hat once (Section 12.1)
# ---------------------------------------------------------------------------
def normalized_adjacency(W: sp.spmatrix) -> sp.spmatrix:
    r"""``A_hat = D_tilde^{-1/2} (A + I) D_tilde^{-1/2}`` with self-loops (Eq. 2)."""
    W = sp.csr_matrix(W)
    N = W.shape[0]
    A_tilde = W + sp.eye(N)
    d = np.asarray(A_tilde.sum(axis=1)).ravel()
    d_inv_sqrt = np.zeros_like(d)
    nz = d > 0
    d_inv_sqrt[nz] = d[nz] ** -0.5
    Dis = sp.diags(d_inv_sqrt)
    return (Dis @ A_tilde @ Dis).tocsr()


def symmetric_laplacian(W: sp.spmatrix) -> sp.spmatrix:
    """Normalised Laplacian ``L_sym = I - A_hat`` (used for phi_N)."""
    N = W.shape[0]
    return (sp.eye(N) - normalized_adjacency(W)).tocsr()


# ---------------------------------------------------------------------------
# Step 2 — indicators and conductance (Definition 1, Theorem 2)
# ---------------------------------------------------------------------------
def indicator(nodes: Sequence[int], N: int) -> np.ndarray:
    r"""Normalised gang indicator ``v_S = 1_S / sqrt(|S|)`` (||v_S|| = 1)."""
    v = np.zeros(N)
    nodes = np.asarray(list(nodes), dtype=int)
    v[nodes] = 1.0 / np.sqrt(len(nodes))
    return v


def normalized_conductance(v: np.ndarray, A_hat: sp.spmatrix) -> float:
    r"""``phi_N = v^T L_sym v = 1 - v^T A_hat v`` (Theorem 2)."""
    return float(1.0 - v @ (A_hat @ v))


def sample_conductance_matched_negative(
    s: int,
    phi_target: float,
    A_hat: sp.spmatrix,
    rng: np.random.Generator,
    n_tries: int = 40,
    tol: float = 0.25,
) -> np.ndarray:
    r"""Draw a random ``s``-vertex set whose normalised conductance matches
    ``phi_target`` (the matching that makes the boundary moment blocks cancel,
    Theorem 5 / "Where the labels come from" in Section 11).

    Returns the best indicator vector found (closest phi_N) over ``n_tries``.
    """
    N = A_hat.shape[0]
    best_v, best_gap = None, np.inf
    for _ in range(n_tries):
        nodes = rng.choice(N, size=s, replace=False)
        v = indicator(nodes, N)
        gap = abs(normalized_conductance(v, A_hat) - phi_target)
        if gap < best_gap:
            best_v, best_gap = v, gap
        if gap <= tol * max(phi_target, 1e-6):
            break
    return best_v


# ---------------------------------------------------------------------------
# Step 3 — accumulate moments by mat-vecs (Definition 6, Section 12.3)
# ---------------------------------------------------------------------------
def moment_matrix(A_hat: sp.spmatrix, v: np.ndarray, K: int) -> np.ndarray:
    r"""Hankel gang moment matrix ``M`` of degree ``K`` (Definition 6).

    ``w_p = A_hat^p v`` are formed by ``K`` repeated sparse multiplies and the
    moments ``m_hat_p = w_a^T w_b`` (for any ``a + b = p``).  ``M[k, l] =
    m_hat_{k+l}`` for ``0 <= k, l <= K``.  Cost ``O(K * nnz)`` — no
    eigendecomposition (the whole point of Section 7/12).
    """
    W = [v]
    for _ in range(2 * K):
        W.append(A_hat @ W[-1])
    # m_hat_p = w_0^T w_p  (== w_a^T w_b for any a+b=p since A_hat is symmetric).
    m = np.array([float(W[0] @ W[p]) for p in range(2 * K + 1)])
    idx = np.add.outer(np.arange(K + 1), np.arange(K + 1))
    return m[idx]


def retained_energy(theta: np.ndarray, M: np.ndarray) -> float:
    r"""Gang energy retained by the soft low-pass: ``C(theta) = theta^T M theta``
    (Definition 5 / Theorem 3).  ``C(e_K) = m_hat_{2K} = C_SGC(K)``."""
    return float(theta @ M @ theta)


def sgc_retained_energy(M: np.ndarray) -> float:
    """Vanilla-SGC retained energy ``C_SGC(K) = m_hat_{2K} = M[K, K]`` (Def. 3)."""
    return float(M[-1, -1])


# ---------------------------------------------------------------------------
# Step 4 — fit the filter (Theorem 5 / Eq. 34)
# ---------------------------------------------------------------------------
@dataclass
class FilterFit:
    theta: np.ndarray  # learned polynomial coefficients (K+1,)
    K: int
    M_pos: np.ndarray  # batch-averaged gang moment matrix
    M_neg: np.ndarray  # batch-averaged conductance-matched negative moment matrix
    energy_sgc: float  # C_SGC(K) = m_hat_{2K}
    energy_learned: float  # C(theta*) on the positive batch
    discriminative_gap: float  # lambda_max(Delta M)


def fit_filter(
    M_pos: np.ndarray,
    M_neg: np.ndarray,
    mode: str = "difference",
    ground_mode_constraint: bool = True,
    project_out_boundary: bool = True,
) -> FilterFit:
    r"""Closed-form maximally-discriminative filter (Section 11, Eq. 34).

    ``mode='difference'``  -> theta* = top eigenvector of  Delta M = M_pos - M_neg
                              (Theorem 5; isolates the type-bearing p>=3 block).
    ``mode='generalized'`` -> theta* = top generalised eigenvector of (M_pos, M_neg).
    ``mode='retain'``      -> theta* = top eigenvector of M_pos (pure retention,
                              Theorem 4 — recovers a smooth low-pass).

    ``project_out_boundary`` enforces the structure of Theorem 5 exactly by
    zeroing the type-blind anti-diagonals ``p = k + l <= 2`` of ``Delta M`` before
    the eigendecomposition.  Conductance matching makes these cancel only
    approximately; zeroing them removes the residual boundary mass so theta*
    aligns with the skewness/higher-moment (band-pass) block as predicted.

    ``ground_mode_constraint`` rescales theta so that ``sum_k theta_k = 1``
    (preserve the smoothest mode g_theta(1)=1, Section 11); skipped when the
    boundary block is projected out (theta then sums to ~0 by construction).
    """
    K = M_pos.shape[0] - 1
    if mode == "difference":
        dM = M_pos - M_neg
        if project_out_boundary:
            idx = np.add.outer(np.arange(K + 1), np.arange(K + 1))
            dM = np.where(idx <= 2, 0.0, dM)
        w, V = np.linalg.eigh(0.5 * (dM + dM.T))
        theta = V[:, -1]
        gap = float(w[-1])
    elif mode == "generalized":
        from scipy.linalg import eigh as geigh

        reg = M_neg + 1e-8 * np.eye(K + 1)
        w, V = geigh(M_pos, reg)
        theta = V[:, -1]
        gap = float(w[-1])
    elif mode == "retain":
        w, V = np.linalg.eigh(0.5 * (M_pos + M_pos.T))
        theta = V[:, -1]
        gap = float((M_pos - M_neg).trace())
        project_out_boundary = False
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if ground_mode_constraint and not project_out_boundary and abs(theta.sum()) > 1e-8:
        theta = theta / theta.sum()
    else:
        theta = theta / (np.linalg.norm(theta) + 1e-12)

    return FilterFit(
        theta=theta,
        K=K,
        M_pos=M_pos,
        M_neg=M_neg,
        energy_sgc=sgc_retained_energy(M_pos),
        energy_learned=retained_energy(theta, M_pos),
        discriminative_gap=gap,
    )


# ---------------------------------------------------------------------------
# Step 5 — build the coarsening target subspace (Eq. 26)
# ---------------------------------------------------------------------------
def apply_polynomial(A_hat: sp.spmatrix, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    r"""``g_theta(A_hat) X = sum_k theta_k A_hat^k X`` by Horner-free accumulation."""
    out = np.zeros_like(X, dtype=float)
    Ak_X = X.astype(float)
    for k in range(len(theta)):
        out += theta[k] * Ak_X
        if k < len(theta) - 1:
            Ak_X = A_hat @ Ak_X
    return out


def build_subspace(
    A_hat: sp.spmatrix,
    theta: np.ndarray,
    width: int = 64,
    X: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""Coarsening target ``Z_theta = g_theta(A_hat) X`` (Eq. 26 / step 5).

    With a random ``X`` each column is the spectral response ``g_theta`` applied
    to a random signal, so ``span(Z_theta)`` is a randomised range finder for the
    modes the learned filter keeps.  This basis is handed to Loukas RSA
    coarsening in place of ``span(U_K)``.
    """
    N = A_hat.shape[0]
    if X is None:
        rng = rng or np.random.default_rng(0)
        X = rng.standard_normal((N, width))
    Z = apply_polynomial(A_hat, theta, X)
    # orthonormalise the basis for numerical conditioning of downstream eigh.
    Q, _ = np.linalg.qr(Z)
    return Q


def sgc_subspace(
    A_hat: sp.spmatrix,
    K: int = 10,
    width: int = 64,
    X: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Vanilla-SGC target ``span(A_hat^K X)`` (Eq. 21) — the theta = e_K special case."""
    theta = np.zeros(K + 1)
    theta[K] = 1.0
    return build_subspace(A_hat, theta, width=width, X=X, rng=rng)


# ---------------------------------------------------------------------------
# Section 12 — full training procedure for better motif detection
# ---------------------------------------------------------------------------
@dataclass
class SGCTarget:
    basis: np.ndarray  # Z_theta*  (N x width) — the RSA target subspace
    fit: FilterFit
    A_hat: sp.spmatrix


def train_sgc_target(
    W: sp.spmatrix,
    positive_node_sets: Sequence[Sequence[int]],
    K: int = 10,
    width: int = 64,
    n_neg_per_pos: int = 1,
    mode: str = "difference",
    ground_mode_constraint: bool = True,
    seed: int = 0,
) -> SGCTarget:
    r"""The training procedure of Section 12, end to end.

    1. Build ``A_hat`` once.
    2. For each positive gang ``S+`` draw conductance-matched negatives ``S-``.
    3. Accumulate the Hankel moment matrices by mat-vecs.
    4. Fit the filter in closed form (top eigenvector of ``Delta M``).
    5. Build and return the target ``Z_theta* = g_theta*(A_hat) X``.

    Parameters
    ----------
    W : sparse adjacency of the host transaction graph.
    positive_node_sets : known / candidate gang vertex sets (training positives).
    K : polynomial degree of the learnable filter (NOT the #eigenvectors).
    width : embedding width ``d`` -> dimension of the produced subspace.
    """
    rng = np.random.default_rng(seed)
    A_hat = normalized_adjacency(W)
    N = A_hat.shape[0]

    M_pos_list: List[np.ndarray] = []
    M_neg_list: List[np.ndarray] = []
    for nodes in positive_node_sets:
        nodes = list(nodes)
        if len(nodes) < 2:
            continue
        v = indicator(nodes, N)
        phi = normalized_conductance(v, A_hat)
        M_pos_list.append(moment_matrix(A_hat, v, K))
        for _ in range(n_neg_per_pos):
            v_neg = sample_conductance_matched_negative(len(nodes), phi, A_hat, rng)
            M_neg_list.append(moment_matrix(A_hat, v_neg, K))

    if not M_pos_list:
        raise ValueError("no usable positive gangs (need >= 2 nodes each)")

    M_pos = np.mean(M_pos_list, axis=0)
    M_neg = np.mean(M_neg_list, axis=0)

    fit = fit_filter(
        M_pos, M_neg, mode=mode, ground_mode_constraint=ground_mode_constraint
    )

    # shared random X so the basis is reproducible across calls.
    X = rng.standard_normal((N, width))
    basis = build_subspace(A_hat, fit.theta, X=X)

    return SGCTarget(basis=basis, fit=fit, A_hat=A_hat)
