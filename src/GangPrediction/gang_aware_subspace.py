"""Gang-Aware Subspace Construction for Graph Coarsening.

This module implements a gang-aware subspace construction algorithm that builds
a basis matrix V specifically designed to preserve "gang signals" during graph
coarsening. The key idea is:

1. Build seed vectors that "light up" known malicious/normal patterns
2. Smooth them with the graph Laplacian (resolvent smoothing)
3. Compress into a small orthonormal basis V (the target subspace R = span(V))
4. Use V in Loukas-style variation coarsening (instead of generic eigenvectors)

This helps detect new gangs by preserving structure similar to known patterns.

Reference: The algorithm is based on the principle that if unseen gangs share
structure with seen ones (similar transaction motifs), their indicator vectors
will lie near the span of training gang signals after smoothing.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torch_geometric.data import Data

from src.GangPrediction.utils.utils import graph_params


def build_gang_aware_basis(
    G: Data,
    malicious_patterns: Dict[str, List[int]],
    normal_patterns: Dict[str, List[int]],
    alpha: float = 1.0,
    k: int = 32,
    add_discrimination_seed: bool = False,
    normalize_seeds: bool = True,
    cg_tol: float = 1e-6,
    cg_maxiter: Optional[int] = None,
    device: Optional[str] = None,
    ensure_orthogonal=True,
) -> torch.Tensor:
    """
    Build a gang-aware orthonormal basis for graph coarsening.

    This function constructs a subspace R = span(V) that preserves "gang signals"
    by smoothing pattern indicator vectors with the graph Laplacian and compressing
    them into a low-dimensional basis.

    Parameters
    ----------
    G : torch_geometric.data.Data
        The input graph with Laplacian L (computed if not present).
    malicious_patterns : Dict[str, List[int]]
        Dictionary mapping pattern_id -> list of node indices for malicious patterns.
    normal_patterns : Dict[str, List[int]]
        Dictionary mapping pattern_id -> list of node indices for normal patterns.
    alpha : float, default=1.0
        Smoothing strength for resolvent smoothing (I + alpha*L)^{-1}.
        Higher alpha = smoother signals.
    k : int, default=32
        Target dimension of the subspace (number of basis vectors).
    add_discrimination_seed : bool, default=True
        If True, add a global discrimination seed (mean_malicious - mean_normal).
    normalize_seeds : bool, default=True
        If True, normalize each seed by 1/sqrt(|S|) to balance pattern sizes.
    cg_tol : float, default=1e-6
        Convergence tolerance for Conjugate Gradient solver.
    cg_maxiter : int, optional
        Maximum CG iterations. If None, uses 2*n.
    device : str, optional
        Device for computation. If None, uses G.x.device or 'cpu'.

    Returns
    -------
    V : torch.Tensor
        Orthonormal basis matrix of shape (n, k) spanning the gang-aware subspace.

    Example
    -------
    >>> V = build_gang_aware_basis(G, alert_patterns, normal_patterns, alpha=1.0, k=32)
    >>> # Use V in coarsening instead of spectral basis B
    >>> # In coarsening_utils.py, replace B = calc_B(G, K) with B = V
    """
    if device is None:
        device = G.x.device if hasattr(G, "x") and G.x is not None else "cpu"

    n = G.num_nodes

    # Ensure graph has Laplacian
    if not hasattr(G, "L") or G.L is None:
        G.W, G.L, G.dw = graph_params(G)

    # L = G.L.to(device)

    # Step 0: Build seed vectors
    S, labels = _build_seed_vectors(
        n=n,
        malicious_patterns=malicious_patterns,
        normal_patterns=normal_patterns,
        add_discrimination_seed=add_discrimination_seed,
        normalize_seeds=normalize_seeds,
        ensure_orthogonal=ensure_orthogonal,
        device=device,
    )

    # Step 1: Smooth seeds using resolvent (I + alpha*L)^{-1}
    # H = _smooth_seeds_resolvent(
    #     L=L,
    #     S=S,
    #     alpha=alpha,
    #     tol=cg_tol,
    #     maxiter=cg_maxiter,
    # )

    # # Step 2: Compress to low-dimensional orthonormal basis via SVD
    # V = _compress_to_basis(H, k=H.shape[1])
    V, R = torch.linalg.qr(S)
    if not ensure_orthogonal:
        tol = 1e-6
        diag = torch.abs(torch.diag(R))
        rank = (diag > tol).sum()

        V = V[:, :rank]

    return V


def build_gang_aware_basis_lda(
    G: Data,
    malicious_patterns: Dict[str, List[int]],
    normal_patterns: Dict[str, List[int]],
    alpha: float = 1.0,
    k: int = 32,
    normalize_seeds: bool = True,
    cg_tol: float = 1e-6,
    cg_maxiter: Optional[int] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Build a gang-aware basis using Fisher LDA for discrimination.

    This variant uses Linear Discriminant Analysis to find directions that
    maximally separate malicious from normal patterns, rather than just
    reconstructing the smoothed signals.

    Parameters
    ----------
    G : torch_geometric.data.Data
        The input graph.
    malicious_patterns : Dict[str, List[int]]
        Malicious pattern dictionary.
    normal_patterns : Dict[str, List[int]]
        Normal pattern dictionary.
    alpha : float, default=1.0
        Smoothing strength.
    k : int, default=32
        Target dimension (will be min(k, min(n_mal, n_nor))).
    normalize_seeds : bool, default=True
        Normalize seeds by pattern size.
    cg_tol : float, default=1e-6
        CG tolerance.
    cg_maxiter : int, optional
        Max CG iterations.
    device : str, optional
        Computation device.

    Returns
    -------
    V : torch.Tensor
        Orthonormal basis matrix of shape (n, k).
    """
    if device is None:
        device = G.x.device if hasattr(G, "x") and G.x is not None else "cpu"

    n = G.num_nodes

    if not hasattr(G, "L") or G.L is None:
        G.W, G.L, G.dw = graph_params(G)

    L = G.L.to(device)

    # Build seeds (without discrimination seed - we'll use LDA instead)
    S, labels = _build_seed_vectors(
        n=n,
        malicious_patterns=malicious_patterns,
        normal_patterns=normal_patterns,
        add_discrimination_seed=False,
        normalize_seeds=normalize_seeds,
        device=device,
    )

    # Smooth seeds
    H = _smooth_seeds_resolvent(L, S, alpha, cg_tol, cg_maxiter)

    # Apply Fisher LDA in H-space, then map back
    V = _compress_with_lda(H, labels, k=k)

    return V


def _build_seed_vectors(
    n: int,
    malicious_patterns: Dict[str, List[int]],
    normal_patterns: Dict[str, List[int]],
    add_discrimination_seed: bool = True,
    normalize_seeds: bool = True,
    device: str = "cpu",
    ensure_orthogonal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build seed vectors from pattern node sets.

    For each pattern S, the seed is:
        s = w * (1/|S|) * indicator(S)
    where w = 1/sqrt(|S|) if normalize_seeds else 1.

    Parameters
    ----------
    n : int
        Number of nodes in the graph.
    malicious_patterns : Dict
        Malicious patterns.
    normal_patterns : Dict
        Normal patterns.
    add_discrimination_seed : bool
        Add mean_malicious - mean_normal as extra seed.
    normalize_seeds : bool
        Weight by 1/sqrt(|S|).
    device : str
        Computation device.
    ensure_orthogonal : bool, default=True
        If True, ensure orthogonal columns by keeping each node in only one pattern.
        Nodes appearing in multiple patterns are assigned to the first pattern encountered.

    Returns
    -------
    S : torch.Tensor
        Seed matrix of shape (n, m) where m = total number of seeds.
    labels : torch.Tensor
        Labels of shape (m,): 1 for malicious, 0 for normal, 2 for discrimination.
    """
    seeds = []
    labels = []

    # Track used nodes to ensure orthogonal columns
    used_nodes = set() if ensure_orthogonal else None

    # Process malicious patterns
    for pattern_id, node_indices in malicious_patterns.items():
        if ensure_orthogonal:
            # Filter out nodes already used in previous patterns
            unique_indices = [idx for idx in node_indices if idx not in used_nodes]
            if len(unique_indices) == 0:
                continue  # Skip pattern if all nodes already used
            used_nodes.update(unique_indices)
            node_indices = unique_indices
        s = _pattern_to_seed(n, node_indices, normalize_seeds, device)
        seeds.append(s)
        labels.append(1)  # malicious

    n_malicious = len(seeds)

    # Process normal patterns
    for pattern_id, node_indices in normal_patterns.items():
        if ensure_orthogonal:
            # Filter out nodes already used in previous patterns
            unique_indices = [idx for idx in node_indices if idx not in used_nodes]
            if len(unique_indices) == 0:
                continue  # Skip pattern if all nodes already used
            used_nodes.update(unique_indices)
            node_indices = unique_indices
        s = _pattern_to_seed(n, node_indices, normalize_seeds, device)
        seeds.append(s)
        labels.append(0)  # normal

    # Add global discrimination seed: mean(malicious) - mean(normal)
    if add_discrimination_seed and n_malicious > 0 and len(seeds) > n_malicious:
        mal_seeds = torch.stack(seeds[:n_malicious], dim=1)  # (n, n_mal)
        nor_seeds = torch.stack(seeds[n_malicious:], dim=1)  # (n, n_nor)

        mean_mal = mal_seeds.mean(dim=1)  # (n,)
        mean_nor = nor_seeds.mean(dim=1)  # (n,)

        s_delta = mean_mal - mean_nor
        # Normalize discrimination seed
        norm = torch.norm(s_delta)
        if norm > 1e-10:
            s_delta = s_delta / norm

        seeds.append(s_delta)
        labels.append(2)  # discrimination

    if len(seeds) == 0:
        raise ValueError(
            "No patterns provided. Need at least one malicious or normal pattern."
        )

    S = torch.stack(seeds, dim=1)  # (n, m)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    return S, labels


def _pattern_to_seed(
    n: int,
    node_indices: List[int],
    normalize: bool,
    device: str,
) -> torch.Tensor:
    """Convert a pattern (list of node indices) to a normalized seed vector."""
    s = torch.zeros(n, dtype=torch.float32, device=device)
    indices = torch.tensor(node_indices, dtype=torch.long, device=device)

    pattern_size = len(node_indices)
    if pattern_size == 0:
        return s

    # Indicator normalized by size
    s[indices] = 1.0 / pattern_size

    # Optional: additional normalization by sqrt(size) for balancing
    if normalize:
        s = s / np.sqrt(pattern_size)

    return s


def _smooth_seeds_resolvent(
    L: torch.Tensor,
    S: torch.Tensor,
    alpha: float,
    tol: float = 1e-6,
    maxiter: Optional[int] = None,
) -> torch.Tensor:
    """
    Smooth seed vectors using resolvent: h = (I + alpha*L)^{-1} s

    Uses Conjugate Gradient since (I + alpha*L) is SPD.

    Parameters
    ----------
    L : torch.Tensor
        Graph Laplacian (sparse).
    S : torch.Tensor
        Seed matrix of shape (n, m).
    alpha : float
        Smoothing strength.
    tol : float
        CG tolerance.
    maxiter : int, optional
        Max iterations.

    Returns
    -------
    H : torch.Tensor
        Smoothed signatures of shape (n, m), each column normalized.
    """
    n, m = S.shape
    device = S.device

    # Build A = I + alpha*L (as operator)
    def A_mv(x):
        """Apply (I + alpha*L) to x."""
        if x.dim() == 1:
            return x + alpha * torch.sparse.mm(L, x.unsqueeze(1)).squeeze(1)
        else:
            return x + alpha * torch.sparse.mm(L, x)

    # Solve (I + alpha*L) H = S using CG
    H = _cg_solve_multi(A_mv, S, tol=tol, maxiter=maxiter)

    # Normalize each column to unit norm
    norms = torch.norm(H, dim=0, keepdim=True).clamp(min=1e-10)
    H = H / norms

    return H


def _cg_solve_multi(
    A_mv,
    B: torch.Tensor,
    tol: float = 1e-6,
    maxiter: Optional[int] = None,
) -> torch.Tensor:
    """
    Solve A X = B for multiple RHS using Conjugate Gradient.

    Parameters
    ----------
    A_mv : callable
        Function that applies the matrix A to a vector/matrix.
    B : torch.Tensor
        Right-hand side matrix of shape (n, d).
    tol : float
        Convergence tolerance.
    maxiter : int, optional
        Maximum iterations.

    Returns
    -------
    X : torch.Tensor
        Solution matrix of shape (n, d).
    """
    n, d = B.shape
    device = B.device
    dtype = B.dtype

    if maxiter is None:
        maxiter = 2 * n

    X = torch.zeros_like(B)
    R = B - A_mv(X)
    P = R.clone()

    rsold = (R * R).sum(dim=0)  # per-column

    for i in range(maxiter):
        AP = A_mv(P)
        alpha_denom = (P * AP).sum(dim=0).clamp(min=1e-30)
        alpha = rsold / alpha_denom

        X = X + P * alpha.unsqueeze(0)
        R = R - AP * alpha.unsqueeze(0)

        rsnew = (R * R).sum(dim=0)

        # Check convergence
        if (rsnew.sqrt() < tol * B.norm(dim=0)).all():
            break

        beta = rsnew / rsold.clamp(min=1e-30)
        P = R + P * beta.unsqueeze(0)
        rsold = rsnew

    return X


def _compress_to_basis(
    H: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Compress smoothed signals to a k-dimensional orthonormal basis via SVD.

    Parameters
    ----------
    H : torch.Tensor
        Smoothed signal matrix of shape (n, m).
    k : int
        Target dimension.

    Returns
    -------
    V : torch.Tensor
        Orthonormal basis of shape (n, min(k, rank(H))).
    """
    n, m = H.shape

    # Limit k to available rank
    k = min(k, m, n)

    # SVD: H ≈ U_k Σ_k V_k^T
    # We want U_k as our basis
    U, S, Vh = torch.linalg.svd(H, full_matrices=False)

    # Take top-k left singular vectors
    V = U[:, :k]

    return V


def _compress_with_lda(
    H: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Compress using Fisher LDA to find discriminative directions.

    Parameters
    ----------
    H : torch.Tensor
        Smoothed signals of shape (n, m).
    labels : torch.Tensor
        Labels of shape (m,): 1 for malicious, 0 for normal.
    k : int
        Target dimension.

    Returns
    -------
    V : torch.Tensor
        Discriminative basis of shape (n, k).
    """
    n, m = H.shape
    device = H.device

    # Separate malicious and normal samples (columns of H)
    mal_mask = labels == 1
    nor_mask = labels == 0

    n_mal = mal_mask.sum().item()
    n_nor = nor_mask.sum().item()

    if n_mal == 0 or n_nor == 0:
        # Fall back to SVD if we don't have both classes
        return _compress_to_basis(H, k)

    H_mal = H[:, mal_mask]  # (n, n_mal)
    H_nor = H[:, nor_mask]  # (n, n_nor)

    # Class means
    mu_mal = H_mal.mean(dim=1)  # (n,)
    mu_nor = H_nor.mean(dim=1)  # (n,)
    mu_all = H.mean(dim=1)  # (n,)

    # Between-class scatter: S_B = sum_c n_c (mu_c - mu)(mu_c - mu)^T
    # This is rank-1 for 2 classes
    diff_mal = (mu_mal - mu_all).unsqueeze(1)  # (n, 1)
    diff_nor = (mu_nor - mu_all).unsqueeze(1)  # (n, 1)
    S_B = n_mal * (diff_mal @ diff_mal.T) + n_nor * (diff_nor @ diff_nor.T)

    # Within-class scatter: S_W = sum_c sum_{x in c} (x - mu_c)(x - mu_c)^T
    H_mal_centered = H_mal - mu_mal.unsqueeze(1)
    H_nor_centered = H_nor - mu_nor.unsqueeze(1)
    S_W = H_mal_centered @ H_mal_centered.T + H_nor_centered @ H_nor_centered.T

    # Regularize S_W
    S_W = S_W + 1e-6 * torch.eye(n, device=device)

    # Solve generalized eigenvalue problem: S_B w = lambda S_W w
    # Equivalent to: S_W^{-1} S_B w = lambda w
    try:
        S_W_inv = torch.linalg.inv(S_W)
        M = S_W_inv @ S_B

        # Get eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eig(M)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real

        # Sort by eigenvalue (descending)
        idx = torch.argsort(eigenvalues, descending=True)

        # For 2-class LDA, we only get 1 discriminative direction
        # But we can augment with PCA directions
        k_lda = min(k, 1)  # Only 1 LDA direction for 2 classes
        V_lda = eigenvectors[:, idx[:k_lda]]

        if k > k_lda:
            # Add PCA directions orthogonal to LDA
            V_pca = _compress_to_basis(H, k - k_lda + 1)[
                :, 1:
            ]  # Skip first (likely similar to LDA)
            V = torch.cat([V_lda, V_pca], dim=1)[:, :k]
        else:
            V = V_lda

        # Orthonormalize
        V, _ = torch.linalg.qr(V)

    except Exception:
        # Fall back to SVD
        V = _compress_to_basis(H, k)

    return V


def calc_B_gang_aware(
    G: Data,
    K: int,
    malicious_patterns: Dict[str, List[int]],
    normal_patterns: Dict[str, List[int]],
    alpha: float = 1.0,
    use_lda: bool = False,
) -> torch.Tensor:
    """
    Drop-in replacement for calc_B that uses gang-aware subspace.

    This function can be used directly in place of the standard calc_B
    in coarsening_utils.py to make coarsening gang-aware.

    Parameters
    ----------
    G : Data
        PyTorch Geometric graph.
    K : int
        Target subspace dimension.
    malicious_patterns : Dict
        Malicious patterns from load_all_patterns.
    normal_patterns : Dict
        Normal patterns from load_all_patterns.
    alpha : float
        Smoothing strength.
    use_lda : bool
        If True, use LDA-based compression for discrimination.

    Returns
    -------
    B : torch.Tensor
        Gang-aware basis matrix of shape (n, K).

    Example
    -------
    >>> # In test_refactored.py, replace:
    >>> # B = calc_B(G, TRAIN_CONFIG["K"])
    >>> # with:
    >>> B = calc_B_gang_aware(G, TRAIN_CONFIG["K"], alert_patterns, normal_patterns)
    """
    if use_lda:
        V = build_gang_aware_basis_lda(
            G=G,
            malicious_patterns=malicious_patterns,
            normal_patterns=normal_patterns,
            alpha=alpha,
            k=K,
        )
    else:
        V = build_gang_aware_basis(
            G=G,
            malicious_patterns=malicious_patterns,
            normal_patterns=normal_patterns,
            alpha=alpha,
            k=K,
        )

    # The Loukas coarsening uses A = L^{1/2} B for variation costs
    # Since we already have a smooth basis, we can either:
    # 1. Return V directly (simpler, still works)
    # 2. Compute L^{1/2} V for variation interpretation

    # For now, return V directly as it's already smooth
    # The variation cost will measure how well V is preserved
    return V


# Convenience function for integration with existing code
def get_gang_aware_basis(
    G: Data,
    alert_patterns: Dict[str, List[int]],
    normal_patterns: Dict[str, List[int]],
    K: int = 32,
    alpha: float = 1.0,
    method: str = "svd",
) -> torch.Tensor:
    """
    Convenience function to get gang-aware basis.

    Parameters
    ----------
    G : Data
        The graph.
    alert_patterns : Dict
        Alert/malicious patterns.
    normal_patterns : Dict
        Normal patterns.
    K : int
        Subspace dimension.
    alpha : float
        Smoothing strength. Larger = smoother patterns.
    method : str
        "svd" for PCA-style, "lda" for Fisher LDA.

    Returns
    -------
    V : torch.Tensor
        Basis matrix of shape (n, K).
    """
    use_lda = method.lower() == "lda"

    # Handle empty patterns gracefully
    if len(alert_patterns) == 0 and len(normal_patterns) == 0:
        print("Warning: No patterns provided. Falling back to spectral basis.")
        from src.GangPrediction.coarsening_utils import calc_B

        return calc_B(G, K)

    return calc_B_gang_aware(
        G=G,
        K=K,
        malicious_patterns=alert_patterns,
        normal_patterns=normal_patterns,
        alpha=alpha,
        use_lda=use_lda,
    )
