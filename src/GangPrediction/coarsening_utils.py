"""Graph coarsening utilities based on variation-aware contractions."""

from copy import deepcopy
import scipy
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_sparse import SparseTensor
from sortedcontainers import SortedList
from tqdm import tqdm

from src.GangPrediction.utils.utils import degree, graph_params, sparse_eye
from src.GangPrediction.maxWeightMatching import maxWeightMatching
from utils.utils import *


def coarsen(
    # G,
    # X=None, # WHT: node features
    Gc: Data,
    K=10,
    r=0.5,
    max_levels=10,
    method="variation_neighborhood",
    algorithm="greedy",
    Uk=None,
    lk=None,
    max_level_r=0.99,
    similarity_threshold=0.0,
    max_epsilon=float("inf"),
):
    """
    This function provides a common interface for coarsening algorithms that contract subgraphs

    Parameters
    ----------
    G : pygsp Graph
    K : int
        The size of the subspace we are interested in preserving.
    r : float between (0,1)
        The desired reduction defined as 1 - n/N.
    method : String
        ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron']
    max_epsilon : float
        The maximum cumulative epsilon allowed.

    Returns
    -------
    C : np.array of size n x N
        The coarsening matrix.
    Gc : pygsp Graph
        The smaller graph.
    Call : list of np.arrays
        Coarsening matrices for each level
    Gall : list of (n_levels+1) pygsp Graphs
        All graphs involved in the multilevel coarsening

    Example
    -------
    C, Gc, Call, Gall = coarsen(G, K=10, r=0.8)
    """
    r = np.clip(r, 0.0, max_level_r)
    N = Gc.num_nodes

    C = sparse_eye(N)
    Gc.W, Gc.L, Gc.dw = graph_params(Gc)

    B = calc_B(Gc, K, Uk, lk)
    iC = None

    Call, Gall = [], []
    epsilon_l, max_eps_in_level, epsilons = 0, 0, [0]
    Gall.append(Gc)

    for level in range(1, max_levels + 1):
        # max_eps_in_level += max_epsilon / max_levels
        max_sigma = (max_epsilon + 1) / (epsilon_l + 1) - 1
        Gc, B, sigma_l = coarse_one_level(
            Gc,
            B,
            K=K,
            method=method,
            algorithm=algorithm,
            similarity_threshold=similarity_threshold,
            level=level,
            r_cur=r,
            max_sigma=max_sigma,
        )

        C = torch.sparse.mm(iC, C)
        Call.append(iC)
        Gall.append(Gc)

        epsilon_l = (sigma_l + 1) * (epsilon_l + 1) - 1
        epsilons.append(epsilon_l)

        if iC.shape[1] - iC.shape[0] <= 2:
            break  # avoid too many levels for so few nodes
        if Gc.num_nodes <= (1 - r) * N:
            break

    return C, Gc, Call, Gall


def coarse_one_level(
    G,
    B,
    K=10,
    method="variation_neighborhoods",
    algorithm="greedy",
    level=1,
    r_cur=None,
    similarity_threshold=0.0,
    max_sigma=float("inf"),
):
    """Coarsen a single level using a variation-based or matching-based policy."""

    done_flag = False

    if "variation" in method:
        if level == 1:
            A = B
        else:
            B = torch.sparse.mm(G.C, B)
            d, V = torch.linalg.eigh(B.T @ G.L @ B)
            X_init = torch.randn(B.shape[1], K, device=G.L.device)
            # d, V = torch.lobpcg(
            #     B.T @ G.L @ B, k=K, X=X_init, largest=False, niter=50, tol=1e-4
            # )
            # d = torch.ones(K, device=G.L.device)
            # V = torch.eye(K, device=G.L.device)
            mask = d < 1e-10
            d[mask] = 1
            # dinvsqrt = d ** (0)
            if method in ["variation_edges", "variation_embedding"]:
                dinvsqrt = d ** (-0.5)
            elif method == "gang_edges":
                dinvsqrt = d ** (+0.5)
            # dinvsqrt = d ** (-0.5)
            dinvsqrt[mask] = 0
            A = B @ V @ torch.diag(dinvsqrt) @ V.T

        if method in ["variation_edges", "gang_edges", "variation_embedding"]:
            coarsening_list, sigma_l, done_flag = contract_variation_edges(
                G,
                A=A,
                r=r_cur,
                algorithm=algorithm,
                similarity_threshold=similarity_threshold,
                max_sigma=max_sigma,
            )
        # elif method == "variation_embedding":
        #     coarsening_list, sigma_l, done_flag = contract_variation_edges(
        #         G,
        #         A=A,
        #         # A=G.embeddings,
        #         r=r_cur,
        #         algorithm=algorithm,
        #         similarity_threshold=similarity_threshold,
        #         max_sigma=max_sigma,
        #     )
        else:
            coarsening_list, sigma_l, done_flag = contract_variation_linear(
                G,
                A=A,
                r=r_cur,
                mode=method,
                similarity_threshold=similarity_threshold,
                max_sigma=max_sigma,
            )

    else:
        weights = get_proximity_measure(G, method, K=K)

        if algorithm == "optimal":
            # the edge-weight should be light at proximal edges
            weights = -weights
            if "rss" not in method:
                weights -= min(weights)
            coarsening_list = matching_optimal(G, weights=weights, r=r_cur)

        elif algorithm == "greedy":
            coarsening_list, sigma_l, done_flag = matching_greedy(
                G,
                weights=weights,
                r=r_cur,
                similarity_threshold=similarity_threshold,
                max_sigma=max_sigma,
            )

    iC = get_coarsening_matrix(coarsening_list, G.num_nodes)

    Gc = construct_G(G, iC)

    return Gc, B, sigma_l, done_flag


def calc_B(Gc, K, U=None):
    """Compute the spectral basis used by variation-based coarsening."""
    if (U is not None) and (U.shape[1] >= K):
        d, V = torch.linalg.eigh(Gc.L.to_dense())
        # X_init = torch.randn(Gc.num_nodes, K, device=Gc.L.device)
        # d, V = torch.lobpcg(Gc.L, k=K, X=X_init, largest=False, niter=200, tol=1e-4)
        # lk = d[:K]
        # Vk = V[:, :K]
        lk = d
        Vk = V

        mask = lk < 0
        lk[mask] = 1
        lsinv = lk ** (+0.5)
        lsinv[mask] = 0

        Uk = U[:, :K]
        B = Uk @ (Uk.T @ Vk @ torch.diag(lsinv)) @ Vk.T
    else:
        # Use sparse eigendecomposition for efficiency (10-100x faster than dense)
        # try:
        #     # Initialize with random vectors for lobpcg
        #     X_init = torch.randn(Gc.num_nodes, K, device=Gc.L.device)
        #     lk, Uk = torch.lobpcg(
        #         Gc.L, k=K, X=X_init, largest=False, niter=200, tol=1e-4
        #     )
        # except Exception as e:
        #     # Fallback to dense if lobpcg fails (e.g., very small graphs)
        #     print(
        #         f"Warning: lobpcg failed ({e}), falling back to dense eigendecomposition"
        #     )
        # d, V = torch.linalg.eigh(Gc.L.to_dense())
        # # d = torch.flip(d, dims=[0])
        # # V = torch.flip(V, dims=[1])
        # lk = d[:K]
        # Uk = V[:, :K]

        # mask = lk < 0
        # lk[mask] = 1
        # lsinv = lk ** (-0.5)
        # lsinv[mask] = 0
        # B = Uk @ torch.diag(lsinv)

        offset = 2 * max(Gc.dw)
        T = offset * sparse_eye(Gc.num_nodes) - Gc.L
        # lk, Uk = torch.linalg.eigh(T, k=K, which="LM", tol=1e-5)
        lk, Uk = torch.lobpcg(T, k=K, largest=True, tol=1e-5)
        lk = torch.flip(offset - lk, [0])
        Uk = torch.flip(Uk, [1])
        mask = lk < 1e-10
        lk[mask] = 1
        lsinv = lk ** (-0.5)
        lsinv[mask] = 0
        B = Uk @ torch.diag(lsinv)
    return B


def calc_B_embedding(Gc, K):
    """Compute a smooth embedding basis from feature-propagated signals."""
    # # Y = AX (or stack [X, AX, A^2 X])
    # Y = ax_stack(Gc.W, Gc.x, powers=1)  # -> [N, d]
    # # L-orthonormal basis of span(Y)
    # B0 = l_orthonormalize(Y, Gc.L, ridge=1e-6)  # [N, k], k = rank(Y) (<= d)
    # Given: sparse L, A, dense X (n x d), target k
    # Y = Gc.W @ Gc.x
    # YtY = Y.T @ Y
    # YtLY = Y.T @ (Gc.L @ Y)
    # Y = Y.cpu().detach().numpy()
    # YtLY = YtLY.cpu().detach().numpy()
    # YtY = YtY.cpu().detach().numpy()

    # eps = 1e-3
    # # Solve (Y^T L Y) v = lambda (Y^T Y + eps I) v
    # w, V = scipy.linalg.eigh(YtLY, YtY + eps * np.eye(YtY.shape[0]))
    # idx = np.argsort(w)[:K]  # k smallest lambdas (smoothest)
    # Vk = V[:, idx]
    # Z = Y @ Vk  # n x k
    # lam = w[idx]
    # B0 = Z @ np.diag(lam**+0.5)  # L-orthonormal basis in span(Y)
    # # -> use B0 in the p.16-style k×k updates at each coarsening level
    # return torch.tensor(B0, dtype=torch.float32, device=Gc.x.device)
    # A = torch.diag(Gc.dw) - Gc.L
    Q = arnoldi_iteration(Gc.L, K)[1]
    return Q.float()
    # return F.normalize(Q, p=2, dim=1)


def arnoldi_iteration(A, m: int, b=None, log=True):
    local_dev = "cpu"
    # local_dev = dev
    """Compute a basis of the (n + 1)-Krylov subspace of the matrix A.

    This is the space spanned by the vectors {b, Ab, ..., A^n b}.

    Parameters
    ----------
    A : array_like
        An m × m array.
    b : array_like
        Initial vector (length m).
    n : int
        One less than the dimension of the Krylov subspace, or equivalently the *degree* of the Krylov space. Must be >= 1.

    Returns
    -------
    Q : numpy.array
        An m x (n + 1) array, where the columns are an orthonormal basis of the Krylov subspace.
    h : numpy.array
        An (n + 1) x n array. A on basis Q. It is upper Hessenberg.
    """
    A = deepcopy(A).double()
    A = A.to_sparse().to(local_dev)
    if b is None:
        # b = torch.ones(A.shape[0], dtype=torch.double, device=dev)
        b = torch.randn(A.shape[0], dtype=torch.double, device=local_dev)
        if torch.sum(b) < 0:
            b = -b
    eps = 1e-12
    h = torch.zeros((m, m), dtype=torch.double, device=local_dev)
    Q = torch.zeros((A.shape[0], m), dtype=torch.double, device=local_dev)
    # Normalize the input vector
    Q[:, 0] = b / torch.norm(b, 2)  # Use it as the first Krylov vector
    if log:
        bar = tqdm(total=m - 1)
    for k in range(1, m):
        v = A @ Q[:, k - 1]  # Generate a new candidate vector
        for j in range(k):  # Subtract the projections on previous vectors
            h[j, k - 1] = Q[:, j].conj() @ v
            v = v - h[j, k - 1] * Q[:, j]

        h[k, k - 1] = torch.norm(v, 2)
        if h[k, k - 1] > eps:  # Add the produced vector to the list, unless
            Q[:, k] = v / h[k, k - 1]
        else:  # If that happens, stop iterating.
            return h, Q
        if log:
            bar.update()

    # h = h.to(dev)
    # Q = Q.to(dev)
    return h, Q


################################################################################
# General coarsening utility functions
################################################################################


def construct_G(G: Data, iC: SparseTensor):
    """Build a coarsened PyG Data graph from a coarsening matrix."""
    C_plus = calc_C_plus(iC)
    L = calc_L(G.L, C_plus)
    # Extract diagonal of sparse Laplacian L (degrees) without densifying the whole matrix
    W, dw = calc_W_deg(L)
    indices = W.indices()
    values = W.values()
    num_nodes = W.size(0)
    # x = torch.sparse.mm(iC, G.x) if G.x is not None else None
    x = coarsen_vector(G.x, iC) if G.x is not None else None
    Gc = Data(x=x, edge_index=indices, edge_weight=values, num_nodes=num_nodes)
    Gc.C = iC
    Gc.C_plus = C_plus
    Gc.L = L
    Gc.dw = dw
    Gc.W = W
    # Gc.soft_y = torch.sparse.mm(iC, G.soft_y) if G.soft_y is not None else None
    if hasattr(G, "soft_y") and G.soft_y is not None:
        Gc.soft_y = coarsen_vector(G.soft_y, iC)
        Gc.y = torch.argmax(Gc.soft_y, dim=1) if Gc.soft_y is not None else Gc.y
    else:
        Gc.soft_y = None
        Gc.y = G.y if hasattr(G, "y") else None
    if hasattr(G, "y_train") and G.y_train is not None:
        Gc.y_train = coarsen_vector(G.y_train, iC) if G.y_train is not None else None
        s = Gc.y_train.sum(1)
        Gc.train_idx = s.nonzero().view(-1) if Gc.y_train is not None else None
        Gc.test_idx = s.eq(0).nonzero().view(-1) if Gc.y_train is not None else None

    # Gc.W, Gc.L, Gc.dw = graph_params(Gc)

    embeddings = (
        # torch.sparse.mm(iC, G.embeddings)
        coarsen_vector(G.embeddings, iC)
        if hasattr(G, "embeddings") and G.embeddings is not None
        else None
    )
    Gc.embeddings = embeddings
    Gc.embeddings = (
        F.normalize(Gc.embeddings, p=2, dim=1) if embeddings is not None else None
    )

    if hasattr(G, "pos") and G.pos is not None:
        # pos = torch.sparse.mm(iC, G.pos)
        Gc.pos = coarsen_vector(G.pos, iC)

    if hasattr(G, "colors") and G.colors is not None:
        Gc.colors = coarsen_vector(G.colors, iC)

    return Gc


def calc_C_plus(C):
    """Compute the left pseudo-inverse of the coarsening matrix."""
    # Pinv = C.T; #Pinv[Pinv>0] = 1
    C_sum = torch.sparse.sum(C * C, dim=1).to_dense()
    inv_C_sum = 1.0 / C_sum
    # Create sparse diagonal matrix
    indices = torch.arange(len(inv_C_sum)).repeat(2, 1)
    D = torch.sparse_coo_tensor(indices, inv_C_sum, (len(inv_C_sum), len(inv_C_sum)))
    C_plus = torch.sparse.mm(C.t(), D)

    return C_plus


def calc_L(L, C_plus):
    """Project the Laplacian to the coarse space."""
    L_c = torch.sparse.mm(torch.sparse.mm(C_plus.t(), L), C_plus)
    L_c = (
        L_c + L_c.t()
    ) / 2  # this is only needed to avoid pygsp complaining for tiny errors
    # Coalesce to ensure consistent sparse format for downstream operations
    if L_c.is_sparse:
        L_c = L_c.coalesce()
    return L_c


def calc_W_deg(S):
    """Extract adjacency and degree diagonal from a sparse Laplacian."""
    if S.layout != torch.sparse_coo:
        raise ValueError("Expected a sparse COO tensor")
    S = S.coalesce()
    n, m = S.shape
    if n != m:
        raise ValueError("Matrix must be square to extract a diagonal")
    idx = S.indices()
    vals = S.values()
    mask = idx[0] == idx[1]
    diag = torch.zeros(n, device=vals.device, dtype=vals.dtype)
    if mask.any():
        diag.index_add_(0, idx[0, mask], vals[mask])
        non_diag_indices = idx[:, ~mask]
        non_diag_values = -vals[~mask]
        W = torch.sparse_coo_tensor(
            non_diag_indices, non_diag_values, (n, n), device=vals.device
        )
    else:
        W = torch.sparse_coo_tensor(idx, vals, (n, n), device=vals.device)

    return W.coalesce(), diag


def similarity(nodes_features):
    """Average pairwise similarity for a node feature block."""
    s = nodes_features @ nodes_features.T
    n = s.shape[0]
    return torch.sum(torch.triu(s, diagonal=1)) / ((n - 1) * n / 2)


def coarsen_vector(x, C):
    """Apply coarsening matrix to features or label probabilities."""
    return C @ x


def get_coarsening_matrix(partitioning, N):
    """
    Create coarsening matrix C using sparse tensor operations.

    Parameters:
    -----------
    G : PyTorch Geometric Data object
        The graph to be coarsened
    partitioning : list
        List of subgraphs to be contracted

    Returns:
    --------
    C : torch.sparse.Tensor
        The coarsening matrix
    """
    # Start from identity and collapse rows for each contracted subgraph.
    # Create sparse identity matrix
    C = sparse_eye(N)

    # Keep track of which rows to preserve
    rows_to_keep = torch.ones(N, dtype=torch.bool)

    # Process each subgraph
    for candidate in partitioning:
        subgraph = candidate["list"]
        nc = len(subgraph)
        if nc <= 1:
            continue

        # Convert subgraph to tensor if needed
        if not isinstance(subgraph, torch.Tensor):
            subgraph = torch.tensor(subgraph)

        # Mark rows to remove
        rows_to_keep[subgraph[1:]] = False

        # Get representative node
        rep_node = subgraph[0].item()

        # Create new entries for the representative node
        rep_indices = torch.stack(
            [
                torch.full((nc,), rep_node),  # Row indices (all rep_node)
                subgraph,  # Column indices
            ]
        )

        rep_values = torch.full((nc,), 1.0 / torch.tensor(nc, dtype=torch.float32))

        # Create sparse tensor for this update
        update = torch.sparse_coo_tensor(rep_indices, rep_values, (N, N))

        # Update the representative row
        # First zero out any existing entries in the row
        mask_indices = torch.tensor([[rep_node], [rep_node]])
        mask_values = torch.ones(1)
        mask = torch.sparse_coo_tensor(mask_indices, mask_values, (N, N))

        # Add the update to C
        C = C + update - mask

    # Keep only the rows that weren't contracted
    keep_indices = torch.nonzero(rows_to_keep).squeeze()

    # Extract the values and indices from C
    C = C.coalesce()  # Ensure C is in COO format
    indices = C.indices()
    values = C.values()

    # Filter rows that need to be kept
    mask = torch.isin(indices[0], keep_indices)
    filtered_indices = indices[:, mask]
    filtered_values = values[mask]

    # Map old row indices to new indices
    row_map = torch.full((N,), -1, dtype=torch.long)
    row_map[keep_indices] = torch.arange(len(keep_indices))
    filtered_indices[0] = row_map[filtered_indices[0]]

    # Create the final sparse tensor
    C_final = torch.sparse_coo_tensor(
        filtered_indices, filtered_values, (len(keep_indices), N)
    )

    return C_final


################################################################################
# Variation-based contraction algorithms
################################################################################
def contract_variation_embeddings(
    G: Data,
    A=None,
    r=0.5,
    algorithm="greedy",
    similarity_threshold=0.0,
    max_sigma=float("inf"),
):
    """
    Sequential contraction with local variation and edge-based families.
    This is a specialized implementation for the edge-based family, that works
    slightly faster than the contract_variation() function, which works for
    any family.

    See contract_variation() for documentation.
    """
    deg, M = (
        G.dw,
        G.num_edges,
    )
    ones = torch.ones(2)
    Pibot = torch.eye(2) - torch.outer(ones, ones) / 2

    # cost function for the edge
    def subgraph_cost(A, edge, w):
        deg_new = 2 * deg[edge] - w
        L = torch.tensor([[deg_new[0], -w], [-w, deg_new[1]]])
        B = Pibot @ A[edge, :]
        structural_cost = torch.linalg.norm(B.T @ L @ B) ** 2

        # semantic cost
        if hasattr(G, "embeddings") and G.embeddings is not None:
            # Pi_C^{\perp} H for the two-node subgraph (edge)
            Bh = Pibot @ G.embeddings[edge, :]

            # compute || Pi_C^{\perp} H ||_{L_C}^2  = || B^T L B ||_F^2
            semantic_cost = torch.linalg.norm(Bh.T @ L @ Bh) ** 2
        else:
            semantic_cost = 0.0

        return [structural_cost, semantic_cost, structural_cost + semantic_cost]

    # edges = np.array(G.get_edge_list()[0:2])
    weights = torch.tensor(
        [subgraph_cost(A, G.edge_index[:, e], G.edge_weight[e])[1] for e in range(M)]
    )
    # weights = torch.zeros(M)
    # for e in range(M):
    #    weights[e] = subgraph_cost_old(G, A, edges[:,e])

    if algorithm == "optimal":
        # identify the minimum weight matching
        coarsening_list = matching_optimal(G, weights=weights, r=r)

    elif algorithm == "greedy":
        # find a heavy weight matching
        coarsening_list, sigma_l, done_flag = matching_greedy(
            G,
            weights=weights,
            r=r,
            similarity_threshold=similarity_threshold,
            max_sigma=max_sigma,
        )

    return coarsening_list, sigma_l, done_flag


def contract_variation_edges(
    G: Data,
    A=None,
    r=0.5,
    algorithm="greedy",
    similarity_threshold=0.0,
    max_sigma=float("inf"),
):
    """
    Sequential contraction with local variation and edge-based families.
    This is a specialized implementation for the edge-based family, that works
    slightly faster than the contract_variation() function, which works for
    any family.

    See contract_variation() for documentation.
    """
    deg, M = (
        G.dw,
        G.num_edges,
    )

    # Vectorized cost computation for all edges at once
    # Mathematical derivation: for 2-node subgraph with Pibot projection,
    # cost = 0.25 * (deg[i] + deg[j])^2 * ||A[i] - A[j]||^4
    src = G.edge_index[0]  # source nodes, shape (M,)
    tgt = G.edge_index[1]  # target nodes, shape (M,)

    # Compute A differences for all edges: shape (M, K)
    A_diff = A[src] - A[tgt]

    # Squared norms of differences: shape (M,)
    diff_norm_sq = torch.sum(A_diff * A_diff, dim=1)

    # Sum of degrees for each edge: shape (M,)
    deg_sum = deg[src] + deg[tgt]

    # Vectorized structural cost: 0.25 * (deg_sum)^2 * ||diff||^4
    weights = 0.25 * (deg_sum**2) * (diff_norm_sq**2)

    if algorithm == "optimal":
        # identify the minimum weight matching
        coarsening_list = matching_optimal(G, weights=weights, r=r)
    elif algorithm == "greedy":
        # find a heavy weight matching
        coarsening_list, sigma_l, done_flag = matching_greedy(
            G,
            weights=weights,
            r=r,
            max_sigma=max_sigma,
            similarity_threshold=similarity_threshold,
        )

    return coarsening_list, sigma_l, done_flag


# TODO(WHT): include features for each node
def contract_variation_linear(
    G: Data,
    A,
    r=0.5,
    mode="neighborhood",
    similarity_threshold=0.0,
    max_sigma=float("inf"),
):
    """
    Sequential contraction with local variation and general families.
    This is an implemmentation that improves running speed,
    at the expense of being more greedy (and thus having slightly larger error).

    See contract_variation() for documentation.
    """

    N, deg, W_lil = G.num_nodes, G.dw, G.W

    # The following is correct only for a single level of coarsening.

    # cost function for the subgraph induced by nodes array
    def subgraph_cost(nodes):
        if not isinstance(nodes, torch.Tensor):
            nodes = torch.tensor(nodes, dtype=torch.long)
        nc = len(nodes)
        if nc <= 1:
            return 0.0

        # Create a sparse permutation matrix P
        # P will be of size nc x N where nc = len(nodes)
        # P[i,j] = 1 if j is the i-th node in 'nodes', otherwise 0
        P_indices = torch.stack(
            [
                torch.arange(nc, device=W_lil.device),  # Row indices
                nodes,  # Column indices
            ]
        )
        P_values = torch.ones(nc, device=W_lil.device)
        P = torch.sparse_coo_tensor(P_indices, P_values, (nc, N), device=W_lil.device)

        # Extract the subgraph weight matrix using P @ W_lil @ P.T
        # First compute P @ W_lil
        PW = torch.sparse.mm(P, W_lil)
        # Then compute (P @ W_lil) @ P.T
        W_sub = torch.sparse.mm(PW, P.t())

        # Compute subgraph degree vector
        d_sub = torch.sparse.sum(W_sub, dim=1).to_dense()

        # Create Laplacian using original degrees and subgraph connections
        diag_values = 2 * deg[nodes] - d_sub
        L_diag = torch.sparse_coo_tensor(
            torch.stack([torch.arange(nc, device=W_lil.device)] * 2),
            diag_values,
            (nc, nc),
            device=W_lil.device,
        )
        L = L_diag - W_sub

        # Extract A submatrix using the same permutation approach
        A_sub = A[nodes, :]

        # Compute projection matrix
        ones = torch.ones(nc, device=W_lil.device) / nc
        P_A_sub = A_sub - torch.outer(ones, torch.sum(A_sub, dim=0))

        # Compute the final cost
        if L.is_sparse:
            L_P_A = torch.sparse.mm(L, P_A_sub)
        else:
            L_P_A = L @ P_A_sub

        structural_cost = torch.norm(P_A_sub.t() @ L_P_A) / (nc - 1)
        return structural_cost

    class CandidateSet:
        def __init__(self, candidate_list):
            self.set = candidate_list
            self.cost = subgraph_cost(candidate_list)

        def __lt__(self, other):
            return self.cost < other.cost

    family = []
    # W_bool = G.A + sparse_eye(G.num_nodes)
    if "neighborhood" in mode:
        for i in range(N):
            # i_set = G.A[i,:].indices # graph_utils.get_neighbors(G, i)
            # i_set = np.append(i_set, i)
            # i_set = W_bool[i, :].indices
            i_set = k_hop_subgraph(i, 1, G.edge_index, num_nodes=G.num_nodes)[0]
            family.append(CandidateSet(i_set))

    if "cliques" in mode:
        import networkx.convert_matrix as nx

        Gnx = nx.from_scipy_sparse_matrix(G.W)
        for clique in nx.find_cliques(Gnx):
            family.append(CandidateSet(torch.tensor(clique)))

    else:
        if "edges" in mode:
            for e in range(0, G.edge_index.shape[1]):
                family.append(CandidateSet(G.edge_index[:, e]))
        if "triangles" in mode:
            triangles = set([])
            edges = G.edge_index
            for e in range(0, edges.shape[1]):
                [u, v] = edges[:, e]
                for w in range(G.num_nodes):
                    if G.W[u, w] > 0 and G.W[v, w] > 0:
                        triangles.add(frozenset([u, v, w]))
            triangles = list(map(lambda x: torch.tensor(list(x)), triangles))
            for triangle in triangles:
                family.append(CandidateSet(triangle))

    family = SortedList(family)
    marked = torch.zeros(G.num_nodes, dtype=torch.bool)

    # ----------------------------------------------------------------------------
    # Construct a (minimum weight) independent set.
    # ----------------------------------------------------------------------------
    coarsening_list = []
    # n, n_target = N, (1-r)*N
    n_reduce = np.floor(r * N)  # how many nodes do we need to reduce/eliminate?
    X = G.embeddings if hasattr(G, "embeddings") else G.x
    max_sigma2 = max_sigma**2
    sigma_l_2 = 0.0
    count = 0
    done_flag = False

    while len(family) > 0:

        i_cset = family.pop(index=0)
        i_set = i_cset.set

        if len(i_set) <= 1:
            continue

        # check cost threshold
        if (sigma_l_2 + i_cset.cost) > max_sigma2:
            if count == 0:
                done_flag = True
            break

        # check if marked
        i_marked = marked[i_set]

        if not any(i_marked):

            n_gain = len(i_set) - 1
            if n_gain > n_reduce:
                continue  # this helps avoid over-reducing

            if X is not None and len(i_set) > 1:
                sim = similarity(X[i_set])
            else:
                sim = 1.0

            # print(f"{sim=}, {len(i_set)=}, {i_set=}, {X[i_set]=}")

            # probability based on similarity merging
            if sim > similarity_threshold:

                # all vertices are unmarked: add i_set to the coarsening list
                marked[i_set] = True
                coarsening_list.append(
                    {"list": i_set, "similarity": sim, "cost": i_cset.cost}
                )
                # n -= len(i_set) - 1
                n_reduce -= n_gain

            # if n <= n_target: break
            if n_reduce <= 0:
                break

            count += 1
            sigma_l_2 += i_cset.cost

        # may be worth to keep this set
        # else:
        #     i_set = i_set[~i_marked]
        #     if len(i_set) > 1:
        #         # todo1: check whether to add to coarsening_list before adding to family
        #         # todo2: currently this will also select contraction sets that are disconnected
        #         # should we eliminate those?
        #         i_cset.set = i_set
        #         i_cset.cost = subgraph_cost(i_set)
        #         family.add(i_cset)
    sigma_l = np.sqrt(sigma_l_2)
    return coarsening_list, sigma_l, done_flag


################################################################################
# Edge-based contraction algorithms
################################################################################


def get_proximity_measure(G: Data, name, K=10):

    N = G.num_nodes
    W = G.W  # np.array(G.W.toarray(), dtype=np.float32)
    deg = degree(G.edge_index, G.num_nodes, G.edge_weight)  # np.sum(W, axis=0)
    edges = G.edge_index
    weights = G.edge_weight
    M = edges.shape[1]

    num_vectors = K  # int(1*K*np.log(K))
    if "lanczos" in name:
        L = G.L  # Assuming this returns a sparse tensor
        # Ensure sparse tensor is coalesced for lobpcg compatibility
        if L.is_sparse:
            L = L.coalesce()
        # torch.lobpcg has issues with sparse tensors in some PyTorch versions.
        # Use dense eigendecomposition for reliability.
        L_dense = L.to_dense() if L.is_sparse else L
        # Get smallest K eigenvalues/eigenvectors (excluding zero eigenvalue)
        l_all, X_all = torch.linalg.eigh(L_dense)
        # Take the smallest K+1 eigenvectors (skip the first one which is constant)
        l_lan = l_all[1 : K + 1]
        X_lan = X_all[:, 1 : K + 1]
    elif "cheby" in name:
        e = "laplacian of G"
        X_cheby = generate_test_vectors(
            G, num_vectors=num_vectors, method="Chebychev", lambda_cut=G.e[K + 1]
        )
    elif "JC" in name:
        X_jc = generate_test_vectors(
            G, num_vectors=num_vectors, method="JC", iterations=20
        )
    elif "GS" in name:
        X_gs = generate_test_vectors(
            G, num_vectors=num_vectors, method="GS", iterations=1
        )
    if "expected" in name:
        X = G.embeddings
        assert not torch.isnan(X).any()
        assert X.shape[0] == N
        K = X.shape[1]

    proximity = torch.zeros(M, dtype=torch.float32)

    # heuristic for mutligrid
    if name == "heavy_edge":
        wmax = torch.tensor(torch.max(G.W, 0).todense())[0] + 1e-5
        for e in range(0, M):
            proximity[e] = weights[e] / max(
                wmax[edges[:, e]]
            )  # select edges with large proximity
        return proximity

    # heuristic for mutligrid
    elif name == "algebraic_JC":
        proximity += torch.inf
        for e in range(0, M):
            i, j = edges[:, e]
            for kIdx in range(num_vectors):
                xk = X_jc[:, kIdx]
                proximity[e] = min(
                    proximity[e], 1 / max(torch.abs(xk[i] - xk[j]) ** 2, 1e-6)
                )  # select edges with large proximity

        return proximity

    # heuristic for mutligrid
    elif name == "affinity_GS":
        c = torch.zeros((N, N))
        for e in range(0, M):
            i, j = edges[:, e]
            c[i, j] = (X_gs[i, :] @ X_gs[j, :].T) ** 2 / (
                (X_gs[i, :] @ X_gs[i, :].T) ** 2 * (X_gs[j, :] @ X_gs[j, :].T) ** 2
            )  # select

        c += c.T
        c -= torch.diag(torch.diag(c))
        for e in range(0, M):
            i, j = edges[:, e]
            proximity[e] = c[i, j] / (max(c[i, :]) * max(c[j, :]))

        return proximity

    for e in range(0, M):
        i, j = edges[:, e]

        if name == "heavy_edge_degree":
            proximity[e] = (
                deg[i] + deg[j] + 2 * G.W[i, j]
            )  # select edges with large proximity

        # loose as little information as possible (custom)
        elif "min_expected_loss" in name:
            for kIdx in range(1, K):
                xk = X[:, kIdx]
                proximity[e] = sum(
                    [proximity[e], (xk[i] - xk[j]) ** 2]
                )  # select edges with small proximity
        # loose as little gradient information as possible (custom)
        elif name == "min_expected_gradient_loss":
            for kIdx in range(1, K):
                xk = X[:, kIdx]
                proximity[e] = sum(
                    [
                        proximity[e],
                        (xk[i] - xk[j]) ** 2 * (deg[i] + deg[j] + 2 * G.W[i, j]),
                    ]
                )  # select edges with small proximity

        # relaxation ensuring that K first eigenspaces are aligned (custom)
        elif name == "rss":
            for kIdx in range(1, K):
                xk = G.U[:, kIdx]
                lk = G.e[kIdx]
                proximity[e] = sum(
                    [
                        proximity[e],
                        (xk[i] - xk[j]) ** 2
                        * ((deg[i] + deg[j] + 2 * G.W[i, j]) / 4)
                        / lk,
                    ]
                )  # select edges with small proximity

        # fast relaxation ensuring that K first eigenspaces are aligned (custom)
        elif name == "rss_lanczos":
            for kIdx in range(1, K):
                xk = X_lan[:, kIdx]
                lk = l_lan[kIdx]
                proximity[e] = sum(
                    [
                        proximity[e],
                        (xk[i] - xk[j]) ** 2
                        * ((deg[i] + deg[j] + 2 * G.W[i, j]) / 4 - 0.5 * (lk + lk))
                        / lk,
                    ]
                )  # select edges with small proximity

        # approximate relaxation ensuring that K first eigenspaces are aligned (custom)
        elif name == "rss_cheby":
            for kIdx in range(num_vectors):
                xk = X_cheby[:, kIdx]
                lk = xk.T @ G.L @ xk
                proximity[e] = sum(
                    [
                        proximity[e],
                        (
                            (xk[i] - xk[j]) ** 2
                            * ((deg[i] + deg[j] + 2 * G.W[i, j]) / 4 - 0 * lk)
                            / lk
                        ),
                    ]
                )  # select edges with small proximity

        # heuristic for mutligrid (algebraic multigrid)
        elif name == "algebraic_GS":
            proximity[e] = torch.inf
            for kIdx in range(num_vectors):
                xk = X_gs[:, kIdx]
                proximity[e] = min(
                    proximity[e], 1 / max(torch.abs(xk[i] - xk[j]) ** 2, 1e-6)
                )  # select edges with large proximity

    if ("rss" in name) or ("expected" in name):
        proximity = -proximity

    return proximity


def generate_test_vectors(
    G, num_vectors=10, method="Gauss-Seidel", iterations=5, lambda_cut=0.1
):

    L = G.L
    N = G.num_nodes
    X = torch.randn(N, num_vectors) / torch.sqrt(N)

    if method == "GS" or method == "Gauss-Seidel":
        # Extract upper triangular part (excluding diagonal)
        indices = L.indices()
        values = L.values()

        # Create mask for upper triangular elements (i < j)
        upper_mask = indices[0] < indices[1]
        # Create mask for lower triangular elements (including diagonal) (i >= j)
        lower_diag_mask = indices[0] >= indices[1]

        # Create L_upper (strictly upper triangular part)
        L_upper_indices = indices[:, upper_mask]
        L_upper_values = values[upper_mask]
        L_upper = torch.sparse_coo_tensor(L_upper_indices, L_upper_values, L.shape)

        # Create L_lower_diag (lower triangular part including diagonal)
        L_lower_diag_indices = indices[:, lower_diag_mask]
        L_lower_diag_values = values[lower_diag_mask]
        L_lower_diag = torch.sparse_coo_tensor(
            L_lower_diag_indices, L_lower_diag_values, L.shape
        )

        # Convert to CSR format for more efficient operations
        L_upper = L_upper.coalesce()
        L_lower_diag = L_lower_diag.coalesce()

        for j in range(num_vectors):
            x = X[:, j]
            for t in range(iterations):
                # Compute right hand side: -L_upper @ x
                rhs = -torch.sparse.mm(L_upper, x.unsqueeze(1)).squeeze()

            # Solve L_lower_diag @ x_new = rhs using forward substitution
            x_new = torch.zeros_like(x)
            for i in range(N):
                # Get row i of L_lower_diag
                row_indices = (L_lower_diag.indices()[0] == i).nonzero().squeeze()
                if len(row_indices.shape) == 0:  # Handle scalar case
                    row_indices = row_indices.unsqueeze(0)

                col_indices = L_lower_diag.indices()[1][row_indices]
                row_values = L_lower_diag.values()[row_indices]

                # Compute x[i] = (rhs[i] - sum(L[i,j] * x[j] for j < i)) / L[i,i]
                # Note: We only need to consider j < i for forward substitution
                diag_idx = (col_indices == i).nonzero().squeeze()
                if (
                    len(diag_idx.shape) == 0 and diag_idx.nelement() > 0
                ):  # Handle scalar case
                    diag_idx = diag_idx.unsqueeze(0)

                if diag_idx.nelement() > 0:  # Check if diagonal element exists
                    diag_val = row_values[diag_idx]
                off_diag_mask = col_indices < i
                if off_diag_mask.any():
                    off_diag_sum = torch.sum(
                        row_values[off_diag_mask] * x_new[col_indices[off_diag_mask]]
                    )
                    x_new[i] = (rhs[i] - off_diag_sum) / diag_val
                else:
                    x_new[i] = rhs[i] / diag_val

            x = x_new

            X[:, j] = x
        return X

    if method == "JC" or method == "Jacobi":

        deg = degree(G.edge_index, G.num_nodes, G.edge_weight)
        indices = torch.arange(len(deg)).repeat(2, 1)
        D = torch.sparse_coo_tensor(indices, deg, (len(deg), len(deg)))
        # D = sp.sparse.diags(deg, 0)
        deginv = deg ** (-1)
        deginv[deginv == torch.inf] = 0
        Dinv = torch.sparse_coo_tensor(indices, deginv, (len(deginv), len(deginv)))
        M = Dinv.dot(D - L)

        for j in range(num_vectors):
            x = X[:, j]
            for t in range(iterations):
                x = 0.5 * x + 0.5 * M.dot(x)
            X[:, j] = x
        return X

    elif method == "Chebychev":
        from pygsp import filters

        f = filters.Filter(G, lambda x: ((x <= lambda_cut) * 1).astype(torch.float32))
        return f.filter(X, method="chebyshev", order=50)


def matching_optimal(edge_index, num_nodes, weights, r=0.4):
    """
    Generates a matching optimally with the objective of minimizing the total
    weight of all edges in the matching.

    Parameters
    ----------
    G : pygsp graph
    weights : np.array(M)
        a weight for each edge
    ratio : float
        The desired dimensionality reduction (ratio = 1 - n/N)

    Notes:
    * The complexity of this is O(N^3)
    * Depending on G, the algorithm might fail to return ratios>0.3
    """
    N = num_nodes

    # the edge set
    edges = edge_index
    M = edges.shape[1]

    max_weight = 1 * torch.max(weights)

    # prepare the input for the minimum weight matching problem
    edge_list = []
    for edgeIdx in range(M):
        [i, j] = edges[:, edgeIdx]
        if i == j:
            continue
        edge_list.append((i, j, max_weight - weights[edgeIdx]))

    assert min(weights) >= 0

    # solve it
    tmp = torch.tensor(maxWeightMatching(edge_list))

    # format output
    m = tmp.shape[0]
    matching = torch.zeros((m, 2), dtype=int)
    matching[:, 0] = torch.arange(m)
    matching[:, 1] = tmp

    # remove null edges and duplicates
    idx = torch.where(tmp != -1)[0]
    matching = matching[idx, :]
    idx = torch.where(matching[:, 0] > matching[:, 1])[0]
    matching = matching[idx, :]

    assert matching.shape[0] >= 1

    # if the returned matching is larger than what is requested, select the min weight subset of it
    matched_weights = torch.zeros(matching.shape[0])
    for mIdx in range(matching.shape[0]):
        i = matching[mIdx, 0]
        j = matching[mIdx, 1]
        eIdx = [
            e
            for e, t in enumerate(edges[:, :].T)
            if ((t == [i, j]).all() or (t == [j, i]).all())
        ]
        matched_weights[mIdx] = weights[eIdx]

    keep = min(int(torch.ceil(r * N)), matching.shape[0])
    if keep < matching.shape[0]:
        idx = torch.argsort(matched_weights)[:keep]
        matching = matching[idx, :]
        weights = matched_weights[idx]

    return matching


def matching_greedy(
    G: Data,
    weights,
    r=0.4,
    similarity_threshold=0.0,
    max_sigma=float("inf"),
):
    """
    Generates a matching greedily by selecting at each iteration the edge
    with the largest weight and then removing all adjacent edges from the
    candidate set.

    Parameters
    ----------
    G : pygsp graph
    weights : np.array(M)
        a weight for each edge
    X : np.array(N, D), optional
        node features, used for computing similarity
    r : float
        The desired dimensionality reduction (r = 1 - n/N)
    similarity_threshold : float, optional
        The threshold for considering two nodes as similar
    max_sigma : float
        The threshold for considering two nodes as similar
    max_cost : float, optional
        The maximum variation cost after full coarsening
    cur_cost : float, optional
        The current variation cost before coarsening

    Notes:
    * The complexity of this is O(M)
    * Depending on G, the algorithm might fail to return ratios>0.3
    """

    done_flag = False
    N = G.num_nodes
    X = G.embeddings if hasattr(G, "embeddings") else G.x

    # the edge set
    edges = G.edge_index.clone()

    idx = torch.argsort(weights)
    edges = edges[:, idx]
    weights = weights[idx]

    # the candidate edge set
    candidate_edges = edges.T.tolist()

    # the matching edge set (this is a list of arrays)
    matching = []
    max_sigma2 = max_sigma**2

    # which vertices have been selected
    marked = torch.zeros(N, dtype=torch.bool)

    n = N
    if r is not None:
        n_target = (1 - r) * N
    else:
        n_target = None
    count = 0
    sigma_l_2 = 0  # cumulative cost sigma^2
    T = len(candidate_edges)
    while count <= T - 1:
        # pop a candidate edge
        [i, j] = candidate_edges[count]
        cost = weights[count]
        count += 1

        # check cost threshold
        if (sigma_l_2 + cost) > max_sigma2:
            if count == 1:
                done_flag = True
            break

        # check if marked
        if any(marked[[i, j]]):
            continue

        if X is not None:
            sim = similarity(X[[i, j]])
        else:
            sim = 1.0

        if sim >= similarity_threshold:
            marked[[i, j]] = True
            n -= 1

            # add it to the matching
            # matching.append(torch.tensor([i, j]))
            matching.append({"list": [i, j], "similarity": sim, "cost": cost})

            sigma_l_2 += cost
        # termination condition
        if r is not None:
            if n <= n_target:
                break

    sigma_l = sigma_l_2**0.5
    return matching, sigma_l, done_flag
