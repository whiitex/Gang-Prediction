"""Graph coarsening utilities based on variation-aware contractions."""

from copy import deepcopy
import scipy
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from tqdm import tqdm

from src.GangPrediction.maxWeightMatching import maxWeightMatching
from src.GangPrediction.utils.utils import *


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
        Gc, B, sigma_l, done_flag = coarse_one_level(
            Gc,
            B,
            method=method,
            algorithm=algorithm,
            level=level,
            r_cur=r,
            max_sigma=max_sigma,
        )

        iC = Gc.C
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
    method="variation_neighborhood",
    algorithm="greedy",
    level=1,
    r_cur=None,
    max_sigma=float("inf"),
    use_label_for_coarsening=False,
):
    """Coarsen a single level using a variation-based or matching-based policy."""

    done_flag = False

    if level == 0:
        A = B
    else:
        d, V = torch.linalg.eigh(B.T @ G.L @ B)
        # X_init = torch.randn(B.shape[1], K, device=G.L.device)
        # d, V = torch.lobpcg(
        #     B.T @ G.L @ B, k=K, X=X_init, largest=False, niter=50, tol=1e-4
        # )
        # d = torch.ones(K, device=G.L.device)
        # V = torch.eye(K, device=G.L.device)
        mask = d < 1e-10
        d[mask] = 1
        # dinvsqrt = d ** (0)
        if method == "gang_edges":
            dinvsqrt = d ** (+0.5)
        else:
            dinvsqrt = d ** (-0.5)
        # dinvsqrt = d ** (-0.5)
        dinvsqrt[mask] = 0
        A = B @ V @ torch.diag(dinvsqrt) @ V.T

    if method == "variation_neighborhood":
        coarsening_list, sigma_l, done_flag = contract_variation_linear(
            G,
            A=A,
            r=r_cur,
            mode=method,
            max_sigma=max_sigma,
        )
    elif method == "variation_star":
        coarsening_list, sigma_l, done_flag = contract_variation_star(
            G,
            A=A,
            r=r_cur,
            max_sigma=max_sigma,
            use_label_for_coarsening=use_label_for_coarsening,
        )
    else:
        coarsening_list, sigma_l, done_flag = contract_variation_edges(
            G,
            A=A,
            r=r_cur,
            algorithm=algorithm,
            max_sigma=max_sigma,
            use_label_for_coarsening=use_label_for_coarsening,
        )

    iC = get_coarsening_matrix(coarsening_list, G.num_nodes)

    Gc, B = construct_G(G, B, iC)

    return Gc, B, sigma_l, done_flag


def calc_L_plus(Gc):
    """Compute the square root of the Laplacian for variation-based coarsening."""
    d, V = torch.linalg.eigh(Gc.L.to_dense())
    # d, V = torch.lobpcg(Gc.L, k=Gc.num_nodes, X=torch.randn(Gc.num_nodes, Gc.num_nodes), largest=False, niter=200, tol=1e-4)
    mask = d < 1e-10
    d[mask] = 1
    dinvsqrt = d ** (-0.5)
    dinvsqrt[mask] = 0
    L_plus = V @ torch.diag(dinvsqrt) @ V.T
    return L_plus


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


def construct_G(G: Data, B: torch.Tensor, iC: SparseTensor):
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

    B = torch.sparse.mm(Gc.C, B)

    return Gc, B


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
    # L_c = (
    #     L_c + L_c.t()
    # ) / 2  # this is only needed to avoid pygsp complaining for tiny errors
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
    max_sigma=float("inf"),
    use_label_for_coarsening=False,
):
    """
    Sequential contraction with local variation and edge-based families.
    This is a specialized implementation for the edge-based family, that works
    slightly faster than the contract_variation() function, which works for
    any family.

    See contract_variation() for documentation.
    """
    deg = G.dw

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

    # Enforce same-label constraint: set weight to infinity for edges connecting different labels
    if use_label_for_coarsening and hasattr(G, "y_pred") and G.y_pred is not None:
        different_labels = G.y_pred[src] != G.y_pred[tgt]
        weights[different_labels] = float("inf")

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
        )

    return coarsening_list, sigma_l, done_flag


def contract_variation_star(
    G: Data,
    A=None,
    r=0.5,
    max_sigma=float("inf"),
    use_label_for_coarsening=False,
):
    """
    Contraction using edge-based variation costs with star-shaped merging.

    Same fast vectorized cost computation as variation_edges, but the greedy
    matching allows a representative node to absorb multiple fresh neighbors
    in a single level. This handles fan-in/fan-out patterns without needing
    multiple hierarchical levels.
    """
    deg = G.dw
    src = G.edge_index[0]
    tgt = G.edge_index[1]

    A_diff = A[src] - A[tgt]
    diff_norm_sq = torch.sum(A_diff * A_diff, dim=1)
    deg_sum = deg[src] + deg[tgt]
    weights = 0.25 * (deg_sum**2) * (diff_norm_sq**2)

    if use_label_for_coarsening and hasattr(G, "y_pred") and G.y_pred is not None:
        different_labels = G.y_pred[src] != G.y_pred[tgt]
        weights[different_labels] = float("inf")

    coarsening_list, sigma_l, done_flag = matching_greedy_star(
        G,
        weights=weights,
        r=r,
        max_sigma=max_sigma,
    )
    return coarsening_list, sigma_l, done_flag


# TODO(WHT): include features for each node
def contract_variation_linear(
    G: Data,
    A,
    r=0.5,
    mode="neighborhood",
    max_sigma=float("inf"),
):
    """
    Sequential contraction with local variation and neighborhood-based families.
    Greedily contracts the lowest-cost 1-hop neighborhood at each step.

    See contract_variation_edges() for the analogous edge-based version.
    """
    N = G.num_nodes
    deg = G.dw
    edge_index = G.edge_index
    edge_weight = G.edge_weight
    device = edge_index.device

    # --- Build adjacency list from edge_index (O(M)) ---
    src, tgt = edge_index[0], edge_index[1]
    adj_nodes = [[] for _ in range(N)]  # neighbor indices
    adj_weights = [[] for _ in range(N)]  # edge weights
    for e in range(edge_index.shape[1]):
        u, v = src[e].item(), tgt[e].item()
        adj_nodes[u].append(v)
        adj_weights[u].append(edge_weight[e].item())

    # --- Compute cost for every 1-hop neighborhood ---
    neighborhoods = [None] * N
    costs = torch.full((N,), float("inf"))

    for i in range(N):
        # Neighborhood = {i} ∪ N(i)
        nbrs = adj_nodes[i]
        nodes = torch.tensor([i] + nbrs, dtype=torch.long, device=device)
        nodes = torch.unique(nodes)
        neighborhoods[i] = nodes
        nc = len(nodes)
        if nc <= 1:
            continue

        # Build small dense W_sub via adjacency lookup
        idx_map = {n.item(): k for k, n in enumerate(nodes)}
        W_sub = torch.zeros(nc, nc, device=device)
        for u_local, u in enumerate(nodes.tolist()):
            for v, w in zip(adj_nodes[u], adj_weights[u]):
                if v in idx_map:
                    W_sub[u_local, idx_map[v]] = w

        # Subgraph Laplacian (same convention as contract_variation_edges)
        d_sub = W_sub.sum(dim=1)
        L_sub = torch.diag(2 * deg[nodes] - d_sub) - W_sub

        # Project A onto orthogonal complement of the constant vector
        A_sub = A[nodes]
        P_A = A_sub - A_sub.mean(dim=0)

        # Cost = ||P_A^T L P_A||_F / (nc - 1)
        L_P_A = L_sub @ P_A
        costs[i] = torch.norm(P_A.T @ L_P_A) / (nc - 1)

    # --- Greedy independent-set selection (sorted by ascending cost) ---
    idx = torch.argsort(costs)

    marked = torch.zeros(N, dtype=torch.bool)
    coarsening_list = []
    n_reduce = np.floor(r * N)
    max_sigma2 = max_sigma**2
    sigma_l_2 = 0.0
    count = 0
    done_flag = False

    for k in range(N):
        i = idx[k].item()
        nodes = neighborhoods[i]
        cost = costs[i].item()

        if len(nodes) <= 1:
            continue
        if cost == float("inf"):
            break

        # check cost threshold
        if (sigma_l_2 + cost) > max_sigma2:
            if count == 0:
                done_flag = True
            break

        # skip if any node already contracted
        if marked[nodes].any():
            continue

        n_gain = len(nodes) - 1
        if n_gain > n_reduce:
            continue  # avoid over-reducing

        marked[nodes] = True
        coarsening_list.append({"list": nodes, "cost": cost})
        n_reduce -= n_gain
        sigma_l_2 += cost
        count += 1

        if n_reduce <= 0:
            break

    sigma_l = np.sqrt(sigma_l_2)
    return coarsening_list, sigma_l, done_flag


################################################################################
# Edge-based contraction algorithms
################################################################################


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

    # the edge set
    edges = G.edge_index

    # Deduplicate bidirectional edges: keep only (u,v) where u < v
    src, tgt = edges[0], edges[1]
    keep = src < tgt
    edges = edges[:, keep]
    weights = weights[keep]

    idx = torch.argsort(weights, stable=False)
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
        if r is not None:
            if n <= n_target:
                break
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

        marked[[i, j]] = True
        n -= 1

        # add it to the matching
        matching.append({"list": [i, j], "cost": cost})

        sigma_l_2 += cost
        # termination condition

    sigma_l = sigma_l_2**0.5
    return matching, sigma_l, done_flag


def matching_greedy_star(
    G: Data,
    weights,
    r=0.4,
    max_sigma=float("inf"),
    deg_ratio=500.5,
):
    """
    Adaptive greedy matching that combines star and pairwise strategies.

    For each edge in ascending cost order, the higher-degree endpoint is the
    candidate representative (rep) and the lower-degree one is the leaf.

    - If rep's degree >= deg_ratio * leaf's degree (hub pattern), rep stays
      open after the merge and can absorb additional fresh neighbors later
      (star behaviour for fan-in / fan-out).
    - Otherwise both nodes are locked after the merge (pairwise behaviour,
      optimal for cycles and regular subgraphs).

    Parameters
    ----------
    deg_ratio : float
        Degree ratio threshold to decide star vs pairwise merging.
        Default 1.5.  Set to inf to get pure pairwise (=variation_edges).
        Set to 1.0 to get pure star.
    """
    done_flag = False
    N = G.num_nodes
    edges = G.edge_index
    deg = G.dw

    # Deduplicate bidirectional edges: keep only (u,v) where u < v
    src, tgt = edges[0], edges[1]
    keep = src < tgt
    edges = edges[:, keep]
    weights = weights[keep]

    idx = torch.argsort(weights, stable=False)
    edges = edges[:, idx]
    weights = weights[idx]

    candidate_edges = edges.T.tolist()
    max_sigma2 = max_sigma**2

    # Node states:
    #   absorbed[i] = True  → node i has been absorbed into another supernode
    #   rep_of[i]   >= 0    → node i is the representative of group rep_of[i]
    absorbed = torch.zeros(N, dtype=torch.bool)
    rep_of = torch.full((N,), -1, dtype=torch.long)

    groups = []  # list of {"members": [int, ...], "cost": float}
    n = N
    n_target = (1 - r) * N if r is not None else None
    sigma_l_2 = 0.0
    count = 0

    T = len(candidate_edges)
    for t in range(T):
        if n_target is not None and n <= n_target:
            break

        [u, v] = candidate_edges[t]
        cost = weights[t].item()

        if (sigma_l_2 + cost) > max_sigma2:
            if count == 0:
                done_flag = True
            break

        # Orient edge so that rep = higher-degree node, leaf = lower-degree node.
        if deg[u] >= deg[v]:
            rep, leaf = u, v
        else:
            rep, leaf = v, u

        # leaf must be fresh (not absorbed, not a representative)
        if absorbed[leaf] or rep_of[leaf] >= 0:
            continue
        # rep must not be absorbed (but can already be a representative)
        if absorbed[rep]:
            continue

        # Decide star vs pairwise based on degree ratio
        is_hub = deg[rep] >= deg_ratio * deg[leaf]

        if rep_of[rep] >= 0:
            # rep is already a representative → extend its group with leaf
            # (only reachable when rep was kept open by a previous hub merge)
            g = rep_of[rep].item()
            groups[g]["members"].append(leaf)
            groups[g]["cost"] += cost
        else:
            # Both fresh → create new group with rep as representative
            g = len(groups)
            groups.append({"members": [rep, leaf], "cost": cost})
            if is_hub:
                rep_of[rep] = g  # keep rep open for more absorptions

        absorbed[leaf] = True
        if not is_hub:
            # Pairwise mode: lock rep too (like variation_edges)
            absorbed[rep] = True
            n -= 2
        else:
            n -= 1

        sigma_l_2 += cost
        count += 1

    coarsening_list = [{"list": g["members"], "cost": g["cost"]} for g in groups]
    sigma_l = sigma_l_2**0.5
    return coarsening_list, sigma_l, done_flag
