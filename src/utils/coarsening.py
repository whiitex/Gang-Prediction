from pygsp import graphs
from graphcoarsening.graph_coarsening.coarsening_utils import *
from graphcoarsening.graph_coarsening.graph_utils import *

def apply_Loukas_coarsening(G: graphs.Graph, X=None, method='variation_neighborhoods', ratio=0.5, K=3, similarity_threshold=0.65, max_levels=10, log_info=False):
    """
    Output:
      - C: coarsening matrix (n, N)
      - Gc: coarsened graph (n, n)
      - Call: all coarsened graphs (levels, n_l, N)
      - Gall: all original graphs (levels, n_l, n_l)
    """
    
    available_methods = ['variation_neighborhoods', 'variation_edges', 'heavy_edge', 'algebraic_JC', 'kron'] 
    if method not in available_methods:
        raise ValueError(f"Unknown coarsening method: {method}. Method must be one of {available_methods}.")
    
    C, Gc, Call, Gall = coarsen(G, X=X, K=K, method=method, r=ratio, similarity_threshold=similarity_threshold)

    if log_info:
        print(f"Coarsening: {method} ", end='')
        print(f"({G.N} n, {G.Ne} e) -> ({Gc.N} n, {Gc.Ne} e); ")
    
    return C, Gc, Call, Gall

def create_pygsp_graph_noduplicatecheck(edges, num_nodes, weights=None) -> graphs.Graph:
    """
    Input:
      - edges: Mx2 NUMPY array, each row is an edge (source, target)
      - weights: Mx1 NUMPY array, weights corresponding to each edge
      - num_nodes: int, total number of nodes in the graph
    Output:
      - G: pygsp Graph object with the adjacency matrix constructed from edges and weights
    Note: This function does not account for duplicate edges.
    """

    if weights is None: weights = np.ones(len(edges))
    
    # adjacency matrix
    adj_matrix = sp.sparse.coo_matrix(
        (weights, (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes)
    )
    adj_matrix = adj_matrix + adj_matrix.T
    
    # pygsp graph
    G = graphs.Graph(adj_matrix)
    G.compute_laplacian()
    # G.set_coordinates()
    
    return G

def create_pygsp_graph(edges, num_nodes, weights=None) -> graphs.Graph:
    if weights is None: weights = np.ones(len(edges))
    
    edges_bidirectional = np.vstack([edges, edges[:, [1, 0]]])  # add reverse edges
    weights_bidirectional = np.hstack([weights, weights])  # duplicate weights
    
    # remove duplicate edges
    edge_set = set()
    unique_edges = []
    unique_weights = []
    mapping = {}
    
    for i, (u, v) in enumerate(edges_bidirectional):
        edge = tuple(sorted([u, v]))
        if edge not in edge_set:
            edge_set.add(edge)
            unique_edges.append([u, v])
            unique_weights.append(weights_bidirectional[i])
            mapping[tuple(sorted([u, v]))] = len(unique_edges) - 1
        else:
            unique_weights[mapping[tuple(sorted([u, v]))]] += weights_bidirectional[i]

    unique_edges = np.array(unique_edges)
    unique_weights = np.array(unique_weights)
    
    # symmetric adjacency matrix
    adj_matrix = sp.sparse.coo_matrix(
        (unique_weights, (unique_edges[:, 0], unique_edges[:, 1])),
        shape=(num_nodes, num_nodes)
    )

    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    adj_matrix = adj_matrix.tocsr()
    
    adj_matrix.setdiag(0)
    adj_matrix.eliminate_zeros()
    
    # pygsp graph
    G = graphs.Graph(adj_matrix)    
    G.compute_laplacian()
    
    return G