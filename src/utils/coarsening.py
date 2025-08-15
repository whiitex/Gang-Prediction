from pygsp import graphs
from graphcoarsening.graph_coarsening.coarsening_utils import *
from graphcoarsening.graph_coarsening.graph_utils import *

def apply_Loukas_coarsening(G: graphs.Graph, X=None, method='variation_neighborhoods', ratio=0.5, K=3, similarity_threshold=0.65, log_info=False):
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

def create_pygsp_graph(edges, num_nodes, weights=None) -> graphs.Graph:
    """
    - edges: Mx2 NUMPY array, each row is an edge (source, target)
    - weights: Mx1 NUMPY array, weights corresponding to each edge
    - num_nodes: int, total number of nodes in the graph
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