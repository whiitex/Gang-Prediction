import numpy as np
import pandas as pd

def load_cora_dataset(log_info=True):
    """
    Output:
      - node_ids: list (N) of node IDs
      - features: np.array (N, d) of features 
      - labels: np.array (N,) of correct labels 
      - edges_idx: np.array (M, 2) of edges
      - idx_map: dict mapping original node IDs to index
    """

    # nodes, features and labels
    content = pd.read_csv("../data/cora/cora.content", sep='\t', header=None)
    
    node_ids = content.iloc[:, 0].values
    features = content.iloc[:, 1:-1].values.astype(np.int32)
    labels = content.iloc[:, -1].values
    
    idx_map = {j: i for i, j in enumerate(node_ids)}
    
    # edges
    edges = pd.read_csv("../data/cora/cora.cites", sep='\t', header=None, names=['target', 'source'])
    
    edges_idx = []
    for _, edge in edges.iterrows():
        if edge['source'] in idx_map and edge['target'] in idx_map:
            edges_idx.append([idx_map[edge['source']], idx_map[edge['target']]])
    
    edges_idx = np.array(edges_idx)

    assert(len(node_ids) == 2708)
    assert(features.shape == (2708, 1433))
    assert(len(labels) == 2708)
    assert(edges_idx.shape == (5429, 2))
    assert(len(np.unique(labels)) == 7)
    
    if log_info:
        print(f"Dataset loaded:")
        print(f"- Nodes: {len(node_ids)}")
        print(f"- Edges: {len(edges_idx)}")
        print(f"- Features: {features.shape[1]}")
        print(f"- Classes: {len(np.unique(labels))}")

    return node_ids, features, labels, edges_idx, idx_map