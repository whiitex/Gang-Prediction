import torch
import numpy as np
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIR = "../data/aml"
TRANSACTION_PATH = DIR + "/transactions.pkl"
FEATURES_PATH = DIR + "/features.pt"

def load_aml_transactions() -> list:
    with open(TRANSACTION_PATH, "rb") as f:
        return pickle.load(f)

def load_aml_features() -> torch.Tensor:
    return torch.load(FEATURES_PATH, map_location=device)

def load_aml_dataset(step_start: int=0, steps_amount: int=100000, extend_features: bool = False) -> dict:
    """
    Parameters:
    - step_start: int, starting step index (0-based)
    - steps_amount: int, number of steps to include in the dataset
    - extend_features: bool, if True, extends features with additional columns for amount and count
        + total amount money received/sent by the node
        + delta amount money 
        + total amount of transactions received/sent
        + total amount of transactions sent

    Returns:
    - edges: Mx2 NUMPY array - each row is an edge (source, target)
    - weights: Mx1 NUMPY array - weights corresponding to each edge
    - num_nodes: int - total number of nodes in the graph
    - features: torch.Tensor of shape (N, D) - N is the number of nodes and D is the feature dimension
    - ground_truth: Nx1 torch.Tensor - N is the number of nodes and the value is the label for each node
    """

    transactions = load_aml_transactions()
    features = load_aml_features()
    num_nodes = features.shape[0]

    steps_amount = min(steps_amount, len(transactions) - step_start)
    if step_start < 0 or step_start >= len(transactions):
        raise ValueError(f"step_start must be in range [0, {len(transactions) - 1}]")
    
    edges, weights = [], []
    ground_truth = torch.zeros(num_nodes, dtype=torch.long, device=device)

    if extend_features:
        features_ext = torch.zeros((num_nodes, features.shape[1] + 4), dtype=features.dtype, device=device)
        features_ext[:, :features.shape[1]] = features
        features = features_ext

    for step in range(step_start, step_start + steps_amount):
        for ts in transactions[step]:
            src, dst, amount, type_, is_sar, alert_id, model_type = ts
            
            if is_sar == 1:
                ground_truth[src] = 1
                ground_truth[dst] = 1
            
            edges.append([src, dst])
            weights.append(amount)

            if extend_features:
                features[src, -4] += amount
                features[dst, -4] += amount
                features[src, -3] += amount
                features[dst, -3] -= amount
                features[src, -2] += 1
                features[dst, -2] += 1
                features[src, -1] += 1

    return {
        "edges": np.array(edges),
        "weights": np.array(weights),
        "num_nodes": num_nodes,
        "features": features,
        "ground_truth": ground_truth
    }