import os
import torch
import pickle
from datasets import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


AML_DATA_DIR_PATH = "./data/"

def load_aml_transactions(dataset_root: str="AMLGentex", name: str="easy") -> list:
    with open(f"{AML_DATA_DIR_PATH}{dataset_root}/{name}/transactions.pkl", "rb") as f:
        return pickle.load(f)

def load_aml_features(dataset_root: str="AMLGentex", name: str="easy") -> torch.Tensor:
    return torch.load(f"{AML_DATA_DIR_PATH}{dataset_root}/{name}/features.pt", map_location=device)

def load_aml_dataset(
        dataset_root: str="AMLGentex", 
        name: str="easy",
        step_start: int=0, 
        steps_amount: int=100000, 
        extend_features: bool=False
) -> dict:
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
        - edges: torch.Tensor of shape (2, M) - each row is an edge (source, target)
        - edge_features: torch.Tensor of shape (M, D) - features corresponding to each edge
        - num_nodes: int - total number of nodes in the graph
        - features: torch.Tensor of shape (N, D) - N is the number of nodes and D is the feature dimension
        - ground_truth: torch.Tensor of shape (N, 1) - N is the number of nodes and the value is the label for each node
    """

    allowed_names = ["easy", "hard", "1M"]
    if name not in allowed_names:
        raise ValueError(f"name must be one of {allowed_names}")
    
    # verify dataset exists, if not - download it and save data
    if not os.path.exists(f"{AML_DATA_DIR_PATH}{dataset_root}/{name}"):
        save_aml_dataset(dataset_root=dataset_root, name=name)

    # load transactions and features from files
    transactions = load_aml_transactions(dataset_root=dataset_root, name=name)
    features = load_aml_features(dataset_root=dataset_root, name=name)
    num_nodes = features.shape[0]

    steps_amount = min(steps_amount, len(transactions) - step_start)
    if step_start < 0 or step_start >= len(transactions):
        raise ValueError(f"step_start must be in range [0, {len(transactions) - 1}]")
    
    edges = [[], []]
    weights = []
    edge_features = []
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

            if extend_features:
                # sent
                features[src, -4] += amount
                features[src, -3] += 1
                # received
                features[dst, -2] += amount
                features[dst, -1] += 1

            edges[0].append(src)
            edges[1].append(dst)
            edge_features.append([amount, type_])
            weights.append(amount)

    # convert to torch tensors
    edges = torch.tensor(edges, dtype=torch.long, device=device)
    edge_features = torch.tensor(edge_features, dtype=torch.float32, device=device)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    return {
        "edges": edges,
        "num_nodes": num_nodes,
        "features": features,
        "weights": weights,
        "edge_features": edge_features,
        "ground_truth": ground_truth
    }


def save_aml_dataset(dataset_root: str="AMLGentex", name: str="easy"):
    print(f"Downloading {dataset_root} {name} to {AML_DATA_DIR_PATH}{dataset_root}/{name}/")

    # create directory if not exists
    if not os.path.exists(f"{AML_DATA_DIR_PATH}"):
        os.mkdir(f"{AML_DATA_DIR_PATH}")
    if not os.path.exists(f"{AML_DATA_DIR_PATH}{dataset_root}"):
        os.mkdir(f"{AML_DATA_DIR_PATH}{dataset_root}")
    if not os.path.exists(f"{AML_DATA_DIR_PATH}{dataset_root}/{name}"):
        os.mkdir(f"{AML_DATA_DIR_PATH}{dataset_root}/{name}")

    # loading (from huggingface) and processing
    ds = load_dataset(f"{dataset_root}/{name}")
    print(ds['train'].column_names)

    nodes_data = [[] for _ in range(2000000)] # N x D list of lists 
    bank_map, idx_bank = {}, 0
    transactions = [[] for _ in range(1000)]
    map_type = {
        'INITALBALANCE': 0,
        'CASH': 1,
        'TRANSFER': 2,
    }

    for r in ds['train']:
        id = r['nameOrig'] + 2
        if r['bankOrig'] not in bank_map:
            bank_map[r['bankOrig']] = idx_bank
            idx_bank += 1
        nodes_data[id] = [bank_map[r['bankOrig']], int(r['phoneChangesOrig']), int(r['daysInBankOrig'])]

        id = r['nameDest'] + 2
        if r['bankDest'] not in bank_map:
            bank_map[r['bankDest']] = idx_bank
            idx_bank += 1
        nodes_data[id] = [bank_map[r['bankDest']], int(r['phoneChangesDest']), int(r['daysInBankDest'])]

        transactions[r['step']].append([
            int(r['nameOrig'] + 2),
            int(r['nameDest'] + 2),
            float(r['amount']),
            int(map_type[r['type']]),
            int(r['isSAR']),
            int(r['alertID']),
            int(r['modelType'])
        ])


    while nodes_data[-1] == []: nodes_data.pop()
    features = torch.tensor(nodes_data, dtype=torch.float32, device=device)
    torch.save(features, f"{AML_DATA_DIR_PATH}{dataset_root}/{name}/features.pt")

    while transactions[-1] == []: transactions.pop()
    with open(f"{AML_DATA_DIR_PATH}{dataset_root}/{name}/transactions.pkl", "wb") as f:
        pickle.dump(transactions, f)
