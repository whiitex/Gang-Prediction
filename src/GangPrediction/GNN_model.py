"""GNN model definition and training/evaluation helpers."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GraphConv, GINConv
import torch.optim as optim

# torch geometric
from torch_geometric.data import Data

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from src.GangPrediction.pattern_models import Pattern
from src.ml.metrics.metrics import average_precision_score

from src.GangPrediction.experiment_utils import (
    _capture_pattern_lineage,
    get_node_to_supernode_mapping,
)

from torch_geometric.nn import MessagePassing


class WeightedGINConv(MessagePassing):
    def __init__(self, nn_mlp, eps=0.0, train_eps=False):
        super().__init__(aggr="add")
        self.nn = nn_mlp
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.tensor([eps], dtype=torch.float))
        else:
            self.register_buffer("eps", torch.tensor([eps], dtype=torch.float))

    def forward(self, x, edge_index, edge_weight=None):
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = out + (1 + self.eps) * x
        return self.nn(out)

    def message(self, x_j, edge_weight):
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j


class GCN(nn.Module):

    # def __init__(self, nfeat, nhid, nclass, dropout=0.5, n_layers=4):
    #     super(GCN, self).__init__()
    #     if n_layers < 2:
    #         raise ValueError("n_layers must be at least 2")
    #     self.dropout = dropout
    #     self.n_layers = n_layers

    #     n_hid1 = 8
    #     n_hid2 = 16
    #     n_hid3 = 32

    #     # Build list of conv layers dynamically
    #     self.convs = nn.ModuleList()
    #     # GINConv requires an MLP as input
    #     self.convs.append(
    #         GINConv(
    #             nn.Sequential(
    #                 nn.Linear(nfeat, n_hid1),
    #                 nn.ReLU(),
    #                 nn.Linear(n_hid1, n_hid1),
    #             ),
    #         )
    #         # GCNConv(nfeat, nhid)
    #     )  # First layer
    #     self.convs.append(
    #         GINConv(
    #             nn.Sequential(
    #                 nn.Linear(n_hid1, n_hid2),
    #                 nn.ReLU(),
    #                 nn.Linear(n_hid2, n_hid2),
    #             ),
    #         )
    #         # GCNConv(nfeat, nhid)
    #     )  # First layer
    #     self.convs.append(
    #         GINConv(
    #             nn.Sequential(
    #                 nn.Linear(n_hid2, n_hid3),
    #                 nn.ReLU(),
    #                 nn.Linear(n_hid3, n_hid3),
    #             ),
    #         )
    #         # GCNConv(nfeat, nhid)
    #     )  # First layer
    #     # for _ in range(n_layers - 2):  # Hidden layers
    #     #     self.convs.append(
    #     #         GINConv(
    #     #             nn.Sequential(
    #     #                 nn.Linear(n_hid1, n_hid2),
    #     #                 nn.ReLU(),
    #     #                 nn.Linear(n_hid2, n_hid2),
    #     #             )
    #     #         )
    #     #         # GCNConv(nhid, nhid)
    #     #     )
    #     self.convs.append(
    #         GINConv(
    #             nn.Sequential(
    #                 nn.Linear(n_hid3, n_hid3),
    #                 nn.ReLU(),
    #                 nn.Linear(n_hid3, nclass),
    #             ),
    #         )
    #         # GCNConv(nhid, nclass)
    #     )  # Output layer

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, n_layers=4):
        super(GCN, self).__init__()
        if n_layers < 2:
            raise ValueError("n_layers must be at least 2")
        self.dropout = dropout
        self.n_layers = n_layers

        # Build list of conv layers dynamically
        self.convs = nn.ModuleList()
        # GINConv requires an MLP as input
        self.convs.append(
            # GINConv(
            WeightedGINConv(
                nn.Sequential(
                    nn.Linear(nfeat, nhid),
                    nn.ReLU(),
                    nn.Linear(nhid, nhid),
                ),
            )
            # GCNConv(nfeat, nhid)
        )  # First layer
        for _ in range(n_layers - 2):  # Hidden layers
            self.convs.append(
                WeightedGINConv(
                    nn.Sequential(
                        nn.Linear(nhid, nhid),
                        nn.ReLU(),
                        nn.Linear(nhid, nhid),
                    )
                )
                # GCNConv(nhid, nhid)
            )
        self.convs.append(
            WeightedGINConv(
                nn.Sequential(
                    nn.Linear(nhid, nhid),
                    nn.ReLU(),
                    nn.Linear(nhid, nclass),
                ),
            )
            # GCNConv(nhid, nclass)
        )  # Output layer

    def forward(self, x, edge_index, edge_weight=None):
        """Compute logits for each node."""
        for i, conv in enumerate(self.convs[:-1]):
            x = F.relu(conv(x, edge_index, edge_weight))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return x

    def get_embeddings(self, x, edge_index, edge_weight=None):
        """Return intermediate node embeddings (pre-classifier)."""
        for conv in self.convs[:-2]:
            x = F.relu(conv(x, edge_index, edge_weight))
        x = self.convs[-2](x, edge_index, edge_weight)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


def train_gnn_1_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    data: Data,
    C_plus=None,
    P=None,
    L=None,
    y=None,
    train_idx: list = None,
    coarse_loss: bool = False,
    class_weights: torch.Tensor = None,
):
    """
    Output:
        - train_loss
        - val_loss
        - val_accuracy
    """
    # One training pass with optional coarse-to-fine projection.

    # data = data.to(next(model.parameters()).device)

    model.train()
    optimizer.zero_grad()

    # S_mp = data.S_mp if hasattr(data, "S_mp") else data.W
    y = y if y is not None else data.y
    train_idx = train_idx if train_idx is not None else data.train_idx

    logits = model(
        data.x,
        data.edge_index,
        data.edge_weight if hasattr(data, "edge_weight") else None,
    )
    if class_weights is not None:
        class_weights = class_weights.to(logits.device)
        logits = logits * class_weights.unsqueeze(0)
    if C_plus is not None:
        logits = C_plus @ logits
        # pred_fine = F.log_softmax(C_plus @ logits, dim=1)
    # else:
    # pred_fine = F.log_softmax(logits, dim=1)
    pred_fine = F.softmax(logits, dim=1)
    # embeddings = model.get_embeddings(data.x, data.edge_index, data.edge_weight if hasattr(data, "edge_weight") else None)

    # train
    embeddings = data.embeddings if hasattr(data, "embeddings") else None
    # embeddings = model.get_embeddings(
    #     data.x,
    #     data.edge_index,
    #     data.edge_weight if hasattr(data, "edge_weight") else None,
    # )
    # embeddings = F.normalize(embeddings, p=2, dim=1)
    loss = criterion(
        logits, y, train_idx, embeddings, coarse_loss=coarse_loss, P=P, L=L
    )
    loss.backward()
    optimizer.step()
    train_acc = accuracy_score(
        y[train_idx].detach().cpu().numpy(),
        pred_fine[train_idx].max(1)[1].detach().cpu().numpy(),
    )

    # validate
    # model.eval()
    # with torch.no_grad():
    #     output = model(data.x, data.edge_index)
    #     # embeddings = model.get_embeddings(data.x, data.edge_index)

    #     # loss_val = criterion(output, data.embeddings, data.y, val_idx)
    #     # pred_val = output[val_idx].max(1)[1]
    #     # acc_val = accuracy_score(data.y[val_idx].cpu().numpy(), pred_val.cpu().numpy())

    return loss.item(), train_acc


def evaluate_model(
    model: nn.Module,
    data: Data,
    C=None,
    C_plus=None,
    original_data=None,
    alert_patterns=None,
    normal_patterns=None,
    alert_thresholds=(0.75, 0.75),
    normal_thresholds=(0.5, 0.5),
):
    """Evaluate the model on test set."""

    model.eval()
    # data = data.to(next(model.parameters()).device)

    with torch.no_grad():
        logits = model(
            data.x,
            data.edge_index,
            data.edge_weight if hasattr(data, "edge_weight") else None,
        )
        #
        output = F.softmax(logits, dim=1)
        # output = F.log_softmax(logits, dim=1)
        # if C_plus is not None:
        #     pred_test = C_plus @ output
        # else:
        #     pred_test = output
        logit_test = logits[data.test_idx]
        pred_test = output[data.test_idx].max(1)[1]
        acc_test = accuracy_score(
            data.y[data.test_idx].cpu().numpy(), pred_test.cpu().numpy()
        )

        # For binary classification with 2-class output, use positive class scores
        logit_test_scores = (
            logit_test[:, 1].cpu().numpy()
            if logit_test.dim() > 1 and logit_test.shape[1] == 2
            else logit_test.cpu().numpy()
        )
        precission = average_precision_score(
            data.y[data.test_idx].cpu().numpy(),
            logit_test_scores,
            recall_span=(0.6, 1.0),
        )

    suspicious_probs = output[:, 1] if output.shape[1] > 1 else output.squeeze()
    predictions_all = output.max(1)[1]

    if C is None:
        return {
            "accuracy_test": acc_test,
            "precision_test": precission,
        }

    # Project logits back to original graph size using C_plus
    logits_fine = C_plus @ logits
    pred_fine = torch.argmax(F.softmax(logits_fine, dim=1), dim=1)

    accuracy_fine = torch.sum(
        pred_fine[original_data.test_idx] == original_data.y[original_data.test_idx]
    ).item() / len(original_data.test_idx)

    # Use positive class PROBABILITIES (not logits) for precision calculation
    # AP requires scores in [0,1] range for proper thresholding
    probs_fine = F.softmax(logits_fine, dim=1)
    prob_fine_scores = (
        probs_fine[:, 1].cpu().numpy()
        if probs_fine.dim() > 1 and probs_fine.shape[1] == 2
        else probs_fine.cpu().numpy()
    )
    # Correct argument order: y_true first, y_scores second
    precission_fine = average_precision_score(
        original_data.y[original_data.test_idx].cpu().numpy(),
        prob_fine_scores[original_data.test_idx.cpu().numpy()],
        recall_span=(0.6, 1.0),
    )

    auc = roc_auc_score(
        original_data.y[original_data.test_idx].cpu().numpy(),
        prob_fine_scores[original_data.test_idx.cpu().numpy()],
    )
    ap = average_precision_score(
        original_data.y[original_data.test_idx].cpu().numpy(),
        prob_fine_scores[original_data.test_idx.cpu().numpy()],
    )

    results = {
        # "predictions": predictions_all,
        "suspicious_probs": suspicious_probs,
        "n_pred_suspicious": (
            int((predictions_all == 1).sum().item()) if output.shape[1] > 1 else 0
        ),
        "mean_prob": float(suspicious_probs.mean().item()),
        "max_prob": float(suspicious_probs.max().item()),
        "accuracy_fine": accuracy_fine,
        "precision_fine": precission_fine,
        "auc_fine": auc,
        "ap_fine": ap,
        "accuracy_test": acc_test,
        "precision_test": precission,
        "num_nodes_coarse": data.num_nodes,
    }

    node_to_supernode = get_node_to_supernode_mapping(C) if C is not None else None

    _capture_pattern_lineage(
        patterns=[*alert_patterns, *normal_patterns],
        node_to_supernode=node_to_supernode,
        pseudo_labels=probs_fine,
    )

    alert_metrics = Pattern.average_metrics(
        alert_patterns,
        majority_threshold=alert_thresholds[0],
        coarsening_threshold=alert_thresholds[1],
    )
    normal_metrics = Pattern.average_metrics(
        normal_patterns,
        majority_threshold=normal_thresholds[0],
        coarsening_threshold=normal_thresholds[1],
    )
    results["alert_metrics"] = alert_metrics
    results["normal_metrics"] = normal_metrics

    alert_pattern_types = set(p.pattern_type for p in alert_patterns)
    normal_pattern_types = set(p.pattern_type for p in normal_patterns)

    for pattern_type in alert_pattern_types:
        type_metrics = Pattern.average_metrics(
            [p for p in alert_patterns if p.pattern_type == pattern_type],
            majority_threshold=alert_thresholds[0],
            coarsening_threshold=alert_thresholds[1],
        )
        results[f"alert_metrics_{pattern_type}"] = type_metrics

    for pattern_type in normal_pattern_types:
        type_metrics = Pattern.average_metrics(
            [p for p in normal_patterns if p.pattern_type == pattern_type],
            majority_threshold=normal_thresholds[0],
            coarsening_threshold=normal_thresholds[1],
        )
        results[f"normal_metrics_{pattern_type}"] = type_metrics

    return results
