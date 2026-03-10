"""GNN model definition and training/evaluation helpers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GraphConv, GINConv
import torch.optim as optim

# torch geometric
from torch_geometric.data import Data

from sklearn.metrics import accuracy_score, classification_report

from src.ml.metrics.metrics import average_precision_score


class GCN(nn.Module):
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
            GINConv(
                nn.Sequential(nn.Linear(nfeat, nhid), nn.ReLU(), nn.Linear(nhid, nhid))
            )
        )  # First layer
        for _ in range(n_layers - 2):  # Hidden layers
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(nhid, nhid), nn.ReLU(), nn.Linear(nhid, nhid)
                    )
                )
            )
        self.convs.append(
            GINConv(
                nn.Sequential(nn.Linear(nhid, nhid), nn.ReLU(), nn.Linear(nhid, nclass))
            )
        )  # Output layer

    def forward(self, x, edge_index, edge_weight=None):
        """Compute logits for each node."""
        for i, conv in enumerate(self.convs[:-1]):
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def get_embeddings(self, x, edge_index, edge_weight=None):
        """Return intermediate node embeddings (pre-classifier)."""
        for conv in self.convs[:-2]:
            x = F.relu(conv(x, edge_index))
        x = self.convs[-2](x, edge_index)
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
    loss = criterion(logits, y, train_idx, embeddings, coarse_loss=coarse_loss)
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


def evaluate_model(model: nn.Module, data: Data, log_info=True):
    """Evaluate the model on test set."""
    model.eval()
    # data = data.to(next(model.parameters()).device)
    S_mp = data.S_mp if hasattr(data, "S_mp") else data.W

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

        if log_info:
            print(f"\nTest Accuracy: {acc_test:.4f}")
            print("\nClassification Report:")
            print(
                classification_report(
                    data.y[data.test_idx].cpu().numpy(), pred_test.cpu().numpy()
                )
            )

    return acc_test, precission, pred_test, logits
