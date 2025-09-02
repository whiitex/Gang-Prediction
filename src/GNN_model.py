import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
import torch.optim as optim

# torch geometric
from torch_geometric.data import Data

from sklearn.metrics import accuracy_score, classification_report


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(nfeat, nhid)
        self.conv2 = SAGEConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
        # return F.log_softmax(x, dim=1)

    def get_embeddings(self, x, edge_index):
        # x = x.to(next(self.parameters()).device)  # send x to the model's device
        # edge_index = edge_index.to(x.device)
        x = F.relu(self.conv1(x, edge_index))
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


def train_gnn_1_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    data: Data,
    C_plus=None,
    y=None,
    train_idx: list = None,
    coarse_loss: bool = False,
):
    """
    Output:
        - train_loss
        - val_loss
        - val_accuracy
    """

    # data = data.to(next(model.parameters()).device)

    model.train()
    optimizer.zero_grad()

    S_mp = data.S_mp if hasattr(data, "S_mp") else data.W
    y = y if y is not None else data.y
    train_idx = train_idx if train_idx is not None else data.train_idx

    logits = model(data.x, S_mp)
    if C_plus is not None:
        pred_fine = F.softmax(C_plus @ logits, dim=1)
        # pred_fine = F.log_softmax(C_plus @ logits, dim=1)
    else:
        pred_fine = F.softmax(logits, dim=1)
        # pred_fine = F.log_softmax(logits, dim=1)
    # embeddings = model.get_embeddings(data.x, data.edge_index)

    # train
    embeddings = data.embeddings if hasattr(data, "embeddings") else None
    loss = criterion(pred_fine, y, train_idx, embeddings, coarse_loss=coarse_loss)
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
        logits = model(data.x, S_mp)
        output = F.softmax(logits, dim=1)
        # output = F.log_softmax(logits, dim=1)
        # if C_plus is not None:
        #     pred_test = C_plus @ output
        # else:
        #     pred_test = output
        pred_test = output[data.test_idx].max(1)[1]
        acc_test = accuracy_score(
            data.y[data.test_idx].cpu().numpy(), pred_test.cpu().numpy()
        )

        if log_info:
            print(f"\nTest Accuracy: {acc_test:.4f}")
            print("\nClassification Report:")
            print(
                classification_report(
                    data.y[data.test_idx].cpu().numpy(), pred_test.cpu().numpy()
                )
            )

    return acc_test, pred_test, logits
