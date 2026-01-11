"""Training entrypoints for the GNN baseline and the coarsening-aware variant."""

import os
import sys
import numpy as np
from tqdm import tqdm

from src.ml.metrics.metrics import average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

# torch
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch_geometric.utils import to_networkx
import networkx as nx

# graph coarsening - Loukas 2020
from src.GangPrediction.coarsening_utils import *
from src.GangPrediction.graph_utils import *

# utils
from src.GangPrediction.utils.utils import *

from src.GangPrediction.GNN_model import GCN, train_gnn_1_epoch, evaluate_model
from src.GangPrediction.coarsening_aware_loss import *


def compute_class_weights(
    labels: torch.Tensor, train_idx: torch.Tensor = None, device: str = "cpu"
) -> torch.Tensor:
    """
    Compute class weights based on inverse frequency for handling class imbalance.

    Args:
        labels: Tensor of shape [N] with class labels
        train_idx: Optional tensor of training indices. If provided, weights are computed
                   only from training labels. Otherwise, uses all labels.
        device: Device to place the weights tensor on

    Returns:
        Tensor of shape [num_classes] with class weights (higher weight for minority class)
    """
    if train_idx is not None:
        train_labels = labels[train_idx]
    else:
        train_labels = labels

    num_classes = len(torch.unique(labels))
    class_counts = torch.zeros(num_classes, device=device)

    for c in range(num_classes):
        class_counts[c] = (train_labels == c).sum().float()

    # Avoid division by zero
    class_counts = torch.clamp(class_counts, min=1.0)

    # Inverse frequency weighting
    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts)

    # Normalize so weights sum to num_classes (maintains loss scale)
    class_weights = class_weights * num_classes / class_weights.sum()

    return class_weights


def train_GNN(
    data,
    epochs,
    lr=0.01,
    wd=5e-4,
    nhid=128,
    dropout=0.1,
    device="cpu",
    use_class_weights=True,
):
    """Train a vanilla GCN on the full-resolution graph."""
    # model
    nclass = len(np.unique(data.y.numpy()))  # 7
    model = GCN(nfeat=data.num_features, nhid=nhid, nclass=nclass, dropout=dropout).to(
        device
    )

    # Compute class weights for imbalanced data
    class_weights = None
    if use_class_weights:
        train_idx = data.train_idx if hasattr(data, "train_idx") else None
        class_weights = compute_class_weights(data.y, train_idx, device=device)
        print(f"Using class weights: {class_weights}")

    # criterion
    criterion = CoarseningAwareLoss(class_weights=class_weights)

    # train data
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    bar = tqdm(total=epochs)
    for epoch in range(epochs):
        train_loss, train_acc = train_gnn_1_epoch(
            model,
            optimizer,
            criterion,
            data,
            coarse_loss=False,
        )
        bar.set_postfix_str(
            f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}"
        )
        bar.update(1)

    return model


def train_GNN_coarsening_aware_loss(
    data: Data,
    levels: int,
    epoch_per_level: int,
    lr=0.01,  # Reduced from 0.01 for stability
    wd=5e-4,
    method="variation_neighborhoods",
    algorithm: str = "greedy",
    similarity_threshold=0.65,
    K=50,
    nhid=128,
    dropout=0.1,
    device="cpu",
    # save_path="results",
    create_gif=True,
    use_class_weights=True,
    grad_clip=1.0,  # Gradient clipping for stability
    warmup_epochs=3,  # Extra epochs at start of each level
):
    """Train a GCN across coarsening levels while applying a coarsening-aware loss."""
    # model
    N = data.num_nodes  # 2708
    nclass = len(np.unique(data.y.numpy()))  # 7
    model = GCN(nfeat=data.num_features, nhid=nhid, nclass=nclass, dropout=dropout).to(
        device
    )

    # Compute class weights for imbalanced data
    class_weights = None
    if use_class_weights:
        train_idx = data.train_idx if hasattr(data, "train_idx") else None
        class_weights = compute_class_weights(data.y, train_idx, device=device)
        print(f"Using class weights: {class_weights}")

    # criterion
    criterion = CoarseningAwareLoss(class_weights=class_weights)

    # train data
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Learning rate scheduler - reduce LR when loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )

    original_data = data
    Gc = data

    # Create one-hot encoding of labels for soft label aggregation at coarse levels.
    device = Gc.y.device
    num_classes = len(torch.unique(Gc.y))

    labels_onehot = torch.zeros(len(Gc.y), num_classes, device=device)
    labels_onehot.scatter_(1, Gc.y.view(-1, 1), 1)
    Gc.soft_y = labels_onehot

    Gc.y_train = torch.zeros_like(Gc.soft_y)
    Gc.y_train[Gc.train_idx, :] = Gc.soft_y[Gc.train_idx, :]

    Gc.y_val = torch.zeros_like(Gc.soft_y)
    Gc.y_val[Gc.val_idx, :] = Gc.soft_y[Gc.val_idx, :]

    Gc.y_test = torch.zeros_like(Gc.soft_y)
    Gc.y_test[Gc.test_idx, :] = Gc.soft_y[Gc.test_idx, :]

    x, ycrs, yfine, prec_l, prec_fine, ylosst, ylossv, valacc, num_nodes_coarse = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # Initialize coarsening matrices and a layout for visualization/debugging.
    C = sparse_eye(N)
    C_plus = sparse_eye(N)
    GG = to_networkx(Gc, to_undirected=True)
    pos = nx.spring_layout(GG)
    pos = [vals for vals in pos.values()]
    Gc.pos = torch.tensor(pos, dtype=torch.float32, device=device)

    if method == "variation_embedding":
        B = calc_B_embedding(Gc, K)
    else:
        B = calc_B(Gc, K)
    iC = None

    Call, Gall, iCs = [], [], []
    Gall.append(Gc)
    # Call.append(C)

    bar = tqdm(total=levels)
    prev_accuracy_fine = 0.0

    for level in range(1, levels + 1):
        ratio = np.log(level ** (4 / 3)) / 100 + 0.01
        # ratio = 1
        # get embeddings from the GNN
        S_mp = Gc.S_mp if hasattr(Gc, "S_mp") else Gc.W
        embeddings = model.get_embeddings(Gc.x, S_mp)
        Gc.embeddings = F.normalize(embeddings, p=2, dim=1)

        Gc, B = coarse_one_level(
            Gc,
            B,
            K=K,
            method=method,
            algorithm=algorithm,
            similarity_threshold=similarity_threshold,
            level=level,
            r_cur=ratio,
        )
        C = torch.sparse.mm(Gc.C, C)
        C_plus = torch.sparse.mm(C_plus, Gc.C_plus)

        Gall.append(Gc)
        Call.append(C)
        iCs.append(iC)

        # Add warmup epochs at level transitions for stability
        total_epochs = epoch_per_level + (warmup_epochs if level <= 5 else 1)

        for epoch in range(total_epochs):
            # train the GNN on the coarsened graph
            train_loss, train_acc = train_gnn_1_epoch(
                model,
                optimizer,
                criterion,
                Gc,
                C_plus,
                original_data.y,
                original_data.train_idx,
                coarse_loss=epoch == 0,
            )

            # Gradient clipping for stability
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Update learning rate based on training loss
        scheduler.step(train_loss)

        # evaluate the model on the coarsened graph
        acc_test_c, precission_c, pred_test_c, logits = evaluate_model(
            model, Gc, log_info=False
        )
        acc_test, precission, pred_test, _ = evaluate_model(
            model, original_data, log_info=False
        )

        pred_fine = torch.argmax(F.softmax(C_plus @ logits, dim=1), dim=1)
        # pred_fine = torch.argmax(F.log_softmax(C_plus @ logits, dim=1), dim=1)

        accuracy_fine = torch.sum(
            pred_fine[original_data.test_idx] == original_data.y[original_data.test_idx]
        ).item() / len(original_data.test_idx)
        precission_fine = average_precision_score(
            pred_fine[original_data.test_idx].cpu().numpy(),
            original_data.y[original_data.test_idx].cpu().numpy(),
            recall_span=(0.6, 1.0),
        )

        # Adaptive coarsening: slow down if accuracy drops significantly
        if level > 1 and (accuracy_fine < prev_accuracy_fine - 0.02):
            ratio *= 0.7  # Reduce coarsening rate
        elif level > 1 and accuracy_fine >= prev_accuracy_fine:
            ratio = min(ratio * 1.1, 0.99)  # Can be more aggressive

        prev_accuracy_fine = accuracy_fine

        bar.set_postfix_str(
            f"{epoch_per_level}/{similarity_threshold:0.2f}, r: {ratio:.2f}, nodes: {Gc.num_nodes} "
            # + f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
            + f"coarse: {acc_test_c:.4f}, orig: {acc_test:.4f} fine: {accuracy_fine:.4f}"
        )
        bar.update(1)

        # ploting data
        x.append(epoch + 1)
        ycrs.append(acc_test_c)
        yfine.append(accuracy_fine)
        ylosst.append(train_loss)
        # ylossv.append(validation_loss)
        # valacc.append(validation_accuracy)
        num_nodes_coarse.append(Gc.num_nodes)
        prec_l.append(precission_c)
        prec_fine.append(precission_fine)

    name = f"data_gnn_CoarseningAwareLoss_V2_th_{similarity_threshold*100:.0f}_epochs_{epoch_per_level}.npy"
    np.save(
        f"{save_path}{name}",
        {
            "x": x,
            "ycrs": ycrs,
            "yfine": yfine,
            "ylosst": ylosst,
            "prec_l": prec_l,
            "prec_fine": prec_fine,
            # "Gall": Gall,
            # "Call": Call,
            # "ylossv": ylossv,
            # "valacc": valacc,
            "num_nodes_coarse": num_nodes_coarse,
            "description": f"Data obtained using: {method=}, {similarity_threshold=}, CoarseningAwareLoss() levels approach",
        },
    )

    return Gall, Call, model, C_plus

    # plt.show()
