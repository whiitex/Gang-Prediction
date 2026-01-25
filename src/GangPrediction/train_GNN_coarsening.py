"""Training entrypoints for the GNN baseline and the coarsening-aware variant."""

import os
import sys
import numpy as np
from tqdm import tqdm

from src.GangPrediction.experiment_utils import evaluate_at_level
from src.ml.metrics.metrics import average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

# torch
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import to_networkx
import networkx as nx

# graph coarsening - Loukas 2020
from src.GangPrediction.coarsening_utils import *
from src.GangPrediction.graph_utils import *

# utils
from src.GangPrediction.utils.utils import *

from src.GangPrediction.GNN_model import GCN, train_gnn_1_epoch, evaluate_model
from src.GangPrediction.coarsening_aware_loss import *


def train_GNN(
    data,
    epochs,
    lr=0.01,
    wd=5e-4,
    nhid=128,
    dropout=0.1,
    device="cpu",
    use_class_weights=True,
    **kwargs,
):
    """Train a vanilla GCN on the full-resolution graph."""
    # model
    nclass = len(np.unique(data.y.numpy()))  # 7
    # Keep nclass as-is for CrossEntropyLoss compatibility (don't reduce to 1 for binary)
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
    lr=0.01,
    wd=5e-4,
    method="variation_neighborhoods",
    algorithm: str = "greedy",
    similarity_threshold=0.0,
    max_epsilon=float("inf"),
    K=50,
    nhid=128,
    dropout=0.1,
    device="cpu",
    use_class_weights=True,
    B=None,
    alert_patterns=None,
    normal_patterns=None,
    alert_thresholds=0.5,
    normal_thresholds=0.5,
    prob_threshold=0.3,
    # Train patterns for contrastive loss (separate from eval patterns)
    alert_train_patterns=None,
    normal_train_patterns=None,
    # Dynamic epoch scheduling parameters
    initial_epochs=5,
    min_epochs=1,
    max_epoch_interval=3,
    loss_window=5,
    loss_threshold=0.01,
    train=False,
    model=None,
    # Pattern type dictionaries for per-type detection tracking
    alert_types=None,
    normal_types=None,
    **kwargs,
):
    """
    Train a GCN across coarsening levels while applying a coarsening-aware loss.

    Simple epoch schedule:
    - Start with initial_epochs (default 5) epochs per coarsening level
    - Gradually reduce to 1 epoch per 3 coarsening levels at the end
    """
    N = data.num_nodes
    nclass = len(np.unique(data.y.numpy()))

    if model is None:
        model = GCN(
            nfeat=data.num_features, nhid=nhid, nclass=nclass, dropout=dropout
        ).to(device=device)

    # Compute class weights for imbalanced data
    class_weights = None
    if use_class_weights:
        train_idx = data.train_idx if hasattr(data, "train_idx") else None
        class_weights = compute_class_weights(data.y, train_idx, device=device)
        # class_weights = torch.tensor([0.1, 0.9])
        print(f"Using class weights: {class_weights}")

    if train:
        # criterion - use train patterns for contrastive loss
        criterion = CoarseningAwareLoss(
            class_weights=class_weights,
            alert_patterns=alert_train_patterns,
            normal_patterns=normal_train_patterns,
        )

        # train data
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        # Learning rate scheduler - reduce LR when loss plateaus
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
        )
        # Track epochs for logging
        train_loss = 0.0  # Initialize for cases when epochs=0
    else:
        model.eval()

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
    alert_rates = []
    alert_rates_gt = []
    normal_rates_gt = []
    # Rate1 and Rate2 for detailed tracking
    alert_rates_rate1 = []
    alert_rates_rate2 = []
    alert_gt_type_rates = []
    normal_gt_type_rates = []
    alert_rates_gt_rate1 = []
    alert_rates_gt_rate2 = []
    normal_rates_gt_rate1 = []
    normal_rates_gt_rate2 = []
    # Per-type detection rates
    alert_model_type_rates = []

    # Initialize coarsening matrices and a layout for visualization/debugging.
    C = sparse_eye(N)
    C_plus = sparse_eye(N)

    if B is None:
        if method == "variation_embedding":
            print("Calculating B with embedding variation...")
            B = calc_B_embedding(Gc, K)
        else:
            print("Calculating B with embedding variation...")
            B = calc_B(Gc, K)

    Call, Gall = [np.eye(N)], [Gc]

    bar = tqdm(total=levels)
    epsilon_l, epsilons = 0, []

    # Dynamic epoch scheduling state
    current_epochs = initial_epochs  # Start with initial_epochs per level
    epoch_interval = 1  # Train every level initially (increases to max_epoch_interval)
    loss_history = []  # Track training losses for convergence detection
    epochs_per_level_history = []  # Track for saving

    for level in range(1, levels + 1):
        # Simple ratio schedule (original formula)
        ratio = np.log(level) / 2000 + 0.0001
        if ratio > 0.0025:
            ratio = 0.0025
        # ratio = 1

        # Get embeddings from the GNN in eval mode with no_grad
        # This avoids injecting dropout randomness and prevents backprop through stale representations
        # model.eval()
        # with torch.no_grad():
        if train:
            model.train()  # Switch back to train mode for the training step
        embeddings = model.get_embeddings(Gc.x, Gc.edge_index, Gc.edge_weight)
        Gc.embeddings = F.normalize(embeddings, p=2, dim=1).detach()

        # max_eps_in_level += max_epsilon / levels
        max_sigma = (max_epsilon + 1) / (epsilon_l + 1) - 1

        Gc, B, sigma_l, done_flag = coarse_one_level(
            Gc,
            B,
            K=K,
            method=method,
            algorithm=algorithm,
            similarity_threshold=similarity_threshold,
            level=level,
            r_cur=ratio,
            max_sigma=max_sigma,
        )
        C = torch.sparse.mm(Gc.C, C)
        C_plus = torch.sparse.mm(C_plus, Gc.C_plus)

        Gall.append(Gc)
        Call.append(C)

        epsilon_l = (sigma_l + 1) * (epsilon_l + 1) - 1
        if done_flag or epsilon_l >= max_epsilon:
            print(
                f"Reached max epsilon {max_epsilon} at level {level} with epsilon_l {epsilon_l:.3f}."
            )
            break
        epsilons.append(epsilon_l)

        if train:
            # Dynamic epoch scheduling: only train if at the right interval
            if level % epoch_interval == 0:
                for epoch in range(current_epochs):
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
                        class_weights=class_weights,
                    )

                # Track loss for convergence detection
                loss_history.append(train_loss)
                epochs_per_level_history.append(current_epochs)

                # Adapt epochs based on loss convergence rate
                if len(loss_history) >= loss_window:
                    recent_losses = loss_history[-loss_window:]
                    # Calculate average loss change rate over the window
                    loss_change_rate = (
                        abs(recent_losses[0] - recent_losses[-1]) / loss_window
                    )

                    # If loss is stabilizing (small changes), reduce training intensity
                    if loss_change_rate < loss_threshold:
                        if current_epochs > min_epochs:
                            # First reduce epochs per level
                            current_epochs = max(min_epochs, current_epochs - 1)
                        elif epoch_interval < max_epoch_interval:
                            # Then increase interval between training (train less frequently)
                            epoch_interval = min(max_epoch_interval, epoch_interval + 1)
                    elif loss_change_rate > loss_threshold * 3:
                        # If loss is changing a lot, we may need more training
                        if epoch_interval > 1:
                            epoch_interval = max(1, epoch_interval - 1)
                        elif current_epochs < initial_epochs:
                            current_epochs = min(initial_epochs, current_epochs + 1)
            else:
                # Skip training at this level, use last known values
                train_loss = loss_history[-1] if loss_history else 0.0
                epochs_per_level_history.append(0)
                scheduler.step(train_loss)

        else:
            train_loss = 0.0  # No training, so loss is 0

        # Evaluate the model on the coarsened graph
        acc_test_c, precission_c, pred_test_c, logits = evaluate_model(
            model, Gc, log_info=False
        )

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

        results = evaluate_at_level(
            model,
            Gc,
            C,
            data.y,
            alert_patterns,
            normal_patterns,
            alert_types=alert_types,
            normal_types=normal_types,
            alert_thresholds=alert_thresholds,
            normal_thresholds=normal_thresholds,
            prob_threshold=prob_threshold,
        )

        bar.set_postfix_str(
            f"ep={current_epochs}, {epsilon_l:.6f}/{max_epsilon:0.3f}, ratio={ratio:.6f}, nodes: {Gc.num_nodes} | "
            f"alert_rate: {results.get('alert_model_rate', 0):.4f}, normal_rate: {results.get('normal_gt_rate', 0):.4f} | "
            f"prec: {precission_fine:.4f}, coarse: {acc_test_c:.4f}, fine: {accuracy_fine:.4f} | "
        )
        bar.update(1)

        # Store data for plotting
        x.append(level)
        ycrs.append(acc_test_c)
        yfine.append(accuracy_fine)
        ylosst.append(train_loss if current_epochs > 0 else ylosst[-1] if ylosst else 0)
        num_nodes_coarse.append(Gc.num_nodes)
        prec_l.append(precission_c)
        prec_fine.append(precission_fine)
        alert_rates.append(results.get("alert_model_rate", 0))
        alert_rates_gt.append(results.get("alert_gt_rate", 0))
        normal_rates_gt.append(results.get("normal_gt_rate", 0))
        # Store rate1 and rate2 for detailed analysis
        alert_rates_rate1.append(results.get("alert_model_rate1", 0))
        alert_rates_rate2.append(results.get("alert_model_rate2", 0))
        alert_rates_gt_rate1.append(results.get("alert_gt_rate1", 0))
        alert_rates_gt_rate2.append(results.get("alert_gt_rate2", 0))
        normal_rates_gt_rate1.append(results.get("normal_gt_rate1", 0))
        normal_rates_gt_rate2.append(results.get("normal_gt_rate2", 0))
        # Store per-type detection rates
        alert_gt_type_rates.append(results.get("alert_gt_type_rates", {}))
        alert_model_type_rates.append(results.get("alert_model_type_rates", {}))
        normal_gt_type_rates.append(results.get("normal_gt_type_rates", {}))

    if train:
        name = f"data_gnn_CoarseningAwareLoss_V2_th_{similarity_threshold*100:.0f}_ep_{max_epsilon}_dynamic_train.npy"
    else:
        name = f"data_gnn_CoarseningAwareLoss_V2_th_{similarity_threshold*100:.0f}_ep_{max_epsilon}_dynamic_inference.npy"
    np.save(
        f"{save_path}{name}",
        {
            "x": x,
            "ycrs": ycrs,
            "yfine": yfine,
            "ylosst": ylosst,
            "prec_l": prec_l,
            "prec_fine": prec_fine,
            "epsilons": epsilons,
            "num_nodes_coarse": num_nodes_coarse,
            "epochs_per_level": epochs_per_level_history,
            "alert_rates": alert_rates,
            "alert_rates_gt": alert_rates_gt,
            "normal_rates_gt": normal_rates_gt,
            "alert_rates_rate1": alert_rates_rate1,
            "alert_rates_rate2": alert_rates_rate2,
            "alert_rates_gt_rate1": alert_rates_gt_rate1,
            "alert_rates_gt_rate2": alert_rates_gt_rate2,
            "normal_rates_gt_rate1": normal_rates_gt_rate1,
            "normal_rates_gt_rate2": normal_rates_gt_rate2,
            # Per-type detection rates
            "alert_gt_type_rates": alert_gt_type_rates,
            "alert_model_type_rates": alert_model_type_rates,
            "normal_gt_type_rates": normal_gt_type_rates,
        },
    )

    return Gall, Call, model, C_plus
