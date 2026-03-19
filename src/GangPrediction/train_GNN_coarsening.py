"""Training entrypoints for the GNN baseline and the coarsening-aware variant."""

import os
import sys
import numpy as np
from tqdm import tqdm

from src.GangPrediction.experiment_utils import get_node_to_supernode_mapping
from src.GangPrediction.gang_aware_subspace import get_gang_aware_basis

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

# torch
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    # Train patterns for contrastive loss (separate from eval patterns)
    alert_train_patterns=None,
    normal_train_patterns=None,
    # Dynamic epoch scheduling parameters
    initial_epochs=5,
    min_epochs=1,
    max_epoch_interval=3,
    loss_window=5,
    loss_threshold=0.01,
    coarse_loss_epochs_per_level=1,
    train=False,
    model=None,
    **kwargs,
):
    """
    Train a GCN across coarsening levels while applying a coarsening-aware loss.

    Simple epoch schedule:
    - Start with initial_epochs (default 5) epochs per coarsening level
    - Gradually reduce to 1 epoch per 3 coarsening levels at the end
    - Apply the coarsening-aware loss for the first
      coarse_loss_epochs_per_level inner epochs at each level

    When use_learned_B=True:
    - B is computed from GNN embeddings (orthonormal basis via SVD)
    - SupernodeEmbeddingLoss enforces nodes in the same pattern to have similar embeddings
    - This allows learning the coarsening structure end-to-end
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
            use_supernode_loss=(
                True if method == "learning_subspace" else False
            ),  # Enable supernode loss when learning B
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

    # Initialize coarsening matrices and a layout for visualization/debugging.
    C = sparse_eye(N)
    C_plus = sparse_eye(N)

    # L_plus = calc_L_half(Gc)  # Precompute L_half for variation-based coarsening

    # Compute basis for coarsening using only TRAIN patterns
    if method == "edge_gangs":
        Uk = get_gang_aware_basis(
            data,
            # alert_patterns=alert_patterns,  # Use only train patterns
            # normal_patterns=normal_patterns,  # Use only train patterns
            alert_patterns=alert_train_patterns,  # Use only train patterns
            normal_patterns=normal_train_patterns,  # Use only train patterns
            K=K,
            alpha=kwargs["alpha"],
            method=kwargs["compression_method"],
        )
        B = calc_B(data, Uk.shape[1], U=Uk)  # Precompute eigenvectors
        # B = calc_B(G, K, U=Uk)  # Precompute eigenvectors
        print(f"Gang-aware basis shape: {B.shape}")
    elif method == "variation_embedding":
        Uk = calc_B_embedding(data, K)
        B = calc_B(data, K, U=Uk)
    elif method == "variation_edges":
        B = calc_B(data, K)
        # B2 = calc_B(G, K)
        # B = torch.concat([B1, B2], dim=1)
    elif method == "learning_subspace":
        # For learning_subspace, we will compute the basis dynamically during training
        B = None
    # else:

    Call, Gall = [np.eye(N)], [Gc]

    bar = tqdm(total=levels)
    epsilon_l, epsilons = 0, []

    # Dynamic epoch scheduling state
    current_epochs = initial_epochs  # Start with initial_epochs per level
    epoch_interval = 1  # Train every level initially (increases to max_epoch_interval)
    loss_history = []  # Track training losses for convergence detection
    epochs_per_level_history = []  # Track for saving
    results_history = []  # Track evaluation results per level
    for level in range(1, levels + 1):
        # Simple ratio schedule (original formula)
        ratio = np.log(level) / 10000 + 0.0001
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

        # Update B from embeddings if using learned B
        if method == "learning_subspace" and train:
            with torch.no_grad():
                G_embeddings = model.get_embeddings(
                    data.x, data.edge_index, data.edge_weight
                )
                # Recompute orthonormal basis from current embeddings
                V = calc_B_from_embeddings(G_embeddings)
                B_C = torch.sparse.mm(C, V)  # Update B for the current coarsened graph
                # B_C = calc_B_from_embeddings(embeddings)

        # max_eps_in_level += max_epsilon / levels
        max_sigma = (max_epsilon + 1) / (epsilon_l + 1) - 1

        Gc, B, sigma_l, done_flag = coarse_one_level(
            Gc,
            B=B_C if method == "learning_subspace" and train else B,
            method=method,
            algorithm=algorithm,
            similarity_threshold=similarity_threshold,
            level=level,
            r_cur=ratio,
            max_sigma=max_sigma,
        )
        C = torch.sparse.mm(Gc.C, C)
        C_plus = torch.sparse.mm(C_plus, Gc.C_plus)

        P = torch.sparse.mm(C_plus, C)

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
                active_coarse_loss_epochs = max(
                    0, min(current_epochs, coarse_loss_epochs_per_level)
                )
                for epoch in range(current_epochs):
                    # train the GNN on the coarsened graph
                    train_loss, train_acc = train_gnn_1_epoch(
                        model,
                        optimizer=optimizer,
                        criterion=criterion,
                        data=Gc,
                        C_plus=C_plus,
                        P=P,
                        L=data.L,
                        original_data=original_data,
                        coarse_loss=epoch < active_coarse_loss_epochs,
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
                # scheduler.step(train_loss)

        else:
            train_loss = 0.0  # No training, so loss is 0

        # Evaluate the model on the coarsened graph and compute pattern metrics.
        results = evaluate_model(
            model,
            Gc,
            C=C,
            C_plus=C_plus,
            original_data=original_data,
            alert_patterns=alert_patterns,
            normal_patterns=normal_patterns,
            alert_thresholds=alert_thresholds,
            normal_thresholds=normal_thresholds,
        )
        results["epsilons"] = epsilons
        results["epochs_per_level"] = epochs_per_level_history
        results_history.append(results)
        alert_rate = results["alert_metrics"].get("detection_rate", 0)
        normal_rate = results["normal_metrics"].get("detection_rate", 0)

        bar.set_postfix_str(
            f"ep={current_epochs}, {epsilon_l:.6f}/{max_epsilon:0.3f}, ratio={ratio:.6f}, nodes: {Gc.num_nodes} | "
            f"alert_rate: {alert_rate:.4f}, normal_rate: {normal_rate:.4f} | "
            f"prec: {results.get('precision_fine', 0):.4f}, coarse: {results.get('accuracy_test', 0):.4f}, fine: {results.get('accuracy_fine', 0):.4f} | "
        )
        bar.update(1)

    return Gall, Call, model, C_plus, results_history
