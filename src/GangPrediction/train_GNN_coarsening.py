"""Training entrypoints for the GNN baseline and the coarsening-aware variant."""

import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.GangPrediction.GNN_model import GCN, train_gnn_1_epoch, evaluate_model
from src.GangPrediction.gang_aware_subspace import get_gang_aware_basis
from src.GangPrediction.experiment_utils import create_subspace
from src.GangPrediction.coarsening_aware_loss import *
from src.GangPrediction.coarsening_utils import *
from src.GangPrediction.graph_utils import *
from src.GangPrediction.utils.utils import *


def train_GNN(
    data,
    epochs,
    lr=0.01,
    wd=5e-4,
    nhid=128,
    dropout=0.1,
    use_class_weights=True,
    num_layers=2,
    GNN_type="GAT",
    use_edge_weights=False,
    **kwargs,
):
    """Train a vanilla GCN on the full-resolution graph."""
    # model
    nclass = len(np.unique(data.y.numpy()))  # 7
    # Keep nclass as-is for CrossEntropyLoss compatibility (don't reduce to 1 for binary)
    model = GCN(
        nfeat=data.num_features,
        nhid=nhid,
        nclass=nclass,
        dropout=dropout,
        num_layers=num_layers,
        GNN_type=GNN_type,
        use_edge_weights=use_edge_weights,
    ).to(device=device)

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
    max_epsilon=float("inf"),
    K=50,
    nhid=128,
    dropout=0.1,
    use_class_weights=True,
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
    epsilon_schedule_power=1.0,
    train=False,
    model=None,
    num_layers=2,
    coarsening_weight=1.0,
    use_label_for_coarsening=False,
    use_super_node_loss=False,
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
            nfeat=data.num_features,
            nhid=nhid,
            nclass=nclass,
            dropout=dropout,
            num_layers=num_layers,
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
            use_supernode_loss=use_super_node_loss
            and (
                True if "learning" in method else False
            ),  # Enable supernode loss when learning B
            coarse_weight=coarsening_weight,
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
    tensor_device = Gc.y.device
    num_classes = len(torch.unique(Gc.y))

    labels_onehot = torch.zeros(len(Gc.y), num_classes, device=tensor_device)
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
    else:
        # For learning_subspace, we will compute the basis dynamically during training
        B = None

    Call, Gall = [np.eye(N)], [Gc]

    bar = tqdm(total=levels)
    epsilon_l, epsilons = 0, []

    # Dynamic epoch scheduling state
    current_epochs = initial_epochs  # Start with initial_epochs per level
    epoch_interval = 1  # Train every level initially (increases to max_epoch_interval)
    loss_history = []  # Track training losses for convergence detection
    epochs_per_level_history = []  # Track for saving
    results_history = []  # Track evaluation results per level
    learned_basis_rows = None
    for level in range(1, levels + 1):
        # Simple ratio schedule (original formula)
        # ratio = np.log(level) / 5000 + 0.0001
        # if ratio > 0.0025:
        #     ratio = 0.0025
        # if level <= 10:
        #     ratio = 0
        # else:
        ratio = 1

        # Get embeddings from the GNN in eval mode with no_grad
        # This avoids injecting dropout randomness and prevents backprop through stale representations
        model.eval()
        # with torch.no_grad():
        # Update B from embeddings if using learned B
        if method == "learning_subspace" and train:
            with torch.no_grad():
                G_embeddings = model.get_embeddings(
                    data.x, data.edge_index, data.edge_weight
                )
                # Recompute orthonormal basis from current embeddings
                V = calc_B_from_embeddings(G_embeddings)
                # learned_basis_rows = V.detach()
                B = torch.sparse.mm(C, V)  # Update B for the current coarsened graph
        elif method == "use_subspaces":
            V = create_subspace(alert_train_patterns, normal_train_patterns, N, device)
            # V = create_subspace(alert_patterns, normal_patterns, N, device)
            B = torch.sparse.mm(C, V)  # Update B for the current coarsened graph
        elif method == "learning_vectors":
            with torch.no_grad():
                G_embeddings = model.get_embeddings(
                    data.x, data.edge_index, data.edge_weight
                )
                V = calc_B_from_embeddings(G_embeddings)
                # learned_basis_rows = V.detach()
                B = torch.sparse.mm(C, V)  # Update B for the current coarsened graph
                # B = 0 * B
        # embeddings = model.get_embeddings(Gc.x, Gc.edge_index, Gc.edge_weight)
        # Gc.embeddings = F.normalize(embeddings, p=2, dim=1).detach()

        # Polynomial epsilon schedule: slow increase early, faster increase later.
        level_progress = level / max(1, levels)
        if (
            np.abs(epsilon_schedule_power) < 1e-8
        ):  # Effectively no scheduling, constant epsilon
            max_eps_in_level = max_epsilon  # No scheduling, constant epsilon
        else:
            pow = level_progress**epsilon_schedule_power
            max_eps_in_level = max_epsilon * pow
        max_sigma = max(0.0, (max_eps_in_level + 1) / (epsilon_l + 1) - 1)

        Gc.pred = model(Gc.x, Gc.edge_index, Gc.edge_weight)
        Gc.y_pred = Gc.pred.argmax(dim=1)

        Gc, B, sigma_l, done_flag = coarse_one_level(
            Gc,
            B=B,
            method=method,
            algorithm=algorithm,
            level=level,
            r_cur=ratio,
            max_sigma=max_sigma,
            use_label_for_coarsening=use_label_for_coarsening,
        )
        C = torch.sparse.mm(Gc.C, C)
        C_plus = torch.sparse.mm(C_plus, Gc.C_plus)

        P = torch.sparse.mm(C_plus, C)

        Gall.append(Gc)
        Call.append(C)

        epsilon_l = (sigma_l + 1) * (epsilon_l + 1) - 1
        # if done_flag or epsilon_l >= max_epsilon:
        #     print(
        #         f"Reached max epsilon {max_epsilon} at level {level} with epsilon_l {epsilon_l:.3f}."
        #     )
        # break
        epsilons.append(epsilon_l)

        level_loss_total = np.nan
        level_loss_cls = np.nan
        level_loss_supernode = np.nan

        if train:
            # Dynamic epoch scheduling: only train if at the right interval
            if level % epoch_interval == 0:
                active_coarse_loss_epochs = max(0, current_epochs)
                epoch_loss_components = {
                    "loss_total": [],
                    "loss_cls": [],
                    "loss_supernode": [],
                }
                for epoch in range(current_epochs):
                    # train the GNN on the coarsened graph
                    train_loss, train_acc, loss_components = train_gnn_1_epoch(
                        model,
                        optimizer=optimizer,
                        criterion=criterion,
                        data=Gc,
                        C_plus=C_plus,
                        P=P,
                        L=data.L,
                        original_data=original_data,
                        coarse_loss=epoch < active_coarse_loss_epochs,
                        return_loss_components=True,
                        # class_weights=class_weights,
                    )

                    epoch_loss_components["loss_total"].append(
                        float(loss_components.get("loss_total", train_loss))
                    )
                    epoch_loss_components["loss_cls"].append(
                        float(loss_components.get("loss_cls", train_loss))
                    )
                    epoch_loss_components["loss_supernode"].append(
                        float(loss_components.get("loss_supernode", 0.0))
                    )

                if epoch_loss_components["loss_total"]:
                    level_loss_total = float(
                        np.mean(epoch_loss_components["loss_total"])
                    )
                    level_loss_cls = float(np.mean(epoch_loss_components["loss_cls"]))
                    level_loss_supernode = float(
                        np.mean(epoch_loss_components["loss_supernode"])
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

                scheduler.step(train_loss)
            else:
                # Skip training at this level, use last known values
                train_loss = loss_history[-1] if loss_history else 0.0
                epochs_per_level_history.append(0)
                scheduler.step(train_loss)

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
            basis_rows=learned_basis_rows,
        )
        results["epsilons"] = epsilons
        results["epochs_per_level"] = epochs_per_level_history
        results["loss_total"] = level_loss_total
        results["loss_cls"] = level_loss_cls
        results["loss_supernode"] = level_loss_supernode
        results_history.append(results)
        alert_rate = results["alert_metrics"].get("detection_rate", 0)
        normal_rate = results["normal_metrics"].get("detection_rate", 0)

        bar.set_postfix_str(
            f"ep={current_epochs}, {epsilon_l:.6f}/{max_epsilon:0.3f}, ratio={ratio:.6f}, nodes: {Gc.num_nodes} | "
            f"alert_rate: {alert_rate:.4f}, normal_rate: {normal_rate:.4f} | "
            f"prec: {results.get('precision_fine', 0):.4f}, coarse: {results.get('accuracy_test', 0):.4f}, fine: {results.get('accuracy_fine', 0):.4f} | "
        )
        bar.update(1)

    if learned_basis_rows is not None:
        model.latest_learned_basis_rows = learned_basis_rows.detach().cpu()

    return Gall, Call, model, C_plus, results_history
