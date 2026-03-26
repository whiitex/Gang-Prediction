"""GNN model definition and training/evaluation helpers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, roc_auc_score

from src.GangPrediction.pattern_models import Pattern
from src.ml.metrics.metrics import average_precision_score
from src.GangPrediction.experiment_utils import (
    _capture_pattern_lineage,
    get_node_to_supernode_mapping,
)
from src.GangPrediction.embedding_diagnostics import (
    compute_embedding_and_basis_diagnostics,
)
from src.GangPrediction.utils.utils import *


def train_gnn_1_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    data: Data,
    C_plus=None,
    P=None,
    L=None,
    original_data=None,
    coarse_loss: bool = False,
    return_loss_components: bool = False,
    # class_weights: torch.Tensor = None,
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
    y = original_data.y if original_data is not None else data.y
    train_idx = original_data.train_idx if original_data is not None else data.train_idx

    logits = model(
        data.x,
        data.edge_index,
        data.edge_weight if hasattr(data, "edge_weight") else None,
    )
    # if class_weights is not None:
    # class_weights = class_weights.to(logits.device)
    # logits = logits * class_weights.unsqueeze(0)
    if C_plus is not None:
        logits = C_plus @ logits
        # pred_fine = F.log_softmax(C_plus @ logits, dim=1)
    # else:
    # pred_fine = F.log_softmax(logits, dim=1)
    pred_fine = F.softmax(logits, dim=1)
    # embeddings = model.get_embeddings(data.x, data.edge_index, data.edge_weight if hasattr(data, "edge_weight") else None)

    # train
    # embeddings = data.embeddings if hasattr(data, "embeddings") else None
    if original_data is not None and coarse_loss:
        embeddings = model.get_embeddings(
            original_data.x,
            original_data.edge_index,
            original_data.edge_weight if hasattr(data, "edge_weight") else None,
        )
    else:
        embeddings = None
    # embeddings = F.normalize(embeddings, p=2, dim=1)
    loss, loss_super_node, loss_cls = criterion(
        logits, y, train_idx, embeddings, coarse_loss=coarse_loss, P=P, L=L
    )
    loss.backward()
    optimizer.step()
    train_acc = accuracy_score(
        y[train_idx].detach().cpu().numpy(),
        pred_fine[train_idx].max(1)[1].detach().cpu().numpy(),
    )

    if return_loss_components:
        components = {
            "loss_total": loss.item(),
            "loss_cls": loss.item(),
            "loss_supernode": 0.0,
        }
        if hasattr(criterion, "latest_loss_components"):
            components.update(criterion.latest_loss_components)
        return loss.item(), train_acc, components

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
    basis_rows=None,
):
    """Evaluate the model on test set."""

    model.eval()
    # data = data.to(next(model.parameters()).device)

    alert_patterns = alert_patterns or []
    normal_patterns = normal_patterns or []

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

    with torch.no_grad():
        emb_source = original_data if original_data is not None else data
        emb_rows = model.get_embeddings(
            emb_source.x,
            emb_source.edge_index,
            emb_source.edge_weight if hasattr(emb_source, "edge_weight") else None,
        )
    diag = compute_embedding_and_basis_diagnostics(
        emb_rows,
        alert_patterns=alert_patterns,
        normal_patterns=normal_patterns,
        basis_rows=basis_rows,
    )
    results["embedding_diagnostics"] = {
        "embedding": diag["embedding"],
        "basis": diag["basis"],
        "pattern_node_count": diag["pattern_node_count"],
        "num_patterns": diag["num_patterns"],
    }

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
