"""PyTorch implementation of collective trainable SGC motif detection.

This module implements the objective in Eq. (48) of ``Graph_Coarsening``:

    max_{||theta||_2 = 1} lambda_min(G(theta)),
    G(theta) = V.T g_theta(A_hat).T g_theta(A_hat) V.

``V`` contains one normalized indicator vector per *training* pattern and
``g_theta(A_hat) = sum_k theta_k A_hat**k``.  The graph operator is sparse and
all propagation, optimization, Gram construction, and evaluation use PyTorch.
No test pattern is used when fitting theta or calibrating the decision threshold.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import json
import math

import torch


@dataclass
class SGCTrainingResult:
    """Parameters and diagnostics returned after optimizing Eq. (48)."""

    theta: torch.Tensor
    objective: float
    vanilla_sgc_objective: float
    train_threshold: float
    degree: int
    epochs: int
    history: List[float]
    train_scores: List[float]
    train_labels: List[str]
    mode: str = "lambda_min"
    feature_aware: bool = False
    separation_ratio: float | None = None
    alert_scores: List[float] = field(default_factory=list)
    normal_scores: List[float] = field(default_factory=list)
    auc: float | None = None


@dataclass
class PatternDetection:
    """Detection result for one held-out pattern."""

    pattern_id: str
    pattern_type: str
    label: str
    retained_energy: float
    separation_energy: float
    score: float
    detected: bool


def normalized_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return sparse symmetric ``A_hat = D^-1/2 (A + I) D^-1/2``.

    The AMLGenTex loader may return a directed transaction graph.  Eq. (48)
    assumes the symmetric normalized adjacency, so directed edges are mirrored.
    Existing reciprocal edges retain their original total weight; one-way edges
    receive the same weight in both directions.
    """

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    device = edge_index.device
    rows, cols = edge_index[0], edge_index[1]
    non_self = rows != cols
    rows, cols = rows[non_self], cols[non_self]
    if edge_weight is None:
        values = torch.ones(rows.numel(), dtype=dtype, device=device)
    else:
        values = edge_weight.to(device=device, dtype=dtype)[non_self]

    # A + A.T, divided by two: reciprocal entries retain their weight while a
    # one-way transaction becomes an undirected edge of half weight per side.
    symmetric_indices = torch.cat(
        [torch.stack((rows, cols)), torch.stack((cols, rows))], dim=1
    )
    symmetric_values = torch.cat((values, values)) * 0.5
    adjacency = torch.sparse_coo_tensor(
        symmetric_indices,
        symmetric_values,
        (num_nodes, num_nodes),
        device=device,
        dtype=dtype,
    ).coalesce()

    loop_nodes = torch.arange(num_nodes, device=device)
    with_loops = torch.sparse_coo_tensor(
        torch.cat((adjacency.indices(), torch.stack((loop_nodes, loop_nodes))), dim=1),
        torch.cat(
            (adjacency.values(), torch.ones(num_nodes, dtype=dtype, device=device))
        ),
        (num_nodes, num_nodes),
        device=device,
        dtype=dtype,
    ).coalesce()

    degree = torch.zeros(num_nodes, dtype=dtype, device=device)
    degree.scatter_add_(0, with_loops.indices()[0], with_loops.values())
    inv_sqrt_degree = degree.clamp_min(torch.finfo(dtype).eps).rsqrt()
    indices = with_loops.indices()
    normalized_values = (
        with_loops.values() * inv_sqrt_degree[indices[0]] * inv_sqrt_degree[indices[1]]
    )
    return torch.sparse_coo_tensor(
        indices,
        normalized_values,
        (num_nodes, num_nodes),
        device=device,
        dtype=dtype,
    ).coalesce()


def pattern_indicator_matrix(
    patterns: Sequence[Any], num_nodes: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Build ``V`` with normalized indicator columns for non-empty patterns."""

    if not patterns:
        raise ValueError("at least one pattern is required")

    columns = []
    for pattern in patterns:
        nodes = torch.as_tensor(pattern.node_indices, dtype=torch.long, device=device)
        nodes = torch.unique(nodes)
        nodes = nodes[(nodes >= 0) & (nodes < num_nodes)]
        if nodes.numel() == 0:
            raise ValueError(f"pattern {pattern.id!r} has no nodes in the graph")
        column = torch.zeros(num_nodes, dtype=dtype, device=device)
        column[nodes] = 1.0 / math.sqrt(nodes.numel())
        columns.append(column)
    return torch.stack(columns, dim=1)


def propagation_stack(
    adjacency: torch.Tensor, signals: torch.Tensor, degree: int
) -> List[torch.Tensor]:
    """Compute ``[V, A_hat V, ..., A_hat**degree V]`` by sparse matvecs."""

    propagated = [signals]
    for _ in range(degree):
        propagated.append(torch.sparse.mm(adjacency, propagated[-1]))
    return propagated


def filter_signals(
    propagated: Sequence[torch.Tensor], theta: torch.Tensor
) -> torch.Tensor:
    """Apply ``g_theta`` to precomputed propagated pattern indicators."""

    if len(propagated) != theta.numel():
        raise ValueError("theta degree does not match the propagated signal stack")
    return sum(coefficient * signal for coefficient, signal in zip(theta, propagated))


def gram_matrix(
    propagated: Sequence[torch.Tensor],
    theta: torch.Tensor,
    features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the (feature-aware) filtered pattern Gram matrix ``G(theta)``.

    With ``features=None`` this is the Eq. (46) structural Gram
    ``G = F.T F`` (equivalently ``Sigma_X = I``).  With node features
    ``X in R^{N x f}`` it becomes the feature-aware sandwich

        G(theta) = P.T g_theta(A_hat) X X.T g_theta(A_hat) P
                 = Y(theta).T Y(theta),  Y(theta) = X.T g_theta(A_hat) P,

    so each pattern is represented by its ``f``-dim feature signature
    ``y_j = X.T g_theta(A_hat) v_j`` instead of its ``N``-dim filtered indicator.
    """

    filtered = filter_signals(propagated, theta)
    if features is not None:
        filtered = features.T @ filtered
    return filtered.T @ filtered


def feature_moment_matrices(
    propagated: Sequence[torch.Tensor],
    features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-pattern feature-moment matrices ``M_j in R^{(K+1) x (K+1)}``.

    ``(M_j)_kl = (X.T A_hat^k v_j).T (X.T A_hat^l v_j)`` so that the feature-aware
    retained energy of pattern ``j`` is the quadratic form ``theta.T M_j theta``.
    With ``features=None`` (``Sigma_X = I``) this reduces to the structural
    moment ``(A_hat^k v_j).T (A_hat^l v_j)``.  Returns shape ``(m, K+1, K+1)``.
    """

    if features is not None:
        signatures = [features.T @ signal for signal in propagated]
    else:
        signatures = list(propagated)
    stacked = torch.stack(signatures, dim=0)  # (K+1, d, m)
    return torch.einsum("kdj,ldj->jkl", stacked, stacked)


def quadratic_scores(moments: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Per-pattern retained energy ``theta.T M_j theta`` (shape ``(m,)``)."""

    theta = theta.to(device=moments.device, dtype=moments.dtype)
    return torch.einsum("jkl,k,l->j", moments, theta, theta)


def discriminative_score_ratio(
    scores: torch.Tensor, is_alert: torch.Tensor, *, eps: float = 1e-8
) -> torch.Tensor:
    """Soft Fisher ratio on the per-pattern energies ``s_j = theta.T M_j theta``.

        J(theta) = (mean_alert(s) - mean_normal(s))**2
                   / (var_alert(s) + var_normal(s) + eps)

    The numerator is squared so the objective is invariant to which class has
    the larger energy, and it is fully differentiable in ``theta``.  Unlike the
    closed-form :func:`fisher_theta` (which separates the *class-mean* moment
    matrices), this maximizes the separation of the actual energy distribution
    that is later thresholded, so it can be optimized by gradient descent.
    """

    pos = scores[is_alert]
    neg = scores[~is_alert]
    mean_gap = pos.mean() - neg.mean()
    spread = pos.var(unbiased=False) + neg.var(unbiased=False) + eps
    return (mean_gap * mean_gap) / spread


def fisher_theta(
    moments: torch.Tensor,
    labels: Sequence[str],
    *,
    ridge: float = 1e-6,
) -> tuple[torch.Tensor, float]:
    """Top generalized eigenvector of the class-mean feature-moment pair.

    Maximizes the alert/normal separation ratio in filter-coefficient space,

        theta* = argmax_theta (theta.T S_+ theta) / (theta.T (S_- + eps I) theta),

    where ``S_+`` and ``S_-`` are the mean feature-moment matrices ``M_j`` over
    the alert and normal training patterns.  This is Fisher/LDA on the
    ``(K+1)``-dim filter coefficients, picking the spectral band *and* feature
    directions where alerts differ most from normals.
    """

    is_alert = torch.tensor(
        [str(label) == "alert" for label in labels], device=moments.device
    )
    if not bool(is_alert.any()) or not bool((~is_alert).any()):
        raise ValueError(
            "Fisher objective needs both alert and normal training patterns"
        )

    s_plus = moments[is_alert].mean(dim=0)
    s_minus = moments[~is_alert].mean(dim=0)
    s_plus = 0.5 * (s_plus + s_plus.T)
    s_minus = 0.5 * (s_minus + s_minus.T)
    eye = torch.eye(s_plus.shape[0], dtype=moments.dtype, device=moments.device)
    chol = torch.linalg.cholesky(s_minus + ridge * eye)
    # whiten: C = L^-1 S_+ L^-T, then theta = L^-T (top eigenvector of C).
    whitened = torch.linalg.solve_triangular(chol, s_plus, upper=False)
    whitened = torch.linalg.solve_triangular(chol, whitened.T, upper=False).T
    whitened = 0.5 * (whitened + whitened.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(whitened)
    top = eigenvectors[:, -1]
    theta = torch.linalg.solve_triangular(chol.T, top.unsqueeze(1), upper=True).squeeze(
        1
    )
    return _unit(theta), float(eigenvalues[-1])


def _roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Rank-based ROC AUC (Mann-Whitney) with 0.5 credit for ties."""

    pos = torch.as_tensor(positive, dtype=torch.float64).flatten()
    neg = torch.as_tensor(negative, dtype=torch.float64).flatten()
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (pos.numel() * neg.numel()))


def score_pattern_energies(
    adjacency: torch.Tensor,
    patterns: Sequence[Any],
    theta: torch.Tensor,
    *,
    degree: int,
    features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Feature-aware retained energy ``theta.T M_j theta`` for each pattern."""

    V = pattern_indicator_matrix(
        patterns, adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device
    )
    propagated = propagation_stack(adjacency, V, degree)
    if features is not None:
        features = features.to(device=adjacency.device, dtype=adjacency.dtype)
        if features.dim() == 1:
            features = features.unsqueeze(1)
    moments = feature_moment_matrices(propagated, features)
    return quadratic_scores(moments, theta)


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(torch.finfo(vector.dtype).eps)


def _pattern_margin_scores(gram: torch.Tensor, ridge: float = 1e-10) -> torch.Tensor:
    """Per-pattern energy that remains after accounting for cross-talk.

    For a positive-definite Gram matrix this is ``1 / diag(G^-1)``: the squared
    norm of each filtered pattern after projection away from all other patterns.
    It is an individual counterpart to the collective ``lambda_min(G)`` target.
    A pseudoinverse gives a stable score if a test set contains duplicate or
    linearly dependent patterns.
    """

    gram = 0.5 * (gram + gram.T)
    stabilized = gram + ridge * torch.eye(
        gram.shape[0], dtype=gram.dtype, device=gram.device
    )
    inverse_diag = torch.diagonal(torch.linalg.pinv(stabilized)).clamp_min(ridge)
    return inverse_diag.reciprocal()


def fit_collective_sgc(
    adjacency: torch.Tensor,
    train_patterns: Sequence[Any],
    *,
    degree: int = 8,
    epochs: int = 400,
    learning_rate: float = 5e-2,
    threshold_quantile: float = 0.05,
    features: torch.Tensor | None = None,
    mode: str = "lambda_min",
    ridge: float = 1e-6,
) -> SGCTrainingResult:
    """Fit unit-norm ``theta`` on training patterns.

    Two options control the Gram and the objective:

    * ``features`` -- when ``None`` the structural Gram ``G = F.T F`` is used
      (``Sigma_X = I``, isotropic/Gaussian features).  When the node-feature
      matrix ``X = graph.x`` is supplied, the feature-aware Gram
      ``G = P.T g_theta(A_hat) X X.T g_theta(A_hat) P`` is used instead.
    * ``mode`` -- ``"lambda_min"`` maximizes Eq. (48) ``lambda_min(G(theta))``
      (retain/resolve every pattern) by projected Adam on the unit sphere;
      ``"fisher"`` instead returns the top generalized eigenvector of the
      class-mean feature-moment pair ``(S_+, S_-)`` to *discriminate* alerts
      from normals (needs both classes in ``train_patterns``);
      ``"discriminative"`` trains ``theta`` by projected Adam to maximize the
      soft Fisher ratio of the per-pattern energies
      :func:`discriminative_score_ratio` (also needs both classes).
    """

    if degree < 0:
        raise ValueError("degree must be non-negative")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 0.0 <= threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile must be in [0, 1]")
    if mode not in ("lambda_min", "fisher", "discriminative"):
        raise ValueError("mode must be 'lambda_min', 'fisher', or 'discriminative'")

    device, dtype = adjacency.device, adjacency.dtype
    V = pattern_indicator_matrix(
        train_patterns, adjacency.shape[0], dtype=dtype, device=device
    )
    propagated = propagation_stack(adjacency, V, degree)
    if features is not None:
        features = features.to(device=device, dtype=dtype)
        if features.dim() == 1:
            features = features.unsqueeze(1)
        if features.shape[0] != adjacency.shape[0]:
            raise ValueError("features must have one row per graph node")
    labels = [str(pattern.label) for pattern in train_patterns]
    moments = feature_moment_matrices(propagated, features)

    initial_theta = torch.zeros(degree + 1, dtype=dtype, device=device)
    initial_theta[-1] = 1.0
    with torch.no_grad():
        vanilla = torch.linalg.eigvalsh(
            gram_matrix(propagated, initial_theta, features)
        )[0].item()

    separation_ratio: float | None = None
    if mode == "fisher":
        best_theta, separation_ratio = fisher_theta(moments, labels, ridge=ridge)
        with torch.no_grad():
            best_objective = torch.linalg.eigvalsh(
                gram_matrix(propagated, best_theta, features)
            )[0].item()
        history: List[float] = [best_objective]
    elif mode == "discriminative":
        is_alert_t = torch.tensor([label == "alert" for label in labels], device=device)
        if not bool(is_alert_t.any()) or not bool((~is_alert_t).any()):
            raise ValueError(
                "discriminative objective needs both alert and normal "
                "training patterns"
            )
        raw_theta = torch.nn.Parameter(initial_theta.clone())
        optimizer = torch.optim.Adam((raw_theta,), lr=learning_rate)
        best_theta = initial_theta.clone()
        best_ratio = float("-inf")
        history = []
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            theta = _unit(raw_theta)
            scores = quadratic_scores(moments, theta)
            ratio = discriminative_score_ratio(scores, is_alert_t)
            if not torch.isfinite(ratio):
                raise FloatingPointError(
                    "non-finite discriminative ratio while fitting SGC"
                )
            (-ratio).backward()
            optimizer.step()

            value = float(ratio.detach().cpu())
            history.append(value)
            if value > best_ratio:
                best_ratio = value
                best_theta = theta.detach().clone()
        separation_ratio = best_ratio
        with torch.no_grad():
            best_objective = torch.linalg.eigvalsh(
                gram_matrix(propagated, best_theta, features)
            )[0].item()
    else:
        raw_theta = torch.nn.Parameter(initial_theta.clone())
        optimizer = torch.optim.Adam((raw_theta,), lr=learning_rate)
        # Keep vanilla SGC as a feasible candidate.  This means optimization
        # cannot report a result worse than the requested SGC baseline if an
        # Adam step is unhelpful for a particular AMLGenTex split.
        history = [vanilla]
        best_theta = initial_theta.clone()
        best_objective = vanilla

        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            theta = _unit(raw_theta)
            current_gram = gram_matrix(propagated, theta, features)
            gram = 0.5 * (current_gram + current_gram.T)
            objective = torch.linalg.eigvalsh(gram)[0]
            if not torch.isfinite(objective):
                raise FloatingPointError("non-finite lambda_min(G) while fitting SGC")
            (-objective).backward()
            optimizer.step()

            value = float(objective.detach().cpu())
            history.append(value)
            if value > best_objective:
                best_objective = value
                best_theta = theta.detach().clone()

    with torch.no_grad():
        final_gram = gram_matrix(propagated, best_theta, features)
        train_scores = _pattern_margin_scores(final_gram)
        threshold = torch.quantile(train_scores, threshold_quantile).item()

    # Report the separation of the *decision* score -- the cross-talk-adjusted
    # margin s_j that ``threshold`` actually gates in ``detect_patterns`` -- so
    # ``auc`` matches the quantity being thresholded.  The raw retained energy
    # ``theta^T M_j theta`` is a different quantity (it ignores cross-talk); the
    # held-out histogram in the runner plots those energies separately.
    is_alert = torch.tensor(
        [label == "alert" for label in labels], device=train_scores.device
    )
    alert_scores = train_scores[is_alert].detach().cpu().tolist()
    normal_scores = train_scores[~is_alert].detach().cpu().tolist()
    auc = (
        _roc_auc(alert_scores, normal_scores)
        if alert_scores and normal_scores
        else None
    )

    return SGCTrainingResult(
        theta=best_theta,
        objective=best_objective,
        vanilla_sgc_objective=vanilla,
        train_threshold=threshold,
        degree=degree,
        epochs=epochs,
        history=history,
        train_scores=train_scores.detach().cpu().tolist(),
        train_labels=labels,
        mode=mode,
        feature_aware=features is not None,
        separation_ratio=separation_ratio,
        alert_scores=alert_scores,
        normal_scores=normal_scores,
        auc=auc,
    )


def detect_patterns(
    adjacency: torch.Tensor,
    test_patterns: Sequence[Any],
    fit: SGCTrainingResult,
    features: torch.Tensor | None = None,
) -> tuple[List[PatternDetection], Dict[str, Dict[str, Any]]]:
    """Score only held-out patterns using a theta learned from training patterns.

    A pattern is detected when its cross-talk-adjusted retained energy is at
    least the lower-quantile training threshold.  Results are grouped by the
    source label first (``alert`` or ``normal``), then pattern type.  This
    prevents a type name shared by both families from being merged in the
    detection report.

    ``features`` must be the *same* node-feature matrix ``X`` used to fit
    ``theta`` and calibrate ``fit.train_threshold``.  Otherwise the held-out
    Gram would use ``Sigma_X = I`` while the threshold was calibrated
    feature-aware (``Sigma_X = X X^T``), scoring test patterns inconsistently.
    """

    if not test_patterns:
        return [], {}

    if fit.feature_aware and features is None:
        raise ValueError(
            "fit is feature-aware (theta and threshold calibrated with X); pass "
            "the same node-feature matrix to detect_patterns via features="
        )

    V = pattern_indicator_matrix(
        test_patterns,
        adjacency.shape[0],
        dtype=adjacency.dtype,
        device=adjacency.device,
    )
    propagated = propagation_stack(adjacency, V, fit.degree)
    theta = fit.theta.to(device=adjacency.device, dtype=adjacency.dtype)
    if features is not None:
        features = features.to(device=adjacency.device, dtype=adjacency.dtype)
        if features.dim() == 1:
            features = features.unsqueeze(1)
        if features.shape[0] != adjacency.shape[0]:
            raise ValueError("features must have one row per graph node")
    gram = gram_matrix(propagated, theta, features)
    retained = torch.diagonal(gram)
    separated = _pattern_margin_scores(gram)

    detections: List[PatternDetection] = []
    grouped: Dict[str, Dict[str, List[PatternDetection]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, pattern in enumerate(test_patterns):
        score = float(separated[index].detach().cpu())
        result = PatternDetection(
            pattern_id=str(pattern.id),
            pattern_type=str(pattern.pattern_type),
            label=str(pattern.label),
            retained_energy=float(retained[index].detach().cpu()),
            separation_energy=score,
            score=score,
            detected=score >= fit.train_threshold,
        )
        detections.append(result)
        grouped[result.label][result.pattern_type].append(result)

    by_label: Dict[str, Dict[str, Any]] = {}
    for label, patterns_by_type in sorted(grouped.items()):
        type_results: Dict[str, Dict[str, float]] = {}
        all_entries: List[PatternDetection] = []
        for pattern_type, entries in sorted(patterns_by_type.items()):
            detected = sum(entry.detected for entry in entries)
            type_results[pattern_type] = {
                "detected": detected,
                "total": len(entries),
                "detection_rate": detected / len(entries),
                "mean_retained_energy": sum(entry.retained_energy for entry in entries)
                / len(entries),
                "mean_separation_energy": sum(
                    entry.separation_energy for entry in entries
                )
                / len(entries),
            }
            all_entries.extend(entries)
        detected = sum(entry.detected for entry in all_entries)
        by_label[label] = {
            "detected": detected,
            "total": len(all_entries),
            "detection_rate": detected / len(all_entries),
            "by_pattern_type": type_results,
        }
    return detections, by_label


def train_and_detect(
    graph: Any,
    train_patterns: Sequence[Any],
    test_patterns: Sequence[Any],
    **fit_kwargs: Any,
) -> tuple[SGCTrainingResult, List[PatternDetection], Dict[str, Dict[str, Any]]]:
    """Convenience entry point for AMLGenTex ``Data`` returned by the loader."""

    adjacency = normalized_adjacency(
        graph.edge_index,
        int(graph.num_nodes),
        getattr(graph, "edge_weight", None),
    )
    fit = fit_collective_sgc(adjacency, train_patterns, **fit_kwargs)
    detections, by_label = detect_patterns(
        adjacency, test_patterns, fit, features=fit_kwargs.get("features")
    )
    return fit, detections, by_label


def save_report(
    path: str | Path,
    fit: SGCTrainingResult,
    detections: Iterable[PatternDetection],
    by_label: Mapping[str, Mapping[str, Any]],
) -> None:
    """Persist a JSON report without serializing GPU tensors."""

    payload = {
        "theta": fit.theta.detach().cpu().tolist(),
        "objective_lambda_min_G": fit.objective,
        "vanilla_sgc_lambda_min_G": fit.vanilla_sgc_objective,
        "training_score_threshold": fit.train_threshold,
        "degree": fit.degree,
        "epochs": fit.epochs,
        "training_pattern_count": len(fit.train_labels),
        "training_pattern_labels": sorted(set(fit.train_labels)),
        "detection_rate_by_label": by_label,
        "patterns": [asdict(detection) for detection in detections],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def plot_diagnostics(
    directory: str | Path,
    fit: SGCTrainingResult,
    detections: Sequence[PatternDetection],
    by_label: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Path]:
    """Write plots that validate optimization and held-out detection behavior.

    The plots deliberately expose the two separate test families: alerts and
    normals.  Matplotlib is imported lazily so SGC fitting itself has only a
    PyTorch dependency.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output: Dict[str, Path] = {}

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(range(len(fit.history)), fit.history, color="tab:blue", linewidth=1.5)
    axis.axhline(
        fit.vanilla_sgc_objective,
        color="tab:gray",
        linestyle="--",
        label="vanilla SGC",
    )
    axis.set(xlabel="optimization step", ylabel=r"$\lambda_{\min}(G(\theta))$")
    axis.set_title("Eq. (48) collective SGC objective")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    output["objective_history"] = directory / "objective_history.png"
    fig.savefig(output["objective_history"], dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    theta = fit.theta.detach().cpu().tolist()
    axis.stem(range(len(theta)), theta, basefmt=" ")
    axis.set(xlabel="polynomial degree k", ylabel=r"$\theta_k$")
    axis.set_title("Learned trainable-SGC coefficients")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    output["theta"] = directory / "theta_coefficients.png"
    fig.savefig(output["theta"], dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    labels = sorted({detection.label for detection in detections})
    for label in labels:
        scores = [
            detection.score for detection in detections if detection.label == label
        ]
        if scores:
            axis.hist(scores, bins="auto", alpha=0.55, label=f"test {label}")
    for label in sorted(set(fit.train_labels)):
        scores = [
            score
            for score, train_label in zip(fit.train_scores, fit.train_labels)
            if train_label == label
        ]
        if scores:
            axis.hist(
                scores,
                bins="auto",
                histtype="step",
                linewidth=2,
                label=f"train {label}",
            )
    axis.axvline(
        fit.train_threshold,
        color="black",
        linestyle="--",
        label="training threshold",
    )
    axis.set(xlabel="cross-talk-adjusted retained energy", ylabel="pattern count")
    axis.set_title("Train-calibrated scores on held-out alert and normal patterns")
    axis.legend()
    fig.tight_layout()
    output["scores"] = directory / "train_test_score_distributions.png"
    fig.savefig(output["scores"], dpi=160)
    plt.close(fig)

    labels = [label for label in ("alert", "normal") if label in by_label]
    if not labels:
        labels = list(by_label)
    if not labels:
        return output
    fig, axes = plt.subplots(
        1, len(labels), figsize=(max(6, 4 * len(labels)), 4), squeeze=False
    )
    for axis, label in zip(axes[0], labels):
        per_type = by_label[label]["by_pattern_type"]
        types = list(per_type)
        rates = [per_type[pattern_type]["detection_rate"] for pattern_type in types]
        bars = axis.bar(
            types, rates, color="tab:red" if label == "alert" else "tab:green"
        )
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("detection rate")
        axis.set_title(f"Held-out {label} patterns")
        axis.tick_params(axis="x", rotation=30)
        for bar, pattern_type in zip(bars, types):
            metrics = per_type[pattern_type]
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{int(metrics['detected'])}/{int(metrics['total'])}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("Detection rate by pattern type and label")
    fig.tight_layout()
    output["detection_rates"] = directory / "detection_rates_by_label_and_type.png"
    fig.savefig(output["detection_rates"], dpi=160)
    plt.close(fig)
    return output
