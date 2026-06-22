from types import SimpleNamespace

import torch

from src.GangPrediction.sgc_detection import (
    detect_patterns,
    fit_collective_sgc,
    normalized_adjacency,
)


def _pattern(identifier, nodes, pattern_type, label="alert"):
    return SimpleNamespace(
        id=identifier,
        node_indices=torch.tensor(nodes),
        pattern_type=pattern_type,
        label=label,
    )


def test_collective_sgc_optimizes_eq48_and_reports_per_type_rates():
    # Two disconnected pairs make the expected motif indicators independent;
    # this keeps the exact collective Gram objective well-conditioned.
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    adjacency = normalized_adjacency(edge_index, num_nodes=4)
    train_patterns = [
        _pattern("train-a", [0, 1], "fan_out", "alert"),
        _pattern("train-b", [2, 3], "cycle", "normal"),
    ]
    test_patterns = [
        _pattern("test-a", [0, 1], "fan_out", "alert"),
        _pattern("test-b", [2, 3], "cycle", "normal"),
    ]

    fit = fit_collective_sgc(
        adjacency,
        train_patterns,
        degree=2,
        epochs=20,
        learning_rate=0.05,
    )
    detections, by_label = detect_patterns(adjacency, test_patterns, fit)

    assert torch.isclose(fit.theta.norm(), torch.tensor(1.0, dtype=fit.theta.dtype))
    assert fit.objective >= fit.vanilla_sgc_objective
    assert len(detections) == 2
    assert set(by_label) == {"alert", "normal"}
    assert by_label["alert"]["by_pattern_type"]["fan_out"]["detection_rate"] == 1.0
    assert by_label["normal"]["by_pattern_type"]["cycle"]["detection_rate"] == 1.0


def test_alert_only_training_does_not_include_normal_patterns():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    adjacency = normalized_adjacency(edge_index, num_nodes=4)
    alert_train = [_pattern("train-alert", [0, 1], "fan_out", "alert")]
    held_out = [
        _pattern("test-alert", [0, 1], "fan_out", "alert"),
        _pattern("test-normal", [2, 3], "cycle", "normal"),
    ]

    fit = fit_collective_sgc(adjacency, alert_train, degree=2, epochs=5)
    _, by_label = detect_patterns(adjacency, held_out, fit)

    assert fit.train_labels == ["alert"]
    assert set(by_label) == {"alert", "normal"}
