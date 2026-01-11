"""Unified plotting utilities for coarsening experiments."""

from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


# Default color palette
COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]


def smooth_curve(y: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply moving average smoothing to a curve."""
    if len(y) < window_size:
        return y
    return uniform_filter1d(y, size=window_size, mode="nearest")


def create_coarsening_plot(
    data_list: List[Dict],
    thresholds: List[float],
    x_key: str,
    y_keys: List[Tuple[str, str, str]],  # (key, style, label_suffix)
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str,
    baseline_value: Optional[float] = None,
    baseline_label: str = "orig",
    coarse_baseline: Optional[float] = None,
    figsize: Tuple[int, int] = (16, 9),
    smooth: bool = False,
    smooth_window: int = 5,
    ylim: Optional[Tuple[float, float]] = None,
    grid: bool = False,
):
    """
    Generic plotting function for coarsening experiment results.

    Args:
        data_list: List of data dicts for each threshold
        thresholds: List of threshold values
        x_key: Key for x-axis data (or None to use indices)
        y_keys: List of (data_key, line_style, label_suffix) tuples
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        save_path: Path to save the figure
        baseline_value: Optional horizontal line for baseline
        baseline_label: Label for baseline line
        coarse_baseline: Optional second baseline (e.g., coarse accuracy)
        figsize: Figure size
        smooth: Whether to apply smoothing
        smooth_window: Window size for smoothing
        ylim: Y-axis limits
        grid: Whether to show grid
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Add baseline horizontal line
    if baseline_value is not None:
        xmax = (
            len(data_list[0][y_keys[0][0]])
            if x_key is None
            else max(data_list[0][x_key])
        )
        ax.hlines(
            y=baseline_value,
            xmin=0,
            xmax=xmax,
            color="black",
            linestyles="--",
            label=baseline_label,
        )

    # Plot each threshold
    for idx, (data, threshold) in enumerate(zip(data_list, thresholds)):
        color = COLORS[idx % len(COLORS)]
        width = 2 if idx == len(thresholds) - 1 else 1

        for y_key, style, label_suffix in y_keys:
            if y_key not in data:
                continue

            y_data = np.array(data[y_key])
            if smooth:
                y_data = smooth_curve(y_data, smooth_window)

            x_data = np.array(data[x_key]) if x_key else np.arange(len(y_data))

            marker = "*" if "Coarse" in label_suffix else "o"
            ax.plot(
                x_data,
                y_data,
                color=color,
                linewidth=width,
                linestyle=style,
                marker=marker,
                markersize=2,
                label=f"{label_suffix} {threshold*100:.0f}%",
            )

    # Add coarse baseline if provided
    if coarse_baseline is not None:
        xmax = (
            len(data_list[0][y_keys[0][0]])
            if x_key is None
            else max(data_list[0].get(x_key, [100]))
        )
        ax.hlines(
            y=coarse_baseline,
            xmin=0,
            xmax=xmax,
            color="gray",
            linestyles="--",
            label="coarse",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    if grid:
        ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path)

    return fig, ax


def plot_accuracy_vs_iteration(
    data_list: List[Dict],
    thresholds: List[float],
    save_path: str,
    baseline_accuracy: float,
    coarse_accuracy: Optional[float] = None,
    smooth: bool = False,
):
    """Plot coarse vs fine accuracy over iterations."""
    return create_coarsening_plot(
        data_list=data_list,
        thresholds=thresholds,
        x_key=None,
        y_keys=[
            ("ycrs", "-", "Coarse"),
            ("yfine", ":", "Fine"),
        ],
        xlabel="Iteration",
        ylabel="Accuracy",
        title="Coarse vs Fine Accuracy over Iterations",
        save_path=save_path,
        baseline_value=baseline_accuracy,
        coarse_baseline=coarse_accuracy,
        smooth=smooth,
    )


def plot_accuracy_vs_nodes(
    data_list: List[Dict],
    thresholds: List[float],
    save_path: str,
    baseline_accuracy: float,
    coarse_accuracy: Optional[float] = None,
    smooth: bool = False,
):
    """Plot coarse vs fine accuracy against number of coarse nodes."""
    return create_coarsening_plot(
        data_list=data_list,
        thresholds=thresholds,
        x_key="num_nodes_coarse",
        y_keys=[
            ("ycrs", "-", "Coarse"),
            ("yfine", ":", "Fine"),
        ],
        xlabel="Number of Coarse Nodes",
        ylabel="Accuracy",
        title="Coarse vs Fine Accuracy over Iterations",
        save_path=save_path,
        baseline_value=baseline_accuracy,
        coarse_baseline=coarse_accuracy,
        smooth=smooth,
    )


def plot_precision_vs_nodes(
    data_list: List[Dict],
    thresholds: List[float],
    save_path: str,
    baseline_precision: float,
    smooth: bool = False,
):
    """Plot coarse vs fine precision against number of coarse nodes."""
    return create_coarsening_plot(
        data_list=data_list,
        thresholds=thresholds,
        x_key="num_nodes_coarse",
        y_keys=[
            ("prec_l", "-", "Coarse"),
            ("prec_fine", ":", "Fine"),
        ],
        xlabel="Number of Coarse Nodes",
        ylabel="Precision",
        title="Coarse vs Precision over Iterations",
        save_path=save_path,
        baseline_value=baseline_precision,
        smooth=smooth,
    )


def plot_pattern_detection(
    data_list: List[Dict],
    thresholds: List[float],
    save_path: str,
    pattern_type: str = "alert",
    include_model: bool = True,
):
    """
    Plot pattern detection rates across coarsening levels.

    Args:
        data_list: List of data dicts for each threshold
        thresholds: List of threshold values
        save_path: Path to save the figure
        pattern_type: "alert" or "normal"
        include_model: Whether to include model-based detection (for alerts)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    gt_key = (
        f"{pattern_type}_pattern_detection_rates_gt"
        if pattern_type == "normal"
        else "pattern_detection_rates_gt"
    )
    model_key = "pattern_detection_rates"

    # Fallback for legacy key names
    if pattern_type == "normal":
        gt_key = "normal_pattern_detection_rates_gt"

    for idx, (data, threshold) in enumerate(zip(data_list, thresholds)):
        color = COLORS[idx % len(COLORS)]
        width = 2 if idx == len(thresholds) - 1 else 1

        # GT-based detection
        if gt_key in data:
            ax.plot(
                data[gt_key],
                color=color,
                linewidth=width,
                marker="o" if pattern_type == "alert" else "s",
                markersize=3,
                label=f"{pattern_type.title()} GT (th={threshold*100:.0f}%)",
            )

        # Model-based detection (alerts only)
        if include_model and pattern_type == "alert" and model_key in data:
            ax.plot(
                data[model_key],
                color=color,
                linewidth=width,
                linestyle="--",
                marker="x",
                markersize=3,
                label=f"Alert Model (th={threshold*100:.0f}%)",
            )

    majority_th = "75%" if pattern_type == "alert" else "50%"
    coarse_th = "75%" if pattern_type == "alert" else "50%"

    ax.set_xlabel("Coarsening Level")
    ax.set_ylabel("Pattern Detection Rate")
    ax.set_title(
        f"{pattern_type.title()} Pattern Detection Rate vs Coarsening Level\n(>{majority_th} {pattern_type} + >{coarse_th} coarsened together)"
    )
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path)

    return fig, ax


def plot_all_results(
    data_list: List[Dict],
    thresholds: List[float],
    save_dir: str,
    epochs_per_level: int,
    baseline_accuracy: float,
    baseline_precision: float,
    coarse_accuracy: Optional[float] = None,
    has_alert_patterns: bool = False,
    has_normal_patterns: bool = False,
    smooth: bool = False,
):
    """
    Generate all standard plots for a coarsening experiment.

    Args:
        data_list: List of data dicts for each threshold
        thresholds: List of threshold values
        save_dir: Directory to save plots
        epochs_per_level: Number of epochs per level (for filename)
        baseline_accuracy: Original model accuracy
        baseline_precision: Original model precision
        coarse_accuracy: Optional coarse model accuracy
        has_alert_patterns: Whether alert pattern data exists
        has_normal_patterns: Whether normal pattern data exists
        smooth: Whether to apply smoothing
    """
    save_dir = Path(save_dir)

    # Accuracy vs iteration
    plot_accuracy_vs_iteration(
        data_list,
        thresholds,
        str(save_dir / f"iterative_custom_loss_accuracy_{epochs_per_level}.png"),
        baseline_accuracy,
        coarse_accuracy,
        smooth,
    )

    # Accuracy vs nodes
    plot_accuracy_vs_nodes(
        data_list,
        thresholds,
        str(
            save_dir
            / f"iterative_custom_loss_accuracy_{epochs_per_level}_with_nodes.png"
        ),
        baseline_accuracy,
        coarse_accuracy,
        smooth,
    )

    # Precision vs nodes
    plot_precision_vs_nodes(
        data_list,
        thresholds,
        str(
            save_dir
            / f"iterative_custom_loss_precission_{epochs_per_level}_with_nodes.png"
        ),
        baseline_precision,
        smooth,
    )

    # Pattern detection plots
    if has_alert_patterns and any("pattern_detection_rates_gt" in d for d in data_list):
        plot_pattern_detection(
            data_list,
            thresholds,
            str(save_dir / f"alert_pattern_detection_rates_{epochs_per_level}.png"),
            pattern_type="alert",
            include_model=True,
        )

    if has_normal_patterns and any(
        "normal_pattern_detection_rates_gt" in d for d in data_list
    ):
        plot_pattern_detection(
            data_list,
            thresholds,
            str(save_dir / f"normal_pattern_detection_rates_{epochs_per_level}.png"),
            pattern_type="normal",
            include_model=False,
        )
