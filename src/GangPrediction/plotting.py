"""Unified plotting utilities for coarsening experiments."""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict

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
DEFAULT_COLOR = COLORS[0]


def _to_series_dict(data: Dict | List[Dict]) -> Dict:
    """Normalize plotting input to a dictionary-of-series.

    Supports both:
    - New format: list of per-level result dictionaries (`results_history`)
    - Legacy format: dict of series
    """
    if isinstance(data, dict):
        return data

    if not data:
        return {}

    out: Dict = {
        "x": [],
        "accuracy_test": [],
        "accuracy_fine": [],
        "precision_test": [],
        "precision_fine": [],
        "num_nodes_coarse": [],
        "alert_detection_rate": [],
        "alert_detection_rate_filtered": [],
        "alert_detection_rate_majority": [],
        "alert_detection_rate_coarsened": [],
        "alert_recall": [],
        "alert_precision": [],
        "alert_f1": [],
        "alert_recall_filtered": [],
        "alert_precision_filtered": [],
        "alert_f1_filtered": [],
        "normal_detection_rate": [],
        "normal_detection_rate_filtered": [],
        "normal_detection_rate_majority": [],
        "normal_detection_rate_coarsened": [],
        "normal_recall": [],
        "normal_precision": [],
        "normal_f1": [],
        "normal_recall_filtered": [],
        "normal_precision_filtered": [],
        "normal_f1_filtered": [],
        "alert_gt_type_rates": [],
        "normal_gt_type_rates": [],
        "epochs_per_level": [],
        "epsilons": [],
    }

    for idx, level in enumerate(data, start=1):
        out["x"].append(idx)
        out["accuracy_test"].append(level.get("accuracy_test", np.nan))
        out["accuracy_fine"].append(level.get("accuracy_fine", np.nan))
        out["precision_test"].append(level.get("precision_test", np.nan))
        out["precision_fine"].append(level.get("precision_fine", np.nan))
        out["num_nodes_coarse"].append(level.get("num_nodes_coarse", np.nan))

        alert_metrics = level.get("alert_metrics", {}) or {}
        normal_metrics = level.get("normal_metrics", {}) or {}

        out["alert_detection_rate"].append(alert_metrics.get("detection_rate", np.nan))
        out["alert_detection_rate_filtered"].append(
            alert_metrics.get("detection_rate_filtered", np.nan)
        )
        out["alert_detection_rate_majority"].append(
            alert_metrics.get("detection_rate", np.nan)
        )
        out["alert_detection_rate_coarsened"].append(
            alert_metrics.get("detection_rate_filtered", np.nan)
        )
        out["alert_recall"].append(alert_metrics.get("recall", np.nan))
        out["alert_precision"].append(alert_metrics.get("precision", np.nan))
        out["alert_f1"].append(alert_metrics.get("f1", np.nan))
        alert_recall_filtered = alert_metrics.get("recall_filtered", np.nan)
        alert_precision_filtered = alert_metrics.get("precision_filtered", np.nan)
        out["alert_recall_filtered"].append(alert_recall_filtered)
        out["alert_precision_filtered"].append(alert_precision_filtered)
        if (
            alert_recall_filtered is not None
            and alert_precision_filtered is not None
            and not np.isnan(alert_recall_filtered)
            and not np.isnan(alert_precision_filtered)
        ):
            out["alert_f1_filtered"].append(
                (2.0 * alert_recall_filtered * alert_precision_filtered)
                / (alert_recall_filtered + alert_precision_filtered + 1e-12)
            )
        else:
            out["alert_f1_filtered"].append(np.nan)

        out["normal_detection_rate"].append(
            normal_metrics.get("detection_rate", np.nan)
        )
        out["normal_detection_rate_filtered"].append(
            normal_metrics.get("detection_rate_filtered", np.nan)
        )
        out["normal_detection_rate_majority"].append(
            normal_metrics.get("detection_rate", np.nan)
        )
        out["normal_detection_rate_coarsened"].append(
            normal_metrics.get("detection_rate_filtered", np.nan)
        )
        out["normal_recall"].append(normal_metrics.get("recall", np.nan))
        out["normal_precision"].append(normal_metrics.get("precision", np.nan))
        out["normal_f1"].append(normal_metrics.get("f1", np.nan))
        normal_recall_filtered = normal_metrics.get("recall_filtered", np.nan)
        normal_precision_filtered = normal_metrics.get("precision_filtered", np.nan)
        out["normal_recall_filtered"].append(normal_recall_filtered)
        out["normal_precision_filtered"].append(normal_precision_filtered)
        if (
            normal_recall_filtered is not None
            and normal_precision_filtered is not None
            and not np.isnan(normal_recall_filtered)
            and not np.isnan(normal_precision_filtered)
        ):
            out["normal_f1_filtered"].append(
                (2.0 * normal_recall_filtered * normal_precision_filtered)
                / (normal_recall_filtered + normal_precision_filtered + 1e-12)
            )
        else:
            out["normal_f1_filtered"].append(np.nan)

        alert_type_rates: Dict = {}
        normal_type_rates: Dict = {}
        for key, metrics in level.items():
            if key.startswith("alert_metrics_") and key != "alert_metrics":
                pattern_type = key.replace("alert_metrics_", "")
                alert_type_rates[pattern_type] = {
                    "rate": metrics.get("detection_rate", 0.0),
                    "detection_rate_filtered": metrics.get(
                        "detection_rate_filtered", np.nan
                    ),
                    "f1": metrics.get("f1", np.nan),
                    "recall": metrics.get("recall", np.nan),
                    "precision": metrics.get("precision", np.nan),
                    "recall_filtered": metrics.get("recall_filtered", np.nan),
                    "precision_filtered": metrics.get("precision_filtered", np.nan),
                    "detected": metrics.get("detected", 0.0),
                    "total": metrics.get("total", 0.0),
                }
            if key.startswith("normal_metrics_") and key != "normal_metrics":
                pattern_type = key.replace("normal_metrics_", "")
                normal_type_rates[pattern_type] = {
                    "rate": metrics.get("detection_rate", 0.0),
                    "detection_rate_filtered": metrics.get(
                        "detection_rate_filtered", np.nan
                    ),
                    "f1": metrics.get("f1", np.nan),
                    "recall": metrics.get("recall", np.nan),
                    "precision": metrics.get("precision", np.nan),
                    "recall_filtered": metrics.get("recall_filtered", np.nan),
                    "precision_filtered": metrics.get("precision_filtered", np.nan),
                    "detected": metrics.get("detected", 0.0),
                    "total": metrics.get("total", 0.0),
                }
        out["alert_gt_type_rates"].append(alert_type_rates)
        out["normal_gt_type_rates"].append(normal_type_rates)

    last_level = data[-1]
    out["epsilons"] = list(last_level.get("epsilons", []))
    out["epochs_per_level"] = list(last_level.get("epochs_per_level", []))

    return out


def _has_numeric_series(data: Dict, key: str) -> bool:
    values = data.get(key)
    if not isinstance(values, list) or not values:
        return False
    return any(v is not None and not np.isnan(v) for v in values)


def _has_type_rates(data: Dict, key: str) -> bool:
    values = data.get(key)
    return isinstance(values, list) and any(isinstance(v, dict) and v for v in values)


def _has_type_metric(data: Dict, key: str, metric: str) -> bool:
    values = data.get(key)
    if not isinstance(values, list):
        return False
    for level_rates in values:
        if not isinstance(level_rates, dict):
            continue
        for type_metrics in level_rates.values():
            if isinstance(type_metrics, dict) and metric in type_metrics:
                val = type_metrics.get(metric)
                if isinstance(val, (int, float, np.floating, np.integer)):
                    if not np.isnan(val):
                        return True
    return False


def smooth_curve(y: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply moving average smoothing to a curve."""
    if len(y) < window_size:
        return y
    return uniform_filter1d(y, size=window_size, mode="nearest")


def create_coarsening_plot(
    data: Dict,
    # threshold: List[float],
    max_epsilon: float,
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
        data: Data dict for the plot
        max_epsilon: Max epsilon value
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
    y_key_for_len = next((k for k, _, _ in y_keys if k in data), None)
    y_len = len(data.get(y_key_for_len, [])) if y_key_for_len else 0
    if baseline_value is not None:
        xmax = y_len if x_key is None else max(data.get(x_key, [0]))
        ax.hlines(
            y=baseline_value,
            xmin=0,
            xmax=xmax,
            color="black",
            linestyles="--",
            label=baseline_label,
        )

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
            # color=color,
            # linewidth=width,
            linestyle=style,
            marker=marker,
            markersize=2,
            label=f"{label_suffix} {max_epsilon:.3f}%",
        )

    # Add coarse baseline if provided
    if coarse_baseline is not None:
        xmax = y_len if x_key is None else max(data.get(x_key, [0]))
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
    data: Dict,
    max_epsilon: float,
    save_path: str,
    baseline_accuracy: float,
    coarse_accuracy: Optional[float] = None,
    smooth: bool = False,
):
    """Plot coarse vs fine accuracy over iterations."""
    return create_coarsening_plot(
        data=data,
        max_epsilon=max_epsilon,
        x_key=None,
        y_keys=[
            ("accuracy_test", "-", "Coarse"),
            ("accuracy_fine", ":", "Fine"),
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
    data: Dict,
    max_epsilon: float,
    save_path: str,
    baseline_accuracy: float,
    coarse_accuracy: Optional[float] = None,
    smooth: bool = False,
):
    """Plot coarse vs fine accuracy against number of coarse nodes."""
    return create_coarsening_plot(
        data=data,
        max_epsilon=max_epsilon,
        x_key="num_nodes_coarse",
        y_keys=[
            ("accuracy_test", "-", "Coarse"),
            ("accuracy_fine", ":", "Fine"),
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
    data: Dict,
    max_epsilon: float,
    save_path: str,
    baseline_precision: float,
    smooth: bool = False,
):
    """Plot coarse vs fine precision against number of coarse nodes."""
    return create_coarsening_plot(
        data=data,
        max_epsilon=max_epsilon,
        x_key="num_nodes_coarse",
        y_keys=[
            ("precision_test", "-", "Coarse"),
            ("precision_fine", ":", "Fine"),
        ],
        xlabel="Number of Coarse Nodes",
        ylabel="Precision",
        title="Coarse vs Precision over Iterations",
        save_path=save_path,
        baseline_value=baseline_precision,
        smooth=smooth,
    )


def plot_pattern_detection(
    data: Dict,
    max_epsilon: float,
    save_path: str,
    pattern_type: str = "alert",
    include_model: bool = True,
    include_sub_rates: bool = True,
    coarse_th: str = "75%",
    majority_th: str = "75%",
):
    """
    Plot pattern detection rates across coarsening levels.

    Args:
        data: Data dict for the plot
        max_epsilon: Max epsilon value
        save_path: Path to save the figure
        pattern_type: "alert" or "normal"
        include_model: Whether to include model-based detection (for alerts)
        include_sub_rates: Whether to include rate1 and rate2 on a separate figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    gt_key = f"{pattern_type}_detection_rate"

    # Keys for majority and coarsened rates
    gt_rate1_key = f"{pattern_type}_detection_rate_majority"
    gt_rate2_key = f"{pattern_type}_detection_rate_coarsened"

    # GT-based detection (combined rate)
    if gt_key in data:
        ax.plot(
            data[gt_key],
            # color=color,
            # linewidth=width,
            marker="o" if pattern_type == "alert" else "s",
            markersize=3,
            label=f"{pattern_type.title()} Detection (ep={max_epsilon:.3f}%)",
        )

    # majority_th = "75%" if pattern_type == "alert" else "50%"
    # coarse_th = "75%" if pattern_type == "alert" else "50%"

    ax.set_xlabel("Coarsening Level")
    ax.set_ylabel("Pattern Detection Rate")
    ax.set_title(
        f"{pattern_type.title()} Pattern Detection Rate vs Coarsening Level\n(>{majority_th} {pattern_type} + >{coarse_th} coarsened together)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path)

    # Create separate figure for rate1 and rate2 if requested
    fig2, ax2 = None, None
    if include_sub_rates:
        fig2, ax2 = plt.subplots(figsize=(12, 6))

        # GT-based rate1 (majority threshold)
        if gt_rate1_key in data:
            ax2.plot(
                data[gt_rate1_key],
                color=DEFAULT_COLOR,
                # linewidth=width,
                linestyle="-",
                marker="^",
                markersize=3,
                label=f"{pattern_type.title()} Majority Rate (ep={max_epsilon:.3f}%)",
            )

        # GT-based rate2 (coarsening threshold)
        if gt_rate2_key in data:
            ax2.plot(
                data[gt_rate2_key],
                color=DEFAULT_COLOR,
                # linewidth=width,
                linestyle="--",
                marker="v",
                markersize=3,
                label=f"{pattern_type.title()} Coarsened Rate (ep={max_epsilon:.3f}%)",
            )

        ax2.set_xlabel("Coarsening Level")
        ax2.set_ylabel("Pattern Detection Rate")
        ax2.set_title(
            f"{pattern_type.title()} Pattern Detection Sub-Rates vs Coarsening Level\n(Majority threshold vs coarsening threshold)"
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save sub-rates figure with modified filename
        sub_rates_path = save_path.replace(".png", "_sub_rates.png").replace(
            ".pdf", "_sub_rates.pdf"
        )
        if sub_rates_path == save_path:
            sub_rates_path = save_path + "_sub_rates"
        fig2.savefig(sub_rates_path)

    return fig, ax, fig2, ax2


def plot_pattern_detection_by_type(
    data: Dict,
    max_epsilon: float,
    save_path: str,
    pattern_type: str = "alert",
    use_model: bool = False,
):
    """
    Plot pattern detection rates broken down by pattern type.

    Args:
        data: Data dict for the max epsilon
        max_epsilon: Max epsilon value
        save_path: Path to save the figure
        pattern_type: "alert" or "normal"
        use_model: Whether to use model-based detection (only for alerts)
    """
    # Determine which key to use for type rates
    if pattern_type == "alert":
        type_rates_key = (
            "alert_model_type_rates" if use_model else "alert_gt_type_rates"
        )
    else:
        type_rates_key = "normal_gt_type_rates"

    # Collect all unique pattern types across all data
    all_pattern_types = set()
    if type_rates_key in data and len(data[type_rates_key]) > 0:
        # type_rates is a list of dicts (one per level)
        for level_rates in data[type_rates_key]:
            all_pattern_types.update(level_rates.keys())

    if not all_pattern_types:
        print(f"No type detection data found for {pattern_type} patterns")
        return None, None

    all_pattern_types = sorted(all_pattern_types)
    n_types = len(all_pattern_types)

    # Create figure with subplots for each pattern type
    n_cols = min(3, n_types)
    n_rows = (n_types + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False
    )

    # Define colors for different max_epsilons
    type_colors = {
        # Alert types (warm colors)
        "fan_out": "#E63333",
        "fan_in": "#E6801A",
        "cycle": "#CC1A80",
        "bipartite": "#991ACC",
        "stack": "#E6CC1A",
        "scatter_gather": "#FF6666",
        "gather_scatter": "#CC4D00",
        # Normal types (cool colors)
        "single": "#3399E6",
        "forward": "#6666E6",
        "mutual": "#1AB3B3",
        "periodical": "#80CC33",
    }

    for type_idx, ptype in enumerate(all_pattern_types):
        row = type_idx // n_cols
        col = type_idx % n_cols
        ax = axes[row, col]

        if type_rates_key not in data:
            continue

        # Extract rates for this pattern type across all levels
        rates = []
        for level_rates in data[type_rates_key]:
            if ptype in level_rates:
                rates.append(level_rates[ptype]["rate"])
            else:
                rates.append(0.0)

        if rates:
            ax.plot(
                rates,
                color=type_colors.get(ptype, DEFAULT_COLOR),
                marker="o",
                markersize=3,
                label=f"eps={max_epsilon:.1f}",
            )

        ax.set_xlabel("Coarsening Level")
        ax.set_ylabel("Detection Rate")
        ax.set_title(f"{ptype.replace('_', ' ').title()}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide empty subplots
    for idx in range(n_types, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    source = "Model" if use_model else "GT"
    fig.suptitle(
        f"{pattern_type.title()} Pattern Detection by Type ({source}-based)",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved per-type detection plot to {save_path}")

    return fig, axes


def plot_pattern_type_comparison(
    data: Dict,
    max_epsilon: float,
    save_path: str,
    pattern_type: str = "alert",
    use_model: bool = False,
    level_idx: int = -1,
):
    """
    Plot bar chart comparing detection rates across pattern types at a specific level.

    Args:
        data: Data dict for the max epsilon
        max_epsilon: Max epsilon value
        save_path: Path to save the figure
        pattern_type: "alert" or "normal"
        use_model: Whether to use model-based detection (only for alerts)
        level_idx: Which coarsening level to show (-1 for final level)
    """
    # Determine which key to use for type rates
    if pattern_type == "alert":
        type_rates_key = (
            "alert_model_type_rates" if use_model else "alert_gt_type_rates"
        )
    else:
        type_rates_key = "normal_gt_type_rates"

    # Collect data for each epsilon
    type_colors = {
        # Alert types (warm colors)
        "fan_out": "#E63333",
        "fan_in": "#E6801A",
        "cycle": "#CC1A80",
        "bipartite": "#991ACC",
        "stack": "#E6CC1A",
        "scatter_gather": "#FF6666",
        "gather_scatter": "#CC4D00",
        # Normal types (cool colors)
        "single": "#3399E6",
        "forward": "#6666E6",
        "mutual": "#1AB3B3",
        "periodical": "#80CC33",
    }

    # Get all pattern types and their rates at the specified level
    all_pattern_types = set()
    # level_data = []

    # if type_rates_key not in data or len(data[type_rates_key]) == 0:
    # level_data.append({})

    level_rates = data[type_rates_key][level_idx]
    all_pattern_types.update(level_rates.keys())
    # level_data.append(level_rates)

    if not all_pattern_types:
        print(f"No type detection data found for {pattern_type} patterns")
        return None, None

    all_pattern_types = sorted(all_pattern_types)
    n_types = len(all_pattern_types)
    # n_eps = len(max_epsilons)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_types)
    # width = 0.8 / n_eps

    rates = []
    counts = []
    bar_colors = []
    for ptype in all_pattern_types:
        if ptype in level_rates:
            rates.append(level_rates[ptype].get("rate", 0.0))
            if "detected" in level_rates[ptype] and "total" in level_rates[ptype]:
                counts.append(
                    f"{level_rates[ptype]['detected']}/{level_rates[ptype]['total']}"
                )
            else:
                counts.append("")
        else:
            rates.append(0.0)
            counts.append("")
        bar_colors.append(type_colors.get(ptype, DEFAULT_COLOR))

    # offset = (idx - n_eps / 2 + 0.5) * width
    bars = ax.bar(
        x,
        rates,
        color=bar_colors,
        # width,
        label=f"eps={max_epsilon:.1f}",
        # color=COLORS[idx % len(COLORS)],
        alpha=0.8,
    )

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if not count:
            continue
        height = bar.get_height()
        ax.annotate(
            count,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=45,
        )

    ax.set_xlabel("Pattern Type")
    ax.set_ylabel("Detection Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [t.replace("_", " ").title() for t in all_pattern_types],
        rotation=45,
        ha="right",
    )
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    level_name = "Final" if level_idx == -1 else f"Level {level_idx}"
    source = "Model" if use_model else "GT"
    ax.set_title(
        f"{pattern_type.title()} Pattern Detection by Type at {level_name} Level ({source}-based)"
    )

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved pattern type comparison to {save_path}")

    return fig, ax


def plot_pattern_type_heatmap(
    data: Dict,
    max_epsilon: float,
    save_path: str,
    pattern_type: str = "alert",
    use_model: bool = False,
):
    """
    Plot heatmap of detection rates by pattern type vs coarsening level.

    Args:
        data: data dicts for each max epsilon
        max_epsilon: The max epsilon value
        save_path: Path to save the figure
        pattern_type: "alert" or "normal"
        use_model: Whether to use model-based detection (only for alerts)
        epsilon_idx: Which epsilon setting to use (-1 for last)
    """
    # Determine which key to use for type rates
    if pattern_type == "alert":
        type_rates_key = (
            "alert_model_type_rates" if use_model else "alert_gt_type_rates"
        )
    else:
        type_rates_key = "normal_gt_type_rates"

    if type_rates_key not in data or len(data[type_rates_key]) == 0:
        print(f"No type detection data found for {pattern_type} patterns")
        return None, None

    # Get all pattern types
    all_pattern_types = set()
    for level_rates in data[type_rates_key]:
        all_pattern_types.update(level_rates.keys())
    all_pattern_types = sorted(all_pattern_types)

    n_levels = len(data[type_rates_key])
    n_types = len(all_pattern_types)

    # Build heatmap matrix
    heatmap = np.zeros((n_types, n_levels))
    for level_idx, level_rates in enumerate(data[type_rates_key]):
        for type_idx, ptype in enumerate(all_pattern_types):
            if ptype in level_rates:
                heatmap[type_idx, level_idx] = level_rates[ptype]["rate"]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Detection Rate")

    # Set labels
    ax.set_yticks(range(n_types))
    ax.set_yticklabels([t.replace("_", " ").title() for t in all_pattern_types])

    # Reduce x-axis labels if too many levels
    if n_levels > 20:
        step = n_levels // 10
        ax.set_xticks(range(0, n_levels, step))
        ax.set_xticklabels(range(0, n_levels, step))
    else:
        ax.set_xticks(range(n_levels))
        ax.set_xticklabels(range(n_levels))

    ax.set_xlabel("Coarsening Level")
    ax.set_ylabel("Pattern Type")

    source = "Model" if use_model else "GT"
    ax.set_title(
        f"{pattern_type.title()} Pattern Detection Heatmap by Type ({source}-based, eps={max_epsilon:.1f})"
    )

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved pattern type heatmap to {save_path}")

    return fig, ax


def plot_pattern_metrics_over_levels(
    data: Dict,
    max_epsilon: float,
    save_path: str,
    pattern_type: str = "alert",
):
    """Plot overall filtered detection and F1 across coarsening levels."""
    det_key = f"{pattern_type}_detection_rate"
    det_filtered_key = f"{pattern_type}_detection_rate_filtered"
    f1_key = f"{pattern_type}_f1"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if det_key in data:
        axes[0].plot(data[det_key], marker="o", markersize=3, label="Detection Rate")
    if det_filtered_key in data:
        axes[0].plot(
            data[det_filtered_key],
            marker="s",
            markersize=3,
            linestyle="--",
            label="Detection Rate Filtered",
        )
    axes[0].set_xlabel("Coarsening Level")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title(f"{pattern_type.title()} Detection Metrics")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if f1_key in data:
        axes[1].plot(
            data[f1_key],
            marker="o",
            markersize=3,
            color=DEFAULT_COLOR,
            label="F1",
        )
    axes[1].set_xlabel("Coarsening Level")
    axes[1].set_ylabel("F1")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title(f"{pattern_type.title()} F1")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        f"{pattern_type.title()} Overall Pattern Metrics (eps={max_epsilon:.3f})",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    return fig, axes


def plot_average_prf_filtered_unfiltered(
    data: Dict,
    max_epsilon: float,
    save_path: str,
    pattern_type: str = "alert",
):
    """Plot average recall/precision/F1 for both filtered and unfiltered on one axis."""
    recall_key = f"{pattern_type}_recall"
    precision_key = f"{pattern_type}_precision"
    f1_key = f"{pattern_type}_f1"
    recall_filtered_key = f"{pattern_type}_recall_filtered"
    precision_filtered_key = f"{pattern_type}_precision_filtered"
    f1_filtered_key = f"{pattern_type}_f1_filtered"

    fig, ax = plt.subplots(figsize=(12, 6))

    series_specs = [
        (recall_key, "Recall", "-", "o"),
        (precision_key, "Precision", "-", "s"),
        (f1_key, "F1", "-", "^"),
        (recall_filtered_key, "Recall (Filtered)", "--", "o"),
        (precision_filtered_key, "Precision (Filtered)", "--", "s"),
        (f1_filtered_key, "F1 (Filtered)", "--", "^"),
    ]

    color_map = {
        "Recall": "tab:blue",
        "Precision": "tab:orange",
        "F1": "tab:green",
        "Recall (Filtered)": "tab:blue",
        "Precision (Filtered)": "tab:orange",
        "F1 (Filtered)": "tab:green",
    }

    plotted_any = False
    for key, label, linestyle, marker in series_specs:
        values = data.get(key, [])
        if not isinstance(values, list) or not values:
            continue
        if not any(v is not None and not np.isnan(v) for v in values):
            continue
        ax.plot(
            values,
            label=label,
            linestyle=linestyle,
            marker=marker,
            markersize=3,
            color=color_map[label],
        )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return None, None

    ax.set_xlabel("Coarsening Level")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"{pattern_type.title()} Average Recall/Precision/F1 (Filtered + Unfiltered, eps={max_epsilon:.3f})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_pattern_type_metrics_combined(
    data: Dict,
    max_epsilon: float,
    save_path: str,
    pattern_type: str = "alert",
    metric_specs: Optional[List[Tuple[str, str]]] = None,
    use_model: bool = False,
):
    """Plot multiple per-type metrics in one figure (lines are pattern types)."""
    if pattern_type == "alert":
        type_rates_key = (
            "alert_model_type_rates" if use_model else "alert_gt_type_rates"
        )
    else:
        type_rates_key = "normal_gt_type_rates"

    if type_rates_key not in data or len(data[type_rates_key]) == 0:
        return None, None

    if metric_specs is None:
        metric_specs = [
            ("rate", "Detection Rate"),
            ("detection_rate_filtered", "Detection Rate Filtered"),
            ("f1", "F1"),
        ]

    all_pattern_types = set()
    for level_rates in data[type_rates_key]:
        all_pattern_types.update(level_rates.keys())
    if not all_pattern_types:
        return None, None
    all_pattern_types = sorted(all_pattern_types)

    type_colors = {
        "fan_out": "#E63333",
        "fan_in": "#E6801A",
        "cycle": "#CC1A80",
        "bipartite": "#991ACC",
        "stack": "#E6CC1A",
        "scatter_gather": "#FF6666",
        "gather_scatter": "#CC4D00",
        "single": "#3399E6",
        "forward": "#6666E6",
        "mutual": "#1AB3B3",
        "periodical": "#80CC33",
    }

    n_metrics = len(metric_specs)
    n_rows = 2 if n_metrics > 3 else 1
    n_cols = int(np.ceil(n_metrics / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, (metric_key, metric_label) in zip(axes, metric_specs):
        plotted_any = False
        for ptype in all_pattern_types:
            values = []
            for level_rates in data[type_rates_key]:
                metrics = (
                    level_rates.get(ptype, {}) if isinstance(level_rates, dict) else {}
                )
                val = (
                    metrics.get(metric_key, np.nan)
                    if isinstance(metrics, dict)
                    else np.nan
                )
                values.append(val)

            if any(v is not None and not np.isnan(v) for v in values):
                ax.plot(
                    values,
                    marker="o",
                    markersize=3,
                    label=ptype.replace("_", " ").title(),
                    color=type_colors.get(ptype, DEFAULT_COLOR),
                )
                plotted_any = True

        ax.set_xlabel("Coarsening Level")
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(True, alpha=0.3)
        if metric_key in {
            "rate",
            "detection_rate_filtered",
            "f1",
            "recall",
            "precision",
        }:
            ax.set_ylim(-0.05, 1.05)
        if plotted_any:
            ax.legend(fontsize=8)

    # Hide unused axes when grid has more slots than metrics.
    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    source = "Model" if use_model else "GT"
    fig.suptitle(
        f"{pattern_type.title()} Per-Type Metrics ({source}-based, eps={max_epsilon:.3f})",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    return fig, axes


def plot_epsilon_vs_level(
    data: Dict,
    max_epsilon: float,
    save_path: str,
):
    """Plot epsilon vs coarsening level."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # if "epsilons" not in data or "x" not in data:
    #     continue

    # color = COLORS[idx % len(COLORS)]
    levels = np.array(data["x"][: len(data["epsilons"])])
    epsilons = np.array(data["epsilons"])

    ax.plot(
        levels,
        epsilons,
        # color=color,
        linewidth=2,
        marker="o",
        markersize=3,
        label=f"max_eps={max_epsilon:.1f}",
    )

    # Add horizontal line for max epsilon
    # ax.axhline(y=max_epsilon, color=color, linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Coarsening Level")
    ax.set_ylabel("Epsilon (ε)")
    ax.set_title("Epsilon vs Coarsening Level")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path)
    print(f"Saved epsilon vs level plot to {save_path}")

    return fig, ax


def plot_epsilon_vs_precision(
    data: Dict,
    max_epsilon: float,
    save_path: str,
    baseline_precision: Optional[float] = None,
):
    """Plot epsilon vs precision."""
    fig, ax = plt.subplots(figsize=(12, 6))

    epsilons = np.array(data["epsilons"])
    precision = np.array(data["precision_fine"][: len(epsilons)])

    ax.plot(
        epsilons,
        precision,
        linewidth=2,
        marker="o",
        markersize=3,
        label=f"max_eps={max_epsilon:.1f}",
    )

    if baseline_precision is not None:
        ax.axhline(
            y=baseline_precision,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Baseline ({baseline_precision:.3f})",
        )

    ax.set_xlabel("Epsilon (ε)")
    ax.set_ylabel("Precision (AP@60-100%)")
    ax.set_title("Precision vs Epsilon")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path)
    print(f"Saved epsilon vs precision plot to {save_path}")

    return fig, ax


def plot_epsilon_vs_rates(
    data: Dict,
    max_epsilon: float,
    save_path: str,
):
    """Plot epsilon vs alert rate and normal rate."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    epsilons = np.array(data.get("epsilons", []))
    if len(epsilons) == 0:
        return fig, axes

    # Alert rate plot
    ax1 = axes[0]

    # Alert rates (model-based)
    if "alert_detection_rate" in data:
        alert_rates = np.array(data["alert_detection_rate"][: len(epsilons)])
        ax1.plot(
            epsilons,
            alert_rates,
            color=DEFAULT_COLOR,
            linewidth=2,
            marker="o",
            markersize=3,
            linestyle="-",
            label=f"Detection (max_eps={max_epsilon:.1f})",
        )

    ax1.set_xlabel("Epsilon (ε)")
    ax1.set_ylabel("Alert Detection Rate")
    ax1.set_title("Alert Rate vs Epsilon")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # Normal rate plot
    ax2 = axes[1]

    # Normal rates (GT-based)
    if "normal_detection_rate" in data:
        normal_rates_gt = np.array(data["normal_detection_rate"][: len(epsilons)])
        ax2.plot(
            epsilons,
            normal_rates_gt,
            color=DEFAULT_COLOR,
            linewidth=2,
            marker="o",
            markersize=3,
            label=f"max_eps={max_epsilon:.1f}",
        )

    ax2.set_xlabel("Epsilon (ε)")
    ax2.set_ylabel("Normal Detection Rate")
    ax2.set_title("Normal Rate vs Epsilon")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    fig.savefig(save_path)
    print(f"Saved epsilon vs rates plot to {save_path}")

    return fig, axes


def plot_auc_vs_coarsening(
    data: Dict,
    max_epsilon: float,
    save_path: str,
    pattern_type: str = "alert",
    include_ap: bool = True,
):
    """
    Plot AUC and Average Precision scores across coarsening levels.

    This shows threshold-independent performance metrics.

    Args:
        data: Data dictionary for the current max epsilon
        max_epsilon: Maximum epsilon value
        save_path: Path to save the figure
        pattern_type: "alert" or "normal"
        include_ap: Whether to include Average Precision in the same plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    auc_key = f"{pattern_type}_auc"
    ap_key = f"{pattern_type}_ap"

    # Pattern-level AUC
    if auc_key in data:
        ax.plot(
            data[auc_key],
            color=DEFAULT_COLOR,
            # linewidth=width,
            marker="o",
            markersize=4,
            linestyle="-",
            label=f"AUC (ep={max_epsilon:.3f}%)",
        )

    # Average Precision
    if include_ap and ap_key in data:
        ax.plot(
            data[ap_key],
            color=DEFAULT_COLOR,
            # linewidth=width,
            marker="s",
            markersize=4,
            linestyle="--",
            alpha=0.7,
            label=f"AP (ep={max_epsilon:.3f}%)",
        )

    ax.set_xlabel("Coarsening Level")
    ax.set_ylabel("Score")
    ax.set_title(
        f"{pattern_type.title()} Pattern Detection - AUC & AP vs Coarsening Level\n(Threshold-Independent)"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    fig.savefig(save_path)
    print(f"Saved AUC plot to {save_path}")

    return fig, ax


def plot_node_auc_vs_coarsening(
    data: Dict,
    max_epsilon: float,
    save_path: str,
    pattern_type: str = "alert",
):
    """
    Plot node-level AUC scores across coarsening levels.

    Args:
        data: Data dictionary for the current max epsilon
        max_epsilon: Maximum epsilon value
        save_path: Path to save the figure
        pattern_type: "alert" or "normal"
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    node_auc_key = f"{pattern_type}_node_auc"
    node_ap_key = f"{pattern_type}_node_ap"

    if node_auc_key in data:
        ax.plot(
            data[node_auc_key],
            color=DEFAULT_COLOR,
            # linewidth=width,
            marker="o",
            markersize=4,
            linestyle="-",
            label=f"Node AUC (ep={max_epsilon:.3f}%)",
        )

    if node_ap_key in data:
        ax.plot(
            data[node_ap_key],
            color=DEFAULT_COLOR,
            # linewidth=width,
            marker="s",
            markersize=4,
            linestyle="--",
            alpha=0.7,
            label=f"Node AP (ep={max_epsilon:.3f}%)",
        )

    ax.set_xlabel("Coarsening Level")
    ax.set_ylabel("Score")
    ax.set_title(f"{pattern_type.title()} Node-Level AUC vs Coarsening Level")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    fig.savefig(save_path)

    return fig, ax


def plot_coarsening_quality_vs_level(
    data: Dict,
    max_epsilon: float,
    save_path: str,
    pattern_type: str = "alert",
):
    """
    Plot mean coarsening ratio (quality) across coarsening levels.

    Higher values mean pattern nodes are better preserved together.

    Args:
        data: Data dictionary for the current max epsilon
        max_epsilon: Maximum epsilon value
        save_path: Path to save the figure
        pattern_type: "alert" or "normal"
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    coarse_key = f"{pattern_type}_coarsening_ratio"

    if coarse_key in data:
        ax.plot(
            data[coarse_key],
            color=DEFAULT_COLOR,
            # linewidth=width,
            marker="^",
            markersize=4,
            label=f"ep={max_epsilon:.3f}%",
        )

    ax.set_xlabel("Coarsening Level")
    ax.set_ylabel("Mean Coarsening Ratio")
    ax.set_title(
        f"{pattern_type.title()} Pattern Coarsening Quality vs Level\n(Higher = Pattern nodes stay together)"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    fig.savefig(save_path)

    return fig, ax


def plot_roc_curves_at_levels(
    data: Dict,
    max_epsilon: float,
    save_path: str,
    pattern_type: str = "alert",
    levels_to_plot: List[int] = None,
):
    """
    Plot ROC curves at specific coarsening levels.

    Args:
        data: Data dictionary for the current max epsilon
        max_epsilon: Maximum epsilon value
        save_path: Path to save the figure
        pattern_type: "alert" or "normal"
        levels_to_plot: Which levels to plot (default: [0, -1] for first and last)
    """
    roc_key = f"{pattern_type}_roc_data"

    if levels_to_plot is None:
        levels_to_plot = [0, -1]

    n_levels = len(levels_to_plot)
    fig, axes = plt.subplots(1, n_levels, figsize=(6 * n_levels, 5))
    if n_levels == 1:
        axes = [axes]

    for ax_idx, level_idx in enumerate(levels_to_plot):
        ax = axes[ax_idx]

        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")

        if roc_key not in data:
            continue

        roc_data = data[roc_key]
        if len(roc_data) == 0:
            continue

        # Handle negative indexing
        actual_idx = level_idx if level_idx >= 0 else len(roc_data) + level_idx
        if actual_idx < 0 or actual_idx >= len(roc_data):
            continue

        roc_data = roc_data[actual_idx]
        if len(roc_data["fpr"]) > 0:
            auc_val = data.get(f"{pattern_type}_auc", [0] * len(roc_data))[actual_idx]
            ax.plot(
                roc_data["fpr"],
                roc_data["tpr"],
                color=DEFAULT_COLOR,
                linewidth=2,
                label=f"ep={max_epsilon:.3f}% (AUC={auc_val:.3f})",
            )

        level_name = (
            "Initial"
            if level_idx == 0
            else "Final" if level_idx == -1 else f"Level {level_idx}"
        )
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{pattern_type.title()} ROC Curve - {level_name}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    fig.savefig(save_path)
    print(f"Saved ROC curves to {save_path}")

    return fig, axes


def plot_auc_comparison_bar(
    data: Dict,
    max_epsilon: float,
    save_path: str,
    pattern_type: str = "alert",
):
    """
    Bar chart comparing AUC at initial vs final coarsening level.

    Args:
        data: List of data dicts for each max epsilon
        max_epsilons: List of max epsilon values
        save_path: Path to save the figure
        pattern_type: "alert" or "normal"
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    auc_key = f"{pattern_type}_auc"

    x = np.arange(1)
    width = 0.35

    initial_aucs = []
    final_aucs = []
    if auc_key in data and len(data[auc_key]) > 0:
        initial_aucs.append(data[auc_key][0])
        final_aucs.append(data[auc_key][-1])
    else:
        initial_aucs.append(0)
        final_aucs.append(0)

    bars1 = ax.bar(
        x - width / 2, initial_aucs, width, label="Initial", color="steelblue"
    )
    bars2 = ax.bar(x + width / 2, final_aucs, width, label="Final", color="indianred")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Max Epsilon (%)")
    ax.set_ylabel("AUC Score")
    ax.set_title(f"{pattern_type.title()} Pattern AUC: Initial vs Final Coarsening")
    ax.set_xticks(x)
    ax.set_xticklabels(f"{max_epsilon:.3f}%")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    fig.savefig(save_path)

    return fig, ax


def plot_all_results(
    data_list: Dict | List[Dict],
    # threshold: float,
    save_dir: str,
    max_epsilon: float,
    # epochs_per_level: int,
    baseline_accuracy: float,
    baseline_precision: float,
    coarse_accuracy: Optional[float] = None,
    has_alert_patterns: bool = False,
    has_normal_patterns: bool = False,
    smooth: bool = False,
    alert_coarse_th: str = "75%",
    alert_majority_th: str = "75%",
    normal_coarse_th: str = "75%",
    normal_majority_th: str = "75%",
    name_prefix: str = "",
):
    """
    Generate all standard plots for a coarsening experiment.

    Args:
        data_list: New results_history (list of dicts) or legacy dict-of-series
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
    data = _to_series_dict(data_list)
    if isinstance(max_epsilon, (list, tuple, np.ndarray)):
        max_epsilon = float(max_epsilon[0]) if len(max_epsilon) > 0 else float("nan")

    # Accuracy vs iteration
    if _has_numeric_series(data, "accuracy_test") or _has_numeric_series(
        data, "accuracy_fine"
    ):
        plot_accuracy_vs_iteration(
            data,
            max_epsilon,
            str(save_dir / f"iterative_custom_loss_accuracy_dynamic{name_prefix}.png"),
            baseline_accuracy,
            coarse_accuracy,
            smooth,
        )

    # Accuracy vs nodes
    if _has_numeric_series(data, "num_nodes_coarse") and (
        _has_numeric_series(data, "accuracy_test")
        or _has_numeric_series(data, "accuracy_fine")
    ):
        plot_accuracy_vs_nodes(
            data,
            max_epsilon,
            str(
                save_dir
                / f"iterative_custom_loss_accuracy_dynamic_with_nodes{name_prefix}.png"
            ),
            baseline_accuracy,
            coarse_accuracy,
            smooth,
        )

    # Precision vs nodes
    if _has_numeric_series(data, "num_nodes_coarse") and (
        _has_numeric_series(data, "precision_test")
        or _has_numeric_series(data, "precision_fine")
    ):
        plot_precision_vs_nodes(
            data,
            max_epsilon,
            str(
                save_dir
                / f"iterative_custom_loss_precission_dynamic_with_nodes{name_prefix}.png"
            ),
            baseline_precision,
            smooth,
        )

    # Epsilon-based plots
    if _has_numeric_series(data, "epsilons"):
        plot_epsilon_vs_level(
            data,
            max_epsilon,
            str(save_dir / f"epsilon_vs_level{name_prefix}.png"),
        )

    if _has_numeric_series(data, "epsilons") and _has_numeric_series(
        data, "precision_fine"
    ):
        plot_epsilon_vs_precision(
            data,
            max_epsilon,
            str(save_dir / f"epsilon_vs_precision{name_prefix}.png"),
            baseline_precision,
        )

    if _has_numeric_series(data, "epsilons") and (
        _has_numeric_series(data, "alert_detection_rate")
        or _has_numeric_series(data, "normal_detection_rate")
    ):
        plot_epsilon_vs_rates(
            data,
            max_epsilon,
            str(save_dir / f"epsilon_vs_rates{name_prefix}.png"),
        )

    # Pattern detection plots
    if _has_numeric_series(data, "alert_detection_rate"):
        plot_pattern_detection(
            data,
            max_epsilon,
            str(save_dir / f"alert_pattern_detection_rates_dynamic{name_prefix}.png"),
            pattern_type="alert",
            include_model=True,
            coarse_th=alert_coarse_th,
            majority_th=alert_majority_th,
        )

    if _has_numeric_series(data, "normal_detection_rate"):
        plot_pattern_detection(
            data,
            max_epsilon,
            str(save_dir / f"normal_pattern_detection_rates_dynamic{name_prefix}.png"),
            pattern_type="normal",
            include_model=False,
            coarse_th=normal_coarse_th,
            majority_th=normal_majority_th,
        )

    # Overall pattern metrics: detection_rate_filtered + f1
    if _has_numeric_series(
        data, "alert_detection_rate_filtered"
    ) or _has_numeric_series(data, "alert_f1"):
        plot_pattern_metrics_over_levels(
            data,
            max_epsilon,
            str(save_dir / f"alert_pattern_metrics_over_levels{name_prefix}.png"),
            pattern_type="alert",
        )

    if _has_numeric_series(
        data, "normal_detection_rate_filtered"
    ) or _has_numeric_series(data, "normal_f1"):
        plot_pattern_metrics_over_levels(
            data,
            max_epsilon,
            str(save_dir / f"normal_pattern_metrics_over_levels{name_prefix}.png"),
            pattern_type="normal",
        )

    # Average PRF (filtered + unfiltered) on the same plot
    if any(
        _has_numeric_series(data, k)
        for k in [
            "alert_recall",
            "alert_precision",
            "alert_f1",
            "alert_recall_filtered",
            "alert_precision_filtered",
            "alert_f1_filtered",
        ]
    ):
        plot_average_prf_filtered_unfiltered(
            data,
            max_epsilon,
            str(save_dir / f"alert_average_prf_filtered_unfiltered{name_prefix}.png"),
            pattern_type="alert",
        )

    if any(
        _has_numeric_series(data, k)
        for k in [
            "normal_recall",
            "normal_precision",
            "normal_f1",
            "normal_recall_filtered",
            "normal_precision_filtered",
            "normal_f1_filtered",
        ]
    ):
        plot_average_prf_filtered_unfiltered(
            data,
            max_epsilon,
            str(save_dir / f"normal_average_prf_filtered_unfiltered{name_prefix}.png"),
            pattern_type="normal",
        )

    # Per-type pattern detection plots
    # Detection rates by type over coarsening levels
    if _has_type_rates(data, "alert_gt_type_rates"):
        plot_pattern_detection_by_type(
            data,
            max_epsilon,
            str(save_dir / f"alert_detection_by_type{name_prefix}.png"),
            pattern_type="alert",
            use_model=False,
        )
        # Bar chart comparison at final level
        plot_pattern_type_comparison(
            data,
            max_epsilon,
            str(save_dir / f"alert_type_comparison_final{name_prefix}.png"),
            pattern_type="alert",
            use_model=False,
            level_idx=-1,
        )
        # Bar chart comparison at initial level
        plot_pattern_type_comparison(
            data,
            max_epsilon,
            str(save_dir / f"alert_type_comparison_initial{name_prefix}.png"),
            pattern_type="alert",
            use_model=False,
            level_idx=0,
        )
        # Heatmap of detection by type and level
        plot_pattern_type_heatmap(
            data,
            max_epsilon,
            str(save_dir / f"alert_type_heatmap{name_prefix}.png"),
            pattern_type="alert",
            use_model=False,
        )
        if _has_type_metric(
            data, "alert_gt_type_rates", "detection_rate_filtered"
        ) or _has_type_metric(data, "alert_gt_type_rates", "f1"):
            plot_pattern_type_metrics_combined(
                data,
                max_epsilon,
                str(save_dir / f"alert_type_metrics_combined{name_prefix}.png"),
                pattern_type="alert",
                metric_specs=[
                    ("rate", "Detection Rate"),
                    ("detection_rate_filtered", "Detection Rate Filtered"),
                    ("recall", "Recall"),
                    ("precision", "Precision"),
                    ("recall_filtered", "Recall Filtered"),
                    ("precision_filtered", "Precision Filtered"),
                    ("f1", "F1"),
                ],
                use_model=False,
            )

    # Detection rates by type over coarsening levels
    if _has_type_rates(data, "normal_gt_type_rates"):
        plot_pattern_detection_by_type(
            data,
            max_epsilon,
            str(save_dir / f"normal_detection_by_type{name_prefix}.png"),
            pattern_type="normal",
            use_model=False,
        )
        # Bar chart comparison at final level
        plot_pattern_type_comparison(
            data,
            max_epsilon,
            str(save_dir / f"normal_type_comparison_final{name_prefix}.png"),
            pattern_type="normal",
            use_model=False,
            level_idx=-1,
        )
        # Bar chart comparison at initial level
        plot_pattern_type_comparison(
            data,
            max_epsilon,
            str(save_dir / f"normal_type_comparison_initial{name_prefix}.png"),
            pattern_type="normal",
            use_model=False,
            level_idx=0,
        )
        # Heatmap of detection by type and level
        plot_pattern_type_heatmap(
            data,
            max_epsilon,
            str(save_dir / f"normal_type_heatmap{name_prefix}.png"),
            pattern_type="normal",
            use_model=False,
        )
        if _has_type_metric(
            data, "normal_gt_type_rates", "detection_rate_filtered"
        ) or _has_type_metric(data, "normal_gt_type_rates", "f1"):
            plot_pattern_type_metrics_combined(
                data,
                max_epsilon,
                str(save_dir / f"normal_type_metrics_combined{name_prefix}.png"),
                pattern_type="normal",
                metric_specs=[
                    ("rate", "Detection Rate"),
                    ("detection_rate_filtered", "Detection Rate Filtered"),
                    ("recall", "Recall"),
                    ("precision", "Precision"),
                    ("recall_filtered", "Recall Filtered"),
                    ("precision_filtered", "Precision Filtered"),
                    ("f1", "F1"),
                ],
                use_model=False,
            )

    # ===== AUC-based plots (threshold-independent) =====

    # Alert pattern AUC plots
    if has_alert_patterns and "alert_auc" in data:
        # AUC vs coarsening level
        plot_auc_vs_coarsening(
            data,
            max_epsilon,
            str(save_dir / f"alert_auc_vs_coarsening{name_prefix}.png"),
            pattern_type="alert",
            include_ap=True,
        )

        # Node-level AUC
        if "alert_node_auc" in data:
            plot_node_auc_vs_coarsening(
                data,
                max_epsilon,
                str(save_dir / f"alert_node_auc_vs_coarsening{name_prefix}.png"),
                pattern_type="alert",
            )

        # Coarsening quality
        if "alert_coarsening_ratio" in data:
            plot_coarsening_quality_vs_level(
                data,
                max_epsilon,
                str(save_dir / f"alert_coarsening_quality{name_prefix}.png"),
                pattern_type="alert",
            )

        # ROC curves at initial and final levels
        if "alert_roc_data" in data:
            plot_roc_curves_at_levels(
                data,
                max_epsilon,
                str(save_dir / f"alert_roc_curves{name_prefix}.png"),
                pattern_type="alert",
                levels_to_plot=[0, -1],
            )

        # AUC comparison bar chart
        plot_auc_comparison_bar(
            data,
            max_epsilon,
            str(save_dir / f"alert_auc_comparison{name_prefix}.png"),
            pattern_type="alert",
        )

    # Normal pattern AUC plots
    if has_normal_patterns and "normal_auc" in data:
        # AUC vs coarsening level
        plot_auc_vs_coarsening(
            data,
            max_epsilon,
            str(save_dir / f"normal_auc_vs_coarsening{name_prefix}.png"),
            pattern_type="normal",
            include_ap=True,
        )

        # Node-level AUC
        if "normal_node_auc" in data:
            plot_node_auc_vs_coarsening(
                data,
                max_epsilon,
                str(save_dir / f"normal_node_auc_vs_coarsening{name_prefix}.png"),
                pattern_type="normal",
            )

        # Coarsening quality
        if "normal_coarsening_ratio" in data:
            plot_coarsening_quality_vs_level(
                data,
                max_epsilon,
                str(save_dir / f"normal_coarsening_quality{name_prefix}.png"),
                pattern_type="normal",
            )

        # ROC curves at initial and final levels
        if "normal_roc_data" in data:
            plot_roc_curves_at_levels(
                data,
                max_epsilon,
                str(save_dir / f"normal_roc_curves{name_prefix}.png"),
                pattern_type="normal",
                levels_to_plot=[0, -1],
            )

        # AUC comparison bar chart
        plot_auc_comparison_bar(
            data,
            max_epsilon,
            str(save_dir / f"normal_auc_comparison{name_prefix}.png"),
            pattern_type="normal",
        )
