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


def smooth_curve(y: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply moving average smoothing to a curve."""
    if len(y) < window_size:
        return y
    return uniform_filter1d(y, size=window_size, mode="nearest")


def create_coarsening_plot(
    data_list: List[Dict],
    # threshold: List[float],
    max_epsilons: List[float],
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
        max_epsilons: List of max epsilon values
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
    for idx, (data, max_epsilon) in enumerate(zip(data_list, max_epsilons)):
        color = COLORS[idx % len(COLORS)]
        width = 2 if idx == len(max_epsilons) - 1 else 1

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
                label=f"{label_suffix} {max_epsilon:.3f}%",
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
    max_epsilons: List[float],
    save_path: str,
    baseline_accuracy: float,
    coarse_accuracy: Optional[float] = None,
    smooth: bool = False,
):
    """Plot coarse vs fine accuracy over iterations."""
    return create_coarsening_plot(
        data_list=data_list,
        max_epsilons=max_epsilons,
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
    max_epsilons: List[float],
    save_path: str,
    baseline_accuracy: float,
    coarse_accuracy: Optional[float] = None,
    smooth: bool = False,
):
    """Plot coarse vs fine accuracy against number of coarse nodes."""
    return create_coarsening_plot(
        data_list=data_list,
        max_epsilons=max_epsilons,
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
    max_epsilons: List[float],
    save_path: str,
    baseline_precision: float,
    smooth: bool = False,
):
    """Plot coarse vs fine precision against number of coarse nodes."""
    return create_coarsening_plot(
        data_list=data_list,
        max_epsilons=max_epsilons,
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
    max_epsilons: List[float],
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
        data_list: List of data dicts for each max epsilon
        max_epsilons: List of max epsilon values
        save_path: Path to save the figure
        pattern_type: "alert" or "normal"
        include_model: Whether to include model-based detection (for alerts)
        include_sub_rates: Whether to include rate1 and rate2 on a separate figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    gt_key = f"{pattern_type}_rates_gt" if pattern_type == "normal" else "alert_rates"

    # Keys for rate1 and rate2
    gt_rate1_key = f"{pattern_type}_rates_gt_rate1"
    gt_rate2_key = f"{pattern_type}_rates_gt_rate2"
    model_rate1_key = "alert_rates_rate1"
    model_rate2_key = "alert_rates_rate2"

    # Fallback for legacy key names
    if pattern_type == "normal":
        gt_key = "normal_rates_gt"

    for idx, (data, max_epsilon) in enumerate(zip(data_list, max_epsilons)):
        color = COLORS[idx % len(COLORS)]
        width = 2 if idx == len(max_epsilons) - 1 else 1

        # GT-based detection (combined rate)
        if gt_key in data:
            ax.plot(
                data[gt_key],
                color=color,
                linewidth=width,
                marker="o" if pattern_type == "alert" else "s",
                markersize=3,
                label=f"{pattern_type.title()} GT (ep={max_epsilon:.3f}%)",
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

        for idx, (data, max_epsilon) in enumerate(zip(data_list, max_epsilons)):
            color = COLORS[idx % len(COLORS)]
            width = 2 if idx == len(max_epsilons) - 1 else 1

            # GT-based rate1 (majority threshold)
            if gt_rate1_key in data:
                ax2.plot(
                    data[gt_rate1_key],
                    color=color,
                    linewidth=width,
                    linestyle="-",
                    marker="^",
                    markersize=3,
                    label=f"{pattern_type.title()} GT Rate1 (ep={max_epsilon:.3f}%)",
                )

            # GT-based rate2 (coarsening threshold)
            if gt_rate2_key in data:
                ax2.plot(
                    data[gt_rate2_key],
                    color=color,
                    linewidth=width,
                    linestyle="--",
                    marker="v",
                    markersize=3,
                    label=f"{pattern_type.title()} GT Rate2 (ep={max_epsilon:.3f}%)",
                )

            # Model-based rate1 (alerts only)
            if include_model and pattern_type == "alert" and model_rate1_key in data:
                ax2.plot(
                    data[model_rate1_key],
                    color=color,
                    linewidth=width,
                    linestyle=":",
                    marker="+",
                    markersize=4,
                    label=f"Alert Model Rate1 (ep={max_epsilon:.3f}%)",
                )

            # Model-based rate2 (alerts only)
            if include_model and pattern_type == "alert" and model_rate2_key in data:
                ax2.plot(
                    data[model_rate2_key],
                    color=color,
                    linewidth=width,
                    linestyle="-.",
                    marker="x",
                    markersize=4,
                    label=f"Alert Model Rate2 (ep={max_epsilon:.3f}%)",
                )

        ax2.set_xlabel("Coarsening Level")
        ax2.set_ylabel("Pattern Detection Rate")
        ax2.set_title(
            f"{pattern_type.title()} Pattern Detection Sub-Rates vs Coarsening Level\n(Rate1: majority threshold, Rate2: coarsening threshold)"
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
    data_list: List[Dict],
    max_epsilons: List[float],
    save_path: str,
    pattern_type: str = "alert",
    use_model: bool = False,
):
    """
    Plot pattern detection rates broken down by pattern type.

    Args:
        data_list: List of data dicts for each max epsilon
        max_epsilons: List of max epsilon values
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
    for data in data_list:
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

        for idx, (data, max_epsilon) in enumerate(zip(data_list, max_epsilons)):
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
                color = COLORS[idx % len(COLORS)]
                width = 2 if idx == len(max_epsilons) - 1 else 1
                ax.plot(
                    rates,
                    color=color,
                    linewidth=width,
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
    data_list: List[Dict],
    max_epsilons: List[float],
    save_path: str,
    pattern_type: str = "alert",
    use_model: bool = False,
    level_idx: int = -1,
):
    """
    Plot bar chart comparing detection rates across pattern types at a specific level.

    Args:
        data_list: List of data dicts for each max epsilon
        max_epsilons: List of max epsilon values
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
    level_data = []

    for data in data_list:
        if type_rates_key not in data or len(data[type_rates_key]) == 0:
            level_data.append({})
            continue

        level_rates = data[type_rates_key][level_idx]
        all_pattern_types.update(level_rates.keys())
        level_data.append(level_rates)

    if not all_pattern_types:
        print(f"No type detection data found for {pattern_type} patterns")
        return None, None

    all_pattern_types = sorted(all_pattern_types)
    n_types = len(all_pattern_types)
    n_eps = len(max_epsilons)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_types)
    width = 0.8 / n_eps

    for idx, (eps_data, max_epsilon) in enumerate(zip(level_data, max_epsilons)):
        rates = []
        counts = []
        for ptype in all_pattern_types:
            if ptype in eps_data:
                rates.append(eps_data[ptype]["rate"])
                counts.append(
                    f"{eps_data[ptype]['detected']}/{eps_data[ptype]['total']}"
                )
            else:
                rates.append(0.0)
                counts.append("0/0")

        offset = (idx - n_eps / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            rates,
            width,
            label=f"eps={max_epsilon:.1f}",
            color=COLORS[idx % len(COLORS)],
            alpha=0.8,
        )

        # Add count labels on bars
        for bar, count in zip(bars, counts):
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
    data_list: List[Dict],
    max_epsilons: List[float],
    save_path: str,
    pattern_type: str = "alert",
    use_model: bool = False,
    epsilon_idx: int = -1,
):
    """
    Plot heatmap of detection rates by pattern type vs coarsening level.

    Args:
        data_list: List of data dicts for each max epsilon
        max_epsilons: List of max epsilon values
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

    data = data_list[epsilon_idx]

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
        f"{pattern_type.title()} Pattern Detection Heatmap by Type ({source}-based, eps={max_epsilons[epsilon_idx]:.1f})"
    )

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved pattern type heatmap to {save_path}")

    return fig, ax


def plot_all_results(
    data_list: List[Dict],
    # threshold: float,
    save_dir: str,
    max_epsilons: List[float],
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
        data_list: List of data dicts for each max epsilon
        max_epsilons: List of max epsilon values
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
        max_epsilons,
        str(save_dir / f"iterative_custom_loss_accuracy_dynamic{name_prefix}.png"),
        baseline_accuracy,
        coarse_accuracy,
        smooth,
    )

    # Accuracy vs nodes
    plot_accuracy_vs_nodes(
        data_list,
        max_epsilons,
        str(
            save_dir
            / f"iterative_custom_loss_accuracy_dynamic_with_nodes{name_prefix}.png"
        ),
        baseline_accuracy,
        coarse_accuracy,
        smooth,
    )

    # Precision vs nodes
    plot_precision_vs_nodes(
        data_list,
        max_epsilons,
        str(
            save_dir
            / f"iterative_custom_loss_precission_dynamic_with_nodes{name_prefix}.png"
        ),
        baseline_precision,
        smooth,
    )

    # Pattern detection plots
    if has_alert_patterns and any("alert_rates_gt" in d for d in data_list):
        plot_pattern_detection(
            data_list,
            max_epsilons,
            str(save_dir / f"alert_pattern_detection_rates_dynamic{name_prefix}.png"),
            pattern_type="alert",
            include_model=True,
            coarse_th=alert_coarse_th,
            majority_th=alert_majority_th,
        )

    if has_normal_patterns and any("normal_rates_gt" in d for d in data_list):
        plot_pattern_detection(
            data_list,
            max_epsilons,
            str(save_dir / f"normal_pattern_detection_rates_dynamic{name_prefix}.png"),
            pattern_type="normal",
            include_model=False,
            coarse_th=normal_coarse_th,
            majority_th=normal_majority_th,
        )

    # Per-type pattern detection plots
    if has_alert_patterns and any("alert_gt_type_rates" in d for d in data_list):
        # Detection rates by type over coarsening levels
        plot_pattern_detection_by_type(
            data_list,
            max_epsilons,
            str(save_dir / f"alert_detection_by_type{name_prefix}.png"),
            pattern_type="alert",
            use_model=False,
        )
        # Bar chart comparison at final level
        plot_pattern_type_comparison(
            data_list,
            max_epsilons,
            str(save_dir / f"alert_type_comparison_final{name_prefix}.png"),
            pattern_type="alert",
            use_model=False,
            level_idx=-1,
        )
        # Bar chart comparison at initial level
        plot_pattern_type_comparison(
            data_list,
            max_epsilons,
            str(save_dir / f"alert_type_comparison_initial{name_prefix}.png"),
            pattern_type="alert",
            use_model=False,
            level_idx=0,
        )
        # Heatmap of detection by type and level
        plot_pattern_type_heatmap(
            data_list,
            max_epsilons,
            str(save_dir / f"alert_type_heatmap{name_prefix}.png"),
            pattern_type="alert",
            use_model=False,
        )

    if has_normal_patterns and any("normal_gt_type_rates" in d for d in data_list):
        # Detection rates by type over coarsening levels
        plot_pattern_detection_by_type(
            data_list,
            max_epsilons,
            str(save_dir / f"normal_detection_by_type{name_prefix}.png"),
            pattern_type="normal",
            use_model=False,
        )
        # Bar chart comparison at final level
        plot_pattern_type_comparison(
            data_list,
            max_epsilons,
            str(save_dir / f"normal_type_comparison_final{name_prefix}.png"),
            pattern_type="normal",
            use_model=False,
            level_idx=-1,
        )
        # Bar chart comparison at initial level
        plot_pattern_type_comparison(
            data_list,
            max_epsilons,
            str(save_dir / f"normal_type_comparison_initial{name_prefix}.png"),
            pattern_type="normal",
            use_model=False,
            level_idx=0,
        )
        # Heatmap of detection by type and level
        plot_pattern_type_heatmap(
            data_list,
            max_epsilons,
            str(save_dir / f"normal_type_heatmap{name_prefix}.png"),
            pattern_type="normal",
            use_model=False,
        )
