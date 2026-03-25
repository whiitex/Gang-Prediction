#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import pandas as pd

from src.GangPrediction.utils.utils import *


REQUIRED_COLS = ["modelID", "accountID", "type"]


def validate_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    return df[REQUIRED_COLS].dropna().copy()


def dataset_stats(df: pd.DataFrame) -> dict:
    node_pattern_counts = df.groupby("accountID")["modelID"].nunique()
    node_type_counts = df.groupby("accountID")["type"].nunique()
    pattern_sizes = df.groupby("modelID")["accountID"].nunique()

    by_type = {}
    for ptype, g in df.groupby("type"):
        npc = g.groupby("accountID")["modelID"].nunique()
        unique_nodes = int(g["accountID"].nunique())
        multi_nodes = int((npc > 1).sum())
        by_type[str(ptype)] = {
            "patterns": int(g["modelID"].nunique()),
            "unique_nodes": unique_nodes,
            "rows": int(len(g)),
            "nodes_in_multiple_patterns": multi_nodes,
            "pct_nodes_in_multiple_patterns": (
                (100.0 * multi_nodes / unique_nodes) if unique_nodes else 0.0
            ),
            "max_patterns_per_node": int(npc.max()) if len(npc) else 0,
            "avg_patterns_per_node": float(npc.mean()) if len(npc) else 0.0,
            "pattern_size_mean": float(
                g.groupby("modelID")["accountID"].nunique().mean()
            ),
            "pattern_size_median": float(
                g.groupby("modelID")["accountID"].nunique().median()
            ),
        }

    unique_nodes = int(df["accountID"].nunique())
    multi_nodes = int((node_pattern_counts > 1).sum())
    multi_type_nodes = int((node_type_counts > 1).sum())

    return {
        "rows": int(len(df)),
        "patterns": int(df["modelID"].nunique()),
        "unique_nodes": unique_nodes,
        "nodes_in_multiple_patterns": multi_nodes,
        "pct_nodes_in_multiple_patterns": (
            (100.0 * multi_nodes / unique_nodes) if unique_nodes else 0.0
        ),
        "max_patterns_per_node": (
            int(node_pattern_counts.max()) if len(node_pattern_counts) else 0
        ),
        "avg_patterns_per_node": (
            float(node_pattern_counts.mean()) if len(node_pattern_counts) else 0.0
        ),
        "nodes_in_multiple_types": multi_type_nodes,
        "pct_nodes_in_multiple_types": (
            (100.0 * multi_type_nodes / unique_nodes) if unique_nodes else 0.0
        ),
        "max_types_per_node": (
            int(node_type_counts.max()) if len(node_type_counts) else 0
        ),
        "pattern_size_mean": float(pattern_sizes.mean()) if len(pattern_sizes) else 0.0,
        "pattern_size_median": (
            float(pattern_sizes.median()) if len(pattern_sizes) else 0.0
        ),
        "by_type": by_type,
    }


def cross_overlap(
    alert_df: pd.DataFrame, normal_df: pd.DataFrame, top_k: int = 10
) -> dict:
    alert_nodes = set(alert_df["accountID"].unique())
    normal_nodes = set(normal_df["accountID"].unique())
    inter = alert_nodes & normal_nodes
    union = alert_nodes | normal_nodes

    # Type-pair overlaps by shared unique accountID
    cross_type = (
        alert_df[["accountID", "type"]]
        .drop_duplicates()
        .merge(
            normal_df[["accountID", "type"]].drop_duplicates(),
            on="accountID",
            suffixes=("_alert", "_normal"),
        )
    )
    combo = (
        cross_type.groupby(["type_alert", "type_normal"])["accountID"]
        .nunique()
        .sort_values(ascending=False)
    )

    return {
        "alert_unique_nodes": int(len(alert_nodes)),
        "normal_unique_nodes": int(len(normal_nodes)),
        "overlap_unique_nodes": int(len(inter)),
        "pct_of_alert_nodes": (
            (100.0 * len(inter) / len(alert_nodes)) if alert_nodes else 0.0
        ),
        "pct_of_normal_nodes": (
            (100.0 * len(inter) / len(normal_nodes)) if normal_nodes else 0.0
        ),
        "jaccard": (len(inter) / len(union)) if union else 0.0,
        "top_type_pair_overlaps": [
            {"alert_type": str(a), "normal_type": str(n), "shared_nodes": int(c)}
            for (a, n), c in combo.head(top_k).items()
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pattern overlap analysis for AMLGentex pattern files"
    )
    parser.add_argument(
        "--experiment",
        default="tutorial_demo12",
        help="Experiment name, e.g. tutorial_demo12",
    )
    parser.add_argument("--workspace", default=".", help="Workspace root path")
    parser.add_argument(
        "--top-k", type=int, default=10, help="Top K type-pair overlaps to report"
    )
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    spatial_dir = workspace / "experiments" / args.experiment / "spatial"

    alert_fp = spatial_dir / "alert_models.csv"
    normal_fp = spatial_dir / "normal_models.csv"

    if not alert_fp.exists() or not normal_fp.exists():
        raise FileNotFoundError(
            f"Missing pattern files.\nalert: {alert_fp.exists()} ({alert_fp})\nnormal: {normal_fp.exists()} ({normal_fp})"
        )

    alert_df = validate_df(pd.read_csv(alert_fp), "alert_models.csv")
    normal_df = validate_df(pd.read_csv(normal_fp), "normal_models.csv")

    result = {
        "experiment": args.experiment,
        "files": {"alert": str(alert_fp), "normal": str(normal_fp)},
        "alert": dataset_stats(alert_df),
        "normal": dataset_stats(normal_df),
        "cross_alert_normal": cross_overlap(alert_df, normal_df, top_k=args.top_k),
    }

    output_path = (
        Path(args.output)
        if args.output
        else workspace / "results" / f"pattern_overlap_report_{args.experiment}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    # Short console summary
    print("=== Pattern Overlap Summary ===")
    print(f"Experiment: {args.experiment}")
    print(
        f"Alert:  patterns={result['alert']['patterns']}, unique_nodes={result['alert']['unique_nodes']}, "
        f"multi_pattern_nodes={result['alert']['nodes_in_multiple_patterns']} "
        f"({result['alert']['pct_nodes_in_multiple_patterns']:.2f}%)"
    )
    print(
        f"Normal: patterns={result['normal']['patterns']}, unique_nodes={result['normal']['unique_nodes']}, "
        f"multi_pattern_nodes={result['normal']['nodes_in_multiple_patterns']} "
        f"({result['normal']['pct_nodes_in_multiple_patterns']:.2f}%)"
    )
    print(
        f"Cross overlap nodes={result['cross_alert_normal']['overlap_unique_nodes']}, "
        f"Jaccard={result['cross_alert_normal']['jaccard']:.4f}"
    )
    print(f"Saved full report to: {output_path}")


if __name__ == "__main__":
    main()
