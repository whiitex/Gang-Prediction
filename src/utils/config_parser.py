"""Utilities for loading and validating the main experiment config."""

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")


REQUIRED_KEYS = (
    "experiment",
    "method",
    "max_levels",
    # "epochs_per_level",
    "threshold",
    "max_epsilon",
    "plot_gifs",
    "train_config",
    "alert_thresholds",
    "normal_thresholds",
    "pattern_split_config",
    # "epsilon_schedule_power",
)


def _validate_thresholds(config: Dict[str, Any], key: str) -> Tuple[float, float]:
    value = config.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"'{key}' must be a list with exactly two numeric values")
    return float(value[0]), float(value[1])


def load_main_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and minimally validate the top-level experiment configuration."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError("Top-level config must be a YAML mapping")

    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    if not isinstance(config["train_config"], dict):
        raise ValueError("'train_config' must be a mapping")

    if not isinstance(config["pattern_split_config"], dict):
        raise ValueError("'pattern_split_config' must be a mapping")

    config["alert_thresholds"] = _validate_thresholds(config, "alert_thresholds")
    config["normal_thresholds"] = _validate_thresholds(config, "normal_thresholds")

    return config
