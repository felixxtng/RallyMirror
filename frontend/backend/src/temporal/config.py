"""Config loading for RallyMirror.

Thin wrapper around ``configs/default.yaml`` so modules and tests share one
loader. Keeps the YAML as the single place to tune paths/labels/projection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load a YAML config into a dict.

    Parameters
    ----------
    path:
        Path to a YAML config. Defaults to ``configs/default.yaml``.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def active_projection(config: Dict[str, Any]) -> Dict[str, str]:
    """Return the active label->phase projection from a loaded config."""
    projection_cfg = config["projection"]
    active_name = projection_cfg["active"]
    return projection_cfg[active_name]
