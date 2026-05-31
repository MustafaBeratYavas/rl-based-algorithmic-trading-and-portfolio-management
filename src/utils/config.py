"""Load project YAML configuration with path and schema safeguards.

Configuration helpers anchor relative paths to the repository, reject root
escapes, validate top-level mapping shape, and copy split-specific environment
settings without mutating caller-owned dictionaries.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import PROJECT_ROOT


class ConfigError(ValueError):
    """Raised when user-provided configuration violates the project contract."""

    pass


def load_yaml_config(
    path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Load a YAML mapping from inside the project root.

    Relative paths are anchored to ``project_root`` and absolute paths must stay
    inside that root. This keeps CLI configuration loading deterministic and
    prevents accidental reads from unrelated host paths.
    """

    # Resolve through the project root so CLI invocations cannot escape the workspace.
    config_path = _resolve_config_path(path, project_root)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    # Require top-level mappings so callers receive a predictable configuration contract.
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a YAML mapping in {config_path}")
    return data


def _resolve_config_path(path: str | Path, project_root: str | Path = PROJECT_ROOT) -> Path:
    """Resolve a config path while rejecting project-root escapes."""
    root = Path(project_root).expanduser().resolve()
    candidate = Path(path).expanduser()
    config_path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not config_path.is_relative_to(root):
        raise ConfigError(f"Configuration path escapes the project root: {path}")
    return config_path


def config_with_data_split(config: dict[str, Any], split: str) -> dict[str, Any]:
    """Return a copy of an environment config pinned to a named dataset split.

    The original mapping is never mutated, which lets training, evaluation, and
    optimization reuse the same base config with different split selections.
    """

    if not split:
        raise ConfigError("Data split name cannot be empty.")

    result = copy.deepcopy(config)
    data_paths = result.get("data_paths")
    if data_paths is None:
        result["data_split"] = split
        return result
    if not isinstance(data_paths, dict):
        raise ConfigError("env config data_paths must be a mapping of split names to paths.")
    if split not in data_paths:
        available = ", ".join(sorted(str(key) for key in data_paths)) or "none"
        raise ConfigError(f"Unknown data split {split!r}. Available splits: {available}")

    result["data_path"] = data_paths[split]
    result["data_split"] = split
    return result


def require_keys(config: dict[str, Any], required_keys: Iterable[str], context: str) -> None:
    """Raise ``ConfigError`` when required orchestration keys are absent."""
    # Validate command-critical keys at startup instead of failing deep in execution.
    missing = [key for key in required_keys if key not in config]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ConfigError(f"Missing required {context} config keys: {joined}")
