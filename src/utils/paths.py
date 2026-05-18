"""Resolve project-relative filesystem paths from a single repository root."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    # Preserve absolute paths for test fixtures while anchoring relative paths to the repo.
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def ensure_directory(path: str | Path) -> Path:
    # Create directories through the same resolver used by runtime file outputs.
    directory = resolve_project_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
