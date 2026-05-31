"""Resolve project-relative filesystem paths from a single repository root."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths against the repository root while preserving absolutes."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def ensure_directory(path: str | Path) -> Path:
    """Create and return a project-resolved directory path."""
    directory = resolve_project_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
