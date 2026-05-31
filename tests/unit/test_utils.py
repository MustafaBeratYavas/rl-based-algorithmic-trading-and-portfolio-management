"""Validate shared path, logging, reproducibility, and configuration utilities."""

from __future__ import annotations

import logging
import random

import pytest

from src.utils.logger import _coerce_log_level, configure_logging_from_config
from src.utils.paths import PROJECT_ROOT, ensure_directory, resolve_project_path
from src.utils.random import make_rng, set_global_seed


# Path tests verify that runtime paths resolve from a stable project root.
def test_resolve_project_path_preserves_absolute_paths(tmp_path) -> None:
    assert resolve_project_path(tmp_path) == tmp_path


def test_resolve_project_path_anchors_relative_paths_to_project_root() -> None:
    assert (
        resolve_project_path("configs/data_config.yaml")
        == PROJECT_ROOT / "configs/data_config.yaml"
    )


def test_project_root_contains_pyproject_toml() -> None:
    # Guard against incorrect root resolution after directory restructuring.
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_ensure_directory_creates_target_directory(tmp_path) -> None:
    target = tmp_path / "nested" / "dir"

    result = ensure_directory(target)

    assert result == target
    assert target.exists()


def test_ensure_directory_is_idempotent_on_existing_directory(tmp_path) -> None:
    target = tmp_path / "existing"
    target.mkdir()

    result = ensure_directory(target)

    assert result == target


# Reproducibility tests keep seeded randomness explicit and repeatable.
def test_set_global_seed_makes_python_and_numpy_reproducible() -> None:
    # The helper should align Python randomness and return a repeatable NumPy Generator.
    first_rng = set_global_seed(123)
    assert first_rng is not None
    first_python = random.random()
    first_numpy = float(first_rng.random())

    second_rng = set_global_seed(123)
    assert second_rng is not None

    assert random.random() == first_python
    assert float(second_rng.random()) == first_numpy


def test_set_global_seed_returns_none_when_seed_is_none() -> None:
    result = set_global_seed(None)

    assert result is None


def test_make_rng_creates_isolated_repeatable_generators() -> None:
    first_rng = make_rng(456)
    second_rng = make_rng(456)

    assert float(first_rng.random()) == float(second_rng.random())


def test_make_rng_accepts_none_seed() -> None:
    rng = make_rng(None)

    # Must return a valid generator even without a fixed seed.
    assert 0.0 <= float(rng.random()) <= 1.0


# Log-level tests preserve the public logging configuration contract.
def test_coerce_log_level_accepts_names_and_rejects_unknown_values() -> None:
    assert _coerce_log_level("INFO") == 20

    with pytest.raises(ValueError, match="Unsupported log level"):
        _coerce_log_level("LOUD")


def test_coerce_log_level_accepts_integer_passthrough() -> None:
    assert _coerce_log_level(30) == 30


def test_coerce_log_level_is_case_insensitive() -> None:
    assert _coerce_log_level("debug") == logging.DEBUG
    assert _coerce_log_level("WARNING") == logging.WARNING


# Logging setup tests ensure minimal configs remain valid for CLI entry points.
def test_configure_logging_from_config_handles_empty_config(tmp_path) -> None:
    # Must not raise even if the config dict contains no logging section.
    configure_logging_from_config({}, default_log_file=str(tmp_path / "test.log"))


def test_configure_logging_from_config_handles_non_dict_config(tmp_path) -> None:
    configure_logging_from_config("not a dict", default_log_file=str(tmp_path / "test.log"))
