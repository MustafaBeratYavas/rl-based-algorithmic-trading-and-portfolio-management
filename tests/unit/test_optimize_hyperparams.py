"""Validate Optuna optimization helpers, search spaces, and trial configuration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scripts.optimize_hyperparams import (
    build_trial_config,
    suggest_hyperparameters,
    validate_study_name,
)
from src.utils.config import ConfigError


# Study name validation
def test_validate_study_name_accepts_safe_name() -> None:
    assert validate_study_name("PPO_portfolio_2026") == "PPO_portfolio_2026"


def test_validate_study_name_accepts_single_character() -> None:
    assert validate_study_name("x") == "x"


def test_validate_study_name_accepts_max_length() -> None:
    name = "a" * 128
    assert validate_study_name(name) == name


@pytest.mark.parametrize(
    "study_name",
    ["", "bad-name", "bad.name", "bad name", "x" * 129, "name@special"],
    ids=["empty", "hyphen", "dot", "space", "too_long", "special_char"],
)
def test_validate_study_name_rejects_unsafe_values(study_name: str) -> None:
    # Study names are constrained because they are persisted into SQLite-backed storage.
    with pytest.raises(ConfigError, match="Optuna study names"):
        validate_study_name(study_name)


# Hyperparameter search space suggestions
def test_suggest_hyperparameters_returns_ppo_keys() -> None:
    trial = MagicMock()
    trial.suggest_float.return_value = 0.001
    trial.suggest_categorical.return_value = 64

    result = suggest_hyperparameters(trial, "PPO")

    expected_keys = {"learning_rate", "gamma", "batch_size", "ent_coef", "n_steps"}
    assert set(result.keys()) == expected_keys


def test_suggest_hyperparameters_returns_a2c_keys() -> None:
    trial = MagicMock()
    trial.suggest_float.return_value = 0.001
    trial.suggest_categorical.return_value = 64

    result = suggest_hyperparameters(trial, "A2C")

    expected_keys = {"learning_rate", "gamma", "ent_coef", "n_steps"}
    assert set(result.keys()) == expected_keys


def test_suggest_hyperparameters_returns_sac_keys() -> None:
    trial = MagicMock()
    trial.suggest_float.return_value = 0.001
    trial.suggest_categorical.return_value = 64

    result = suggest_hyperparameters(trial, "SAC")

    expected_keys = {
        "learning_rate",
        "batch_size",
        "gamma",
        "tau",
        "learning_starts",
        "buffer_size",
    }
    assert set(result.keys()) == expected_keys


def test_suggest_hyperparameters_is_case_insensitive() -> None:
    trial = MagicMock()
    trial.suggest_float.return_value = 0.001
    trial.suggest_categorical.return_value = 64

    result_lower = suggest_hyperparameters(trial, "ppo")
    result_upper = suggest_hyperparameters(trial, "PPO")

    assert set(result_lower.keys()) == set(result_upper.keys())


def test_suggest_hyperparameters_rejects_unsupported_algorithm() -> None:
    trial = MagicMock()

    with pytest.raises(ValueError, match="Unsupported algorithm"):
        suggest_hyperparameters(trial, "DQN")


# Trial configuration builder
def test_build_trial_config_merges_hyperparams_with_train_config() -> None:
    trial = MagicMock()
    trial.suggest_float.return_value = 0.0005
    trial.suggest_categorical.return_value = 128

    train_config = {
        "algorithm": "PPO",
        "total_timesteps": 100_000,
        "verbose": 1,
    }

    result = build_trial_config(trial, train_config)

    # Optuna proposals must override baseline values.
    assert result["verbose"] == 0
    assert result["total_timesteps"] == 100_000
    assert "learning_rate" in result


def test_build_trial_config_preserves_algorithm_from_train_config() -> None:
    trial = MagicMock()
    trial.suggest_float.return_value = 0.001
    trial.suggest_categorical.return_value = 64

    train_config = {"algorithm": "SAC", "verbose": 1}

    result = build_trial_config(trial, train_config)

    assert result["algorithm"] == "SAC"


def test_build_trial_config_defaults_to_ppo_when_algorithm_not_set() -> None:
    trial = MagicMock()
    trial.suggest_float.return_value = 0.001
    trial.suggest_categorical.return_value = 64

    result = build_trial_config(trial, {"verbose": 1})

    # Default algorithm is PPO, so PPO search space keys should be present.
    assert "learning_rate" in result
    assert "n_steps" in result
