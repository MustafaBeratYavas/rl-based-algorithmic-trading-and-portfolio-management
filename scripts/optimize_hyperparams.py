"""Coordinate Optuna hyperparameter search for portfolio agents.

The CLI keeps train and validation environments isolated, constrains each search
space to SB3-supported parameters, and persists the study database so interrupted
optimization runs can resume deterministically.
"""

from __future__ import annotations

import argparse
import re
from logging import Logger
from typing import Any

import optuna
from stable_baselines3.common.evaluation import evaluate_policy

from src.envs.portfolio_env import PortfolioEnv
from src.models.agent_factory import AgentFactory
from src.utils.config import ConfigError, config_with_data_split, load_yaml_config
from src.utils.logger import configure_logging_from_config, get_logger
from src.utils.paths import ensure_directory
from src.utils.random import set_global_seed


def suggest_hyperparameters(trial: optuna.Trial, algorithm: str) -> dict[str, Any]:
    """Return the Optuna search space supported by the selected SB3 algorithm."""
    algorithm = algorithm.upper()
    # Restrict each search space to parameters accepted by the selected SB3 algorithm.
    if algorithm == "PPO":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "gamma": trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
        }
    if algorithm == "A2C":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "gamma": trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995]),
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [5, 16, 64, 256]),
        }
    if algorithm == "SAC":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "gamma": trial.suggest_categorical("gamma", [0.95, 0.98, 0.99, 0.995]),
            "tau": trial.suggest_float("tau", 0.005, 0.05, log=True),
            "learning_starts": trial.suggest_categorical("learning_starts", [100, 500, 1000]),
            "buffer_size": trial.suggest_categorical("buffer_size", [50_000, 100_000, 250_000]),
        }
    raise ValueError(f"Unsupported algorithm for optimization: {algorithm}")


def parse_args() -> argparse.Namespace:
    """Parse optimization configs, split names, and study controls."""
    parser = argparse.ArgumentParser(description="Optimize portfolio RL hyperparameters.")
    parser.add_argument("--env-config", default="configs/env_config.yaml")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--train-data-split", default="train")
    parser.add_argument("--eval-data-split", default="validation")
    return parser.parse_args()


def build_trial_config(trial: optuna.Trial, train_config: dict[str, Any]) -> dict[str, Any]:
    """Merge sampled hyperparameters over the immutable baseline train config."""
    algorithm = train_config.get("algorithm", "PPO")
    hyperparams = suggest_hyperparameters(trial, algorithm)
    # Let Optuna proposals override baseline values without mutating the loaded config.
    return {
        **train_config,
        **hyperparams,
        "verbose": 0,
    }


def run_trial(
    trial: optuna.Trial,
    train_env_config: dict[str, Any],
    eval_env_config: dict[str, Any],
    train_config: dict[str, Any],
    logger: Logger,
) -> float:
    """Train and evaluate one Optuna trial with isolated train/eval environments."""
    config = build_trial_config(trial, train_config)
    train_env = None
    eval_env = None

    try:
        train_env = PortfolioEnv(train_env_config)
        eval_env = PortfolioEnv(eval_env_config)
        # Keep trials shorter than final training so the search remains practical.
        model = AgentFactory.create_agent(
            train_config.get("algorithm", "PPO"), train_env, config, verbose=0
        )
        model.learn(total_timesteps=int(train_config.get("optimization_timesteps", 20_000)))

        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=3, deterministic=True)
        if isinstance(mean_reward, list):
            return float(sum(mean_reward) / len(mean_reward)) if mean_reward else -10000.0
        return float(mean_reward)
    except (ValueError, RuntimeError, TypeError) as exc:
        logger.warning("Trial failed due to: %s", exc)
        return -10000.0
    finally:
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()


def validate_study_name(study_name: str) -> str:
    """Validate study names before they reach Optuna's SQLite storage layer."""
    if not re.fullmatch(r"[A-Za-z0-9_]{1,128}", study_name):
        raise ConfigError(
            "Optuna study names may contain only letters, numbers, and '_', "
            "and must be at most 128 characters long."
        )
    return study_name


def main() -> None:
    """Run or resume the persistent hyperparameter optimization study."""
    args = parse_args()
    env_config = load_yaml_config(args.env_config)
    train_env_config = config_with_data_split(env_config, args.train_data_split)
    eval_env_config = config_with_data_split(env_config, args.eval_data_split)
    train_config = load_yaml_config(args.train_config)
    configure_logging_from_config(train_config)
    logger = get_logger("hyperparam_optimization")
    set_global_seed(train_config.get("seed"))

    logger.info("Starting hyperparameter optimization study...")
    study_path = ensure_directory("logs/optuna_studies")
    # Persist the study database so interrupted optimization runs can resume safely.
    study_name = validate_study_name(
        args.study_name or f"{train_config.get('algorithm', 'PPO').lower()}_portfolio_optimization"
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=f"sqlite:///{study_path / 'study.db'}",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: run_trial(trial, train_env_config, eval_env_config, train_config, logger),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    logger.info("Optimization completed.")
    logger.info("Best Trial:")
    logger.info("  Value (Mean Reward): %s", study.best_trial.value)
    logger.info("  Hyperparameters:")
    for key, value in study.best_trial.params.items():
        logger.info("    %s: %s", key, value)


if __name__ == "__main__":
    main()
