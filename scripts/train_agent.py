"""Train a Stable-Baselines3 portfolio agent from project configuration."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from stable_baselines3.common.env_util import make_vec_env

from src.envs.portfolio_env import PortfolioEnv
from src.models.agent_factory import AgentFactory
from src.models.callbacks import SaveBestModelCallback
from src.utils.config import config_with_data_split, load_yaml_config, require_keys
from src.utils.logger import configure_logging_from_config, get_logger
from src.utils.paths import resolve_project_path
from src.utils.random import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a portfolio management RL agent.")
    parser.add_argument("--env-config", default="configs/env_config.yaml")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--data-split", default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config = config_with_data_split(load_yaml_config(args.env_config), args.data_split)
    train_config = load_yaml_config(args.train_config)
    configure_logging_from_config(train_config)
    logger = get_logger("train_agent")
    require_keys(train_config, ["algorithm", "total_timesteps", "model_save_path"], "training")

    seed = train_config.get("seed")
    set_global_seed(seed)

    logger.info("Initializing Portfolio Environment...")
    n_envs = int(train_config.get("n_envs", 1))
    env: Any
    # Use vectorized environments only when configured; single-env runs stay easier to debug.
    env = None
    try:
        if n_envs > 1:
            env = make_vec_env(
                lambda: PortfolioEnv(copy.deepcopy(env_config)), n_envs=n_envs, seed=seed
            )
        else:
            env = PortfolioEnv(env_config)

        logger.info("Creating %s agent...", train_config["algorithm"])
        model = AgentFactory.create_agent(train_config["algorithm"], env, train_config)

        # Track rolling reward checkpoints while preserving the final model as the canonical output.
        callback = SaveBestModelCallback(
            check_freq=int(train_config.get("checkpoint_frequency", 10_000)),
            save_path=train_config.get("checkpoint_dir", "models/checkpoints"),
        )

        logger.info(
            "Starting training for %s timesteps on data split %s...",
            train_config["total_timesteps"],
            env_config.get("data_split", args.data_split),
        )
        model.learn(total_timesteps=int(train_config["total_timesteps"]), callback=callback)

        # Keep config paths extension-neutral because SB3 appends the archive suffix on save.
        save_path = resolve_project_path(Path(train_config["model_save_path"]))
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model.save(str(save_path))
        logger.info("Training completed. Model saved to %s", save_path)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
