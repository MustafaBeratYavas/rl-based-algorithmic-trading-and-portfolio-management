"""Persist best-model checkpoints during Stable-Baselines3 training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.utils.logger import get_logger
from src.utils.paths import ensure_directory


class SaveBestModelCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = Path(save_path)
        self.best_mean_reward = -np.inf
        self.app_logger = get_logger(__name__)

    def _init_callback(self) -> None:
        if self.save_path is not None:
            self.save_path = ensure_directory(self.save_path)

    def _on_step(self) -> bool:
        # Use SB3 episode summaries when monitor data is available for reward tracking.
        ep_info_buffer = self.model.ep_info_buffer
        if self.n_calls % self.check_freq == 0 and ep_info_buffer:
            mean_reward = float(np.mean([ep_info["r"] for ep_info in ep_info_buffer]))

            # Save only on improvement so checkpoint churn stays bounded.
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_model_path = self.save_path / "best_model"
                self.model.save(best_model_path)

                if self.verbose > 0:
                    self.app_logger.info(
                        "New best mean reward: %.4f. Model saved to %s",
                        self.best_mean_reward,
                        best_model_path,
                    )
        return True
