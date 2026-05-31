"""Define interchangeable reward strategies for portfolio allocation episodes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.evaluation.metrics import calculate_sharpe_ratio


class BaseRewardStrategy(ABC):
    """Reward interface used by ``PortfolioEnv`` without coupling it to SB3."""

    @abstractmethod
    def calculate_reward(self, returns_history: list, current_return: float) -> float:
        """Return the reward for the current step given the realized return path."""
        raise NotImplementedError


class DailyReturnReward(BaseRewardStrategy):
    """Use the immediate portfolio return as the reward signal."""

    def calculate_reward(self, returns_history: list, current_return: float) -> float:
        """Return the unshaped current return."""
        return current_return


class SharpeRatioReward(BaseRewardStrategy):
    """Shape rewards with realized risk-adjusted return once volatility exists."""

    def __init__(self, risk_free_rate: float = 0.0, annualization_factor: int = 252):
        """Store risk-free assumptions used by the Sharpe reward calculation."""
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

    def calculate_reward(self, returns_history: list, current_return: float) -> float:
        """Return immediate reward early, then transition to path-aware Sharpe reward."""
        # Fall back to the immediate return until the sample can define volatility.
        if len(returns_history) < 2:
            return current_return

        returns_array = np.array(returns_history, dtype=float)
        return calculate_sharpe_ratio(
            returns_array,
            risk_free_rate=self.risk_free_rate,
            annualization_factor=self.annualization_factor,
            annualize=False,
        )
