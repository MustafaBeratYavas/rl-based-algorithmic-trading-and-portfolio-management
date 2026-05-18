"""Define interchangeable reward strategies for portfolio allocation episodes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.evaluation.metrics import calculate_sharpe_ratio


class BaseRewardStrategy(ABC):
    # Establish the reward interface used by PortfolioEnv without coupling it to SB3.
    @abstractmethod
    def calculate_reward(self, returns_history: list, current_return: float) -> float:
        raise NotImplementedError


class DailyReturnReward(BaseRewardStrategy):
    # Expose raw daily return for experiments that should avoid path-dependent shaping.
    def calculate_reward(self, returns_history: list, current_return: float) -> float:
        return current_return


class SharpeRatioReward(BaseRewardStrategy):
    def __init__(self, risk_free_rate: float = 0.0, annualization_factor: int = 252):
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

    def calculate_reward(self, returns_history: list, current_return: float) -> float:
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
