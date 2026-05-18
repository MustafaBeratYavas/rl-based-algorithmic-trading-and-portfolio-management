"""Validate reward strategy behavior across short and volatile return histories."""

from __future__ import annotations

import numpy as np

from src.envs.reward_schemes import DailyReturnReward, SharpeRatioReward


def test_daily_return_reward_returns_current_return() -> None:
    reward = DailyReturnReward()

    assert reward.calculate_reward([0.01, 0.02], 0.03) == 0.03


def test_sharpe_reward_uses_daily_risk_free_rate() -> None:
    returns = [0.01, 0.02, -0.005, 0.015]
    reward = SharpeRatioReward(risk_free_rate=0.252, annualization_factor=252)

    expected = (np.mean(returns) - 0.001) / np.std(returns, ddof=1)

    assert np.isclose(reward.calculate_reward(returns, returns[-1]), expected)


def test_sharpe_reward_returns_current_return_until_history_is_available() -> None:
    reward = SharpeRatioReward()

    assert reward.calculate_reward([0.02], 0.02) == 0.02


def test_sharpe_reward_returns_zero_for_constant_returns() -> None:
    reward = SharpeRatioReward()

    assert reward.calculate_reward([0.01, 0.01, 0.01], 0.01) == 0.0


def test_sharpe_reward_handles_negative_return_series() -> None:
    returns = [-0.01, -0.02, -0.015, -0.03]
    reward = SharpeRatioReward()

    result = reward.calculate_reward(returns, returns[-1])

    assert result < 0.0
    assert np.isfinite(result)
