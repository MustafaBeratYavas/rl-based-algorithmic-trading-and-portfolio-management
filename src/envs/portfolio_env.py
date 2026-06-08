"""Simulate long-only portfolio allocation for Stable-Baselines3 agents.

PortfolioEnv owns the accounting loop: target allocation normalization,
transaction frictions, cash handling, market drift, reward calculation, and
episode termination. The environment consumes prevalidated tensors so step-time
logic stays focused on portfolio state transitions.
"""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.data_provider import EnvironmentDataProvider, EnvironmentDataset
from src.envs.reward_schemes import BaseRewardStrategy, DailyReturnReward, SharpeRatioReward
from src.utils.logger import get_logger


class PortfolioEnv(gym.Env):
    """Gymnasium environment for long-only portfolio rebalancing.

    The environment treats actions as target allocations over risky assets plus
    cash, applies transaction frictions before market growth, and exposes only
    trailing engineered features to the agent. Raw prices remain internal
    accounting inputs for returns, costs, and benchmark-compatible state.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: dict[str, Any],
        dataset: EnvironmentDataset | None = None,
        data_provider: EnvironmentDataProvider | None = None,
    ):
        """Initialize simulation economics, spaces, reward strategy, and aligned tensors."""
        super().__init__()
        self.logger = get_logger(__name__)
        self.config = config

        # Read simulation economics from config so experiments remain reproducible.
        self.initial_balance = config.get("initial_balance", 100000.0)
        self.lookback_window = config.get("lookback_window", 30)
        self.transaction_fee_pct = config.get("transaction_fee_pct", 0.001)
        self.buy_fee_pct = config.get("buy_fee_pct", self.transaction_fee_pct)
        self.sell_fee_pct = config.get("sell_fee_pct", self.transaction_fee_pct)
        self.slippage_pct = config.get("slippage_pct", 0.0)
        self.market_impact_pct = config.get("market_impact_pct", 0.0)
        self.termination_threshold = config.get("termination_threshold", 0.1)
        self.risk_free_rate = config.get("risk_free_rate", 0.0)
        self.daily_risk_free_rate = self.risk_free_rate / 252
        self.action_epsilon = config.get("action_epsilon", 1e-8)
        self.render_mode = config.get("render_mode")
        self.feature_cols = config.get("features", EnvironmentDataProvider.DEFAULT_FEATURES)
        self.price_column = config.get("price_column", "Close")

        self.start_date = config.get("start_date", "2000-01-01")
        self.end_date = config.get("end_date", "2099-12-31")

        self._load_dataset(dataset, data_provider)

        self.n_assets = len(self.tickers)
        self.n_features = len(self.feature_cols)

        # Actions represent target allocations across risky assets plus a cash component.
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets + 1,),
            dtype=np.float32,
        )

        # Observations expose trailing market features and the currently held allocation.
        self.observation_space = spaces.Dict(
            {
                "market_history": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.lookback_window, self.n_assets, self.n_features),
                    dtype=np.float32,
                ),
                "portfolio_weights": spaces.Box(
                    low=0.0, high=1.0, shape=(self.n_assets + 1,), dtype=np.float32
                ),
            }
        )

        # Reward strategies are selected by name to keep environment dynamics pluggable.
        reward_strategy_name = config.get("reward_strategy", "sharpe").lower()
        if reward_strategy_name == "sharpe":
            self.reward_strategy: BaseRewardStrategy = SharpeRatioReward(self.risk_free_rate)
        else:
            self.reward_strategy = DailyReturnReward()

        self._reset_state()

    def _load_dataset(
        self,
        dataset: EnvironmentDataset | None,
        data_provider: EnvironmentDataProvider | None,
    ) -> None:
        """Load tensors from injected data or from the configured provider boundary."""
        loaded_dataset = dataset or (data_provider or EnvironmentDataProvider(self.config)).load()
        self.data_matrix = loaded_dataset.data_matrix
        self.close_prices = loaded_dataset.close_prices
        self.tickers = loaded_dataset.tickers
        self.dates = loaded_dataset.dates

    def _reset_state(self) -> None:
        """Restore the episode ledger to the canonical all-cash initial state."""
        # Start each episode fully in cash so the first action performs the initial allocation.
        self.current_step = self.lookback_window
        self.portfolio_value = self.initial_balance

        self.weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.weights[-1] = 1.0

        self.returns_history: list[float] = []
        self.done = False

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the episode ledger and return a defensive initial observation."""
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), self._get_info()

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Return the trailing market window and current allocation weights."""
        if self.current_step < self.lookback_window or self.current_step > len(self.data_matrix):
            raise RuntimeError(
                "PortfolioEnv observation step is outside the available data matrix: "
                f"current_step={self.current_step}, data_rows={len(self.data_matrix)}"
            )
        # Return defensive copies so wrappers cannot mutate internal environment state.
        market_history = self.data_matrix[
            self.current_step - self.lookback_window : self.current_step
        ]
        return {
            "market_history": market_history.astype(np.float32, copy=True),
            "portfolio_weights": self.weights.astype(np.float32, copy=True),
        }

    def _get_info(self, date_step: int | None = None) -> dict[str, Any]:
        """Build diagnostic metadata while bounding date lookup to available rows."""
        info_step = self.current_step if date_step is None else date_step
        bounded_step = min(max(info_step, 0), len(self.dates) - 1)
        return {
            "portfolio_value": self.portfolio_value,
            "current_step": self.current_step,
            "returns_history_len": len(self.returns_history),
            "date": str(self.dates[bounded_step].date()),
            "weights": self.weights.copy(),
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply one allocation, account for costs and drift, and emit Gym step output."""
        if self.done:
            raise RuntimeError("Cannot call step() after the episode has terminated or truncated.")
        if self.current_step >= len(self.close_prices):
            raise RuntimeError(
                "PortfolioEnv current_step is outside the available price matrix: "
                f"current_step={self.current_step}, price_rows={len(self.close_prices)}"
            )

        # Normalize model output before computing turnover and portfolio growth.
        new_weights = self._normalize_action(action)
        transaction_costs = self._calculate_transaction_costs(new_weights)

        current_prices = self.close_prices[self.current_step]
        previous_prices = self.close_prices[self.current_step - 1]

        price_returns = np.divide(
            current_prices - previous_prices,
            previous_prices,
            out=np.zeros_like(current_prices, dtype=np.float32),
            where=previous_prices > 0,
        )
        # Combine risky asset returns with the configured cash return in one growth vector.
        component_returns = np.concatenate(
            [price_returns.astype(np.float64), np.array([self.daily_risk_free_rate])]
        )
        component_growth = 1.0 + component_returns
        portfolio_growth = float(np.dot(new_weights.astype(np.float64), component_growth))

        previous_value = self.portfolio_value
        net_value = max(previous_value - transaction_costs, 0.0)
        self.portfolio_value = max(net_value * portfolio_growth, 0.0)

        if previous_value > 0:
            step_return = (self.portfolio_value - previous_value) / previous_value
        else:
            step_return = 0.0
        self.returns_history.append(step_return)

        reward = self.reward_strategy.calculate_reward(self.returns_history, step_return)
        self.weights = self._drift_weights(new_weights, component_growth, portfolio_growth)

        valued_step = self.current_step
        self.current_step += 1

        termination_value = self.initial_balance * self.termination_threshold
        terminated = bool(self.portfolio_value <= termination_value)
        truncated = bool(self.current_step >= len(self.data_matrix))
        self.done = terminated or truncated

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info(date_step=valued_step),
        )

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert arbitrary model output into a non-negative allocation vector that sums to one."""
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action_array.shape}"
            )

        action_array = np.nan_to_num(action_array, nan=0.0, posinf=1.0, neginf=0.0)
        action_array = np.clip(action_array, 0.0, 1.0)
        total = float(action_array.sum(dtype=np.float64))
        if total < self.action_epsilon:
            weights = np.zeros_like(action_array, dtype=np.float32)
            weights[-1] = 1.0
            return weights
        return (action_array / total).astype(np.float32)

    def _calculate_transaction_costs(self, target_weights: np.ndarray) -> float:
        """Calculate fees, slippage, and market impact from risky-asset turnover."""
        # Charge costs on risky-asset turnover; cash absorbs residual allocation without fees.
        weight_delta = target_weights[:-1] - self.weights[:-1]
        buy_turnover = np.maximum(weight_delta, 0.0).sum()
        sell_turnover = np.maximum(-weight_delta, 0.0).sum()
        gross_turnover = buy_turnover + sell_turnover
        fee_rate = (buy_turnover * self.buy_fee_pct) + (sell_turnover * self.sell_fee_pct)
        slippage_rate = gross_turnover * self.slippage_pct
        impact_rate = (gross_turnover**2) * self.market_impact_pct
        return float((fee_rate + slippage_rate + impact_rate) * self.portfolio_value)

    def _drift_weights(
        self,
        target_weights: np.ndarray,
        component_growth: np.ndarray,
        portfolio_growth: float,
    ) -> np.ndarray:
        """Revalue post-trade weights after asset and cash growth, falling back to cash."""
        # Revalue target weights after market movement so the next rebalance pays real turnover.
        if portfolio_growth <= self.action_epsilon or not np.isfinite(portfolio_growth):
            weights = np.zeros(self.n_assets + 1, dtype=np.float32)
            weights[-1] = 1.0
            return weights

        drifted_weights = target_weights.astype(np.float64) * component_growth / portfolio_growth
        drifted_weights = np.nan_to_num(drifted_weights, nan=0.0, posinf=0.0, neginf=0.0)
        drifted_weights = np.clip(drifted_weights, 0.0, 1.0)
        total = float(drifted_weights.sum())
        if total <= self.action_epsilon:
            weights = np.zeros(self.n_assets + 1, dtype=np.float32)
            weights[-1] = 1.0
            return weights
        return cast(np.ndarray, (drifted_weights / total).astype(np.float32))

    def render(self) -> None:
        """Log the current account state without requiring a graphical backend."""
        # Keep rendering logger-based so training jobs remain safe in headless environments.
        info = self._get_info()
        self.logger.info(
            "step=%s date=%s portfolio_value=%.2f weights=%s",
            info["current_step"],
            info["date"],
            info["portfolio_value"],
            np.round(info["weights"], 4).tolist(),
        )

    def close(self) -> None:
        """Satisfy the Gymnasium cleanup hook; no external resources are held."""
        pass
