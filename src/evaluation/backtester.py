"""Evaluate trained policies against deterministic backtest baselines.

Backtester coordinates model prediction, environment stepping, realized return
reconstruction, financial metric calculation, and passive benchmark generation.
It keeps reporting model-agnostic by deriving returns from portfolio values
rather than from the training reward signal.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from gymnasium import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from src.evaluation.metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from src.utils.logger import get_logger


@runtime_checkable
class PortfolioEnvLike(Protocol):
    """Minimal unwrapped environment surface required for benchmark generation."""

    initial_balance: float
    tickers: list[str]
    close_prices: np.ndarray
    lookback_window: int


class Backtester:
    """Evaluate a trained policy and compare it against transparent baselines.

    The backtester reads only the Gym contract and a small PortfolioEnv-compatible
    surface. Reports recompute realized returns from portfolio values so metrics
    remain independent of the reward shaping used during training.
    """

    def __init__(self, env: Env, model: BaseAlgorithm):
        """Attach the policy/environment pair and initialize reusable result buffers."""
        self.env = env
        self.model = model
        self.logger = get_logger(__name__)
        self.portfolio_values: list[float] = []
        self.returns: list[float] = []

    def run_backtest(self) -> dict:
        """Run one deterministic evaluation episode and return a metric report."""
        self.logger.info("Starting backtest simulation...")
        # Reset accumulated paths so a Backtester instance can be reused safely.
        self.portfolio_values = []
        self.returns = []
        obs, _ = self.env.reset()
        done = False
        truncated = False

        # Read initial capital from the environment contract rather than duplicating config.
        unwrapped_env = self._portfolio_env()
        initial_value = float(unwrapped_env.initial_balance)
        self.portfolio_values.append(initial_value)

        while not (done or truncated):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = self.env.step(action)

            # Recompute realized returns from portfolio value to keep reports model-agnostic.
            current_value = info.get("portfolio_value", 0)
            self.portfolio_values.append(current_value)

            if len(self.portfolio_values) > 1:
                previous_value = self.portfolio_values[-2]
                step_return = (
                    (self.portfolio_values[-1] - previous_value) / previous_value
                    if previous_value
                    else 0.0
                )
                self.returns.append(step_return)

        self.logger.info("Backtest simulation completed.")
        return self._generate_report()

    def _generate_report(self) -> dict:
        """Aggregate realized account paths into financial metrics and passive baselines."""
        # Convert collected paths once so metric functions operate on stable arrays.
        values_array = np.array(self.portfolio_values)
        returns_array = np.array(self.returns)

        initial_value = values_array[0]
        final_value = values_array[-1]
        total_return = (final_value - initial_value) / initial_value

        max_dd = calculate_max_drawdown(values_array)
        sharpe = calculate_sharpe_ratio(returns_array)
        sortino = calculate_sortino_ratio(returns_array)
        benchmarks = self._generate_benchmarks()
        equal_weight = benchmarks["equal_weight"]

        report = {
            "Initial Balance": initial_value,
            "Final Balance": final_value,
            "Total Return (%)": total_return * 100,
            "Max Drawdown (%)": max_dd * 100,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Equal Weight Final Balance": equal_weight["final_balance"],
            "Equal Weight Total Return (%)": equal_weight["total_return_pct"],
            "Equal Weight Initial Cost": equal_weight["initial_cost"],
            "Alpha vs Equal Weight (%)": (total_return * 100) - equal_weight["total_return_pct"],
        }
        if "spy_buy_and_hold" in benchmarks:
            spy_benchmark = benchmarks["spy_buy_and_hold"]
            report["SPY Buy Hold Final Balance"] = spy_benchmark["final_balance"]
            report["SPY Buy Hold Total Return (%)"] = spy_benchmark["total_return_pct"]
        if "stock_bond_60_40" in benchmarks:
            stock_bond_benchmark = benchmarks["stock_bond_60_40"]
            report["60/40 Final Balance"] = stock_bond_benchmark["final_balance"]
            report["60/40 Total Return (%)"] = stock_bond_benchmark["total_return_pct"]

        self.logger.info("--- BACKTEST REPORT ---")
        for key, value in report.items():
            formatted_value = f"{value:.4f}" if isinstance(value, float) else value
            self.logger.info("%s: %s", key, formatted_value)

        return report

    def _generate_benchmarks(self) -> dict[str, dict[str, float]]:
        """Build available passive benchmarks from the environment ticker universe."""
        unwrapped_env = self._portfolio_env()
        tickers = list(unwrapped_env.tickers)
        benchmarks = {
            "equal_weight": self._generate_equal_weight_benchmark(),
        }
        if "SPY" in tickers:
            benchmarks["spy_buy_and_hold"] = self._generate_static_weight_benchmark({"SPY": 1.0})
        if "SPY" in tickers and "TLT" in tickers:
            benchmarks["stock_bond_60_40"] = self._generate_static_weight_benchmark(
                {"SPY": 0.60, "TLT": 0.40}
            )
        return benchmarks

    def _generate_equal_weight_benchmark(self) -> dict[str, float]:
        """Return a passive equal-weight benchmark using all configured assets."""
        unwrapped_env = self._portfolio_env()
        tickers = list(unwrapped_env.tickers)
        if not tickers:
            return {"final_balance": 0.0, "total_return_pct": 0.0, "initial_cost": 0.0}

        asset_weight = 1.0 / len(tickers)
        return self._generate_static_weight_benchmark(dict.fromkeys(tickers, asset_weight))

    def _generate_static_weight_benchmark(
        self, target_weights_by_ticker: dict[str, float]
    ) -> dict[str, float]:
        """Simulate one initial rebalance followed by buy-and-hold growth."""
        # Model static benchmarks as one initial rebalance followed by buy-and-hold growth.
        unwrapped_env = self._portfolio_env()
        close_prices = unwrapped_env.close_prices
        if len(close_prices) == 0:
            return {"final_balance": 0.0, "total_return_pct": 0.0, "initial_cost": 0.0}

        start_step = unwrapped_env.lookback_window
        initial_balance = float(unwrapped_env.initial_balance)
        prices = close_prices[start_step:]
        if len(prices) < 2:
            return {"final_balance": initial_balance, "total_return_pct": 0.0, "initial_cost": 0.0}

        tickers = list(unwrapped_env.tickers)
        weight_vector = np.array(
            [target_weights_by_ticker.get(ticker, 0.0) for ticker in tickers],
            dtype=np.float64,
        )
        total_risky_weight = float(weight_vector.sum())
        if total_risky_weight > 1.0 + 1e-9:
            raise ValueError("Benchmark risky weights cannot exceed 100%.")

        initial_cost = self._calculate_initial_allocation_cost(initial_balance, weight_vector)
        investable_balance = max(initial_balance - initial_cost, 0.0)
        price_growth = np.divide(
            prices[-1], prices[0], out=np.ones_like(prices[-1]), where=prices[0] > 0
        )

        cash_weight = max(1.0 - total_risky_weight, 0.0)
        periods = max(len(prices) - 1, 0)
        daily_risk_free = float(getattr(unwrapped_env, "daily_risk_free_rate", 0.0))
        cash_growth = (1.0 + daily_risk_free) ** periods

        portfolio_growth = float(np.dot(weight_vector, price_growth) + cash_weight * cash_growth)
        final_balance = float(investable_balance * portfolio_growth)
        total_return = (
            (final_balance - initial_balance) / initial_balance if initial_balance else 0.0
        )
        return {
            "final_balance": final_balance,
            "total_return_pct": total_return * 100,
            "initial_cost": float(initial_cost),
        }

    def _calculate_initial_allocation_cost(
        self, initial_balance: float, risky_weights: np.ndarray
    ) -> float:
        """Apply benchmark entry costs using the environment's current friction settings."""
        unwrapped_env = self._portfolio_env()
        buy_turnover = float(np.maximum(risky_weights, 0.0).sum())
        fee_rate = buy_turnover * float(getattr(unwrapped_env, "buy_fee_pct", 0.0))
        slippage_rate = buy_turnover * float(getattr(unwrapped_env, "slippage_pct", 0.0))
        impact_rate = (buy_turnover**2) * float(getattr(unwrapped_env, "market_impact_pct", 0.0))
        return (fee_rate + slippage_rate + impact_rate) * initial_balance

    def _portfolio_env(self) -> PortfolioEnvLike:
        """Return the unwrapped environment after validating the required protocol."""
        unwrapped_env = self.env.unwrapped
        if not isinstance(unwrapped_env, PortfolioEnvLike):
            raise TypeError(
                "Backtester requires a PortfolioEnv-compatible Gymnasium environment; "
                f"got {type(unwrapped_env).__name__}."
            )
        return unwrapped_env
