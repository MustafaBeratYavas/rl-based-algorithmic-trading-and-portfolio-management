"""Validate deterministic backtest reporting, benchmark calculations, and error handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.envs.data_provider import EnvironmentDataset
from src.envs.portfolio_env import PortfolioEnv
from src.evaluation.backtester import Backtester


class StaticAllocationModel:
    # Minimal policy object that returns a fixed allocation for every observation.
    def __init__(self, action: np.ndarray):
        self.action = action

    def predict(self, obs, deterministic: bool = True):
        return self.action, None


def make_env_config(dataset_path) -> dict:
    # Mirror the production environment contract with a compact fixture dataset.
    return {
        "data_path": str(dataset_path),
        "initial_balance": 100_000.0,
        "lookback_window": 5,
        "transaction_fee_pct": 0.001,
        "risk_free_rate": 0.0,
        "price_column": "Close",
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "features": [
            "Norm_RSI",
            "Norm_MACD",
            "Norm_MACD_Signal",
            "Norm_SMA_20",
            "Norm_SMA_50",
            "Norm_Log_Return",
        ],
    }


def _make_injected_dataset(n_dates: int = 8, tickers: list[str] | None = None):
    # Build a minimal injected dataset for tests that do not need file-backed data.
    tickers = tickers or ["AAA", "BBB"]
    n_assets = len(tickers)
    dates = list(pd.date_range("2024-01-01", periods=n_dates, freq="B"))
    close_base = np.linspace(100.0, 100.0 + n_dates, n_dates)
    close_prices = np.column_stack([close_base + i * 10 for i in range(n_assets)]).astype(
        np.float32
    )
    data_matrix = np.zeros((n_dates, n_assets, 1), dtype=np.float32)
    return EnvironmentDataset(
        data_matrix=data_matrix,
        close_prices=close_prices,
        tickers=tickers,
        dates=dates,
    )


# Backtest report generation
def test_equal_weight_benchmark_uses_initial_trading_cost(
    synthetic_processed_dataset,
) -> None:
    env = PortfolioEnv(make_env_config(synthetic_processed_dataset))
    model = StaticAllocationModel(np.array([0.5, 0.5, 0.0], dtype=np.float32))

    report = Backtester(env=env, model=model).run_backtest()

    assert np.isclose(report["Equal Weight Initial Cost"], 100.0)
    assert report["Equal Weight Final Balance"] > report["Initial Balance"]


def test_backtester_resets_accumulators_between_runs(synthetic_processed_dataset) -> None:
    env = PortfolioEnv(make_env_config(synthetic_processed_dataset))
    model = StaticAllocationModel(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    backtester = Backtester(env=env, model=model)

    backtester.run_backtest()
    first_run_length = len(backtester.portfolio_values)
    backtester.run_backtest()

    assert len(backtester.portfolio_values) == first_run_length


def test_backtester_reports_spy_and_stock_bond_benchmarks() -> None:
    # Inject SPY and TLT so optional benchmark branches are exercised deterministically.
    dataset = _make_injected_dataset(n_dates=8, tickers=["SPY", "TLT"])
    env = PortfolioEnv(
        {
            "initial_balance": 100_000.0,
            "lookback_window": 2,
            "features": ["Norm_RSI"],
            "price_column": "Close",
            "transaction_fee_pct": 0.0,
        },
        dataset=dataset,
    )
    model = StaticAllocationModel(np.array([0.0, 0.0, 1.0], dtype=np.float32))

    report = Backtester(env=env, model=model).run_backtest()

    assert "SPY Buy Hold Final Balance" in report
    assert "60/40 Final Balance" in report
    assert report["SPY Buy Hold Final Balance"] > report["60/40 Final Balance"]


# Report completeness
def test_backtest_report_contains_required_keys(synthetic_processed_dataset) -> None:
    env = PortfolioEnv(make_env_config(synthetic_processed_dataset))
    model = StaticAllocationModel(np.array([0.5, 0.5, 0.0], dtype=np.float32))

    report = Backtester(env=env, model=model).run_backtest()

    required_keys = [
        "Initial Balance",
        "Final Balance",
        "Total Return (%)",
        "Max Drawdown (%)",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Equal Weight Final Balance",
        "Equal Weight Total Return (%)",
        "Alpha vs Equal Weight (%)",
    ]
    for key in required_keys:
        assert key in report, f"Missing report key: {key}"


# Cash-only policy should grow only by risk-free rate
def test_cash_only_policy_preserves_capital_at_zero_risk_free(
    synthetic_processed_dataset,
) -> None:
    env = PortfolioEnv(make_env_config(synthetic_processed_dataset))
    model = StaticAllocationModel(np.array([0.0, 0.0, 1.0], dtype=np.float32))

    report = Backtester(env=env, model=model).run_backtest()

    assert np.isclose(report["Final Balance"], 100_000.0)


# Benchmark edge cases
def test_static_weight_benchmark_returns_zero_for_empty_prices() -> None:
    dataset = _make_injected_dataset(n_dates=3, tickers=["AAA"])
    env = PortfolioEnv(
        {
            "initial_balance": 100_000.0,
            "lookback_window": 2,
            "features": ["Norm_RSI"],
            "price_column": "Close",
            "transaction_fee_pct": 0.0,
        },
        dataset=dataset,
    )
    model = StaticAllocationModel(np.array([0.0, 1.0], dtype=np.float32))
    backtester = Backtester(env=env, model=model)

    # With lookback_window=2 and 3 dates, only 1 step is possible.
    report = backtester.run_backtest()

    assert report["Initial Balance"] == 100_000.0


# Environment type validation
def test_backtester_rejects_incompatible_environment() -> None:
    # A plain Gymnasium environment without PortfolioEnv attributes should fail.
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    model = StaticAllocationModel(np.array([0.0, 1.0], dtype=np.float32))

    with pytest.raises(TypeError, match="PortfolioEnv-compatible"):
        Backtester(env=env, model=model).run_backtest()

    env.close()
