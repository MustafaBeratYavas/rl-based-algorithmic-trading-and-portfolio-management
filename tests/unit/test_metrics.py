"""Validate financial metric edge cases and scale conventions."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    FinancialMetrics,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)


def test_max_drawdown_handles_empty_series() -> None:
    assert FinancialMetrics.calculate_max_drawdown(np.array([])) == 0.0


def test_max_drawdown_calculates_peak_to_trough_loss() -> None:
    values = np.array([100.0, 120.0, 90.0, 130.0])

    assert np.isclose(FinancialMetrics.calculate_max_drawdown(values), 0.25)


def test_max_drawdown_is_zero_for_monotonic_increase() -> None:
    values = np.array([100.0, 110.0, 120.0, 130.0])

    assert calculate_max_drawdown(values) == 0.0


def test_max_drawdown_handles_near_total_loss() -> None:
    values = np.array([100.0, 50.0, 0.01])

    assert np.isclose(calculate_max_drawdown(values), 0.9999)


def test_sharpe_ratio_can_return_daily_or_annualized_scale() -> None:
    # Verify that annualization is an explicit scaling choice, not baked into the sample.
    returns = np.array([0.01, 0.02, -0.005, 0.015])

    daily_sharpe = calculate_sharpe_ratio(
        returns,
        risk_free_rate=0.252,
        annualization_factor=252,
        annualize=False,
    )
    annualized_sharpe = calculate_sharpe_ratio(
        returns,
        risk_free_rate=0.252,
        annualization_factor=252,
        annualize=True,
    )

    assert np.isclose(annualized_sharpe, daily_sharpe * np.sqrt(252))


def test_sharpe_ratio_uses_sample_standard_deviation() -> None:
    returns = np.array([0.01, 0.02, -0.005, 0.015])

    result = calculate_sharpe_ratio(returns, annualize=False)

    assert np.isclose(result, np.mean(returns) / np.std(returns, ddof=1))


def test_sharpe_ratio_returns_zero_for_single_return() -> None:
    assert calculate_sharpe_ratio(np.array([0.05])) == 0.0


def test_sharpe_ratio_returns_zero_for_constant_returns() -> None:
    returns = np.array([0.01, 0.01, 0.01, 0.01])

    assert calculate_sharpe_ratio(returns) == 0.0


def test_sortino_ratio_returns_zero_for_insufficient_data() -> None:
    assert calculate_sortino_ratio(np.array([0.01])) == 0.0


def test_sortino_ratio_returns_zero_when_no_downside_exists() -> None:
    returns = np.array([0.01, 0.02, 0.03, 0.04])

    assert calculate_sortino_ratio(returns) == 0.0


def test_sortino_ratio_is_finite_with_downside_volatility() -> None:
    returns = np.array([0.01, -0.03, 0.02, -0.02, 0.01])

    result = calculate_sortino_ratio(returns, annualize=False)

    assert result != 0.0
    assert np.isfinite(result)


def test_financial_metrics_wrapper_delegates_sortino_ratio() -> None:
    returns = np.array([0.01, -0.02, 0.03, -0.01])

    assert np.isclose(
        FinancialMetrics.calculate_sortino_ratio(returns),
        calculate_sortino_ratio(returns),
    )
