"""Calculate finite risk and return metrics for backtest reports.

Metric helpers intentionally return stable numeric fallbacks for degenerate
samples so short smoke tests and failed strategies still produce serializable,
comparable reports.
"""

from __future__ import annotations

import numpy as np


def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    """Return the largest peak-to-trough loss in an account-value path.

    Empty inputs and zero-capital prefixes return ``0.0`` rather than ``nan`` so
    downstream reports remain finite.
    """
    if len(portfolio_values) == 0:
        return 0.0
    peak = portfolio_values[0]
    max_drawdown = 0.0

    for value in portfolio_values:
        if value > peak:
            peak = value
        # A zero peak cannot define drawdown, so skip until capital becomes positive again.
        if peak == 0:
            continue
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return float(max_drawdown)


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
    annualize: bool = True,
) -> float:
    """Calculate a finite Sharpe ratio from periodic returns.

    Degenerate samples return ``0.0`` instead of propagating ``nan`` so reports
    remain serializable and comparable across short smoke tests and full runs.
    """

    if len(returns) < 2:
        return 0.0
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    if not np.isfinite(std_return) or std_return == 0:
        return 0.0
    # Convert annual risk-free rates to the daily scale used by return samples.
    daily_risk_free = risk_free_rate / annualization_factor
    sharpe = (mean_return - daily_risk_free) / std_return
    if annualize:
        sharpe *= np.sqrt(annualization_factor)
    return float(sharpe)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
    annualize: bool = True,
) -> float:
    """Calculate a finite Sortino ratio using downside volatility only.

    Returns ``0.0`` when there is insufficient downside history to define a
    stable denominator.
    """
    if len(returns) < 2:
        return 0.0
    mean_return = np.mean(returns)
    daily_risk_free = risk_free_rate / annualization_factor
    downside_returns = returns[returns < daily_risk_free]

    # Sortino is undefined without at least two downside observations.
    if len(downside_returns) < 2:
        return 0.0

    downside_std = np.std(downside_returns, ddof=1)
    if not np.isfinite(downside_std) or downside_std == 0:
        return 0.0
    sortino = (mean_return - daily_risk_free) / downside_std
    if annualize:
        sortino *= np.sqrt(annualization_factor)
    return float(sortino)


class FinancialMetrics:
    """Backward-compatible namespace around the functional metric API."""

    @staticmethod
    def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
        """Delegate max drawdown calculation to the functional API."""
        return calculate_max_drawdown(portfolio_values)

    @staticmethod
    def calculate_sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualization_factor: int = 252,
        annualize: bool = True,
    ) -> float:
        """Delegate Sharpe ratio calculation to the functional API."""
        return calculate_sharpe_ratio(
            returns,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor,
            annualize=annualize,
        )

    @staticmethod
    def calculate_sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualization_factor: int = 252,
        annualize: bool = True,
    ) -> float:
        """Delegate Sortino ratio calculation to the functional API."""
        return calculate_sortino_ratio(
            returns,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor,
            annualize=annualize,
        )
