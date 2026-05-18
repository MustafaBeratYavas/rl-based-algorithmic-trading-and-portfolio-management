"""Calculate risk and return metrics used by backtest reports."""

from __future__ import annotations

import numpy as np


def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
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
    @staticmethod
    def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
        return calculate_max_drawdown(portfolio_values)

    @staticmethod
    def calculate_sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualization_factor: int = 252,
        annualize: bool = True,
    ) -> float:
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
        return calculate_sortino_ratio(
            returns,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor,
            annualize=annualize,
        )
