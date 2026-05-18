"""Expose backtesting, financial metrics, and diagnostic visualization tools."""

from src.evaluation.backtester import Backtester
from src.evaluation.metrics import (
    FinancialMetrics,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from src.evaluation.visualizer import BacktestVisualizer

__all__ = [
    "BacktestVisualizer",
    "Backtester",
    "FinancialMetrics",
    "calculate_max_drawdown",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
]
