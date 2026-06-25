"""Generate publication-style portfolio research charts from deterministic backtests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm

from src.envs.portfolio_env import PortfolioEnv
from src.evaluation.visualizer import _load_pyplot


@dataclass(frozen=True)
class PolicyTrace:
    """Time-aligned policy path collected from one deterministic evaluation episode."""

    return_dates: pd.DatetimeIndex
    portfolio_values: np.ndarray
    allocation_dates: pd.DatetimeIndex
    weights: np.ndarray
    tickers: tuple[str, ...]
    close_prices: np.ndarray
    lookback_window: int
    initial_balance: float
    buy_fee_pct: float
    slippage_pct: float
    market_impact_pct: float
    daily_risk_free_rate: float


RETURN_COLORS = {
    "PPO Agent": "#1f5a83",
    "SPY Benchmark": "#2f7f80",
    "Equal Weight Portfolio": "#746f63",
}

ALLOCATION_COLORS = {
    "SPY": "#376789",
    "QQQ": "#90aabf",
    "XLK": "#5c9f92",
    "XLF": "#a095b3",
    "XLV": "#c8bb79",
    "XLE": "#bb9676",
    "XLY": "#86b7ad",
    "TLT": "#718fa0",
    "GLD": "#d1b35f",
    "Cash": "#d5d8d9",
}


def run_deterministic_policy_trace(env: PortfolioEnv, model: BaseAlgorithm) -> PolicyTrace:
    """Run a trained policy once and collect values, dates, and allocation weights."""
    obs, _ = env.reset()
    done = False
    truncated = False

    initial_date = pd.Timestamp(env.dates[env.lookback_window - 1])
    return_dates = [initial_date]
    portfolio_values = [float(env.initial_balance)]
    allocation_dates: list[pd.Timestamp] = []
    allocation_weights: list[np.ndarray] = []

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, _reward, done, truncated, info = env.step(action)

        return_dates.append(pd.Timestamp(info["date"]))
        portfolio_values.append(float(info["portfolio_value"]))
        allocation_dates.append(pd.Timestamp(info["date"]))
        allocation_weights.append(np.asarray(info["weights"], dtype=np.float64))

    return PolicyTrace(
        return_dates=pd.DatetimeIndex(return_dates),
        portfolio_values=np.asarray(portfolio_values, dtype=np.float64),
        allocation_dates=pd.DatetimeIndex(allocation_dates),
        weights=np.vstack(allocation_weights).astype(np.float64, copy=False),
        tickers=tuple(env.tickers),
        close_prices=env.close_prices.astype(np.float64, copy=True),
        lookback_window=int(env.lookback_window),
        initial_balance=float(env.initial_balance),
        buy_fee_pct=float(env.buy_fee_pct),
        slippage_pct=float(env.slippage_pct),
        market_impact_pct=float(env.market_impact_pct),
        daily_risk_free_rate=float(env.daily_risk_free_rate),
    )


def save_cumulative_return_analysis(
    trace: PolicyTrace,
    output_path: str | Path,
    *,
    width: int = 1920,
    height: int = 1080,
    dpi: int = 160,
) -> Path:
    """Save PPO, SPY, and equal-weight cumulative returns using the requested canvas."""
    plt = _load_pyplot()
    ticker_count = len(trace.tickers)
    benchmark_curves = {
        "SPY Benchmark": static_weight_curve(trace, {"SPY": 1.0}),
        "Equal Weight Portfolio": static_weight_curve(
            trace, dict.fromkeys(trace.tickers, 1.0 / ticker_count)
        ),
    }

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    _style_axis(ax)
    ax.set_title("Cumulative Portfolio Return Analysis", fontsize=20, weight="bold", pad=12)
    ax.set_xlabel("Out-of-Sample Evaluation Period", fontsize=14)
    ax.set_ylabel("Cumulative Return", fontsize=14)

    ppo_returns = _cumulative_returns(trace.portfolio_values, trace.initial_balance)
    ax.plot(
        trace.return_dates,
        ppo_returns * 100.0,
        label="PPO Agent",
        color=RETURN_COLORS["PPO Agent"],
        linewidth=2.8,
    )
    for label, values in benchmark_curves.items():
        ax.plot(
            trace.return_dates,
            _cumulative_returns(values, trace.initial_balance) * 100.0,
            label=label,
            color=RETURN_COLORS[label],
            linewidth=2.8,
        )

    _format_date_axis(ax)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=100.0, decimals=0))
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=11)
    fig.text(
        0.075,
        0.04,
        "Deterministic out-of-sample backtest; transaction costs included",
        color="#687079",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.15)

    return _save_figure(fig, output_path)


def save_dynamic_allocation_chart(
    trace: PolicyTrace,
    output_path: str | Path,
    *,
    display_tickers: Sequence[str] | None = None,
    width: int = 1920,
    height: int = 1080,
    dpi: int = 160,
) -> Path:
    """Save the policy's dynamic allocation path as a stacked percentage area chart."""
    plt = _load_pyplot()
    labels, series = _allocation_series(trace, display_tickers)
    colors = [ALLOCATION_COLORS.get(label, "#8f969b") for label in labels]

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    _style_axis(ax)
    ax.stackplot(trace.allocation_dates, series, labels=labels, colors=colors, alpha=0.78)
    ax.set_title("Dynamic Portfolio Allocation Over Time", fontsize=20, weight="bold", pad=12)
    ax.set_xlabel("Out-of-Sample Evaluation Period", fontsize=14)
    ax.set_ylabel("Portfolio Weight", fontsize=14)
    ax.set_ylim(0.0, 1.0)
    ax.margins(x=0.0, y=0.0)

    _format_date_axis(ax)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=5,
        frameon=True,
        framealpha=0.95,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.20)

    return _save_figure(fig, output_path)


def static_weight_curve(
    trace: PolicyTrace,
    target_weights_by_ticker: Mapping[str, float],
) -> np.ndarray:
    """Build a buy-and-hold benchmark curve aligned to the policy return dates."""
    weight_vector = np.array(
        [target_weights_by_ticker.get(ticker, 0.0) for ticker in trace.tickers],
        dtype=np.float64,
    )
    total_risky_weight = float(weight_vector.sum())
    if total_risky_weight > 1.0 + 1e-9:
        raise ValueError("Benchmark risky weights cannot exceed 100%.")

    initial_cost = _initial_allocation_cost(trace, weight_vector)
    investable_balance = max(trace.initial_balance - initial_cost, 0.0)

    base_step = trace.lookback_window - 1
    prices = trace.close_prices[base_step:]
    if len(prices) <= 1:
        return np.asarray([trace.initial_balance], dtype=np.float64)

    price_growth = np.divide(
        prices[1:],
        prices[0],
        out=np.ones_like(prices[1:], dtype=np.float64),
        where=prices[0] > 0,
    )
    cash_weight = max(1.0 - total_risky_weight, 0.0)
    cash_growth = (1.0 + trace.daily_risk_free_rate) ** np.arange(1, len(prices))
    portfolio_growth = price_growth @ weight_vector + cash_weight * cash_growth
    values = investable_balance * portfolio_growth
    return np.concatenate([[trace.initial_balance], values.astype(np.float64, copy=False)])


def _initial_allocation_cost(trace: PolicyTrace, risky_weights: np.ndarray) -> float:
    buy_turnover = float(np.maximum(risky_weights, 0.0).sum())
    fee_rate = buy_turnover * trace.buy_fee_pct
    slippage_rate = buy_turnover * trace.slippage_pct
    impact_rate = (buy_turnover**2) * trace.market_impact_pct
    return float((fee_rate + slippage_rate + impact_rate) * trace.initial_balance)


def _allocation_series(
    trace: PolicyTrace,
    display_tickers: Sequence[str] | None,
) -> tuple[list[str], list[np.ndarray]]:
    ticker_to_index = {ticker: index for index, ticker in enumerate(trace.tickers)}
    ordered_tickers = [
        ticker
        for ticker in (display_tickers or trace.tickers)
        if ticker in ticker_to_index
    ]
    labels = [*ordered_tickers, "Cash"]
    series = [trace.weights[:, ticker_to_index[ticker]] for ticker in ordered_tickers]
    series.append(trace.weights[:, -1])
    return labels, series


def _cumulative_returns(values: np.ndarray, initial_balance: float) -> np.ndarray:
    return np.divide(values, initial_balance, out=np.ones_like(values), where=initial_balance != 0) - 1.0


def _style_axis(ax: Any) -> None:
    ax.grid(True, color="#c7cdd3", linewidth=1.0, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_color("#7e8790")
        spine.set_linewidth(1.2)
    ax.tick_params(axis="both", labelsize=11, colors="#303740")
    ax.set_facecolor("white")


def _format_date_axis(ax: Any) -> None:
    plt = _load_pyplot()
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(bymonth=[4, 10]))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    ax.margins(x=0.0)


def _save_figure(fig: Any, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    _load_pyplot().close(fig)
    return path
