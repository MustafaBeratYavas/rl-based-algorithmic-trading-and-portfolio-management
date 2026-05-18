"""Exercise the full build, train, and evaluate pipeline as an opt-in smoke test."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.data.processor import DataProcessor
from src.envs.portfolio_env import PortfolioEnv
from src.evaluation.backtester import Backtester
from src.models.agent_factory import AgentFactory

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION") != "1",
        reason="Set RUN_INTEGRATION=1 to run the full pipeline smoke test.",
    ),
]


def test_build_train_evaluate_smoke_pipeline(tmp_path) -> None:
    # Use synthetic OHLCV files so the smoke test covers the pipeline without network access.
    raw_path = tmp_path / "raw"
    processed_path = tmp_path / "processed"
    raw_path.mkdir()

    dates = pd.date_range("2024-01-01", periods=90, freq="B")
    for ticker, offset in [("AAA", 0.0), ("BBB", 5.0)]:
        close = np.linspace(100.0 + offset, 120.0 + offset, len(dates))
        frame = pd.DataFrame(
            {
                "Open": close - 0.2,
                "High": close + 0.5,
                "Low": close - 0.5,
                "Close": close,
                "Volume": np.full(len(dates), 1_000_000),
            },
            index=dates,
        )
        frame.to_csv(raw_path / f"{ticker}.csv")

    # Process raw prices into normalized features before constructing the environment.
    data_config = {
        "tickers": ["AAA", "BBB"],
        "price_column": "Close",
        "paths": {"raw_data": raw_path, "processed_data": processed_path},
        "indicators": {"rsi_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
        "splits": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "normalization": {"enabled": True},
    }
    DataProcessor(data_config).process_all()

    # Train for the minimum useful SB3 run, then verify backtesting completes.
    env_config = {
        "data_path": processed_path / "processed_dataset.parquet",
        "lookback_window": 5,
        "initial_balance": 10_000.0,
        "transaction_fee_pct": 0.001,
        "price_column": "Close",
        "features": [
            "Norm_RSI",
            "Norm_MACD",
            "Norm_MACD_Signal",
            "Norm_SMA_20",
            "Norm_SMA_50",
            "Norm_Log_Return",
        ],
    }
    env = PortfolioEnv(env_config)
    model = AgentFactory.create_agent(
        "PPO",
        env,
        {
            "n_steps": 8,
            "batch_size": 8,
            "learning_rate": 0.001,
            "gamma": 0.95,
            "verbose": 0,
        },
        verbose=0,
    )
    model.learn(total_timesteps=16)

    report = Backtester(env, model).run_backtest()

    assert report["Final Balance"] > 0
