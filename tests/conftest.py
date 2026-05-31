"""Provide shared pytest fixtures for deterministic portfolio datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_processed_dataset(tmp_path):
    """Create a compact processed parquet panel that satisfies the env feature contract."""
    dates = pd.date_range("2024-01-01", periods=45, freq="B")
    records = []
    for ticker, offset in [("AAA", 0.0), ("BBB", 10.0)]:
        for idx, date in enumerate(dates):
            close = 100.0 + offset + idx
            previous_close = close - 1.0
            log_return = np.log(close / previous_close) if previous_close > 0 else 0.0
            records.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "Close": close,
                    "Norm_RSI": 0.1 + idx / 100,
                    "Norm_MACD": 0.2 + idx / 100,
                    "Norm_MACD_Signal": 0.3 + idx / 100,
                    "Norm_SMA_20": 0.4 + idx / 100,
                    "Norm_SMA_50": 0.5 + idx / 100,
                    "Norm_Log_Return": log_return,
                }
            )

    df = pd.DataFrame(records).set_index("Date")
    path = tmp_path / "processed_dataset.parquet"
    df.to_parquet(path)
    return path
