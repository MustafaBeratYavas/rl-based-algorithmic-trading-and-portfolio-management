"""Assemble processed parquet artifacts into PortfolioEnv tensors.

EnvironmentDataProvider owns the data-loading boundary for simulations,
validating feature contracts, ticker coverage, shared trading calendars, and
finite price/feature rows before the Gym environment receives dense arrays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import resolve_project_path


@dataclass(frozen=True)
class EnvironmentDataset:
    """Immutable, aligned tensors consumed by ``PortfolioEnv``.

    ``data_matrix`` and ``close_prices`` share the same date and ticker axes. This
    invariant lets the environment compute observations, returns, and rebalancing
    costs without revalidating the long-form parquet input on every step.
    """

    data_matrix: np.ndarray
    close_prices: np.ndarray
    tickers: list[str]
    dates: list[pd.Timestamp]


class EnvironmentDataProvider:
    """Load processed parquet data and build the dense environment tensors.

    This class is the boundary between persisted research artifacts and the Gym
    environment. It validates feature contracts, ticker coverage, date filters,
    and finite price/feature rows before simulation state is initialized.
    """

    DEFAULT_FEATURES = [
        "Norm_RSI",
        "Norm_MACD",
        "Norm_MACD_Signal",
        "Norm_SMA_20",
        "Norm_SMA_50",
        "Norm_Log_Return",
    ]

    def __init__(self, config: dict[str, Any]):
        """Capture feature, price, ticker, date, and lookback requirements from config."""
        self.config = config
        self.logger = get_logger(__name__)
        self.feature_cols = list(config.get("features", self.DEFAULT_FEATURES))
        self.price_column = config.get("price_column", "Close")
        self.expected_tickers = sorted(config.get("tickers", []))
        self.start_date = pd.Timestamp(config.get("start_date", "2000-01-01"))
        self.end_date = pd.Timestamp(config.get("end_date", "2099-12-31"))
        self.lookback_window = int(config.get("lookback_window", 30))

    def load(self) -> EnvironmentDataset:
        """Load, validate, align, and return tensors for deterministic simulation."""
        # Keep file access and panel construction outside the Gym environment boundary.
        self._validate_feature_contract()
        resolved_data_path = self._resolve_data_path()
        metadata = self._load_metadata(resolved_data_path)

        df = pd.read_parquet(resolved_data_path)
        if df.empty:
            raise ValueError(f"Processed dataset is empty: {resolved_data_path}")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        original_range = self._date_range_label(df)
        required_columns = set(self.feature_cols) | {self.price_column, "Ticker"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(f"Processed dataset is missing columns: {sorted(missing_columns)}")

        df = df.sort_index()
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
        df = df.replace([np.inf, -np.inf], np.nan)
        if df.empty:
            raise ValueError(
                "No rows available for configured environment dates "
                f"{self.start_date.date()} through {self.end_date.date()} in "
                f"{resolved_data_path}. Dataset covers {original_range}; "
                f"metadata covers {self._metadata_range_label(metadata)}."
            )

        # Pin the ticker axis to configuration order when a universe is explicitly provided.
        available_tickers = sorted(df["Ticker"].dropna().unique().tolist())
        if self.expected_tickers:
            missing_tickers = sorted(set(self.expected_tickers).difference(available_tickers))
            if missing_tickers:
                raise ValueError(
                    "Processed dataset is missing configured tickers after date filtering: "
                    f"{missing_tickers}"
                )
            df = df[df["Ticker"].isin(self.expected_tickers)]
            tickers = self.expected_tickers
        else:
            tickers = available_tickers
        if not tickers:
            raise ValueError("Processed dataset does not contain any tickers.")

        # Convert the long dataset into aligned per-ticker frames before stacking arrays.
        ticker_frames = self._build_ticker_frames(df, tickers, required_columns)
        common_dates = self._shared_dates(ticker_frames)
        data_matrix, close_prices = self._build_matrices(ticker_frames, tickers, common_dates)
        dates, data_matrix, close_prices = self._drop_invalid_rows(
            common_dates, data_matrix, close_prices
        )

        minimum_dates = self.lookback_window + 2
        if len(dates) < minimum_dates:
            raise ValueError(
                "Not enough valid dates to initialize PortfolioEnv: "
                f"got {len(dates)}, need at least {minimum_dates}. "
                f"Configured range is {self.start_date.date()} through {self.end_date.date()}; "
                f"dataset covers {original_range}."
            )

        self.logger.info(
            "Loaded environment data with %s dates, %s assets, and %s features.",
            len(dates),
            len(tickers),
            len(self.feature_cols),
        )
        return EnvironmentDataset(
            data_matrix=data_matrix,
            close_prices=close_prices,
            tickers=tickers,
            dates=dates,
        )

    def _validate_feature_contract(self) -> None:
        """Ensure observations use engineered features, not raw accounting prices."""
        if not self.feature_cols:
            raise ValueError("PortfolioEnv requires at least one feature column.")
        if self.price_column in self.feature_cols:
            raise ValueError(
                f"price_column={self.price_column!r} must not also appear in features. "
                "Use normalized indicator columns for observations and reserve the raw "
                "price column for return and transaction calculations."
            )

    def _resolve_data_path(self) -> Path:
        """Resolve the configured processed dataset path and require it to exist."""
        data_path = self.config.get("data_path")
        if data_path is None:
            raise ValueError("PortfolioEnv requires a data_path config value.")

        resolved_data_path = resolve_project_path(data_path)
        if not resolved_data_path.exists():
            raise FileNotFoundError(f"Processed dataset not found: {resolved_data_path}")
        return resolved_data_path

    @staticmethod
    def _load_metadata(data_path: Path) -> dict[str, Any]:
        """Load optional dataset metadata without making it a hard runtime dependency."""
        # Metadata is advisory; corrupted or missing metadata must not block data loading.
        metadata_path = data_path.parent / "dataset_metadata.json"
        if not metadata_path.exists():
            return {}
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(metadata, dict):
            return {}
        return cast(dict[str, Any], metadata)

    @staticmethod
    def _date_range_label(df: pd.DataFrame) -> str:
        """Return a human-readable date range for diagnostics."""
        if df.empty:
            return "no rows"
        return f"{df.index.min().date()} through {df.index.max().date()}"

    @staticmethod
    def _metadata_range_label(metadata: dict[str, Any]) -> str:
        """Return the metadata date range or an unavailable marker."""
        start_date = metadata.get("start_date")
        end_date = metadata.get("end_date")
        if not start_date or not end_date:
            return "unavailable"
        return f"{start_date} through {end_date}"

    def _build_ticker_frames(
        self,
        df: pd.DataFrame,
        tickers: list[str],
        required_columns: set[str],
    ) -> dict[str, pd.DataFrame]:
        """Build one de-duplicated feature/price frame per ticker in stable order."""
        # Deduplicate by date per ticker so the latest vendor row wins deterministically.
        ticker_frames: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            ticker_data = df[df["Ticker"] == ticker].sort_index()
            ticker_data = ticker_data[~ticker_data.index.duplicated(keep="last")]
            ticker_frames[ticker] = ticker_data[list(required_columns - {"Ticker"})]
        return ticker_frames

    @staticmethod
    def _shared_dates(ticker_frames: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
        """Find the common trading calendar required by dense tensor stacking."""
        date_sets = [set(ticker_data.index) for ticker_data in ticker_frames.values()]
        common_dates = sorted(set.intersection(*date_sets)) if date_sets else []
        if not common_dates:
            raise ValueError("No common trading dates are available across configured tickers.")
        return common_dates

    def _build_matrices(
        self,
        ticker_frames: dict[str, pd.DataFrame],
        tickers: list[str],
        common_dates: list[pd.Timestamp],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stack feature and price panels using a stable ticker/date axis order."""
        feature_panels = []
        price_panels = []
        for ticker in tickers:
            ticker_data = ticker_frames[ticker].reindex(common_dates).ffill()
            feature_panels.append(ticker_data[self.feature_cols].to_numpy(dtype=np.float32))
            price_panels.append(ticker_data[self.price_column].to_numpy(dtype=np.float32))

        data_matrix = np.stack(feature_panels, axis=1)
        close_prices = np.stack(price_panels, axis=1)
        return data_matrix, close_prices

    @staticmethod
    def _drop_invalid_rows(
        common_dates: list[pd.Timestamp],
        data_matrix: np.ndarray,
        close_prices: np.ndarray,
    ) -> tuple[list[pd.Timestamp], np.ndarray, np.ndarray]:
        """Remove rows that would make observations or return math undefined."""
        valid_feature_rows = np.isfinite(data_matrix).all(axis=(1, 2))
        valid_price_rows = np.isfinite(close_prices).all(axis=1) & (close_prices > 0).all(axis=1)
        valid_rows = valid_feature_rows & valid_price_rows

        dates = [date for date, is_valid in zip(common_dates, valid_rows, strict=True) if is_valid]
        return (
            dates,
            data_matrix[valid_rows].astype(np.float32, copy=False),
            close_prices[valid_rows].astype(np.float32, copy=False),
        )
