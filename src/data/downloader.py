"""Fetch raw market data and enforce the first pipeline quality gate.

YFinanceDownloader coordinates vendor access, retry/backoff behavior, response
shape validation, and CSV persistence. It deliberately stops before feature
engineering so downstream data-processing decisions remain isolated and auditable.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from src.utils.exceptions import DownloadError
from src.utils.logger import get_logger
from src.utils.paths import ensure_directory


class YFinanceDownloader:
    """Download and persist raw market data under a strict OHLCV contract.

    The downloader is intentionally responsible for vendor tolerance only:
    retries, backoff, response shape validation, and raw CSV persistence. Feature
    cleaning and split decisions stay in ``DataProcessor`` so data lineage remains
    easy to audit.
    """

    REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}

    def __init__(self, config: dict[str, Any]):
        """Initialize vendor settings, quality thresholds, and the raw output directory."""
        self.config = config
        self.logger = get_logger(__name__)
        self.tickers = config.get("tickers", [])
        self.start_date = config.get("start_date")
        self.end_date = config.get("end_date")
        self.raw_path = ensure_directory(config.get("paths", {}).get("raw_data", "data/raw"))

        # Keep network tolerance explicit so operators can tune retries per environment.
        download_config = config.get("download", {})
        self.retries = int(download_config.get("retries", 3))
        self.backoff_seconds = float(download_config.get("backoff_seconds", 2.0))
        self.minimum_rows = int(download_config.get("minimum_rows", 30))
        self.max_nan_ratio = float(download_config.get("max_nan_ratio", 0.05))
        self.strict = bool(download_config.get("strict", True))
        self.auto_adjust = bool(download_config.get("auto_adjust", True))

    def fetch_data(self) -> None:
        """Download configured tickers and apply the configured failure policy.

        Raises:
            ValueError: If the ticker universe is empty.
            DownloadError: If strict mode is enabled and any ticker cannot be persisted.
        """
        # Reject empty ticker universes before the pipeline can appear to succeed.
        if not self.tickers:
            raise ValueError("No tickers configured for data download.")

        self.logger.info("Starting data download process.")
        failed_tickers: list[str] = []
        for ticker in self.tickers:
            if not self._download_ticker(ticker):
                failed_tickers.append(ticker)

        if failed_tickers and self.strict:
            raise DownloadError(
                f"Failed to download required tickers after retries: {', '.join(failed_tickers)}"
            )
        if failed_tickers:
            self.logger.warning(
                "Continuing with partial data because download.strict=false. Failed: %s",
                ", ".join(failed_tickers),
            )
        self.logger.info("Data download process completed.")

    def _download_ticker(self, ticker: str) -> bool:
        """Download and persist one ticker, returning whether a valid CSV was written."""
        # Retry transient provider and network failures before marking the ticker unavailable.
        for attempt in range(self.retries):
            try:
                self.logger.info(
                    "Downloading %s (attempt %s/%s)", ticker, attempt + 1, self.retries
                )
                df = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=self.auto_adjust,
                )

                # Flatten yfinance's occasional MultiIndex response while rejecting ambiguity.
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if df.columns.duplicated().any():
                    duplicates = sorted(set(df.columns[df.columns.duplicated()].tolist()))
                    raise ValueError(f"{ticker} response contains duplicate columns: {duplicates}")

                # Validate the raw response before persisting data consumed by later stages.
                self._validate_download(df, ticker)

                file_path = self.raw_path / f"{ticker}.csv"
                df.to_csv(file_path)
                self.logger.info("Successfully saved %s to %s", ticker, file_path)
                return True

            except (
                ValueError,
                KeyError,
                TimeoutError,
                ConnectionError,
                OSError,
                requests.exceptions.RequestException,
            ) as exc:
                self.logger.warning("Download failed for %s: %s", ticker, exc)
                if attempt < self.retries - 1:
                    time.sleep(self.backoff_seconds * (2**attempt))
                    continue
                self.logger.error("Failed to download %s after %s attempts.", ticker, self.retries)
        return False

    def _validate_download(self, df: pd.DataFrame, ticker: str) -> None:
        """Validate required OHLCV shape, row count, missing-data density, and prices."""
        if df.empty:
            raise ValueError(f"No data retrieved for {ticker}.")

        missing = self.REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"{ticker} response is missing required columns: {sorted(missing)}")

        if len(df) < self.minimum_rows:
            raise ValueError(f"{ticker} has only {len(df)} rows; minimum is {self.minimum_rows}.")

        # Bound missing-data density so forward filling cannot hide poor vendor responses.
        nan_ratio = df[list(self.REQUIRED_COLUMNS)].isna().mean().max()
        if nan_ratio > self.max_nan_ratio:
            raise ValueError(
                f"{ticker} NaN ratio {nan_ratio:.2%} exceeds {self.max_nan_ratio:.2%}."
            )

        # Reject non-positive prices because return calculations assume a positive price base.
        if (df["Close"].dropna() <= 0).any():
            raise ValueError(f"{ticker} contains non-positive close prices.")
