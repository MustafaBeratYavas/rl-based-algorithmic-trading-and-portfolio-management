"""Define domain exceptions for pipeline failure boundaries."""

from __future__ import annotations


# Keep pipeline failures grouped under one domain exception for callers and CLIs.
class RLPortfolioError(Exception):
    pass


# Raised when an external market-data provider cannot satisfy the download contract.
class DownloadError(RLPortfolioError):
    pass


# Raised when raw market data cannot be converted into model-ready artifacts.
class ProcessingError(RLPortfolioError):
    pass


# Raised when training orchestration fails before a model artifact can be trusted.
class TrainingError(RLPortfolioError):
    pass
