"""Render equity and drawdown diagnostics for backtest outputs."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.paths import ensure_directory


@lru_cache(maxsize=1)
def _load_pyplot() -> Any:
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "Backtest chart rendering requires matplotlib. Install project runtime "
            "dependencies with `python -m pip install .`."
        ) from exc
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


class BacktestVisualizer:
    def __init__(self, output_dir: str | Path = "logs/evaluation"):
        self.output_dir = ensure_directory(output_dir)

    def save_equity_curve(
        self, portfolio_values: Iterable[float], file_name: str = "equity_curve.png"
    ) -> Path:
        # Plot the reported account-value path without resampling or smoothing.
        plt = _load_pyplot()
        values = np.asarray(list(portfolio_values), dtype=float)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(values, linewidth=2)
        ax.set_title("Portfolio Equity Curve")
        ax.set_xlabel("Step")
        ax.set_ylabel("Portfolio Value")
        ax.grid(True, alpha=0.3)
        return self._save(fig, file_name)

    def save_drawdown(
        self, portfolio_values: Iterable[float], file_name: str = "drawdown.png"
    ) -> Path:
        # Compute drawdown from the running peak to expose path-dependent risk.
        plt = _load_pyplot()
        values = np.asarray(list(portfolio_values), dtype=float)
        peaks = np.maximum.accumulate(values)
        drawdown = np.divide(peaks - values, peaks, out=np.zeros_like(values), where=peaks != 0)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(np.arange(len(drawdown)), drawdown * 100, alpha=0.35)
        ax.set_title("Portfolio Drawdown")
        ax.set_xlabel("Step")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        return self._save(fig, file_name)

    def _save(self, fig: Any, file_name: str) -> Path:
        # Centralize export settings so all diagnostics use the same output quality.
        output_path = self.output_dir / file_name
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        _load_pyplot().close(fig)
        return output_path
