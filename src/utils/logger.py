"""Configure shared console and file logging for project commands.

Logging setup is process-wide and idempotent, ensuring CLI entry points, data
pipeline stages, and training jobs share one formatter without duplicating
handlers across repeated imports.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

from src.utils.paths import ensure_directory, resolve_project_path

_CONFIGURED = False
_LOCK = threading.Lock()


def configure_logging(
    level: int | str = logging.INFO,
    log_file: str | Path = "logs/pipeline.log",
) -> None:
    """Configure process-wide console and file logging exactly once."""
    global _CONFIGURED
    # Protect one-time handler setup so concurrent callers cannot duplicate log output.
    with _LOCK:
        if _CONFIGURED:
            return

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        root_logger = logging.getLogger()
        root_logger.setLevel(_coerce_log_level(level))

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Validate the log directory immediately while delaying file creation until first use.
        file_path = resolve_project_path(log_file)
        ensure_directory(file_path.parent)
        file_handler = logging.FileHandler(file_path, encoding="utf-8", delay=True)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        _CONFIGURED = True


def configure_logging_from_config(
    config: dict, default_log_file: str = "logs/pipeline.log"
) -> None:
    """Configure logging from an optional ``logging`` config mapping."""
    # Let command configs tune logging without coupling callers to logging internals.
    logging_config = config.get("logging", {}) if isinstance(config, dict) else {}
    configure_logging(
        level=logging_config.get("level", logging.INFO),
        log_file=logging_config.get("log_file", default_log_file),
    )


def get_logger(name: str) -> logging.Logger:
    """Return a configured project logger by name."""
    configure_logging()
    return logging.getLogger(name)


def _coerce_log_level(level: int | str) -> int:
    """Normalize integer or string log-level input into a logging constant."""
    if isinstance(level, int):
        return level

    level_name = level.upper()
    if level_name not in logging._nameToLevel:
        raise ValueError(f"Unsupported log level: {level!r}")
    return int(logging._nameToLevel[level_name])
