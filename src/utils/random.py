"""Centralize reproducibility helpers for experiment randomness."""

from __future__ import annotations

import os
import random

import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Create an isolated NumPy random generator for project-owned draws."""
    return np.random.default_rng(seed)


def set_global_seed(seed: int | None) -> np.random.Generator | None:
    if seed is None:
        return None

    # Align process-level libraries while keeping NumPy's legacy global state untouched.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    rng = make_rng(seed)

    try:
        import torch
    except ImportError:
        return rng

    # Mirror CPU seeds to CUDA devices when training is GPU-backed.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return rng
