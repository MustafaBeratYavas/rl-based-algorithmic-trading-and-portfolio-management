"""Centralize reproducibility helpers for experiment randomness.

Random utilities align Python, project-owned NumPy generators, and optional torch
seeding so training orchestration can be deterministic without forcing every
helper to mutate global RNG state directly.
"""

from __future__ import annotations

import os
import random

import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Create an isolated NumPy random generator for project-owned draws."""
    return np.random.default_rng(seed)


def set_global_seed(seed: int | None) -> np.random.Generator | None:
    """Seed process-level RNGs used by orchestration and optional torch training.

    A ``None`` seed leaves global randomness untouched. NumPy's modern generator
    is returned for project-owned draws while library-level seeds are aligned for
    reproducible Stable-Baselines3 and torch behavior.
    """

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
