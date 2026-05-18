"""Create and load supported Stable-Baselines3 agents from project configuration."""

from __future__ import annotations

from typing import Any, TypeAlias

from gymnasium import Env
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from src.utils.paths import resolve_project_path

AlgorithmClass: TypeAlias = type[BaseAlgorithm]


class AgentFactory:
    _ALGORITHMS: dict[str, AlgorithmClass] = {
        "PPO": PPO,
        "SAC": SAC,
        "A2C": A2C,
    }

    _PARAMETERS: dict[str, set[str]] = {
        "PPO": {
            "learning_rate",
            "n_steps",
            "batch_size",
            "gamma",
            "ent_coef",
            "gae_lambda",
            "clip_range",
            "max_grad_norm",
            "seed",
        },
        "A2C": {
            "learning_rate",
            "n_steps",
            "gamma",
            "ent_coef",
            "gae_lambda",
            "max_grad_norm",
            "seed",
        },
        "SAC": {
            "learning_rate",
            "buffer_size",
            "learning_starts",
            "batch_size",
            "tau",
            "gamma",
            "train_freq",
            "gradient_steps",
            "ent_coef",
            "seed",
        },
    }

    @classmethod
    def supported_algorithms(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._ALGORITHMS))

    @classmethod
    def create_agent(
        cls,
        algorithm: str,
        env: Env | VecEnv,
        config: dict[str, Any],
        verbose: int | None = None,
    ) -> BaseAlgorithm:
        algorithm_name = cls._normalize_algorithm(algorithm)
        algorithm_class = cls._ALGORITHMS[algorithm_name]
        # Pass only algorithm-specific constructor parameters to avoid SB3 config leakage.
        model_kwargs = cls._extract_model_kwargs(algorithm_name, config)
        tensorboard_log = config.get("tensorboard_log")
        if tensorboard_log is not None:
            model_kwargs["tensorboard_log"] = str(resolve_project_path(tensorboard_log))
        model_kwargs["verbose"] = config.get("verbose", 1 if verbose is None else verbose)

        return algorithm_class("MultiInputPolicy", env, **model_kwargs)

    @classmethod
    def load_agent(
        cls,
        algorithm: str,
        model_path: str,
        env: Env | VecEnv | None = None,
        **kwargs: Any,
    ) -> BaseAlgorithm:
        algorithm_name = cls._normalize_algorithm(algorithm)
        resolved_path = resolve_project_path(model_path)
        # Accept extension-neutral config paths because SB3 persists models as zip archives.
        if resolved_path.suffix != ".zip":
            resolved_path = resolved_path.with_suffix(".zip")
        if not resolved_path.exists():
            raise FileNotFoundError(f"Trained model not found: {resolved_path}")
        return cls._ALGORITHMS[algorithm_name].load(str(resolved_path), env=env, **kwargs)

    @classmethod
    def _normalize_algorithm(cls, algorithm: str) -> str:
        algorithm_name = algorithm.upper()
        if algorithm_name not in cls._ALGORITHMS:
            # Include supported values so configuration mistakes are actionable at startup.
            supported = ", ".join(cls.supported_algorithms())
            raise ValueError(f"Algorithm {algorithm!r} is not supported. Supported: {supported}")
        return algorithm_name

    @classmethod
    def _extract_model_kwargs(cls, algorithm: str, config: dict[str, Any]) -> dict[str, Any]:
        allowed_parameters = cls._PARAMETERS[algorithm]
        return {
            key: value
            for key, value in config.items()
            if key in allowed_parameters and value is not None
        }
