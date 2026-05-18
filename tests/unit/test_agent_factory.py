"""Validate SB3 agent factory algorithm selection and config filtering."""

from __future__ import annotations

from src.models.agent_factory import AgentFactory


class FakeAlgorithm:
    # Minimal SB3-compatible constructor and loader used to isolate factory behavior.
    def __init__(self, policy, env, **kwargs):
        self.policy = policy
        self.env = env
        self.kwargs = kwargs

    @classmethod
    def load(cls, path, env=None, **kwargs):
        instance = cls("loaded", env, **kwargs)
        instance.loaded_path = path
        return instance


def test_unsupported_algorithm_error_lists_supported_values() -> None:
    try:
        AgentFactory.create_agent("DQN", env=None, config={})  # type: ignore[arg-type]
    except ValueError as exc:
        message = str(exc)
        assert "DQN" in message
        assert "PPO" in message
        assert "SAC" in message
    else:
        raise AssertionError("Expected unsupported algorithms to raise ValueError.")


def test_supported_algorithms_returns_sorted_tuple() -> None:
    result = AgentFactory.supported_algorithms()

    assert result == tuple(sorted(result))
    assert {"A2C", "PPO", "SAC"}.issubset(result)


def test_extract_model_kwargs_filters_algorithm_specific_parameters() -> None:
    # PPO must ignore off-policy and project-level configuration keys.
    config = {
        "learning_rate": 0.001,
        "batch_size": 64,
        "buffer_size": 1000,
        "tensorboard_log": "logs/tb_logs",
        "model_save_path": "models/final/model",
    }

    result = AgentFactory._extract_model_kwargs("PPO", config)

    assert result == {"learning_rate": 0.001, "batch_size": 64}


def test_create_agent_resolves_tensorboard_path_and_verbose(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(AgentFactory._ALGORITHMS, "PPO", FakeAlgorithm)

    result = AgentFactory.create_agent(
        "ppo",
        env=object(),  # type: ignore[arg-type]
        config={"learning_rate": 0.001, "tensorboard_log": tmp_path / "tb", "verbose": 0},
        verbose=2,
    )

    assert isinstance(result, FakeAlgorithm)
    assert result.policy == "MultiInputPolicy"
    assert result.kwargs["learning_rate"] == 0.001
    assert result.kwargs["tensorboard_log"] == str(tmp_path / "tb")
    assert result.kwargs["verbose"] == 0


def test_load_agent_accepts_extensionless_model_path_and_reports_missing_file(tmp_path) -> None:
    model_path = tmp_path / "missing_model"

    try:
        AgentFactory.load_agent("PPO", str(model_path))
    except FileNotFoundError as exc:
        assert str(model_path.with_suffix(".zip")) in str(exc)
    else:
        raise AssertionError("Expected missing model path to raise FileNotFoundError.")


def test_load_agent_resolves_extensionless_existing_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(AgentFactory._ALGORITHMS, "PPO", FakeAlgorithm)
    model_path = tmp_path / "trained_model.zip"
    model_path.write_bytes(b"placeholder")

    result = AgentFactory.load_agent("PPO", str(model_path.with_suffix("")), custom=True)

    assert isinstance(result, FakeAlgorithm)
    assert result.loaded_path == str(model_path)
    assert result.kwargs == {"custom": True}
