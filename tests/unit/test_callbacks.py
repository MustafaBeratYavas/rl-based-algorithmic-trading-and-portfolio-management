"""Validate best-model checkpoint behavior for training callbacks."""

from __future__ import annotations

from src.models.callbacks import SaveBestModelCallback


class FakeModel:
    """Minimal SB3-like surface used by ``SaveBestModelCallback`` tests."""

    def __init__(self, rewards):
        """Seed the episode-info buffer with deterministic reward summaries."""
        self.ep_info_buffer = [{"r": reward} for reward in rewards]
        self.saved_paths = []

    def save(self, path) -> None:
        """Record checkpoint paths instead of writing model archives."""
        self.saved_paths.append(path)


def test_save_best_model_callback_persists_improved_reward(tmp_path) -> None:
    model = FakeModel([1.0, 2.0, 3.0])
    callback = SaveBestModelCallback(check_freq=1, save_path=str(tmp_path), verbose=0)
    callback.model = model
    callback.n_calls = 1
    callback._init_callback()

    assert callback._on_step() is True

    assert callback.best_mean_reward == 2.0
    assert model.saved_paths == [tmp_path / "best_model"]


def test_save_best_model_callback_skips_non_improving_rewards(tmp_path) -> None:
    model = FakeModel([1.0, 2.0, 3.0])
    callback = SaveBestModelCallback(check_freq=1, save_path=str(tmp_path), verbose=0)
    callback.model = model
    callback.n_calls = 1
    callback.best_mean_reward = 10.0
    callback._init_callback()

    assert callback._on_step() is True

    assert model.saved_paths == []


def test_save_best_model_callback_respects_check_frequency(tmp_path) -> None:
    model = FakeModel([5.0])
    callback = SaveBestModelCallback(check_freq=10, save_path=str(tmp_path), verbose=0)
    callback.model = model
    callback.n_calls = 3
    callback._init_callback()

    callback._on_step()

    assert model.saved_paths == []
