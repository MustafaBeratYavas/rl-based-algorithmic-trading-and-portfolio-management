"""Validate configuration loading, path safety, and split selection helpers."""

from __future__ import annotations

import pytest

from src.utils.config import ConfigError, config_with_data_split, load_yaml_config, require_keys


def test_load_yaml_config_raises_on_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_yaml_config("does/not/exist.yaml")


def test_load_yaml_config_rejects_relative_path_traversal() -> None:
    with pytest.raises(ConfigError, match="escapes the project root"):
        load_yaml_config("../outside.yaml")


def test_load_yaml_config_rejects_absolute_path_outside_project(tmp_path) -> None:
    # Use an explicit path guaranteed to be outside the project root on any platform.
    import tempfile
    from pathlib import Path

    system_temp = Path(tempfile.gettempdir()).resolve()
    config_file = system_temp / "rl_test_outside.yaml"
    try:
        config_file.write_text("key: value\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="escapes the project root"):
            load_yaml_config(config_file)
    finally:
        config_file.unlink(missing_ok=True)


def test_load_yaml_config_rejects_non_mapping_yaml(tmp_path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("- item1\n- item2\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="YAML mapping"):
        load_yaml_config(config_file, project_root=tmp_path)


def test_load_yaml_config_returns_mapping_for_valid_yaml(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "good.yaml"
    config_file.write_text("key: value\nnested:\n  answer: 42\n", encoding="utf-8")

    result = load_yaml_config("configs/good.yaml", project_root=tmp_path)

    assert result["key"] == "value"
    assert result["nested"]["answer"] == 42


def test_require_keys_raises_on_missing_keys() -> None:
    with pytest.raises(ConfigError, match="algorithm"):
        require_keys({"learning_rate": 0.001}, ["algorithm", "learning_rate"], "training")


def test_require_keys_passes_when_all_keys_exist() -> None:
    require_keys({"a": 1, "b": 2}, ["a", "b"], "test")


def test_config_with_data_split_sets_selected_data_path() -> None:
    # Split selection should return a copy and leave the caller's config untouched.
    config = {
        "data_path": "data/processed/processed_dataset.parquet",
        "data_paths": {
            "train": "data/processed/train_dataset.parquet",
            "test": "data/processed/test_dataset.parquet",
        },
    }

    result = config_with_data_split(config, "test")

    assert result["data_path"] == "data/processed/test_dataset.parquet"
    assert result["data_split"] == "test"
    assert config["data_path"] == "data/processed/processed_dataset.parquet"


def test_config_with_data_split_without_data_paths_records_split_only() -> None:
    config = {"data_path": "data/processed/processed_dataset.parquet"}

    result = config_with_data_split(config, "validation")

    assert result["data_path"] == "data/processed/processed_dataset.parquet"
    assert result["data_split"] == "validation"
    assert "data_split" not in config


def test_config_with_data_split_rejects_empty_split_name() -> None:
    with pytest.raises(ConfigError, match="cannot be empty"):
        config_with_data_split({}, "")


def test_config_with_data_split_rejects_non_mapping_data_paths() -> None:
    with pytest.raises(ConfigError, match="must be a mapping"):
        config_with_data_split({"data_paths": ["train.parquet"]}, "train")


def test_config_with_data_split_rejects_unknown_split() -> None:
    with pytest.raises(ConfigError, match="Unknown data split"):
        config_with_data_split({"data_paths": {"train": "train.parquet"}}, "test")
