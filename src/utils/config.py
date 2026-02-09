"""Configuration helpers for the centralized pipeline runner."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml

ConfigDict = dict[str, object]


def load_config(config_path: str | Path) -> ConfigDict:
    """Load YAML config into a dictionary.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Parsed configuration mapping.

    Raises:
        ValueError: If the config root is not a mapping.
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")
    return cast(ConfigDict, raw)


def get_section(config: ConfigDict, section_name: str) -> ConfigDict:
    """Return a required mapping section from the root config.

    Args:
        config: Root configuration mapping.
        section_name: Required section key.

    Returns:
        Section mapping.

    Raises:
        ValueError: If the section is missing or not a mapping.
    """
    value = config.get(section_name)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{section_name}' must be a mapping")
    return cast(ConfigDict, value)


def extract_model_kwargs(config: ConfigDict) -> tuple[str, ConfigDict]:
    """Extract model name and model kwargs from global config.

    Args:
        config: Root configuration mapping.

    Returns:
        Tuple of model name and model kwargs.

    Raises:
        ValueError: If the model name is invalid.
    """
    model_config = get_section(config, "model_config")
    model_name = model_config.get("model")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_config.model must be a non-empty string")
    kwargs = dict(model_config)
    kwargs.pop("model", None)
    return model_name.lower(), kwargs
