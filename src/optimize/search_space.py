"""Search-space schema and config patching helpers for optimization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from src.utils.config import ConfigDict, as_bool, as_float, as_int, as_str, get_section

ParameterType = Literal["float", "int", "categorical"]


class TrialSuggestProtocol(Protocol):
    """Protocol for trial-like suggest methods."""

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        step: float | None = None,
    ) -> float:
        """Suggest float value."""

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        log: bool = False,
        step: int | None = None,
    ) -> int:
        """Suggest int value."""

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        """Suggest categorical value."""


@dataclass(frozen=True)
class SearchParameter:
    """One hyperparameter definition from ``optimization.search_space``."""

    name: str
    path: str
    parameter_type: ParameterType
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    choices: tuple[object, ...] = ()


def parse_search_space(raw_search_space: object) -> list[SearchParameter]:
    """Parse the configured search-space list with validation.

    Args:
        raw_search_space: Raw ``optimization.search_space`` payload.

    Returns:
        Parsed search parameters.

    Raises:
        ValueError: If schema is invalid.
    """
    if not isinstance(raw_search_space, Sequence) or isinstance(raw_search_space, (str, bytes)):
        raise ValueError("optimization.search_space must be a sequence")

    parsed: list[SearchParameter] = []
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    for index, raw_item in enumerate(raw_search_space):
        field_prefix = f"optimization.search_space[{index}]"
        item = _ensure_mapping(raw_item, field_prefix)
        parsed_item = _parse_search_item(item=item, field_prefix=field_prefix)
        if parsed_item.name in seen_names:
            raise ValueError(f"Duplicate search parameter name: {parsed_item.name}")
        if parsed_item.path in seen_paths:
            raise ValueError(f"Duplicate search parameter path: {parsed_item.path}")
        seen_names.add(parsed_item.name)
        seen_paths.add(parsed_item.path)
        parsed.append(parsed_item)

    if not parsed:
        raise ValueError("optimization.search_space must contain at least one item")
    return parsed


def extend_with_nas_lite(
    *,
    root_config: ConfigDict,
    base_search_space: list[SearchParameter],
) -> list[SearchParameter]:
    """Append default NAS-lite architecture parameters when enabled.

    NAS-lite phase-1 is implemented as architecture-parameter HPO.

    Args:
        root_config: Root runtime config.
        base_search_space: Already parsed user search space.

    Returns:
        Extended search-space list.
    """
    nas_cfg_raw = root_config.get("nas_lite")
    if not isinstance(nas_cfg_raw, Mapping):
        return base_search_space
    nas_cfg = cast(ConfigDict, nas_cfg_raw)
    if not as_bool(nas_cfg.get("enabled", False), "nas_lite.enabled"):
        return base_search_space
    method = as_str(nas_cfg.get("method", "arch_params_hpo"), "nas_lite.method").lower()
    if method != "arch_params_hpo":
        raise ValueError("nas_lite.method must be 'arch_params_hpo' for phase-1")

    model_cfg = get_section(root_config, "model_config")
    model_name = as_str(model_cfg.get("model", ""), "model_config.model").lower()
    if model_name not in {"v3", "v4", "v5"}:
        return base_search_space

    existing_paths = {item.path for item in base_search_space}
    merged = list(base_search_space)
    for default in _default_nas_parameters():
        if default.path in existing_paths:
            continue
        if not dot_path_exists(root_config, default.path):
            continue
        merged.append(default)
        existing_paths.add(default.path)
    return merged


def sample_parameter(*, trial: object, parameter: SearchParameter) -> object:
    """Sample one parameter from a trial-like object.

    The trial object only needs ``suggest_float/suggest_int/suggest_categorical``.

    Args:
        trial: Optuna-like trial object.
        parameter: Search-space definition.

    Returns:
        Sampled value.
    """
    trial_suggest = cast(TrialSuggestProtocol, trial)
    if parameter.parameter_type == "float":
        if parameter.low is None or parameter.high is None:
            raise ValueError(f"Float parameter {parameter.name} requires low/high")
        if parameter.step is not None:
            return trial_suggest.suggest_float(
                parameter.name,
                float(parameter.low),
                float(parameter.high),
                log=parameter.log,
                step=float(parameter.step),
            )
        return trial_suggest.suggest_float(
            parameter.name,
            float(parameter.low),
            float(parameter.high),
            log=parameter.log,
        )

    if parameter.parameter_type == "int":
        if parameter.low is None or parameter.high is None:
            raise ValueError(f"Int parameter {parameter.name} requires low/high")
        if parameter.step is not None:
            return trial_suggest.suggest_int(
                parameter.name,
                int(parameter.low),
                int(parameter.high),
                log=parameter.log,
                step=int(parameter.step),
            )
        return trial_suggest.suggest_int(
            parameter.name,
            int(parameter.low),
            int(parameter.high),
            log=parameter.log,
        )

    return trial_suggest.suggest_categorical(parameter.name, list(parameter.choices))


def apply_search_parameters(
    *,
    base_config: ConfigDict,
    sampled_values: Mapping[str, object],
    search_space: Sequence[SearchParameter],
) -> ConfigDict:
    """Return a deep-copied config patched with sampled values.

    Args:
        base_config: Root config template.
        sampled_values: Mapping from search parameter names to sampled values.
        search_space: Parsed search-space definitions.

    Returns:
        Trial-specific config clone.
    """
    trial_config = _deep_copy_mapping(base_config)
    for parameter in search_space:
        if parameter.name not in sampled_values:
            raise ValueError(f"Sampled values missing key: {parameter.name}")
        value = sampled_values[parameter.name]
        set_dot_path_value(trial_config, parameter.path, value)
    return trial_config


def dot_path_exists(config: Mapping[str, object], dot_path: str) -> bool:
    """Return whether a nested dotted path exists in a mapping."""
    keys = _split_path(dot_path)
    cursor: object = config
    for key in keys:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return False
        cursor = cursor[key]
    return True


def set_dot_path_value(config: ConfigDict, dot_path: str, value: object) -> None:
    """Set a nested value in-place by dotted path.

    Args:
        config: Mutable config mapping.
        dot_path: Dot path (e.g. ``training_config.optimizer.lr``).
        value: Value to set.

    Raises:
        ValueError: If path is invalid or missing.
    """
    keys = _split_path(dot_path)
    cursor: object = config
    for key in keys[:-1]:
        if not isinstance(cursor, dict):
            raise ValueError(f"Path parent is not a mapping for: {dot_path}")
        if key not in cursor:
            raise ValueError(f"Unknown config path: {dot_path}")
        child = cursor[key]
        if not isinstance(child, dict):
            raise ValueError(f"Path parent is not a mapping for: {dot_path}")
        cursor = child

    final_key = keys[-1]
    if not isinstance(cursor, dict):
        raise ValueError(f"Path parent is not a mapping for: {dot_path}")
    if final_key not in cursor:
        raise ValueError(f"Unknown config path: {dot_path}")
    cursor[final_key] = value


def _parse_search_item(*, item: ConfigDict, field_prefix: str) -> SearchParameter:
    """Parse one search-space item."""
    name = as_str(item.get("name", ""), f"{field_prefix}.name")
    path = as_str(item.get("path", ""), f"{field_prefix}.path")
    parameter_type = as_str(item.get("type", ""), f"{field_prefix}.type").lower()
    if parameter_type not in {"float", "int", "categorical"}:
        raise ValueError(f"{field_prefix}.type must be float/int/categorical")

    if parameter_type == "categorical":
        choices_raw = item.get("choices")
        if not isinstance(choices_raw, Sequence) or isinstance(choices_raw, (str, bytes)):
            raise ValueError(f"{field_prefix}.choices must be a sequence")
        choices = tuple(choices_raw)
        if not choices:
            raise ValueError(f"{field_prefix}.choices must not be empty")
        return SearchParameter(
            name=name,
            path=path,
            parameter_type="categorical",
            choices=choices,
        )

    low = item.get("low")
    high = item.get("high")
    if parameter_type == "float":
        low_float = as_float(low, f"{field_prefix}.low")
        high_float = as_float(high, f"{field_prefix}.high")
        if high_float <= low_float:
            raise ValueError(f"{field_prefix}.high must be greater than low")
        step_raw = item.get("step")
        step_value = as_float(step_raw, f"{field_prefix}.step") if step_raw is not None else None
        if step_value is not None and step_value <= 0:
            raise ValueError(f"{field_prefix}.step must be > 0")
        return SearchParameter(
            name=name,
            path=path,
            parameter_type="float",
            low=low_float,
            high=high_float,
            step=step_value,
            log=as_bool(item.get("log", False), f"{field_prefix}.log"),
        )

    low_int = as_int(low, f"{field_prefix}.low")
    high_int = as_int(high, f"{field_prefix}.high")
    if high_int < low_int:
        raise ValueError(f"{field_prefix}.high must be >= low")
    step_raw_int = item.get("step")
    step_int = as_int(step_raw_int, f"{field_prefix}.step") if step_raw_int is not None else None
    if step_int is not None and step_int <= 0:
        raise ValueError(f"{field_prefix}.step must be > 0")
    return SearchParameter(
        name=name,
        path=path,
        parameter_type="int",
        low=low_int,
        high=high_int,
        step=step_int,
        log=as_bool(item.get("log", False), f"{field_prefix}.log"),
    )


def _ensure_mapping(raw_value: object, field_name: str) -> ConfigDict:
    """Cast raw value to mapping with error context."""
    if not isinstance(raw_value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return dict(raw_value)


def _split_path(dot_path: str) -> list[str]:
    """Split and validate dotted config paths."""
    if not dot_path or dot_path.strip() == "":
        raise ValueError("Search path must be a non-empty string")
    keys = [token.strip() for token in dot_path.split(".")]
    if not keys or any(not token for token in keys):
        raise ValueError(f"Invalid search path: {dot_path}")
    return keys


def _deep_copy_mapping(mapping: Mapping[str, object]) -> ConfigDict:
    """Recursively deep copy nested mapping/list config structures."""
    copied: ConfigDict = {}
    for key, value in mapping.items():
        copied[key] = _deep_copy_value(value)
    return copied


def _deep_copy_value(value: object) -> object:
    """Recursively deep copy one config value."""
    if isinstance(value, Mapping):
        return _deep_copy_mapping(cast(Mapping[str, object], value))
    if isinstance(value, list):
        return [_deep_copy_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_deep_copy_value(item) for item in value)
    return value


def _default_nas_parameters() -> list[SearchParameter]:
    """Return default architecture parameters for phase-1 NAS-lite."""
    return [
        SearchParameter(
            name="nas_d_model",
            path="model_config.d_model",
            parameter_type="categorical",
            choices=(128, 192, 256),
        ),
        SearchParameter(
            name="nas_encoder_layers",
            path="model_config.encoder_layers",
            parameter_type="int",
            low=1,
            high=4,
        ),
        SearchParameter(
            name="nas_cross_attn_layers",
            path="model_config.cross_attn_layers",
            parameter_type="int",
            low=1,
            high=4,
        ),
    ]


__all__ = [
    "ParameterType",
    "SearchParameter",
    "apply_search_parameters",
    "dot_path_exists",
    "extend_with_nas_lite",
    "parse_search_space",
    "sample_parameter",
    "set_dot_path_value",
]
