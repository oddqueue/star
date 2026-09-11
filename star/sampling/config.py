from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from pathlib import Path

from omegaconf import OmegaConf


_GENERATION_KEYS = {
    "class_id",
    "num_samples",
    "num_order_workers",
    "per_process_batch_size",
    "seed",
}
_COMPLETION_KEYS = {
    "masks",
    "num_dataloader_workers",
    "num_samples",
    "num_order_workers",
    "per_process_batch_size",
    "pretokenized_jsonl",
    "seed",
    "source_record_index",
}
_UINT32_MAX = (1 << 32) - 1
_INTEGER_RULES = {
    "class_id": (0, None),
    "num_dataloader_workers": (0, None),
    "num_order_workers": (0, None),
    "num_samples": (1, None),
    "per_process_batch_size": (1, None),
    "seed": (0, _UINT32_MAX),
    "source_record_index": (0, None),
}
_SAMPLING_HYPERPARAMETERS = (
    "temperature",
    "guidance_scale",
    "guidance_scale_power",
)
_OPTIONAL_INTEGER_OPTIONS = {"class_id", "source_record_index"}


def _load_config_from_cli(supported_inference_keys):
    cli_config = OmegaConf.from_cli()
    if "config" not in cli_config:
        raise ValueError(
            "A model config is required, for example config=path/to/model.yaml"
        )
    yaml_config = OmegaConf.load(cli_config.config)
    _validate_cli_override_paths(yaml_config, cli_config)
    config = OmegaConf.merge(yaml_config, cli_config)
    _validate_inference_keys(config, supported_inference_keys)
    return config


def load_generation_config_from_cli():
    config = _load_config_from_cli(_GENERATION_KEYS)
    _validate_inference_integers(config)
    _validate_sampling_hyperparameters(config)
    return config


def load_completion_config_from_cli():
    config = _load_config_from_cli(_COMPLETION_KEYS)
    _validate_completion_inputs(config)
    _validate_inference_integers(config)
    _validate_sampling_hyperparameters(config)
    return config


def inference_value(config, name: str, default=None):
    section = config.get("inference")
    if section is None:
        return default
    if not isinstance(section, Mapping):
        raise ValueError("inference must be a mapping.")
    return section.get(name, default)


def _validate_cli_override_paths(yaml_config, cli_config) -> None:
    yaml_tree = OmegaConf.to_container(yaml_config, resolve=False)
    cli_tree = OmegaConf.to_container(cli_config, resolve=False)
    unknown = []
    for path in _leaf_paths(cli_tree):
        if path == ("config",) or path[0] == "inference":
            continue
        if not _path_exists(yaml_tree, path):
            unknown.append(".".join(path))
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown config override(s): {names}.")


def _leaf_paths(value, prefix=()):
    if isinstance(value, Mapping) and value:
        for name, child in value.items():
            yield from _leaf_paths(child, (*prefix, str(name)))
        return
    yield prefix


def _path_exists(tree, path) -> bool:
    for name in path:
        if not isinstance(tree, Mapping) or name not in tree:
            return False
        tree = tree[name]
    return True


def _validate_inference_keys(config, supported_keys) -> None:
    section = config.get("inference")
    if section is None:
        return
    if not isinstance(section, Mapping):
        raise ValueError("inference must be a mapping.")
    unknown = sorted(set(section.keys()) - supported_keys)
    if unknown:
        names = ", ".join(f"inference.{name}" for name in unknown)
        raise ValueError(f"Unknown sampling option(s): {names}.")


def _validate_inference_integers(config) -> None:
    section = config.get("inference")
    if section is None:
        return
    for name, (minimum, maximum) in _INTEGER_RULES.items():
        if name not in section:
            continue
        value = section[name]
        if value is None and name in _OPTIONAL_INTEGER_OPTIONS:
            continue
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(
                f"inference.{name} must be an integer, got {value!r}."
            )
        if minimum is not None and value < minimum:
            qualifier = "positive" if minimum == 1 else "non-negative"
            raise ValueError(
                f"inference.{name} must be {qualifier}, got {value}."
            )
        if maximum is not None and value > maximum:
            raise ValueError(
                f"inference.{name} must be between {minimum} and {maximum}, "
                f"got {value}."
            )


def _validate_sampling_hyperparameters(config) -> None:
    for name in _SAMPLING_HYPERPARAMETERS:
        path = f"model.generator.{name}"
        try:
            value = OmegaConf.select(config, path)
        except Exception as exc:
            raise ValueError(f"{path} must be a finite number.") from exc
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{path} must be a finite number, got {value!r}.")
        if name == "temperature" and value <= 0:
            raise ValueError(f"{path} must be positive, got {value}.")
        if name == "guidance_scale_power" and value < 0:
            raise ValueError(f"{path} must be non-negative, got {value}.")


def _validate_completion_inputs(config) -> None:
    for name in ("pretokenized_jsonl", "masks"):
        value = inference_value(config, name, None)
        if value is None:
            raise ValueError(f"inference.{name} is required for completion.")
        if not isinstance(value, (str, Path)) or not value:
            raise ValueError(
                f"inference.{name} must name an existing file, got {value!r}."
            )
        if not Path(value).is_file():
            raise ValueError(
                f"inference.{name} does not exist or is not a file: {value}."
            )
