from __future__ import annotations

import json
import math
import numbers
import random
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from star.sampling.traversal import (
    corner_indices,
    grid_adjacency,
    is_connected,
)


def _grid_dimensions(height: int, width: int) -> tuple[int, int, int]:
    if (
        isinstance(height, (bool, np.bool_))
        or not isinstance(height, (numbers.Integral, np.integer))
        or isinstance(width, (bool, np.bool_))
        or not isinstance(width, (numbers.Integral, np.integer))
    ):
        raise TypeError("height and width must be integers.")
    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive.")
    return height, width, height * width


def normalize_token_grid_shape(token_grid_shape) -> tuple[int, int]:
    if isinstance(token_grid_shape, (str, bytes)):
        raise TypeError("token_grid_shape must contain exactly two integers.")
    try:
        if len(token_grid_shape) != 2:
            raise TypeError("token_grid_shape must contain exactly two integers.")
        height, width = token_grid_shape
    except TypeError as exc:
        raise TypeError(
            "token_grid_shape must contain exactly two integers."
        ) from exc
    height, width, _ = _grid_dimensions(height, width)
    return height, width


@lru_cache(maxsize=None)
def _grid_bit_layout(height: int, width: int) -> tuple[int, int, int, int]:
    num_vertices = height * width
    all_bits = (1 << num_vertices) - 1
    left_edge_bits = sum(1 << (row * width) for row in range(height))
    right_edge_bits = sum(
        1 << (row * width + width - 1) for row in range(height)
    )
    corner_bits = (
        1
        | (1 << (width - 1))
        | (1 << ((height - 1) * width))
        | (1 << (num_vertices - 1))
    )
    return all_bits, left_edge_bits, right_edge_bits, corner_bits


def _adjacent_bits(height: int, width: int, vertices: int) -> int:
    all_bits, left_edge_bits, right_edge_bits, _ = _grid_bit_layout(height, width)
    return (
        (vertices >> width)
        | ((vertices << width) & all_bits)
        | ((vertices & ~left_edge_bits) >> 1)
        | ((vertices & ~right_edge_bits) << 1)
    ) & all_bits


def _set_bit_at_rank(bits: int, rank: int) -> int:
    remaining = bits
    for _ in range(rank):
        remaining &= remaining - 1
    return remaining & -remaining


def completion_mask_target_size(num_vertices: int, mask_ratio: float) -> int:
    if (
        isinstance(num_vertices, (bool, np.bool_))
        or not isinstance(num_vertices, (numbers.Integral, np.integer))
    ):
        raise TypeError("num_vertices must be an integer.")
    num_vertices = int(num_vertices)
    if num_vertices <= 0:
        raise ValueError("num_vertices must be positive.")
    if isinstance(mask_ratio, (bool, np.bool_)) or not isinstance(
        mask_ratio, (numbers.Real, np.floating)
    ):
        raise TypeError("mask_ratio must be a real number.")
    mask_ratio = float(mask_ratio)
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(
            f"mask_ratio must be between 0 and 1, got {mask_ratio}."
        )
    return math.ceil(num_vertices * mask_ratio)


def grow_connected_mask_bits(
    height: int,
    width: int,
    target_size: int,
    rng,
) -> int:
    height, width, num_vertices = _grid_dimensions(height, width)
    if (
        isinstance(target_size, (bool, np.bool_))
        or not isinstance(target_size, (numbers.Integral, np.integer))
    ):
        raise TypeError("target_size must be an integer.")
    target_size = int(target_size)
    if not 1 <= target_size <= num_vertices:
        raise ValueError(
            f"target_size must be in [1, {num_vertices}], got {target_size}."
        )
    rng = random if rng is None else rng

    start = rng.randrange(num_vertices)
    mask_bits = 1 << start
    frontier_bits = _adjacent_bits(height, width, mask_bits) & ~mask_bits

    while mask_bits.bit_count() < target_size:
        frontier_size = frontier_bits.bit_count()
        if frontier_size == 0:
            raise RuntimeError(
                "The completion-mask frontier became empty before target size."
            )
        selected_bit = _set_bit_at_rank(
            frontier_bits, rng.randrange(frontier_size)
        )
        frontier_bits ^= selected_bit
        mask_bits |= selected_bit
        frontier_bits |= _adjacent_bits(height, width, selected_bit) & ~mask_bits

    return mask_bits


def is_valid_completion_mask_bits(
    height: int,
    width: int,
    mask_bits: int,
) -> bool:
    height, width, _ = _grid_dimensions(height, width)
    if (
        isinstance(mask_bits, (bool, np.bool_))
        or not isinstance(mask_bits, (numbers.Integral, np.integer))
    ):
        raise TypeError("mask_bits must be an integer.")
    mask_bits = int(mask_bits)
    all_bits, _, _, corner_bits = _grid_bit_layout(height, width)
    if mask_bits <= 0 or mask_bits & ~all_bits:
        return False

    observed_bits = all_bits ^ mask_bits
    if observed_bits == 0 or observed_bits & corner_bits == 0:
        return False

    reached = observed_bits & -observed_bits
    wave = reached
    while wave:
        wave = _adjacent_bits(height, width, wave) & observed_bits & ~reached
        reached |= wave
    return reached == observed_bits


def _mask_entry_to_set(entry, sequence_length: int) -> set[int]:
    if isinstance(entry, set):
        if any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (numbers.Integral, np.integer))
            for index in entry
        ):
            raise TypeError("Mask indices must be integers, not booleans or floats.")
        if any(index < 0 or index >= sequence_length for index in entry):
            raise ValueError(f"Mask indices must be in [0, {sequence_length - 1}].")
        return entry

    if not isinstance(entry, list):
        raise TypeError(
            "Each mask must be a list or set of regenerated token indices, "
            f"got {type(entry).__name__}."
        )
    if any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (numbers.Integral, np.integer))
        for index in entry
    ):
        raise TypeError("Mask indices must be integers, not booleans or floats.")
    indices = [int(index) for index in entry]
    if len(indices) != len(set(indices)):
        raise ValueError("A custom mask contains duplicate indices.")
    if any(index < 0 or index >= sequence_length for index in indices):
        raise ValueError(f"Mask indices must be in [0, {sequence_length - 1}].")
    return set(indices)


def validate_masks(masks: Sequence[set[int]], token_grid_shape) -> None:
    height, width = normalize_token_grid_shape(token_grid_shape)
    sequence_length = height * width
    adjacency = grid_adjacency(height, width)
    vertices = set(range(sequence_length))
    corners = set(corner_indices(height, width))

    for sample_index, mask in enumerate(masks):
        if not mask:
            raise ValueError(f"Mask {sample_index} is empty.")
        if mask == vertices:
            raise ValueError(f"Mask {sample_index} covers every token.")
        observed = vertices - mask
        if corners.issubset(mask):
            raise ValueError(f"Mask {sample_index} covers all four corners.")
        if not is_connected(adjacency, observed):
            raise ValueError(
                f"The observed complement of mask {sample_index} is disconnected."
            )


def normalize_masks(
    masks: Sequence[list[int] | set[int]],
    batch_size: int,
    token_grid_shape,
) -> list[set[int]]:
    if not isinstance(masks, (list, tuple)):
        raise TypeError("masks must be a batch of index lists or sets.")
    entries = list(masks)

    height, width = normalize_token_grid_shape(token_grid_shape)
    sequence_length = height * width
    normalized = [
        _mask_entry_to_set(entry, sequence_length) for entry in entries
    ]
    if len(normalized) == 1 and batch_size > 1:
        normalized = [normalized[0] for _ in range(batch_size)]
    if len(normalized) != batch_size:
        raise ValueError(
            f"Received {len(normalized)} masks for a batch of {batch_size}."
        )
    validate_masks(normalized, (height, width))
    return normalized


def masks_to_tensor(masks: Sequence[set[int]], token_grid_shape):
    import torch

    height, width = normalize_token_grid_shape(token_grid_shape)
    sequence_length = height * width
    result = torch.zeros((len(masks), sequence_length), dtype=torch.bool)
    for sample_index, mask in enumerate(masks):
        result[sample_index, sorted(mask)] = True
    return result


def load_masks(path: str | Path, token_grid_shape) -> list[set[int]]:
    height, width = normalize_token_grid_shape(token_grid_shape)
    sequence_length = height * width
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        entries = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        entries = [json.loads(path.read_text(encoding="utf-8"))]
    else:
        raise ValueError("Custom masks must use .json or .jsonl.")

    if not entries:
        raise ValueError("A custom mask file must contain at least one mask.")
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"indices"}:
            raise ValueError(
                'Each mask file entry must be exactly {"indices": [...]}.'
            )

    masks = [
        _mask_entry_to_set(entry["indices"], sequence_length)
        for entry in entries
    ]
    validate_masks(masks, (height, width))
    return masks
