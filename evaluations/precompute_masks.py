from __future__ import annotations

import argparse
import json
import math
import numbers
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from star.sampling.masks import (  # noqa: E402
    completion_mask_target_size,
    grow_connected_mask_bits,
    is_valid_completion_mask_bits,
)


_UINT32_MAX = (1 << 32) - 1
_MASK_PROPOSAL_SEED_DOMAIN = 2
_ACCEPTANCE_HINTS_16X16 = {
    26: 0.411,
    52: 0.0864,
    77: 0.0240,
    103: 0.00867,
    128: 0.00372,
    154: 0.00137,
    180: 0.00068,
    205: 0.00052,
    231: 0.000935,
}


def _proposal_seed(
    base_seed: int,
    grid_height: int,
    grid_width: int,
    target_size: int,
    proposal_index: int,
) -> int:
    entropy = [
        proposal_index,
        target_size,
        grid_height,
        _MASK_PROPOSAL_SEED_DOMAIN,
        base_seed,
    ]
    if grid_width != grid_height:
        entropy.append(grid_width)
    return int(
        np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0]
    )


def _run_proposal_batch(
    arguments: tuple[int, int, int, int, int, int],
) -> tuple[int, bytes]:
    (
        grid_height,
        grid_width,
        target_size,
        start_index,
        proposal_count,
        base_seed,
    ) = arguments
    byte_width = (grid_height * grid_width + 7) // 8
    packed = bytearray()
    for offset in range(proposal_count):
        proposal_index = start_index + offset
        rng = random.Random(
            _proposal_seed(
                base_seed,
                grid_height,
                grid_width,
                target_size,
                proposal_index,
            )
        )
        mask_bits = grow_connected_mask_bits(
            grid_height,
            grid_width,
            target_size,
            rng,
        )
        if is_valid_completion_mask_bits(grid_height, grid_width, mask_bits):
            packed.extend(proposal_index.to_bytes(8, "little"))
            packed.extend(mask_bits.to_bytes(byte_width, "little"))
    return proposal_count, bytes(packed)


def _unpack_accepted(
    packed: bytes,
    byte_width: int,
) -> Iterable[tuple[int, int]]:
    record_width = 8 + byte_width
    if len(packed) % record_width:
        raise RuntimeError("A worker returned a malformed packed-mask payload.")
    for offset in range(0, len(packed), record_width):
        proposal_index = int.from_bytes(packed[offset : offset + 8], "little")
        mask_bits = int.from_bytes(
            packed[offset + 8 : offset + record_width], "little"
        )
        yield proposal_index, mask_bits


def _acceptance_hint(
    grid_height: int,
    grid_width: int,
    target_size: int,
) -> float:
    if grid_height == grid_width == 16:
        return _ACCEPTANCE_HINTS_16X16.get(target_size, 0.001)
    return 0.05


def _collect_masks(
    *,
    pool: ProcessPoolExecutor,
    grid_height: int,
    grid_width: int,
    target_size: int,
    count: int,
    seed: int,
    proposals_per_task: int,
    max_proposals: int,
) -> tuple[list[tuple[int, int]], int, int, float]:
    byte_width = (grid_height * grid_width + 7) // 8
    accepted: list[tuple[int, int]] = []
    proposals_executed = 0
    next_proposal_index = 0
    rate = _acceptance_hint(grid_height, grid_width, target_size)
    started = time.perf_counter()

    while len(accepted) < count:
        remaining_capacity = max_proposals - proposals_executed
        if remaining_capacity <= 0:
            raise RuntimeError(
                f"Found only {len(accepted)} accepted masks after "
                f"{max_proposals} proposals. Increase --max-proposals."
            )

        remaining_masks = count - len(accepted)
        desired_proposals = math.ceil(1.2 * remaining_masks / max(rate, 1e-12))
        proposal_budget = min(
            remaining_capacity,
            max(proposals_per_task, desired_proposals),
        )
        tasks = []
        unassigned = proposal_budget
        while unassigned > 0:
            task_count = min(proposals_per_task, unassigned)
            tasks.append(
                (
                    grid_height,
                    grid_width,
                    target_size,
                    next_proposal_index,
                    task_count,
                    seed,
                )
            )
            next_proposal_index += task_count
            unassigned -= task_count

        for proposed, packed in pool.map(_run_proposal_batch, tasks, chunksize=1):
            proposals_executed += proposed
            accepted.extend(_unpack_accepted(packed, byte_width))

        accepted.sort(key=lambda item: item[0])
        if accepted:
            empirical_rate = len(accepted) / proposals_executed
            rate = max(rate, empirical_rate)

    selected = accepted[:count]
    proposals_scanned = selected[-1][0] + 1
    elapsed_seconds = time.perf_counter() - started
    return selected, proposals_scanned, proposals_executed, elapsed_seconds


def _mask_indices(mask_bits: int) -> list[int]:
    indices = []
    while mask_bits:
        bit = mask_bits & -mask_bits
        indices.append(bit.bit_length() - 1)
        mask_bits ^= bit
    return indices


def _validate_arguments(
    *,
    mask_ratios: Sequence[float],
    count: int,
    grid_height: int,
    grid_width: int | None = None,
    seed: int,
    workers: int,
    proposals_per_task: int,
    max_proposals: int,
) -> list[float]:
    if isinstance(count, bool) or not isinstance(count, numbers.Integral) or count <= 0:
        raise ValueError("count must be a positive integer.")
    if grid_width is None:
        grid_width = grid_height
    for value, name in (
        (grid_height, "grid_height"),
        (grid_width, "grid_width"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")
    if int(grid_height) * int(grid_width) > _UINT32_MAX:
        raise ValueError(
            "grid_height * grid_width must fit in an unsigned 32-bit integer."
        )
    if (
        isinstance(seed, bool)
        or not isinstance(seed, numbers.Integral)
        or not 0 <= seed <= _UINT32_MAX
    ):
        raise ValueError("seed must be an integer in [0, 2**32 - 1].")
    if (
        isinstance(workers, bool)
        or not isinstance(workers, numbers.Integral)
        or workers <= 0
    ):
        raise ValueError("workers must be a positive integer.")
    if (
        isinstance(proposals_per_task, bool)
        or not isinstance(proposals_per_task, numbers.Integral)
        or proposals_per_task <= 0
    ):
        raise ValueError("proposals_per_task must be a positive integer.")
    if (
        isinstance(max_proposals, bool)
        or not isinstance(max_proposals, numbers.Integral)
        or max_proposals <= 0
        or max_proposals > _UINT32_MAX + 1
    ):
        raise ValueError("max_proposals must be an integer in [1, 2**32].")
    if count > max_proposals:
        raise ValueError(
            "count must not exceed max_proposals because each proposal can "
            "produce at most one mask."
        )

    normalized_mask_ratios = [float(mask_ratio) for mask_ratio in mask_ratios]
    if not normalized_mask_ratios:
        raise ValueError("At least one mask ratio is required.")
    if any(
        not math.isfinite(mask_ratio) or not 0.0 < mask_ratio < 1.0
        for mask_ratio in normalized_mask_ratios
    ):
        raise ValueError(
            "Every mask ratio must be finite and strictly between 0 and 1."
        )
    if len(set(normalized_mask_ratios)) != len(normalized_mask_ratios):
        raise ValueError("Mask ratios must be unique.")
    mask_ratio_labels = [
        f"{mask_ratio:.12g}" for mask_ratio in normalized_mask_ratios
    ]
    if len(set(mask_ratio_labels)) != len(mask_ratio_labels):
        raise ValueError("Mask ratios must have unique canonical filename labels.")
    num_vertices = int(grid_height) * int(grid_width)
    if any(
        completion_mask_target_size(num_vertices, mask_ratio) == num_vertices
        for mask_ratio in normalized_mask_ratios
    ):
        raise ValueError(
            "A mask ratio rounds to a full-grid mask and cannot leave an observed "
            "corner."
        )
    return normalized_mask_ratios


def precompute_masks(
    output_dir: str | Path,
    *,
    mask_ratios: Sequence[float],
    count: int,
    grid_height: int = 16,
    grid_width: int | None = None,
    seed: int = 42,
    workers: int = 1,
    proposals_per_task: int = 1000,
    max_proposals: int = 1_000_000_000,
) -> Path:
    mask_ratios = _validate_arguments(
        mask_ratios=mask_ratios,
        count=count,
        grid_height=grid_height,
        grid_width=grid_width,
        seed=seed,
        workers=workers,
        proposals_per_task=proposals_per_task,
        max_proposals=max_proposals,
    )
    count = int(count)
    grid_height = int(grid_height)
    grid_width = grid_height if grid_width is None else int(grid_width)
    seed = int(seed)
    workers = int(workers)
    proposals_per_task = int(proposals_per_task)
    max_proposals = int(max_proposals)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    runs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for mask_ratio in mask_ratios:
            target_size = completion_mask_target_size(
                grid_height * grid_width,
                mask_ratio,
            )
            masks, proposals_scanned, proposals_executed, elapsed_seconds = (
                _collect_masks(
                    pool=pool,
                    grid_height=grid_height,
                    grid_width=grid_width,
                    target_size=target_size,
                    count=count,
                    seed=seed,
                    proposals_per_task=proposals_per_task,
                    max_proposals=max_proposals,
                )
            )
            mask_ratio_label = f"{mask_ratio:.12g}"
            output_path = output_dir / f"masks_ratio_{mask_ratio_label}.jsonl"
            with output_path.open("x", encoding="utf-8") as handle:
                for _, mask_bits in masks:
                    record = {"indices": _mask_indices(mask_bits)}
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")

            runs.append(
                {
                    "mask_ratio": mask_ratio,
                    "target": target_size,
                    "count": count,
                    "proposals_scanned": proposals_scanned,
                    "proposals_executed": proposals_executed,
                    "acceptance_rate": count / proposals_scanned,
                    "elapsed_seconds": elapsed_seconds,
                    "file": output_path.name,
                }
            )

    manifest = {
        "grid": {
            "height": grid_height,
            "width": grid_width,
            "num_vertices": grid_height * grid_width,
        },
        "seed": seed,
        "workers": workers,
        "proposals_per_task": proposals_per_task,
        "runs": runs,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mask-ratios",
        nargs="+",
        type=float,
        default=[index / 10 for index in range(1, 10)],
    )
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--grid-height", type=int, default=16)
    parser.add_argument("--grid-width", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    parser.add_argument("--proposals-per-task", type=int, default=1000)
    parser.add_argument("--max-proposals", type=int, default=1_000_000_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    manifest = precompute_masks(
        arguments.output_dir,
        mask_ratios=arguments.mask_ratios,
        count=arguments.count,
        grid_height=arguments.grid_height,
        grid_width=arguments.grid_width,
        seed=arguments.seed,
        workers=arguments.workers,
        proposals_per_task=arguments.proposals_per_task,
        max_proposals=arguments.max_proposals,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
