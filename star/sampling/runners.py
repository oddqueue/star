from __future__ import annotations

import operator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader

from star.data import PretokenizedDatasetJSONL

from .api import STARSampler
from .config import inference_value
from .masks import load_masks
from .output import (
    create_npz,
    prepare_output,
    save_image_batch,
)
from .runtime import (
    DistributedRuntime,
    broadcast_error,
    local_output_indices,
    run_rank_synchronized,
    seed_everything,
)


def _integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def _inference_int(config, name: str, default: int) -> int:
    return _integer(inference_value(config, name, default), f"inference.{name}")


def _positive_int(config, name: str, default: int) -> int:
    value = _inference_int(config, name, default)
    if value <= 0:
        raise ValueError(f"inference.{name} must be positive, got {value}.")
    return value


def _seed(config) -> int:
    value = _inference_int(config, "seed", 42)
    if not 0 <= value <= (1 << 32) - 1:
        raise ValueError(
            "inference.seed must be between 0 and 4294967295, "
            f"got {value}."
        )
    return value


def _batch_ranges(output_indices: np.ndarray, batch_size: int):
    for start in range(0, len(output_indices), batch_size):
        yield output_indices[start : start + batch_size]


def run_generation(
    config,
    runtime: DistributedRuntime,
    sampler: STARSampler,
) -> Path:
    num_samples = _positive_int(config, "num_samples", 50_000)
    batch_size = _positive_int(config, "per_process_batch_size", 50)
    seed = _seed(config)
    class_id = inference_value(config, "class_id", None)
    if class_id is not None:
        class_id = _integer(class_id, "inference.class_id")
        if not 0 <= class_id < sampler.generator.condition_num_classes:
            raise ValueError(f"inference.class_id is out of range: {class_id}.")

    output_dir = prepare_output(config.experiment.output_dir, runtime)
    seed_everything(seed + runtime.rank)
    output_indices = local_output_indices(
        num_samples,
        runtime.rank,
        runtime.world_size,
    )
    num_classes = int(sampler.generator.condition_num_classes)

    for batch_output_indices in _batch_ranges(output_indices, batch_size):
        if class_id is None:
            class_ids = batch_output_indices % num_classes
        else:
            class_ids = np.full(
                len(batch_output_indices),
                class_id,
                dtype=np.int64,
            )
        result = sampler.generate(
            class_ids,
            sample_ids=batch_output_indices.tolist(),
        )
        save_image_batch(output_dir, batch_output_indices, result.images)

    # This is deliberately a success-only rendezvous. A rank-local sampling
    # exception propagates immediately so torchrun can terminate its peers;
    # rank zero cannot publish an NPZ unless every rank reached this point.
    runtime.barrier()
    error = None
    if runtime.is_main_process:
        try:
            create_npz(output_dir, num_samples)
        except Exception as exc:
            error = f"Generation post-processing failed: {type(exc).__name__}: {exc}"
    broadcast_error(error)
    return output_dir


def _completion_dataloader(
    dataset,
    output_indices: np.ndarray,
    batch_size: int,
    num_workers: int,
):
    batch_sampler = BatchSampler(
        output_indices.tolist(), batch_size=batch_size, drop_last=False
    )
    worker_arguments = {}
    if num_workers > 0:
        # CUDA and NCCL are initialized before completion reaches this path.
        # Spawn CPU-only readers instead of inheriting accelerator state via
        # the Linux DataLoader default (fork).
        worker_arguments["multiprocessing_context"] = "spawn"
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        **worker_arguments,
    )


def _run_completion_once(
    config,
    sampler: STARSampler,
    dataset,
    output_indices: np.ndarray,
    output_dir: Path,
    *,
    masks: list[set[int]],
    source_record_index: int | None = None,
) -> None:
    batch_size = _positive_int(config, "per_process_batch_size", 50)

    if source_record_index is None:
        num_workers = _inference_int(config, "num_dataloader_workers", 0)
        if num_workers < 0:
            raise ValueError("inference.num_dataloader_workers cannot be negative.")
        batches = zip(
            _batch_ranges(output_indices, batch_size),
            _completion_dataloader(
                dataset,
                output_indices,
                batch_size,
                num_workers,
            ),
        )
    else:
        source_class_id, source_token_ids = dataset[source_record_index]

        def repeated_batches():
            for batch_output_indices in _batch_ranges(output_indices, batch_size):
                count = len(batch_output_indices)
                class_ids = (
                    torch.as_tensor(source_class_id).reshape(1).expand(count)
                )
                token_ids = (
                    torch.as_tensor(source_token_ids)
                    .reshape(1, -1)
                    .expand(count, -1)
                )
                yield batch_output_indices, (class_ids, token_ids)

        batches = repeated_batches()

    processed = 0
    for batch_output_indices, batch in batches:
        class_ids, token_ids = batch
        if len(masks) == 1:
            batch_masks = [masks[0] for _ in batch_output_indices]
        else:
            batch_masks = [masks[int(index)] for index in batch_output_indices]

        result = sampler.complete(
            class_ids,
            token_ids,
            batch_masks,
            sample_ids=batch_output_indices.tolist(),
        )
        save_image_batch(output_dir, batch_output_indices, result.images)
        processed += len(batch_output_indices)
    if processed != len(output_indices):
        raise RuntimeError(
            "Completion processed "
            f"{processed} local samples, expected {len(output_indices)}."
        )


def _prepare_completion_inputs(config, runtime, sampler):
    pretokenized_jsonl = inference_value(config, "pretokenized_jsonl", None)
    if pretokenized_jsonl is None:
        raise ValueError("inference.pretokenized_jsonl is required for completion.")
    dataset = PretokenizedDatasetJSONL(pretokenized_jsonl)
    source_record_index_value = inference_value(
        config,
        "source_record_index",
        None,
    )
    source_record_index = (
        None
        if source_record_index_value is None
        else _integer(
            source_record_index_value,
            "inference.source_record_index",
        )
    )
    if source_record_index is not None and not (
        0 <= source_record_index < len(dataset)
    ):
        raise ValueError(
            "inference.source_record_index must be in "
            f"[0, {len(dataset) - 1}], got {source_record_index}."
        )

    num_samples = _inference_int(config, "num_samples", len(dataset))
    if num_samples <= 0:
        raise ValueError(
            f"inference.num_samples must be positive, got {num_samples}."
        )
    if source_record_index is None and num_samples > len(dataset):
        raise ValueError(
            f"inference.num_samples must be no greater than {len(dataset)}"
            + f", got {num_samples}."
        )
    seed = _seed(config)
    output_indices = local_output_indices(
        num_samples,
        runtime.rank,
        runtime.world_size,
    )

    masks_path = inference_value(config, "masks", None)
    if masks_path is None:
        raise ValueError("inference.masks is required for completion.")
    masks = load_masks(masks_path, sampler.token_grid_shape)
    if len(masks) not in (1, num_samples):
        raise ValueError(
            "A mask file must contain one broadcast mask or one mask "
            f"per output ({num_samples}), got {len(masks)}."
        )

    return (
        dataset,
        source_record_index,
        num_samples,
        seed,
        output_indices,
        masks,
    )


def run_completion(
    config,
    runtime: DistributedRuntime,
    sampler: STARSampler,
) -> Path:
    (
        dataset,
        source_record_index,
        num_samples,
        seed,
        output_indices,
        masks,
    ) = run_rank_synchronized(
        runtime,
        lambda: _prepare_completion_inputs(config, runtime, sampler),
        "Completion input setup failed",
    )

    output_dir = prepare_output(config.experiment.output_dir, runtime)
    seed_everything(seed + runtime.rank)
    _run_completion_once(
        config,
        sampler,
        dataset,
        output_indices,
        output_dir,
        masks=masks,
        source_record_index=source_record_index,
    )

    # As in generation, only a completely successful set of ranks may enter
    # rank-zero post-processing. Mid-run exceptions retain their original
    # type and traceback and are left to torchrun's peer termination.
    runtime.barrier()
    error = None
    if runtime.is_main_process:
        try:
            create_npz(output_dir, num_samples)
        except Exception as exc:
            error = f"Completion post-processing failed: {type(exc).__name__}: {exc}"
    broadcast_error(error)
    return output_dir
