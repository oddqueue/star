from __future__ import annotations

import _imp
import multiprocessing
import os
import operator
import sys
import threading
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from star.sampling.traversal import (
    _order_worker_process_id,
    _sample_seeded_bfs_spanning_tree_order,
    _sample_seeded_postfix_bfs_order,
)
from .masks import normalize_token_grid_shape, validate_masks


_UINT32_MAX = (1 << 32) - 1
_GENERATION_SEED_DOMAIN = 0
_COMPLETION_SEED_DOMAIN = 1
_SPAWN_PATH_LOCK = threading.Lock()


def _integer(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def _uint32(value, name: str) -> int:
    value = _integer(value, name)
    if not 0 <= value <= _UINT32_MAX:
        raise ValueError(
            f"{name} must be between 0 and {_UINT32_MAX}, got {value!r}."
        )
    return value


def _sample_seed(
    base_seed: int,
    stream_id: int,
    sample_id: int,
    *,
    domain_id: int,
) -> int:
    entropy = [
        sample_id,
        stream_id,
        domain_id,
        base_seed,
    ]
    return int(
        np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0]
    )


def _is_installed_package_directory(entry: object) -> bool:
    if not isinstance(entry, (str, bytes, os.PathLike)):
        return False
    path = os.path.normpath(os.fsdecode(os.fspath(entry)))
    components = path.split(os.sep)
    return (
        ("site-packages" in components or "dist-packages" in components)
        and os.path.isfile(os.path.join(path, "__init__.py"))
    )


@contextmanager
def _clean_spawn_sys_path():
    _imp.acquire_lock()
    original = list(sys.path)
    try:
        sys.path[:] = [
            entry
            for entry in original
            if not _is_installed_package_directory(entry)
        ]
        try:
            yield
        finally:
            sys.path[:] = original
    finally:
        _imp.release_lock()


@dataclass(frozen=True)
class _PendingOrders:
    owner_id: int
    batch_size: int
    sequence_length: int
    futures: tuple[Future, ...]


class ParallelOrderSampler:
    def __init__(
        self,
        num_workers: int = 0,
        base_seed: int = 42,
        stream_id: int = 0,
    ) -> None:
        num_workers = _integer(num_workers, "num_workers")
        base_seed = _uint32(base_seed, "base_seed")
        stream_id = _uint32(stream_id, "stream_id")
        if num_workers < 0:
            raise ValueError(
                f"num_workers must be a non-negative integer, got {num_workers!r}."
            )
        self.num_workers = num_workers
        self.base_seed = base_seed
        self.stream_id = stream_id
        self._next_sample_ids = {
            "generation": 0,
            "completion": 0,
        }
        self._state_lock = threading.Lock()
        self._closed = False
        self._executor = None

        if self.num_workers:
            context = multiprocessing.get_context("spawn")
            self._executor = ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=context,
            )
            try:
                # ProcessPoolExecutor starts children lazily. Start every child
                # while it captures a clean import path. Spawn keeps this safe
                # even when the parent rank has already initialized CUDA.
                with _SPAWN_PATH_LOCK, _clean_spawn_sys_path():
                    ready = [
                        self._executor.submit(_order_worker_process_id)
                        for _ in range(self.num_workers)
                    ]
                    for future in ready:
                        future.result()
            except BaseException:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
                self._closed = True
                raise

    def __enter__(self) -> "ParallelOrderSampler":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    def _reserve_sample_ids(
        self,
        batch_size: int,
        sample_ids: Sequence[int] | None,
        *,
        kind: str,
    ) -> tuple[int, ...]:
        batch_size = _integer(batch_size, "batch_size")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}.")
        if sample_ids is not None:
            if len(sample_ids) != batch_size:
                raise ValueError(
                    "sample_ids must contain one ID per batch row, got "
                    f"{len(sample_ids)} IDs for batch_size={batch_size}."
                )
            normalized = tuple(
                _uint32(sample_id, "sample_id") for sample_id in sample_ids
            )
            return normalized

        with self._state_lock:
            if self._closed:
                raise RuntimeError("ParallelOrderSampler is closed.")
            start = self._next_sample_ids[kind]
            end = start + batch_size
            if end - 1 > _UINT32_MAX:
                raise ValueError(
                    f"Implicit sample IDs must not exceed {_UINT32_MAX}."
                )
            self._next_sample_ids[kind] = end
        return tuple(range(start, end))

    def _submit_tasks(
        self,
        tasks: Sequence[tuple],
        worker,
        *,
        sequence_length: int,
    ) -> _PendingOrders:
        # Keep close() from shutting down the executor between two rows of the
        # same logical batch. Serial sampling uses the same lock so submission
        # and shutdown have one unambiguous lifecycle contract.
        with self._state_lock:
            if self._closed:
                raise RuntimeError("ParallelOrderSampler is closed.")
            futures = []
            for task in tasks:
                if self._executor is None:
                    future = Future()
                    try:
                        future.set_result(worker(task))
                    except BaseException as exc:
                        future.set_exception(exc)
                else:
                    future = self._executor.submit(worker, task)
                futures.append(future)

        return _PendingOrders(
            owner_id=id(self),
            batch_size=len(tasks),
            sequence_length=sequence_length,
            futures=tuple(futures),
        )

    def submit_generation_orders(
        self,
        batch_size: int,
        token_grid_shape,
        *,
        sample_ids: Sequence[int] | None = None,
    ) -> _PendingOrders:
        height, width = normalize_token_grid_shape(token_grid_shape)
        sequence_length = height * width
        normalized_ids = self._reserve_sample_ids(
            batch_size,
            sample_ids,
            kind="generation",
        )
        tasks = []
        for slot, sample_id in enumerate(normalized_ids):
            tasks.append(
                (
                    slot,
                    sample_id,
                    height,
                    width,
                    _sample_seed(
                        self.base_seed,
                        self.stream_id,
                        sample_id,
                        domain_id=_GENERATION_SEED_DOMAIN,
                    ),
                )
            )
        return self._submit_tasks(
            tasks,
            _sample_seeded_bfs_spanning_tree_order,
            sequence_length=sequence_length,
        )

    def resolve(
        self,
        pending: _PendingOrders,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if not isinstance(pending, _PendingOrders):
            raise TypeError(
                "pending must be an order handle created by this sampler."
            )
        if pending.owner_id != id(self):
            raise ValueError("Pending orders belong to a different sampler instance.")

        rows = [None] * pending.batch_size
        for future in pending.futures:
            slot, order = future.result()
            if not 0 <= slot < pending.batch_size or rows[slot] is not None:
                raise RuntimeError(f"Order worker returned invalid batch slot {slot}.")
            if len(order) != pending.sequence_length:
                raise RuntimeError(
                    f"Order worker returned {len(order)} tokens; "
                    f"expected {pending.sequence_length}."
                )
            rows[slot] = order
        if any(row is None for row in rows):
            raise RuntimeError("One or more order workers returned no result.")

        orders = torch.tensor(rows, dtype=torch.long)
        if device is None:
            return orders
        target = torch.device(device)
        if target.type == "cpu":
            return orders.to(target)
        if target.type == "cuda" and torch.cuda.is_available():
            orders = orders.pin_memory()
        return orders.to(target, non_blocking=orders.is_pinned())

    def sample_generation_orders(
        self,
        batch_size: int,
        token_grid_shape,
        *,
        sample_ids: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        pending = self.submit_generation_orders(
            batch_size,
            token_grid_shape,
            sample_ids=sample_ids,
        )
        return self.resolve(pending, device=device)

    def sample_completion_orders(
        self,
        masks: Sequence[set[int]],
        token_grid_shape,
        *,
        sample_ids: Sequence[int] | None = None,
        device: torch.device | str | None = None,
        max_attempts: int = 1000,
    ) -> torch.Tensor:
        if not masks:
            raise ValueError("At least one completion mask is required.")
        height, width = normalize_token_grid_shape(token_grid_shape)
        sequence_length = height * width
        max_attempts = _integer(max_attempts, "max_attempts")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")
        normalized_masks = [set(mask) for mask in masks]
        validate_masks(normalized_masks, (height, width))
        normalized_ids = self._reserve_sample_ids(
            len(normalized_masks),
            sample_ids,
            kind="completion",
        )

        tasks = [
            (
                slot,
                sample_id,
                height,
                width,
                tuple(sorted(mask)),
                max_attempts,
                _sample_seed(
                    self.base_seed,
                    self.stream_id,
                    sample_id,
                    domain_id=_COMPLETION_SEED_DOMAIN,
                ),
            )
            for slot, (sample_id, mask) in enumerate(
                zip(normalized_ids, normalized_masks)
            )
        ]
        pending = self._submit_tasks(
            tasks,
            _sample_seeded_postfix_bfs_order,
            sequence_length=sequence_length,
        )
        return self.resolve(pending, device=device)
