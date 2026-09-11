from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, TypeVar

import numpy as np
import torch
import torch.distributed as dist


_T = TypeVar("_T")


@dataclass(frozen=True)
class DistributedRuntime:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    initialized_here: bool

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier(device_ids=[self.local_rank])

    def close(self, *, graceful: bool = True) -> None:
        if graceful and self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def configure_torch() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("STAR sampling requires a CUDA device.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)


def initialize_runtime() -> DistributedRuntime:
    launched_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible."
        )
    torch.cuda.set_device(local_rank)

    initialized_here = False
    if launched_distributed and not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        initialized_here = True

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return DistributedRuntime(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=torch.device("cuda", local_rank),
        initialized_here=initialized_here,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def local_output_indices(
    num_samples: int,
    rank: int,
    world_size: int,
) -> np.ndarray:
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("Invalid distributed rank/world_size pair.")
    return np.arange(rank, num_samples, world_size, dtype=np.int64)


def broadcast_error(error: str | None) -> None:
    if dist.is_initialized():
        payload = [error]
        dist.broadcast_object_list(payload, src=0)
        error = payload[0]
    if error is not None:
        raise RuntimeError(error)


def raise_if_any_rank_failed(
    runtime: DistributedRuntime, error: str | None
) -> None:
    if dist.is_initialized():
        errors = [None] * runtime.world_size
        dist.all_gather_object(errors, error)
    else:
        errors = [error]
    for rank, rank_error in enumerate(errors):
        if rank_error is not None:
            raise RuntimeError(f"Sampling failed on rank {rank}: {rank_error}")


def run_rank_synchronized(
    runtime: DistributedRuntime,
    operation: Callable[[], _T],
    description: str,
) -> _T:
    if runtime.world_size == 1:
        return operation()
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            "A multi-rank sampling runtime requires an initialized process group."
        )

    result = None
    local_error = None
    try:
        result = operation()
    except Exception as exc:
        local_error = f"{description}: {type(exc).__name__}: {exc}"
    raise_if_any_rank_failed(runtime, local_error)
    return result
