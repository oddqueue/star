from __future__ import annotations

import os


_MULTI_NODE_COUNT_VARIABLES = (
    "GROUP_WORLD_SIZE",
    "NNODES",
    "NUM_MACHINES",
    "SLURM_NNODES",
    "SLURM_JOB_NUM_NODES",
)
_NONZERO_NODE_RANK_VARIABLES = (
    "GROUP_RANK",
    "NODE_RANK",
    "MACHINE_RANK",
    "SLURM_NODEID",
)
_WORLD_AND_LOCAL_SIZE_VARIABLES = (
    ("WORLD_SIZE", "LOCAL_WORLD_SIZE"),
    ("OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE"),
)


def _environment_integer(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _multi_node_launch_reason() -> str | None:
    for name in _MULTI_NODE_COUNT_VARIABLES:
        value = _environment_integer(name)
        if value is not None and value > 1:
            return f"{name}={value}"

    for name in _NONZERO_NODE_RANK_VARIABLES:
        value = _environment_integer(name)
        if value is not None and value > 0:
            return f"{name}={value}"

    for world_name, local_name in _WORLD_AND_LOCAL_SIZE_VARIABLES:
        world_size = _environment_integer(world_name)
        local_world_size = _environment_integer(local_name)
        if (
            world_size is not None
            and local_world_size is not None
            and local_world_size > 0
            and world_size > local_world_size
        ):
            return (
                f"{world_name}={world_size} exceeds "
                f"{local_name}={local_world_size}"
            )
    return None


def require_single_node() -> None:
    reason = _multi_node_launch_reason()
    if reason is not None:
        raise RuntimeError(
            "Multi-node execution is not supported by this STAR release "
            f"(detected {reason}). Use a single node; single-node multi-GPU "
            "execution remains supported."
        )
