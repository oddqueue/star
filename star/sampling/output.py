from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .runtime import DistributedRuntime, broadcast_error


def prepare_output(output_dir: str | Path, runtime: DistributedRuntime) -> Path:
    output_dir = Path(output_dir)
    error = None
    if runtime.is_main_process:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            error = f"Could not prepare output directory {output_dir}: {exc}"
    broadcast_error(error)
    runtime.barrier()
    return output_dir


def save_image_batch(
    output_dir: str | Path,
    output_indices: np.ndarray,
    images: np.ndarray,
) -> None:
    for index, image in zip(output_indices.tolist(), images):
        Image.fromarray(image).save(Path(output_dir) / f"{index:06d}.png")


def create_npz(output_dir: str | Path, num_samples: int = 50_000) -> Path:
    output_dir = Path(output_dir)
    samples = []
    for index in tqdm(range(num_samples), desc="Building .npz file from samples"):
        with Image.open(output_dir / f"{index:06d}.png") as image:
            samples.append(np.asarray(image).astype(np.uint8))
    samples = np.stack(samples)
    if (
        samples.ndim != 4
        or samples.shape[0] != num_samples
        or samples.shape[-1] != 3
    ):
        raise RuntimeError(
            f"Expected {num_samples} RGB images, got an array with shape "
            f"{samples.shape}."
        )
    npz_path = Path(f"{output_dir}.npz")
    np.savez(npz_path, arr_0=samples)
    return npz_path
