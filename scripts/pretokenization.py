from __future__ import annotations

import argparse
import datetime
import json
import numbers
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm.auto import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from star.modeling.tokenizer import PretrainedTokenizer
from star.utils.distributed import require_single_node
import star.utils.misc as misc
from star.utils.training_assets import (
    DEFAULT_TOKENIZER_FILENAME,
    DEFAULT_TOKENIZER_REPO_ID,
    resolve_training_asset,
)


_FINAL_JSONL_NAME = "pretokenized.jsonl"
_INTERNAL_RECORD_KEYS = {
    "_local_record_index",
    "source_index",
    "view_index",
    "source_path",
    "class_id",
    "tokens",
}


class _DistributedAssetContext:
    def __init__(self, args):
        self.local_process_index = getattr(args, "gpu", 0)

    @staticmethod
    def wait_for_everyone():
        if misc.is_dist_avail_and_initialized():
            dist.barrier()


def _image_shape(image_size):
    if isinstance(image_size, numbers.Integral) and not isinstance(
        image_size, bool
    ):
        dimensions = (int(image_size),)
    else:
        try:
            dimensions = tuple(image_size)
        except TypeError as exc:
            raise ValueError("--image-size requires one or two integers") from exc
    if len(dimensions) not in (1, 2):
        raise ValueError("--image-size requires one or two integers")
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, numbers.Integral)
        or dimension <= 0
        for dimension in dimensions
    ):
        raise ValueError("--image-size dimensions must be positive integers")
    if len(dimensions) == 1:
        return dimensions[0], dimensions[0]
    return int(dimensions[0]), int(dimensions[1])


def center_crop_arr(pil_image, image_size):
    target_height, target_width = _image_shape(image_size)
    while all(
        source >= 2 * target
        for source, target in zip(
            pil_image.size,
            (target_width, target_height),
        )
    ):
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = max(
        target_width / pil_image.size[0],
        target_height / pil_image.size[1],
    )
    pil_image = pil_image.resize(
        (
            max(target_width, round(pil_image.size[0] * scale)),
            max(target_height, round(pil_image.size[1] * scale)),
        ),
        resample=Image.BICUBIC,
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - target_height) // 2
    crop_x = (arr.shape[1] - target_width) // 2
    return Image.fromarray(
        arr[
            crop_y : crop_y + target_height,
            crop_x : crop_x + target_width,
        ]
    )


class _CenterCrop:
    def __init__(self, image_size):
        self.image_size = _image_shape(image_size)

    def __call__(self, image):
        return center_crop_arr(image, self.image_size)


class _StackCropsToTensor:
    def __call__(self, crops):
        return torch.stack([transforms.ToTensor()(crop) for crop in crops])


class ImageFolderWithSourceInfo(datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        source_path = Path(path).relative_to(self.root).as_posix()
        return sample, target, source_path, index


def get_args_parser():
    parser = argparse.ArgumentParser("Pretokenize ImageNet validation images")
    parser.add_argument(
        "--batch-size",
        default=128,
        type=int,
        help=(
            "source-image batch size per rank; TenCrop passes 10 times this "
            "many images to the tokenizer encoder"
        ),
    )

    parser.add_argument(
        "--image-size",
        default=(256,),
        nargs="+",
        type=int,
        help="input image size as SIZE or HEIGHT WIDTH",
    )
    parser.add_argument(
        "--data-path",
        default="./data/imagenet",
        type=str,
        help="ImageNet root containing val/",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help=(
            "device to use for tokenization; distributed bare cuda selects "
            "LOCAL_RANK, and an explicit cuda:N must match LOCAL_RANK"
        ),
    )
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--num-workers", default=10, type=int)
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient host-to-GPU copies.",
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_false",
        dest="pin_memory",
    )
    parser.set_defaults(pin_memory=True)

    parser.add_argument(
        "--output-dir",
        required=True,
        help="output directory for pretokenized.jsonl",
    )
    parser.add_argument(
        "--tokenizer-checkpoint",
        default=DEFAULT_TOKENIZER_FILENAME,
        help=(
            "local MaskGIT-VQ tokenizer checkpoint; when absent, download "
            f"{DEFAULT_TOKENIZER_REPO_ID}/{DEFAULT_TOKENIZER_FILENAME}"
        ),
    )

    crop_group = parser.add_mutually_exclusive_group(required=True)
    crop_group.add_argument(
        "--ten-crop",
        action="store_true",
        help="encode torchvision TenCrop views",
    )
    crop_group.add_argument(
        "--single-crop",
        action="store_true",
        help="encode one ADM center crop",
    )

    return parser


def _resolve_pretokenization_device(device_spec, *, distributed, local_rank):
    device = torch.device(device_spec)
    if not distributed or device.type != "cuda":
        return device

    if device.index is None:
        return torch.device("cuda", local_rank)
    if device.index != local_rank:
        raise ValueError(
            f"--device {device_spec!r} selects CUDA device {device.index}, but "
            f"distributed pretokenization assigned LOCAL_RANK={local_rank}; "
            f"use --device cuda or --device cuda:{local_rank}"
        )
    return device


def _crop_mode(args):
    if args.ten_crop:
        return "ten_crop"
    return "single_crop"


def _views_per_image(mode):
    return {"single_crop": 1, "ten_crop": 10}[mode]


def _build_transform(mode, image_size):
    image_size = _image_shape(image_size)
    if mode == "ten_crop":
        return transforms.Compose(
            [
                _CenterCrop(
                    tuple(int(dimension * 1.1) for dimension in image_size)
                ),
                transforms.TenCrop(image_size),
                _StackCropsToTensor(),
            ]
        )
    return transforms.Compose([_CenterCrop(image_size), transforms.ToTensor()])


def _expand_batch(samples, targets, source_indices, source_paths, mode):
    source_indices = [int(index) for index in source_indices]
    source_paths = list(source_paths)
    if mode == "ten_crop":
        samples_all = samples.flatten(0, 1)
        targets_all = targets.repeat_interleave(10)
        sources_all = [index for index in source_indices for _ in range(10)]
        paths_all = [path for path in source_paths for _ in range(10)]
        views_all = [view for _ in source_indices for view in range(10)]
    else:
        samples_all = samples
        targets_all = targets
        sources_all = source_indices
        paths_all = source_paths
        views_all = [0] * len(source_indices)
    return samples_all, targets_all, sources_all, paths_all, views_all


def _iter_internal_shard(path):
    previous_index = -1
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} at line {line_number}") from exc
            if not isinstance(record, dict) or set(record) != _INTERNAL_RECORD_KEYS:
                raise ValueError(
                    f"invalid internal record schema in {path} at line {line_number}"
                )
            local_record_index = record["_local_record_index"]
            if isinstance(local_record_index, bool) or not isinstance(
                local_record_index,
                int,
            ):
                raise ValueError(
                    f"non-integer record index in {path} at line {line_number}"
                )
            if local_record_index <= previous_index:
                raise ValueError(f"non-increasing record indices in {path}")
            previous_index = local_record_index
            yield local_record_index, record


def _distributed_source_indices(dataset_size, *, world_size, rank):
    sampler = torch.utils.data.DistributedSampler(
        range(dataset_size),
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    return list(sampler)


def _expected_rank_records(
    dataset_size, *, batch_size, mode, world_size, rank
):
    source_indices = _distributed_source_indices(
        dataset_size,
        world_size=world_size,
        rank=rank,
    )
    for start in range(0, len(source_indices), batch_size):
        batch = source_indices[start : start + batch_size]
        if mode == "ten_crop":
            yield from (
                (source_index, view_index)
                for source_index in batch
                for view_index in range(10)
            )
        else:
            yield from ((source_index, 0) for source_index in batch)


def _validate_internal_record(
    record,
    *,
    expected_local_record_index,
    expected_source_index,
    expected_view_index,
    dataset_samples,
    dataset_root,
    expected_token_count,
    codebook_size,
):
    if record["_local_record_index"] != expected_local_record_index:
        raise ValueError(
            "pretokenization shard contains a missing or duplicate local record: "
            f"expected {expected_local_record_index}, "
            f"got {record['_local_record_index']}"
        )
    if (
        record["source_index"] != expected_source_index
        or record["view_index"] != expected_view_index
    ):
        raise ValueError(
            f"record {expected_local_record_index} has inconsistent source/view identity"
        )
    expected_path, expected_class = dataset_samples[expected_source_index]
    expected_path = Path(expected_path).relative_to(dataset_root).as_posix()
    if record["source_path"] != expected_path:
        raise ValueError(
            f"record {expected_local_record_index} has an unexpected source path"
        )

    class_id = record["class_id"]
    if (
        isinstance(class_id, bool)
        or not isinstance(class_id, int)
        or class_id != int(expected_class)
        or class_id < 0
    ):
        raise ValueError(
            f"record {expected_local_record_index} has an invalid class_id"
        )
    tokens = record["tokens"]
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(
            f"record {expected_local_record_index} must contain a non-empty token list"
        )
    if expected_token_count is not None and len(tokens) != expected_token_count:
        raise ValueError(
            f"record {expected_local_record_index} contains {len(tokens)} tokens; "
            f"expected {expected_token_count}"
        )
    if any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or token < 0
        or token >= codebook_size
        for token in tokens
    ):
        raise ValueError(
            f"record {expected_local_record_index} contains an invalid token"
        )
    return len(tokens)


def _atomic_write_json(path, payload):
    path = Path(path)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.incomplete"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def merge_rank_shards(
    shard_paths,
    output_path,
    *,
    dataset_samples,
    dataset_root,
    batch_size,
    mode,
    codebook_size,
):
    output_path = Path(output_path)
    dataset_root = Path(dataset_root)
    shard_paths = [Path(path) for path in shard_paths]
    world_size = len(shard_paths)
    if world_size <= 0:
        raise ValueError("at least one rank shard is required")
    if (
        isinstance(codebook_size, bool)
        or not isinstance(codebook_size, numbers.Integral)
        or codebook_size <= 0
    ):
        raise ValueError("codebook_size must be a positive integer")
    codebook_size = int(codebook_size)
    expected_names = [f"rank-{rank:05d}.jsonl" for rank in range(world_size)]
    shard_paths.sort(key=lambda path: path.name)
    actual_names = [path.name for path in shard_paths]
    if actual_names != expected_names:
        raise ValueError(
            "rank shards must be a complete zero-based set: "
            f"expected {expected_names}, got {actual_names}"
        )
    samples_per_rank = len(
        _distributed_source_indices(
            len(dataset_samples),
            world_size=world_size,
            rank=0,
        )
    )
    sampler_total_size = samples_per_rank * world_size
    expected_count = sampler_total_size * _views_per_image(mode)
    temporary = output_path.parent / (
        f".{output_path.name}.{uuid.uuid4().hex}.incomplete"
    )
    written = 0
    expected_token_count = None
    try:
        with temporary.open("x", encoding="utf-8") as output:
            for rank, shard_path in enumerate(shard_paths):
                expected_records = iter(
                    _expected_rank_records(
                        len(dataset_samples),
                        batch_size=batch_size,
                        mode=mode,
                        world_size=world_size,
                        rank=rank,
                    )
                )
                rank_written = 0
                for local_record_index, record in _iter_internal_shard(shard_path):
                    try:
                        source_index, view_index = next(expected_records)
                    except StopIteration as exc:
                        raise ValueError(
                            f"rank {rank} shard has more records than expected"
                        ) from exc
                    token_count = _validate_internal_record(
                        record,
                        expected_local_record_index=rank_written,
                        expected_source_index=source_index,
                        expected_view_index=view_index,
                        dataset_samples=dataset_samples,
                        dataset_root=dataset_root,
                        expected_token_count=expected_token_count,
                        codebook_size=codebook_size,
                    )
                    if expected_token_count is None:
                        expected_token_count = token_count
                    public_record = {
                        "class_id": record["class_id"],
                        "tokens": record["tokens"],
                    }
                    line = json.dumps(public_record) + "\n"
                    output.write(line)
                    rank_written += 1
                    written += 1
                try:
                    next(expected_records)
                except StopIteration:
                    pass
                else:
                    raise ValueError(
                        f"rank {rank} shard has fewer records than expected"
                    )
            if written != expected_count:
                raise ValueError(
                    f"expected {expected_count} records, merged {written}"
                )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"records": written}


def _distributed_value_from_rank_zero(value, rank):
    if not misc.is_dist_avail_and_initialized():
        return value
    payload = [value if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _write_rank_shard(
    *,
    shard_path,
    data_loader,
    tokenizer,
    device,
    mode,
):
    temporary = shard_path.with_suffix(shard_path.suffix + ".incomplete")
    processed = 0
    token_grid_shape = None
    try:
        with temporary.open("x", encoding="utf-8") as output:
            for samples, targets, source_paths, source_indices in tqdm(data_loader):
                samples_all, targets_all, sources_all, paths_all, views_all = (
                    _expand_batch(
                        samples,
                        targets,
                        source_indices,
                        source_paths,
                        mode,
                    )
                )
                samples_all = samples_all.to(device, non_blocking=True)
                token_ids = tokenizer.encode(samples_all)
                expanded_rows = len(sources_all)
                if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 3:
                    raise ValueError(
                        "tokenizer output must have shape "
                        "[batch, grid_height, grid_width]"
                    )
                token_ids = token_ids.cpu()
                if (
                    token_ids.dtype == torch.bool
                    or token_ids.is_floating_point()
                    or token_ids.is_complex()
                ):
                    raise ValueError("tokenizer output must contain integer token IDs")
                current_grid_shape = tuple(token_ids.shape[1:])
                if any(dimension <= 0 for dimension in current_grid_shape):
                    raise ValueError("tokenizer output grid dimensions must be positive")
                if token_grid_shape is None:
                    token_grid_shape = current_grid_shape
                elif current_grid_shape != token_grid_shape:
                    raise ValueError(
                        "tokenizer output grid shape changed between batches: "
                        f"{token_grid_shape} != {current_grid_shape}"
                    )
                if not (
                    token_ids.shape[0]
                    == expanded_rows
                    == len(targets_all)
                    == len(paths_all)
                    == len(views_all)
                ):
                    raise ValueError(
                        "tokenizer output and expanded batch metadata have "
                        "different lengths"
                    )
                flat_token_ids = token_ids.reshape(expanded_rows, -1)
                for position in range(expanded_rows):
                    record = {
                        "_local_record_index": processed,
                        "source_index": sources_all[position],
                        "view_index": views_all[position],
                        "source_path": paths_all[position],
                        "class_id": int(targets_all[position]),
                        "tokens": flat_token_ids[position].tolist(),
                    }
                    output.write(json.dumps(record) + "\n")
                    processed += 1
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, shard_path)
    finally:
        temporary.unlink(missing_ok=True)
    return processed


def _write_rank_shard_and_wait_for_success(
    *,
    shard_path,
    data_loader,
    tokenizer,
    device,
    mode,
):
    processed = _write_rank_shard(
        shard_path=shard_path,
        data_loader=data_loader,
        tokenizer=tokenizer,
        device=device,
        mode=mode,
    )
    if misc.is_dist_avail_and_initialized():
        dist.barrier()
    return processed


@torch.no_grad()
def main(args):
    if args.output_dir is None or not str(args.output_dir).strip():
        raise ValueError("--output-dir must not be empty")
    image_size = _image_shape(args.image_size)
    require_single_node()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    misc.init_distributed_mode(args)

    rank = misc.get_rank()
    world_size = misc.get_world_size()
    distributed = bool(getattr(args, "distributed", False) or world_size > 1)
    local_rank = int(getattr(args, "gpu", os.environ.get("LOCAL_RANK", 0)))
    device = _resolve_pretokenization_device(
        args.device,
        distributed=distributed,
        local_rank=local_rank,
    )
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)

    mode = _crop_mode(args)
    transform = _build_transform(mode, image_size)

    dataset = ImageFolderWithSourceInfo(
        os.path.join(args.data_path, "val"), transform=transform
    )
    print(dataset)
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    print("Sampler = %s" % str(sampler))
    data_loader_arguments = {}
    if args.num_workers > 0:
        # Distributed initialization has already created a CUDA context. Spawn
        # fresh CPU-only workers instead of forking that accelerator state.
        data_loader_arguments["multiprocessing_context"] = "spawn"
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        **data_loader_arguments,
    )

    tokenizer_checkpoint = resolve_training_asset(
        args.tokenizer_checkpoint,
        repo_id=DEFAULT_TOKENIZER_REPO_ID,
        filename=DEFAULT_TOKENIZER_FILENAME,
        accelerator=_DistributedAssetContext(args),
    )
    tokenizer = PretrainedTokenizer(tokenizer_checkpoint)
    tokenizer.to(device)

    final_path = output_dir / _FINAL_JSONL_NAME
    lock_path = output_dir / ".pretokenization.lock"
    run_id = _distributed_value_from_rank_zero(uuid.uuid4().hex, rank)
    preflight_error = None
    if rank == 0:
        if final_path.exists():
            preflight_error = (
                "refusing to overwrite existing pretokenization output: "
                f"{final_path.name}"
            )
        elif lock_path.exists():
            preflight_error = (
                f"another or incomplete pretokenization run owns {lock_path}; "
                "inspect it before removing the lock"
            )
        else:
            try:
                lock_path.mkdir()
                _atomic_write_json(
                    lock_path / "owner.json",
                    {"run_id": run_id, "world_size": world_size},
                )
            except Exception as exc:
                preflight_error = (
                    f"could not acquire pretokenization output lock: "
                    f"{type(exc).__name__}: {exc}"
                )
    preflight_error = _distributed_value_from_rank_zero(preflight_error, rank)
    if preflight_error is not None:
        raise FileExistsError(preflight_error)

    staging_dir = output_dir / f".pretokenization-{run_id}.incomplete"
    staging_error = None
    if rank == 0:
        try:
            staging_dir.mkdir()
        except Exception as exc:
            staging_error = (
                f"could not create staging directory: {type(exc).__name__}: {exc}"
            )
    staging_error = _distributed_value_from_rank_zero(staging_error, rank)
    if staging_error is not None:
        raise RuntimeError(staging_error)
    if misc.is_dist_avail_and_initialized():
        dist.barrier()
    shard_path = staging_dir / f"rank-{rank:05d}.jsonl"

    print(f"Start pretokenizing token IDs on rank {rank}")
    start_time = time.time()
    # Synchronize only successful shard writers. Encoding failures propagate
    # immediately so torchrun terminates peer ranks, and rank zero cannot merge
    # or publish a final JSONL unless every shard completed successfully.
    processed = _write_rank_shard_and_wait_for_success(
        shard_path=shard_path,
        data_loader=data_loader,
        tokenizer=tokenizer,
        device=device,
        mode=mode,
    )
    print(f"rank {rank} processed {processed} records")

    # A train-scale ten-crop merge can take longer than the process-group
    # collective timeout. All shards are durable now, so every rank closes the
    # group and rank zero performs the CPU-only merge without making the other
    # ranks wait inside an NCCL collective.
    if misc.is_dist_avail_and_initialized():
        dist.destroy_process_group()
    if rank != 0:
        return

    try:
        shard_paths = [
            staging_dir / f"rank-{shard_rank:05d}.jsonl"
            for shard_rank in range(world_size)
        ]
        missing = [str(path) for path in shard_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing completed rank shards: {missing}")

        staged_final_path = staging_dir / _FINAL_JSONL_NAME
        merge_rank_shards(
            shard_paths,
            staged_final_path,
            dataset_samples=dataset.samples,
            dataset_root=dataset.root,
            batch_size=args.batch_size,
            mode=mode,
            codebook_size=tokenizer.codebook_size,
        )
        if final_path.exists():
            raise FileExistsError(
                "pretokenization output appeared while the run held its lock"
            )
        os.replace(staged_final_path, final_path)
    except Exception as exc:
        raise RuntimeError(
            f"rank 0 merge: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        shutil.rmtree(staging_dir)
    except OSError as exc:
        print(f"warning: could not remove staging directory {staging_dir}: {exc}")
    try:
        shutil.rmtree(lock_path)
    except OSError as exc:
        print(f"warning: could not remove output lock {lock_path}: {exc}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Pretokenization time {total_time_str}")


if __name__ == "__main__":
    parsed_args = get_args_parser().parse_args()
    main(parsed_args)
