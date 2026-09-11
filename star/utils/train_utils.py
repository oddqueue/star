import json
import math
import numbers
import os
import random
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader

from star.data import PretokenizedDatasetJSONL
from star.modeling.losses import ARLoss
from star.modeling.generator import STARGenerator
from star.modeling.tokenizer import PretrainedTokenizer
from star.sampling.core import sample_fn
from star.sampling.orders import ParallelOrderSampler
from star.utils.checkpoint_loading import load_raw_state_dict
from star.utils.viz_utils import make_generation_preview_grid


_EXACT_RESUME_SCHEMA_VERSION = 2
_CHECKPOINT_COMPLETE_FILENAME = "COMPLETE"
_CHECKPOINT_PHASE = "optimizer_step_complete_before_next_batch"
_CLI_ONLY_OVERRIDE_PATHS = {
    ("dataset", "params", "jsonl_index_cache_dir"),
    ("experiment", "generator_init_checkpoint"),
    ("training", "num_order_workers"),
}
_UINT32_MAX = (1 << 32) - 1
_POSITIVE_INTEGER_CONFIG_PATHS = (
    "training.per_process_batch_size",
    "training.gradient_accumulation_steps",
    "training.max_train_steps",
    "experiment.save_every",
    "experiment.generate_every",
    "experiment.log_every",
    "experiment.log_grad_norm_every",
)
_OPTIONAL_NONNEGATIVE_INTEGER_CONFIG_PATHS = (
    "dataset.params.num_dataloader_workers",
    "training.num_order_workers",
)


def _unknown_cli_override_paths(cli_conf, yaml_conf, prefix=()):
    unknown_paths = []
    for key, cli_value in cli_conf.items():
        key = str(key)
        path = (*prefix, key)
        if not prefix and key == "config":
            continue
        if path in _CLI_ONLY_OVERRIDE_PATHS:
            continue
        if not OmegaConf.is_dict(yaml_conf) or key not in yaml_conf:
            unknown_paths.append(".".join(path))
            continue
        yaml_value = yaml_conf[key]
        if OmegaConf.is_dict(cli_value):
            if not OmegaConf.is_dict(yaml_value):
                unknown_paths.append(".".join(path))
                continue
            unknown_paths.extend(
                _unknown_cli_override_paths(cli_value, yaml_value, path)
            )
    return unknown_paths


def get_config():
    cli_conf = OmegaConf.from_cli()
    if "config" not in cli_conf:
        raise ValueError(
            "A training config is required, for example config=configs/star_b.yaml."
        )
    yaml_conf = OmegaConf.load(cli_conf.config)
    unknown_paths = _unknown_cli_override_paths(cli_conf, yaml_conf)
    if unknown_paths:
        raise ValueError(
            "Unknown training CLI override(s): "
            + ", ".join(sorted(unknown_paths))
        )
    config = OmegaConf.merge(yaml_conf, cli_conf)
    _validate_training_config(config)
    return config


def _validate_training_config(config):
    for path in _POSITIVE_INTEGER_CONFIG_PATHS:
        value = OmegaConf.select(config, path)
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or value <= 0
        ):
            raise ValueError(
                f"{path} must be a positive integer, got {value!r}."
            )

    for path in _OPTIONAL_NONNEGATIVE_INTEGER_CONFIG_PATHS:
        value = OmegaConf.select(config, path)
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or value < 0
        ):
            raise ValueError(
                f"{path} must be a non-negative integer, got {value!r}."
            )

    seed = OmegaConf.select(config, "training.seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, numbers.Integral)
        or not 0 <= seed <= _UINT32_MAX
    ):
        raise ValueError(
            "training.seed must be an integer between 0 and "
            f"{_UINT32_MAX}, got {seed!r}."
        )

    max_grad_norm = OmegaConf.select(config, "training.max_grad_norm")
    if max_grad_norm is not None and (
        isinstance(max_grad_norm, bool)
        or not isinstance(max_grad_norm, numbers.Real)
        or not math.isfinite(max_grad_norm)
        or max_grad_norm <= 0
    ):
        raise ValueError(
            "training.max_grad_norm must be null or a positive finite number, "
            f"got {max_grad_norm!r}."
        )

    class_label_dropout = OmegaConf.select(
        config,
        "model.generator.class_label_dropout",
    )
    if (
        isinstance(class_label_dropout, bool)
        or not isinstance(class_label_dropout, numbers.Real)
        or not math.isfinite(class_label_dropout)
        or not 0 <= class_label_dropout <= 1
    ):
        raise ValueError(
            "model.generator.class_label_dropout must be a finite number between "
            f"0 and 1, got {class_label_dropout!r}."
        )


class ValueMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val


class ExactResumeState:
    def __init__(self, manifest):
        self.manifest = manifest
        self.global_step = 0
        self.consumed_batches = 0

    def state_dict(self):
        return {
            "schema_version": _EXACT_RESUME_SCHEMA_VERSION,
            "global_step": self.global_step,
            "consumed_batches": self.consumed_batches,
            "manifest": self.manifest,
        }

    def validate_state_dict(self, state_dict):
        if not isinstance(state_dict, Mapping):
            raise ValueError("STAR exact-resume state must be a mapping.")
        if state_dict.get("schema_version") != _EXACT_RESUME_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported STAR exact-resume schema: "
                f"{state_dict.get('schema_version')!r}."
            )
        saved_manifest = state_dict.get("manifest")
        differing_keys = _manifest_mismatches(self.manifest, saved_manifest)
        if differing_keys:
            raise ValueError(
                "Checkpoint is incompatible with this exact-resume runtime. "
                "Differing manifest fields: " + ", ".join(differing_keys)
            )
        global_step = state_dict.get("global_step")
        consumed_batches = state_dict.get("consumed_batches")
        for value, name in (
            (global_step, "global_step"),
            (consumed_batches, "consumed_batches"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Checkpoint {name} must be a non-negative integer."
                )

    def load_state_dict(self, state_dict):
        self.validate_state_dict(state_dict)
        self.global_step = state_dict["global_step"]
        self.consumed_batches = state_dict["consumed_batches"]


def _manifest_mismatches(expected, saved, prefix=""):
    if isinstance(expected, Mapping):
        if not isinstance(saved, Mapping):
            return [prefix or "manifest"]
        differing_keys = []
        for key in sorted(set(expected) | set(saved), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected or key not in saved:
                differing_keys.append(path)
                continue
            differing_keys.extend(
                _manifest_mismatches(expected[key], saved[key], path)
            )
        return differing_keys
    if saved != expected:
        return [prefix or "manifest"]
    return []


def build_exact_resume_manifest(
    config,
    accelerator,
    train_dataloader,
    *,
    data_seed,
):
    return {
        "schema_version": _EXACT_RESUME_SCHEMA_VERSION,
        "checkpoint_phase": _CHECKPOINT_PHASE,
        "world_size": int(accelerator.num_processes),
        "num_batches_per_epoch": len(train_dataloader),
        "per_process_batch_size": int(
            config.training.per_process_batch_size
        ),
        "gradient_accumulation_steps": int(
            config.training.gradient_accumulation_steps
        ),
        "split_batches": False,
        "drop_last": True,
        "data_seed": int(data_seed),
        "dataset": {"num_records": len(train_dataloader.dataset)},
        "order_sampler": {
            "algorithm": "bfs_spanning_tree",
            "base_seed": int(data_seed),
            "token_grid_shape": [
                int(value)
                for value in config.model.generator.token_grid_shape
            ],
            "stream_id": "accelerator.process_index",
            "sample_id_scheme": "consumed_local_batches_times_batch_size",
        },
    }


def set_training_dataloader_epoch(dataloader, epoch, data_seed):
    if hasattr(dataloader, "set_epoch"):
        dataloader.set_epoch(epoch)
    generator = getattr(dataloader, "generator", None)
    if generator is not None:
        generator.manual_seed(int(data_seed) + int(epoch))


@contextmanager
def preserve_training_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def log_model_summary(model, logger, accelerator):
    if not accelerator.is_main_process:
        return

    from torchinfo import summary

    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    token_ids = (
        torch.arange(model.image_seq_len, dtype=torch.long, device=device)
        .remainder(model.target_codebook_size)
        .unsqueeze(0)
    )
    class_ids = torch.zeros(1, dtype=torch.long, device=device)
    height, width = model.token_grid_shape
    if height * width != model.image_seq_len:
        raise ValueError("The STAR token grid does not match image_seq_len.")
    # Fixed corner-rooted BFS over a spanning tree whose top row is the trunk
    # and whose remaining nodes attach vertically. Within one BFS depth, visit
    # the lower-column node first.
    bfs_order = sorted(
        range(model.image_seq_len),
        key=lambda index: (index // width + index % width, index % width),
    )
    orders = torch.tensor(bfs_order, dtype=torch.long, device=device).unsqueeze(0)
    was_training = model.training
    try:
        with preserve_training_rng_state(), torch.no_grad():
            condition_ids = model.preprocess_condition(
                class_ids,
                cond_drop_prob=0.0,
            )
            model_summary = summary(
                model,
                input_data=(token_ids, condition_ids, orders),
                depth=7,
                col_names=(
                    "input_size",
                    "output_size",
                    "num_params",
                    "params_percent",
                    "kernel_size",
                    "mult_adds",
                ),
                verbose=0,
            )
            logger.info(model_summary)
    finally:
        model.train(was_training)


def create_pretrained_tokenizer(config, accelerator):
    pretrained_tokenizer = PretrainedTokenizer(
        config.model.vq_model.tokenizer_checkpoint
    )
    configured_codebook_size = int(config.model.vq_model.codebook_size)
    if pretrained_tokenizer.codebook_size != configured_codebook_size:
        raise ValueError(
            "Generator and tokenizer codebook sizes do not match: "
            f"{configured_codebook_size} != "
            f"{pretrained_tokenizer.codebook_size}."
        )
    pretrained_tokenizer.to(accelerator.device)
    return pretrained_tokenizer


def create_model_and_loss_module(config, logger, accelerator):
    logger.info("Creating model and loss module.")
    model = STARGenerator(config)

    if config.experiment.get("generator_init_checkpoint", ""):
        checkpoint_path = config.experiment.generator_init_checkpoint
        state_dict = load_raw_state_dict(checkpoint_path)
        load_result = model.load_state_dict(state_dict, strict=False)
        logger.info(
            "Loaded generator initialization checkpoint from "
            f"{checkpoint_path}: {load_result}"
        )

    loss_module = ARLoss(config=config)
    log_model_summary(model, logger, accelerator)
    return model, loss_module


def create_optimizer(config, logger, model):
    logger.info("Creating optimizer.")
    optimizer_config = config.optimizer.params

    def exclude(name, parameter):
        return (
            parameter.ndim < 2
            or "ln" in name
            or "bias" in name
            or "norm" in name
            or "embed" in name
        )

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [
        parameter
        for name, parameter in named_parameters
        if exclude(name, parameter) and parameter.requires_grad
    ]
    rest_params = [
        parameter
        for name, parameter in named_parameters
        if not exclude(name, parameter) and parameter.requires_grad
    ]
    optimizer = AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {
                "params": rest_params,
                "weight_decay": optimizer_config.weight_decay,
            },
        ],
        lr=optimizer_config.learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
    )
    return optimizer


def create_lr_scheduler(config, logger, accelerator, optimizer):
    logger.info("Creating learning-rate scheduler.")
    num_training_steps = (
        config.training.max_train_steps * accelerator.num_processes
    )
    num_warmup_steps = (
        config.lr_scheduler.params.warmup_steps
        * accelerator.num_processes
    )
    base_lr = config.optimizer.params.learning_rate
    end_lr = config.lr_scheduler.params.end_lr

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        ratio = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return (end_lr + (base_lr - end_lr) * ratio) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_dataloader(config, logger):
    logger.info("Creating dataloaders.")
    dataset_config = config.dataset.params
    pretokenized_jsonl = dataset_config.get("pretokenized_jsonl", "")
    if not pretokenized_jsonl:
        raise ValueError("STAR training requires a pretokenized JSONL dataset.")

    num_workers = dataset_config.get("num_dataloader_workers", 0)
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise TypeError(
            "dataset.params.num_dataloader_workers must be an integer."
        )
    if num_workers < 0:
        raise ValueError(
            "dataset.params.num_dataloader_workers cannot be negative."
        )
    data_seed = config.training.get("seed")
    if isinstance(data_seed, bool) or not isinstance(data_seed, int):
        raise ValueError(
            "STAR exact-resume training requires training.seed to be an integer."
        )
    data_generator = torch.Generator()
    data_generator.manual_seed(data_seed)
    train_dataloader = DataLoader(
        PretokenizedDatasetJSONL(
            pretokenized_jsonl,
            index_cache_dir=dataset_config.get("jsonl_index_cache_dir"),
        ),
        batch_size=config.training.per_process_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        generator=data_generator,
    )
    return train_dataloader


def auto_resume(
    config,
    logger,
    accelerator,
    resume_state,
    num_batches_per_epoch,
):
    first_epoch = 0
    first_batch = 0
    accelerator.wait_for_everyone()
    output_dir = Path(config.experiment.output_dir)
    if config.experiment.resume:
        complete_checkpoints = []
        non_exact_checkpoints = []
        for path in output_dir.glob("checkpoint-*"):
            if not path.is_dir():
                continue
            try:
                checkpoint_step = int(path.name.removeprefix("checkpoint-"))
            except ValueError:
                continue
            if checkpoint_step < 0:
                continue
            if (path / _CHECKPOINT_COMPLETE_FILENAME).is_file():
                complete_checkpoints.append((checkpoint_step, path))
            else:
                non_exact_checkpoints.append((checkpoint_step, path))
        logger.info(
            "Complete checkpoints are: "
            f"{[str(path) for _, path in complete_checkpoints]}"
        )
        if not complete_checkpoints and non_exact_checkpoints:
            _, highest_non_exact = max(
                non_exact_checkpoints,
                key=lambda item: item[0],
            )
            raise RuntimeError(
                "Exact auto-resume found only non-exact checkpoints; "
                "it will not perform a coarse resume. If the model file is valid, "
                "start a new run with experiment.generator_init_checkpoint="
                f"{highest_non_exact / 'unwrapped_model' / 'pytorch_model.bin'}."
            )
        if complete_checkpoints:
            checkpoint_step, checkpoint_path = max(
                complete_checkpoints,
                key=lambda item: item[0],
            )
            load_checkpoint(
                checkpoint_path,
                accelerator,
                logger=logger,
                resume_state=resume_state,
            )
            newer_non_exact = [
                (step, path)
                for step, path in non_exact_checkpoints
                if step > checkpoint_step
            ]
            if newer_non_exact and accelerator.is_main_process:
                _, highest_ignored = max(
                    newer_non_exact,
                    key=lambda item: item[0],
                )
                logger.warning(
                    "Resumed from the latest complete exact checkpoint "
                    f"{checkpoint_path}; ignored newer non-exact checkpoint "
                    f"{highest_ignored}."
                )
            first_epoch, first_batch = divmod(
                resume_state.consumed_batches,
                num_batches_per_epoch,
            )
        else:
            logger.info("Training from scratch.")
    else:
        existing_checkpoints = sorted(
            path for path in output_dir.glob("checkpoint-*") if path.is_dir()
        )
        if existing_checkpoints:
            raise RuntimeError(
                "Refusing to start a fresh run in an output directory that "
                "already contains checkpoints: "
                + ", ".join(str(path) for path in existing_checkpoints)
            )
    return resume_state.global_step, first_epoch, first_batch


def training_target_reached(global_step, max_train_steps, logger, accelerator):
    if global_step < max_train_steps:
        return False
    if accelerator.is_main_process:
        if global_step == max_train_steps:
            logger.info(
                "The resumed checkpoint already reached the requested training "
                f"target ({global_step} steps); skipping training."
            )
        else:
            logger.warning(
                "The resumed checkpoint is beyond the requested training target "
                f"({global_step} > {max_train_steps}); skipping training."
            )
    return True


def train_one_epoch_generator(
    config,
    logger,
    accelerator,
    model,
    loss_module,
    optimizer,
    lr_scheduler,
    train_dataloader,
    tokenizer,
    global_step,
    order_sampler: ParallelOrderSampler,
    training_state: ExactResumeState,
):
    if order_sampler is None:
        raise ValueError("STAR training requires an explicit order_sampler.")
    if training_state.global_step != global_step:
        raise ValueError(
            "global_step does not match the registered exact-resume state: "
            f"{global_step} != {training_state.global_step}."
        )
    batch_time_meter = ValueMeter()
    data_time_meter = ValueMeter()
    end = time.time()
    model.train()

    use_order_prefetch = getattr(order_sampler, "num_workers", 0) > 0
    pending_orders = None
    num_train_batches = len(train_dataloader) if use_order_prefetch else None

    for batch_index, batch in enumerate(train_dataloader):
        class_ids, token_ids = batch
        token_ids = token_ids.to(
            accelerator.device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )
        class_ids = class_ids.to(
            accelerator.device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )
        data_time_meter.update(time.time() - end)

        unwrap_model = accelerator.unwrap_model(model)
        expected_token_shape = (class_ids.shape[0], unwrap_model.image_seq_len)
        if token_ids.ndim != 2 or tuple(token_ids.shape) != expected_token_shape:
            raise ValueError(
                "Training token_ids must have shape "
                f"{expected_token_shape}, got {tuple(token_ids.shape)}."
            )

        with accelerator.accumulate(model):
            condition_ids = unwrap_model.preprocess_condition(
                class_ids,
                cond_drop_prob=config.model.generator.class_label_dropout,
            )
            current_batch_size = token_ids.shape[0]
            configured_batch_size = int(config.training.per_process_batch_size)
            if current_batch_size != configured_batch_size:
                raise RuntimeError(
                    "Exact UST sample IDs require every per-rank training "
                    "batch to match training.per_process_batch_size: "
                    f"{current_batch_size} != {configured_batch_size}."
                )
            sample_id_start = training_state.consumed_batches * current_batch_size
            order_sample_ids = range(
                sample_id_start,
                sample_id_start + current_batch_size,
            )
            next_global_step = global_step + 1
            will_save_checkpoint = (
                accelerator.sync_gradients
                and next_global_step % config.experiment.save_every == 0
            )
            if use_order_prefetch:
                if pending_orders is None:
                    pending_orders = order_sampler.submit_generation_orders(
                        current_batch_size,
                        unwrap_model.token_grid_shape,
                        sample_ids=order_sample_ids,
                    )
                orders = order_sampler.resolve(
                    pending_orders,
                    device=token_ids.device,
                )
                pending_orders = None
                reaches_max_steps = (
                    accelerator.sync_gradients
                    and next_global_step >= config.training.max_train_steps
                )
                if (
                    batch_index + 1 < num_train_batches
                    and not reaches_max_steps
                    and not will_save_checkpoint
                ):
                    next_sample_id_start = (
                        (training_state.consumed_batches + 1)
                        * current_batch_size
                    )
                    next_sample_ids = range(
                        next_sample_id_start,
                        next_sample_id_start + current_batch_size,
                    )
                    pending_orders = order_sampler.submit_generation_orders(
                        current_batch_size,
                        unwrap_model.token_grid_shape,
                        sample_ids=next_sample_ids,
                    )
            else:
                orders = order_sampler.sample_generation_orders(
                    token_ids.shape[0],
                    unwrap_model.token_grid_shape,
                    sample_ids=order_sample_ids,
                    device=token_ids.device,
                )
            logits, target_token_ids = model(
                token_ids,
                condition_ids,
                orders,
            )
            loss, loss_dict = loss_module(logits, target_token_ids)

            gen_logs = {}
            for key, value in loss_dict.items():
                gen_logs["train/" + key] = accelerator.gather(value).mean().item()
            accelerator.backward(loss)

            if (
                config.training.max_grad_norm is not None
                and accelerator.sync_gradients
            ):
                accelerator.clip_grad_norm_(
                    model.parameters(), config.training.max_grad_norm
                )

            optimizer.step()
            lr_scheduler.step()

            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

        training_state.consumed_batches += 1

        if accelerator.sync_gradients:
            global_step += 1
            training_state.global_step = global_step
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if global_step % config.experiment.log_every == 0:
                samples_per_second_per_process = (
                    config.training.gradient_accumulation_steps
                    * config.training.per_process_batch_size
                    / batch_time_meter.val
                )
                lr = lr_scheduler.get_last_lr()[0]
                logger.info(
                    f"Data (t): {data_time_meter.val:0.4f}, "
                    f"{samples_per_second_per_process:0.2f}/s/process "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step} "
                    f"Loss: {gen_logs['train/loss']:0.4f} "
                    f"Accuracy: {gen_logs['train/correct_tokens']:0.4f} "
                )
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/process": samples_per_second_per_process,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(gen_logs)
                accelerator.log(logs, step=global_step)
                batch_time_meter.reset()
                data_time_meter.reset()

            if global_step % config.experiment.save_every == 0:
                save_checkpoint(
                    model,
                    config.experiment.output_dir,
                    accelerator,
                    global_step,
                    logger=logger,
                    resume_state=training_state,
                )

            if (
                global_step % config.experiment.generate_every == 0
                and accelerator.is_main_process
            ):
                with preserve_training_rng_state():
                    save_training_preview(
                        model,
                        tokenizer,
                        accelerator,
                        global_step,
                        config.experiment.output_dir,
                        logger=logger,
                        config=config,
                    )

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    "Finishing training: Global step is >= Max train steps: "
                    f"{global_step} >= {config.training.max_train_steps}"
                )
                break

    return global_step


@torch.no_grad()
def save_training_preview(
    model, tokenizer, accelerator, global_step, output_dir, logger, config
):
    model.eval()
    tokenizer.eval()
    logger.info("Generating images...")
    unwrapped_model = accelerator.unwrap_model(model)
    preview_class_ids = torch.tensor(
        [1, 7, 282, 604, 724, 179, 751, 404, 850],
        dtype=torch.long,
        device=accelerator.device,
    ) % unwrapped_model.condition_num_classes
    preview_class_ids = torch.cat(
        [
            preview_class_ids,
            torch.randint(
                0,
                unwrapped_model.condition_num_classes,
                size=(1,),
                device=accelerator.device,
            ),
        ]
    )
    preview_count = preview_class_ids.shape[0]
    preview_sample_start = global_step * preview_count
    with ParallelOrderSampler(
        num_workers=0,
        base_seed=int(config.training.seed),
    ) as preview_order_sampler:
        orders = preview_order_sampler.sample_generation_orders(
            preview_count,
            unwrapped_model.token_grid_shape,
            sample_ids=range(
                preview_sample_start,
                preview_sample_start + preview_count,
            ),
            device=accelerator.device,
        )
    generated_images = sample_fn(
        unwrapped_model,
        tokenizer,
        class_ids=preview_class_ids,
        guidance_scale=config.model.generator.guidance_scale,
        guidance_scale_power=config.model.generator.guidance_scale_power,
        temperature=config.model.generator.temperature,
        device=accelerator.device,
        return_tensor=True,
        orders=orders,
    )
    preview_image = make_generation_preview_grid(generated_images)

    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {"Train Generated": [preview_image]}, step=global_step
        )

    root = Path(output_dir) / "train_generated_images"
    os.makedirs(root, exist_ok=True)
    filename = f"{global_step:08}_s-generated.png"
    preview_image.save(os.path.join(root, filename))
    model.train()


def _read_checkpoint_json(path, description):
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Completed checkpoint has corrupt {description}: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise RuntimeError(
            f"Completed checkpoint {description} must be a JSON object: {path}"
        )
    return value


def _checkpoint_step(value, description, checkpoint_path):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(
            f"Completed checkpoint has invalid {description}: {checkpoint_path}"
        )
    return value


def _read_complete_checkpoint_metadata(checkpoint_path):
    marker_path = checkpoint_path / _CHECKPOINT_COMPLETE_FILENAME
    if not marker_path.is_file():
        raise RuntimeError(
            "Refusing to load an incomplete checkpoint without a completion "
            f"marker: {checkpoint_path}"
        )
    marker = _read_checkpoint_json(marker_path, "completion marker")
    if marker.get("schema_version") != _EXACT_RESUME_SCHEMA_VERSION:
        raise RuntimeError(
            "Completed checkpoint has an unsupported marker schema: "
            f"{checkpoint_path}"
        )
    marker_step = _checkpoint_step(
        marker.get("global_step"),
        "completion-marker global_step",
        checkpoint_path,
    )
    metadata = _read_checkpoint_json(
        checkpoint_path / "metadata.json",
        "metadata",
    )
    metadata_step = _checkpoint_step(
        metadata.get("global_step"),
        "metadata global_step",
        checkpoint_path,
    )
    if marker_step != metadata_step:
        raise RuntimeError(
            "Completed checkpoint marker and metadata disagree on global_step: "
            f"{checkpoint_path}"
        )
    checkpoint_name = checkpoint_path.name
    try:
        directory_step = int(checkpoint_name.removeprefix("checkpoint-"))
    except ValueError as error:
        raise RuntimeError(
            "Completed checkpoint directory has an invalid step suffix: "
            f"{checkpoint_path}"
        ) from error
    if (
        not checkpoint_name.startswith("checkpoint-")
        or directory_step < 0
        or directory_step != marker_step
    ):
        raise RuntimeError(
            "Completed checkpoint directory suffix, marker, and metadata "
            f"disagree on global_step: {checkpoint_path}"
        )
    return metadata


def _resume_cursor_matches(resume_state, saved_state):
    return (
        resume_state.global_step == saved_state["global_step"]
        and resume_state.consumed_batches == saved_state["consumed_batches"]
    )


def save_checkpoint(
    model,
    output_dir,
    accelerator,
    global_step,
    logger,
    resume_state,
):
    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    if resume_state.global_step != global_step:
        raise ValueError(
            "Cannot save a checkpoint whose step differs from the exact-resume "
            f"state: {global_step} != {resume_state.global_step}."
        )

    if (save_path / _CHECKPOINT_COMPLETE_FILENAME).is_file():
        metadata = _read_complete_checkpoint_metadata(save_path)
        saved_resume_state = metadata.get("exact_resume_state")
        if saved_resume_state is None:
            raise RuntimeError(
                "Completed checkpoint is missing exact-resume metadata: "
                f"{save_path}"
            )
        resume_state.validate_state_dict(saved_resume_state)
        if not _resume_cursor_matches(resume_state, saved_resume_state):
            raise RuntimeError(
                "The completed checkpoint has a different training cursor: "
                f"{save_path}"
            )
        logger.info(f"Checkpoint already complete at {save_path}; skipping save.")
        accelerator.wait_for_everyone()
        return
    if save_path.exists():
        raise RuntimeError(
            "Refusing to overwrite a checkpoint without a completion marker: "
            f"{save_path}"
        )

    staging_token = uuid.uuid4().hex if accelerator.is_main_process else None
    if accelerator.num_processes > 1:
        from accelerate.utils import broadcast_object_list

        payload = [staging_token]
        broadcast_object_list(payload, from_process=0)
        staging_token = payload[0]
    staging_path = Path(output_dir) / (
        f".{save_path.name}.incomplete-{staging_token}"
    )
    if accelerator.is_main_process:
        staging_path.mkdir(parents=True, exist_ok=False)
    accelerator.wait_for_everyone()

    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained_weights(
            staging_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
    accelerator.save_state(staging_path)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        metadata = {
            "global_step": global_step,
            "exact_resume_state": resume_state.state_dict(),
        }
        with (staging_path / "metadata.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
        with (staging_path / _CHECKPOINT_COMPLETE_FILENAME).open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                {
                    "schema_version": _EXACT_RESUME_SCHEMA_VERSION,
                    "global_step": global_step,
                },
                stream,
                sort_keys=True,
            )
            stream.write("\n")
        os.replace(staging_path, save_path)
        logger.info(f"Saved complete state to {save_path}")
    accelerator.wait_for_everyone()


def load_checkpoint(
    checkpoint_path,
    accelerator,
    logger,
    resume_state,
):
    checkpoint_path = Path(checkpoint_path)
    metadata = _read_complete_checkpoint_metadata(checkpoint_path)
    expected_resume_state = metadata.get("exact_resume_state")
    global_step = metadata["global_step"]
    if expected_resume_state is None:
        raise RuntimeError(
            f"Checkpoint is missing STAR exact-resume state: {checkpoint_path}"
        )
    resume_state.validate_state_dict(expected_resume_state)
    if expected_resume_state["global_step"] != global_step:
        raise RuntimeError(
            "Checkpoint metadata global_step does not match exact-resume state."
        )

    logger.info(f"Load checkpoint from {checkpoint_path}")
    accelerator.load_state(checkpoint_path, strict=True)
    if not _resume_cursor_matches(resume_state, expected_resume_state):
        raise RuntimeError(
            "Accelerate custom state does not match checkpoint metadata."
        )
    logger.info(f"Resuming at global_step {global_step}")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach()
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)
