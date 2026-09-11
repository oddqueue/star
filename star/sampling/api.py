from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from star.modeling.generator import STARGenerator
from star.modeling.tokenizer import PretrainedTokenizer
from star.utils.checkpoint_loading import (
    MASKGIT_VQ_TOKENIZER,
    STAR_GENERATOR_CHECKPOINTS,
    load_raw_state_dict,
    resolve_checkpoint_file,
)

from .core import sample_fn
from .masks import masks_to_tensor, normalize_masks
from .orders import ParallelOrderSampler


def _as_integer_tensor(values, *, name: str, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(values, device=device)
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(f"{name} must contain integer values.")
    return tensor.to(dtype=torch.long)


@dataclass(frozen=True)
class SamplingResult:
    images: np.ndarray
    token_ids: torch.Tensor


class STARSampler:
    def __init__(
        self,
        generator,
        tokenizer,
        config,
        device="cuda",
        *,
        order_sampler: ParallelOrderSampler | None = None,
    ):
        self.device = torch.device(device)
        if int(generator.target_codebook_size) != int(tokenizer.codebook_size):
            raise ValueError(
                "Generator and tokenizer codebook sizes do not match: "
                f"{generator.target_codebook_size} != {tokenizer.codebook_size}."
            )
        self.generator = generator.eval().to(self.device)
        self.generator.requires_grad_(False)
        self.tokenizer = tokenizer.eval().to(self.device)
        self.tokenizer.requires_grad_(False)
        self.config = config
        self.order_sampler = (
            order_sampler
            if order_sampler is not None
            else ParallelOrderSampler(num_workers=0)
        )

    @classmethod
    def from_pretrained(
        cls,
        config,
        generator_checkpoint: str | Path | None = None,
        tokenizer_checkpoint: str | Path | None = None,
        device="cuda",
        *,
        num_order_workers: int = 0,
        order_seed: int = 42,
    ) -> "STARSampler":
        if generator_checkpoint is None:
            generator_checkpoint = config.experiment.generator_checkpoint
            generator_fallback = (
                STAR_GENERATOR_CHECKPOINTS.get(generator_checkpoint)
                if isinstance(generator_checkpoint, str)
                else None
            )
            generator_checkpoint = resolve_checkpoint_file(
                generator_checkpoint,
                fallback=generator_fallback,
            )
        else:
            # Like an explicit tokenizer checkpoint, an explicit generator
            # checkpoint is always a local path.
            generator_checkpoint = resolve_checkpoint_file(generator_checkpoint)
        if tokenizer_checkpoint is None:
            tokenizer_checkpoint = resolve_checkpoint_file(
                config.model.vq_model.tokenizer_checkpoint,
                fallback=MASKGIT_VQ_TOKENIZER,
            )
        else:
            # An explicitly supplied path is always local, even when its
            # basename happens to match the built-in tokenizer filename.
            tokenizer_checkpoint = resolve_checkpoint_file(tokenizer_checkpoint)

        generator = STARGenerator(config)
        generator.load_state_dict(
            load_raw_state_dict(generator_checkpoint),
            strict=True,
        )
        tokenizer = PretrainedTokenizer(tokenizer_checkpoint)
        order_sampler = ParallelOrderSampler(
            num_workers=num_order_workers,
            base_seed=order_seed,
        )
        try:
            return cls(
                generator,
                tokenizer,
                config,
                device=device,
                order_sampler=order_sampler,
            )
        except BaseException:
            order_sampler.close()
            raise

    def __enter__(self) -> "STARSampler":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        self.order_sampler.close()

    @property
    def sequence_length(self) -> int:
        return int(self.generator.image_seq_len)

    @property
    def token_grid_shape(self) -> tuple[int, int]:
        return tuple(self.generator.token_grid_shape)

    def _sampling_kwargs(self) -> dict:
        return {
            "temperature": self.config.model.generator.temperature,
            "guidance_scale": self.config.model.generator.guidance_scale,
            "guidance_scale_power": self.config.model.generator.guidance_scale_power,
        }

    def _prepare_class_ids(self, class_ids) -> torch.Tensor:
        class_ids = _as_integer_tensor(
            class_ids,
            name="class_ids",
            device=self.device,
        ).reshape(-1)
        num_classes = int(self.generator.condition_num_classes)
        if class_ids.numel() == 0:
            raise ValueError("At least one class ID is required.")
        if int(class_ids.min()) < 0 or int(class_ids.max()) >= num_classes:
            raise ValueError(f"Class IDs must be in [0, {num_classes - 1}].")
        return class_ids

    @torch.no_grad()
    def generate(
        self,
        class_ids,
        *,
        sample_ids: Sequence[int] | None = None,
    ) -> SamplingResult:
        class_ids = self._prepare_class_ids(class_ids)
        orders = self.order_sampler.sample_generation_orders(
            class_ids.shape[0],
            self.token_grid_shape,
            sample_ids=sample_ids,
            device=self.device,
        )
        images, generated_token_ids = sample_fn(
            generator=self.generator,
            tokenizer=self.tokenizer,
            class_ids=class_ids,
            device=self.device,
            orders=orders,
            return_token_ids=True,
            **self._sampling_kwargs(),
        )
        return SamplingResult(
            images=images,
            token_ids=generated_token_ids.detach().cpu(),
        )

    @torch.no_grad()
    def complete(
        self,
        class_ids,
        token_ids,
        masks: Sequence[list[int] | set[int]],
        *,
        sample_ids: Sequence[int] | None = None,
    ) -> SamplingResult:
        class_ids = self._prepare_class_ids(class_ids)
        token_ids = _as_integer_tensor(
            token_ids,
            name="token_ids",
            device=self.device,
        )
        if token_ids.shape != (class_ids.shape[0], self.sequence_length):
            raise ValueError(
                "token_ids must have shape "
                f"[{class_ids.shape[0]}, {self.sequence_length}], "
                f"got {tuple(token_ids.shape)}."
            )
        if int(token_ids.min()) < 0 or int(token_ids.max()) >= int(
            self.generator.target_codebook_size
        ):
            raise ValueError("token_ids contain values outside the tokenizer codebook.")

        normalized_masks = normalize_masks(
            masks,
            class_ids.shape[0],
            self.token_grid_shape,
        )
        mask_tensor = masks_to_tensor(normalized_masks, self.token_grid_shape)
        orders = self.order_sampler.sample_completion_orders(
            normalized_masks,
            self.token_grid_shape,
            sample_ids=sample_ids,
            device=self.device,
        )
        images, generated_token_ids = sample_fn(
            generator=self.generator,
            tokenizer=self.tokenizer,
            class_ids=class_ids,
            token_ids=token_ids,
            masks=normalized_masks,
            orders=orders,
            device=self.device,
            return_token_ids=True,
            **self._sampling_kwargs(),
        )

        generated_token_ids_cpu = generated_token_ids.detach().cpu()
        source_token_ids_cpu = token_ids.detach().cpu()
        if not torch.equal(
            generated_token_ids_cpu[~mask_tensor],
            source_token_ids_cpu[~mask_tensor],
        ):
            raise RuntimeError("Completion changed one or more observed tokens.")
        return SamplingResult(
            images=images,
            token_ids=generated_token_ids_cpu,
        )
