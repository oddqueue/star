from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path


@dataclass(frozen=True)
class HuggingFaceFileSpec:
    repo_id: str
    filename: str
    revision: str

    def __post_init__(self) -> None:
        if not self.repo_id or "/" not in self.repo_id:
            raise ValueError("repo_id must be a non-empty Hugging Face repository ID")
        if not self.filename or Path(self.filename).name != self.filename:
            raise ValueError("filename must be a non-empty basename")
        if not self.revision:
            raise ValueError("revision must not be empty")


MASKGIT_VQ_TOKENIZER = HuggingFaceFileSpec(
    repo_id="fun-research/TiTok",
    filename="maskgit-vqgan-imagenet-f16-256.bin",
    revision="ab646ed225080a3acb7c78440a574d7f67f16fa7",
)

STAR_GENERATOR_CHECKPOINTS = {
    spec.repo_id: spec
    for spec in (
        HuggingFaceFileSpec(
            repo_id="oddqueue/STAR-B",
            filename="pytorch_model.bin",
            revision="996da16cd312d49a4ad2e1efd76e5eb52731a886",
        ),
        HuggingFaceFileSpec(
            repo_id="oddqueue/STAR-L",
            filename="pytorch_model.bin",
            revision="94203667d48ab30031a495523d4083e71571eacc",
        ),
        HuggingFaceFileSpec(
            repo_id="oddqueue/STAR-XL",
            filename="pytorch_model.bin",
            revision="d0a40a6ab1b74a4a09e6ebf7f659c87128a5bff7",
        ),
        HuggingFaceFileSpec(
            repo_id="oddqueue/STAR-XXL",
            filename="pytorch_model.bin",
            revision="40afb41291ed946fda1c99351b7bafcfb51cb0e1",
        ),
    )
}


def download_hf_file(spec: HuggingFaceFileSpec) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required to download the default checkpoint"
        ) from exc

    downloaded_path = Path(
        hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.filename,
            revision=spec.revision,
        )
    ).expanduser()
    if not downloaded_path.is_file():
        raise FileNotFoundError(
            "Hugging Face returned a non-file checkpoint path: "
            f"{downloaded_path}"
        )

    return downloaded_path.resolve()


def resolve_checkpoint_file(
    configured_path: str | PathLike,
    *,
    fallback: HuggingFaceFileSpec | None = None,
) -> Path:
    if not isinstance(configured_path, (str, PathLike)):
        raise TypeError("Checkpoint must be a path string or PathLike value")
    if not str(configured_path).strip():
        raise ValueError("Checkpoint path must not be empty")

    is_fallback_reference = (
        fallback is not None
        and isinstance(configured_path, str)
        and configured_path in (fallback.filename, fallback.repo_id)
    )
    local_path = Path(configured_path).expanduser()
    if local_path.is_file():
        return local_path.resolve()

    if is_fallback_reference:
        return download_hf_file(fallback)

    raise FileNotFoundError(
        f"Configured checkpoint does not exist or is not a file: {local_path}"
    )


def load_raw_state_dict(checkpoint_path: str | PathLike) -> Mapping:
    import torch

    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        raise RuntimeError(
            "This PyTorch version cannot safely load release checkpoints; "
            "install PyTorch 2.0 or newer."
        ) from exc

    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"Expected a raw state dict in {checkpoint_path}, "
            f"got {type(state_dict).__name__}"
        )
    if any(not isinstance(name, str) for name in state_dict):
        raise TypeError(
            f"Expected string parameter names in state dict: {checkpoint_path}"
        )
    return state_dict
