from __future__ import annotations

from os import PathLike
from pathlib import Path

from star.utils.checkpoint_loading import MASKGIT_VQ_TOKENIZER


DEFAULT_TOKENIZER_REPO_ID = MASKGIT_VQ_TOKENIZER.repo_id
DEFAULT_TOKENIZER_FILENAME = MASKGIT_VQ_TOKENIZER.filename
DEFAULT_PRETOKENIZATION_REPO_ID = "yucornetto/RAR"
DEFAULT_PRETOKENIZATION_FILENAME = "maskgitvq.jsonl"

_PINNED_REVISIONS = {
    (DEFAULT_TOKENIZER_REPO_ID, DEFAULT_TOKENIZER_FILENAME): (
        MASKGIT_VQ_TOKENIZER.revision
    ),
    (DEFAULT_PRETOKENIZATION_REPO_ID, DEFAULT_PRETOKENIZATION_FILENAME): (
        "0aed4bfc11a134679915e9c740283cba34e43182"
    ),
}


def _distributed_asset_error_messages(local_error):
    local_message = None
    if local_error is not None:
        local_message = f"{type(local_error).__name__}: {local_error}"

    # Keep this module usable by simple single-process utilities without making
    # their accelerator-like context implement another synchronization API.
    try:
        import torch.distributed as dist
    except ImportError:
        return [local_message]

    if not (dist.is_available() and dist.is_initialized()):
        return [local_message]

    messages = [None] * dist.get_world_size()
    dist.all_gather_object(messages, local_message)
    return messages


def _raise_distributed_asset_errors(local_error):
    messages = _distributed_asset_error_messages(local_error)
    failures = [
        f"rank {rank}: {message}"
        for rank, message in enumerate(messages)
        if message is not None
    ]
    if failures:
        if len(messages) == 1 and local_error is not None:
            # Preserve the original exception type and message outside DDP.
            raise local_error
        error = RuntimeError(
            "Training asset resolution failed on distributed rank(s): "
            + "; ".join(failures)
        )
        if local_error is not None:
            raise error from local_error
        raise error


def resolve_training_asset(
    configured_path,
    *,
    repo_id: str,
    filename: str,
    accelerator,
) -> str:
    path = None
    is_default_reference = False
    validation_error = None
    try:
        if configured_path is None:
            raise ValueError("Training asset path must be configured.")
        if not isinstance(configured_path, (str, PathLike)):
            raise TypeError(
                "Training asset must be a path string or PathLike value."
            )
        if not str(configured_path).strip():
            raise ValueError("Training asset path must not be empty.")

        path = Path(configured_path).expanduser()
        is_default_reference = (
            isinstance(configured_path, str) and configured_path == filename
        )
    except Exception as exc:
        validation_error = exc
    _raise_distributed_asset_errors(validation_error)

    if not is_default_reference:
        resolved_path = None
        resolution_error = None
        try:
            if not path.is_file():
                raise FileNotFoundError(
                    "Configured training asset does not exist or is not a file: "
                    f"{path}"
                )
            resolved_path = str(path.resolve())
        except Exception as exc:
            resolution_error = exc
        _raise_distributed_asset_errors(resolution_error)
        return resolved_path

    revision = _PINNED_REVISIONS.get((repo_id, filename))
    download_arguments = {"repo_id": repo_id, "filename": filename}
    if revision is not None:
        download_arguments["revision"] = revision

    download_setup_error = None
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        download_setup_error = exc
    _raise_distributed_asset_errors(download_setup_error)

    download_error = None
    if accelerator.local_process_index == 0 and not path.is_file():
        try:
            hf_hub_download(**download_arguments)
        except Exception as exc:  # Re-raised after every process reaches the barrier.
            download_error = exc

    accelerator.wait_for_everyone()
    resolved_path = None
    resolution_error = None
    try:
        if path.is_file():
            resolved_path = str(path.resolve())
        else:
            cache_arguments = dict(download_arguments)
            cache_arguments["local_files_only"] = True
            try:
                cached_path = Path(hf_hub_download(**cache_arguments))
            except Exception as exc:
                message = (
                    f"Could not resolve Hugging Face asset {repo_id}/{filename} "
                    "from the Hugging Face cache after the download attempt."
                )
                raise RuntimeError(message) from (download_error or exc)

            if not cached_path.is_file():
                raise FileNotFoundError(
                    "Hugging Face returned a non-file training asset path: "
                    f"{cached_path}"
                )
            resolved_path = str(cached_path.resolve())

    except Exception as exc:
        resolution_error = exc
    _raise_distributed_asset_errors(resolution_error)
    return resolved_path
