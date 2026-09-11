from __future__ import annotations

import json
import operator
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no POSIX flock.
    fcntl = None


_OFFSET_DTYPE = np.dtype("<u8")
_CACHE_LOAD_ERRORS = (EOFError, FileNotFoundError, OSError, ValueError)
_STAR_JSONL_CACHE_SUBDIR = Path("star") / "jsonl-index"
_INDEX_METADATA_VERSION = 1


def default_jsonl_index_cache_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        if not xdg_cache_home:
            xdg_cache_home = Path.home() / ".cache"
        hf_home = Path(xdg_cache_home).expanduser() / "huggingface"
    return Path(hf_home).expanduser() / _STAR_JSONL_CACHE_SUBDIR


class PretokenizedDatasetJSONL(Dataset):
    def __init__(self, data_path, index_cache_dir=None):
        super().__init__()
        path = Path(data_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Pretokenized JSONL is not a file: {path}")

        self.jsonl_file = str(path.resolve())
        self._source_signature = self._stat_signature()
        self._index_path = None
        self._metadata_path = None
        self._file = None
        self._file_pid = None
        if index_cache_dir is None:
            index_cache_dir = default_jsonl_index_cache_dir()
        self._offsets = self._load_or_build_offsets(index_cache_dir)
        self.num_records = len(self._offsets)

    def _stat_signature(self):
        stat = os.stat(self.jsonl_file)
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    @staticmethod
    def _signature_payload(signature):
        return {
            "device": signature[0],
            "inode": signature[1],
            "size": signature[2],
            "mtime_ns": signature[3],
            "ctime_ns": signature[4],
        }

    def _source_cache_dir(self, cache_dir):
        source = Path(self.jsonl_file)
        if source.anchor == os.sep:
            root_name = "root"
        else:  # pragma: no cover - retained for non-POSIX local use.
            root_name = source.anchor.replace(":", "").strip("/\\") or "root"
        source_parts = source.parts[1:] if source.anchor else source.parts
        return cache_dir / "by-path" / root_name / Path(*source_parts)

    @contextmanager
    def _cache_lock(self, lock_path):
        with lock_path.open("a+b") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load_or_build_offsets(self, index_cache_dir):
        if isinstance(index_cache_dir, str) and not index_cache_dir.strip():
            raise ValueError("JSONL index cache directory must not be empty.")

        cache_dir = Path(index_cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not cache_dir.is_dir():
            raise NotADirectoryError(f"JSONL index cache is not a directory: {cache_dir}")

        source_cache_dir = self._source_cache_dir(cache_dir.resolve())
        source_cache_dir.mkdir(parents=True, exist_ok=True)
        index_path = source_cache_dir / "offsets.npy"
        metadata_path = source_cache_dir / "source.json"
        lock_path = source_cache_dir / "build.lock"
        self._index_path = str(index_path.resolve())
        self._metadata_path = str(metadata_path.resolve())

        try:
            return self._load_cached_offsets()
        except _CACHE_LOAD_ERRORS:
            with self._cache_lock(lock_path):
                try:
                    return self._load_cached_offsets()
                except _CACHE_LOAD_ERRORS:
                    offsets = self._build_offsets()
                    self._write_offsets_atomically(index_path, offsets)
                    self._write_metadata_atomically(metadata_path)
                    return self._load_cached_offsets()

    def _load_cached_offsets(self):
        try:
            with open(self._metadata_path, "r", encoding="utf-8") as stream:
                metadata = json.load(stream)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid JSONL index metadata: {self._metadata_path}"
            ) from exc
        if (
            not isinstance(metadata, dict)
            or metadata.get("format_version") != _INDEX_METADATA_VERSION
            or metadata.get("source")
            != self._signature_payload(self._source_signature)
        ):
            raise ValueError(f"Stale JSONL offset index: {self._index_path}")
        if self._stat_signature() != self._source_signature:
            raise RuntimeError(
                f"Pretokenized JSONL changed while loading its index: {self.jsonl_file}"
            )
        offsets = np.load(self._index_path, mmap_mode="r", allow_pickle=False)
        if offsets.ndim != 1 or offsets.dtype != _OFFSET_DTYPE:
            raise ValueError(f"Invalid JSONL offset index: {self._index_path}")
        size = self._source_signature[2]
        if len(offsets) == 0:
            raise ValueError(f"Pretokenized JSONL is empty: {self.jsonl_file}")
        if int(offsets[0]) != 0 or int(offsets[-1]) >= size:
            raise ValueError(f"Invalid JSONL offsets in: {self._index_path}")
        if len(offsets) > 1 and not np.all(offsets[1:] > offsets[:-1]):
            raise ValueError(f"Non-monotonic JSONL offsets in: {self._index_path}")
        return offsets

    def _build_offsets(self):
        offsets = []
        position = 0
        with open(self.jsonl_file, "rb") as jsonl_file:
            for line_number, line in enumerate(jsonl_file, start=1):
                if not line.strip():
                    raise ValueError(
                        f"Empty JSONL record at line {line_number}: {self.jsonl_file}"
                    )
                offsets.append(position)
                position += len(line)

        if not offsets:
            raise ValueError(f"Pretokenized JSONL is empty: {self.jsonl_file}")
        if self._stat_signature() != self._source_signature:
            raise RuntimeError(
                f"Pretokenized JSONL changed while indexing: {self.jsonl_file}"
            )
        return np.asarray(offsets, dtype=_OFFSET_DTYPE)

    def _write_offsets_atomically(self, index_path, offsets):
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{index_path.name}.",
                suffix=".tmp",
                dir=index_path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                np.save(temporary_file, offsets, allow_pickle=False)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, index_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _write_metadata_atomically(self, metadata_path):
        payload = {
            "format_version": _INDEX_METADATA_VERSION,
            "source": self._signature_payload(self._source_signature),
        }
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{metadata_path.name}.",
                suffix=".tmp",
                dir=metadata_path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(payload, temporary_file, sort_keys=True)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, metadata_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _open_file(self):
        current_pid = os.getpid()
        if self._file is None or self._file_pid != current_pid:
            self.close()
            if self._stat_signature() != self._source_signature:
                raise RuntimeError(
                    "Pretokenized JSONL changed after the dataset was created; "
                    "construct a new dataset to rebuild its offsets."
                )
            self._file = open(self.jsonl_file, "rb")
            self._file_pid = current_pid
        return self._file

    def __len__(self):
        return self.num_records

    def __getitem__(self, idx):
        idx = operator.index(idx)
        if idx < 0 or idx >= self.num_records:
            raise IndexError(
                f"JSONL record index {idx} is outside [0, {self.num_records})."
            )
        jsonl_file = self._open_file()
        jsonl_file.seek(int(self._offsets[idx]))
        line = jsonl_file.readline()
        try:
            data = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid JSON at line {idx + 1} in {self.jsonl_file}"
            ) from exc
        if not isinstance(data, dict) or "class_id" not in data or "tokens" not in data:
            raise ValueError(
                f"JSONL record {idx} must contain class_id and tokens."
            )
        return torch.tensor(data["class_id"]), torch.tensor(data["tokens"])

    def close(self):
        jsonl_file = getattr(self, "_file", None)
        if jsonl_file is not None:
            jsonl_file.close()
        self._file = None
        self._file_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_file_pid"] = None
        if self._index_path is not None:
            state["_offsets"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._offsets is None:
            self._offsets = self._load_cached_offsets()

    def __del__(self):
        self.close()
