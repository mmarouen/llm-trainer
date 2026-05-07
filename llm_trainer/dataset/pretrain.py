from __future__ import annotations

import io
import os
import random
from typing import Iterator, List, Optional, Tuple
import numpy as np
import torch
from logging import Logger
from torch.utils.data import DataLoader, IterableDataset
import gcsfs  # type: ignore


def _get_shard_token_counts(paths: List[str], dtype: np.dtype = np.int32) -> List[int]:
    fs = gcsfs.GCSFileSystem()
    bytes_per_token = 4

    counts = []
    for path in paths:
        info = fs.info(f"{path}")
        token_count = (info["size"] - 128) // bytes_per_token
        counts.append(token_count)
    return counts

def _list_shards(bucket: str, prefix: str, size: int=None) -> Tuple[List[str], List[str]]:
    """Return sorted list of shard prefixes (gs://bucket/prefix/shard_N/)."""
    fs = gcsfs.GCSFileSystem(token="google_default")

    input_ids_filenames = sorted(fs.glob(f'{bucket}/{prefix}/shard*_input_ids.npy'))
    labels_filenames = sorted(fs.glob(f'{bucket}/{prefix}/shard*_labels.npy'))
    if size:
        n_shards = min(len(input_ids_filenames), size)
        input_ids_filenames = input_ids_filenames[:n_shards]
        labels_filenames = labels_filenames[:n_shards]
    return input_ids_filenames, labels_filenames


def _load_npy_from_gcs(gcs_path: str, fs: gcsfs.GCSFileSystem) -> np.ndarray:
    """Load a .npy file directly from GCS into a numpy array."""
    #fs = gcsfs.GCSFileSystem(token="google_default")
    with fs.open(gcs_path, "rb") as f:
        return np.load(io.BytesIO(f.read()))


def _chunk_array(arr: np.ndarray, chunk_size: int) -> np.ndarray:
    """
    Truncate `arr` to the largest multiple of `chunk_size` and reshape.
    Returns shape (num_chunks, chunk_size).
    """
    n_full = (len(arr) // chunk_size) * chunk_size
    return arr[:n_full].reshape(-1, chunk_size)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShardedNpyDataset(IterableDataset):
    """
    IterableDataset that streams flat .npy shards from a GCP bucket.

    Design choices
    --------------
    * IterableDataset is preferred over map-style Dataset because shards are
      ~1 B tokens each we never want to materialise the full index in RAM.
    * Each worker loads a disjoint subset of shards so there is no redundant
      I/O and no cross-worker locking.
    * Chunks from one shard are yielded in order; shard order is shuffled per
      epoch when `shuffle_shards=True`.
    * Attention masks are synthesised as ones tensors on the fly no storage
      overhead and no extra GCS reads.
    """

    def __init__(
        self,
        bucket: str,
        gcp_prefix: str,
        chunk_size: int,
        shuffle_shards: bool = True,
        shuffle_chunks: bool = False,
        seed: Optional[int] = None,
        demo_mode: bool = False
    ) -> None:
        super().__init__()
        self.bucket = bucket
        self.gcp_prefix = gcp_prefix
        self.chunk_size = chunk_size
        self.shuffle_shards = shuffle_shards
        self.shuffle_chunks = shuffle_chunks
        self.seed = seed
        self.demo_mode=demo_mode
        self.input_ids_paths: List[str]
        self.labels_paths: List[str]
        self.dataset_folder = os.path.join(self.bucket, self.gcp_prefix)
        self.input_ids_paths, self.labels_paths = _list_shards(bucket=self.bucket, prefix=self.gcp_prefix, size=2 if demo_mode else None)
        self._tokens_per_shard = _get_shard_token_counts(self.input_ids_paths)
        self.total_tokens = sum(self._tokens_per_shard)
        self.total_chunks = self.total_tokens // self.chunk_size
        if not self.input_ids_paths:
            raise ValueError(
                f"No shards found at gs://{bucket}/{gcp_prefix}. "
                "Check bucket name, prefix, and credentials."
            )

    def __len__(self) -> int:
        return self.total_chunks

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        fs = gcsfs.GCSFileSystem(token="google_default")
        pairs = list(zip(self.input_ids_paths, self.labels_paths))
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            pairs = pairs[worker_id::num_workers]


        # Per-epoch shuffle -------------------------------------------
        rng = random.Random(self.seed)
        if self.shuffle_shards:
            rng.shuffle(pairs)

        # Stream chunks -----------------------------------------------
        for ids_path, lbl_path in pairs:
            yield from self._iter_shard(ids_path, lbl_path, rng, fs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_shard(self, input_ids_path: str, labels_path: str, rng: random.Random, fs: gcsfs.GCSFileSystem) -> Iterator[dict]:
        """Load one shard and yield (input_ids, labels, attention_mask) dicts."""

        raw_ids    = _load_npy_from_gcs(input_ids_path, fs)
        raw_labels = _load_npy_from_gcs(labels_path, fs)

        assert len(raw_ids) == len(raw_labels), (
            f"Shard {input_ids_path}: input_ids length {len(raw_ids)} "
            f"!= labels length {len(raw_labels)}"
        )

        chunked_ids    = _chunk_array(raw_ids,    self.chunk_size)
        chunked_labels = _chunk_array(raw_labels, self.chunk_size)

        indices = list(range(len(chunked_ids)))
        if self.shuffle_chunks:
            rng.shuffle(indices)

        ones = torch.ones(self.chunk_size, dtype=torch.long)

        for i in indices:
            yield {
                "input_ids":      torch.from_numpy(chunked_ids[i].astype(np.int64)),
                "labels":         torch.from_numpy(chunked_labels[i].astype(np.int64)),
                "attention_mask": ones,
            }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        tokens_per_shard = [f'{val:,}' for val in self._tokens_per_shard]
        return (
            f"ShardedNpyDataset("
            f"Demo mode={self.demo_mode}"
            f"bucket={self.bucket!r}, "
            f"prefix={self.gcp_prefix!r}, "
            f"chunk_size={self.chunk_size}, "
            f"n_chunks={self.__len__():,}"
            f"n_shards={len(self.input_ids_paths)}, "
            f"tokens per shard={tokens_per_shard}) "
            f"total tokens={self.total_tokens:,}"
        )


def build_pretrain_dataloader(
    bucket: str,
    gcp_prefix: str,
    chunk_size: int,
    logger: Logger,
    batch_size: int = 8,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    shuffle_shards: bool = True,
    shuffle_chunks: bool = False,
    seed: Optional[int] = 42,
    pin_memory: bool = True,
    demo_mode: bool = False
) -> DataLoader:
    """
    Build a DataLoader ready to feed an A100 training loop.

    Parameters
    ----------
    bucket          GCS bucket name (without gs://)
    shard_prefix    Prefix inside the bucket, e.g. "train/shards/"
    chunk_size      Token window size; must divide evenly into shard length
                    (trailing tokens are silently dropped)
    batch_size      Sequences per batch
    num_workers     Parallel shard-loading processes. 4 is a good default
                    for A100; set 0 to load in the main process (debugging).
    prefetch_factor Batches queued per worker (num_workers * prefetch_factor
                    batches live in CPU memory at any time).
    shuffle_shards  Randomise shard order each epoch.
    shuffle_chunks  Randomise chunk order within each shard (higher RAM use).
    seed            Base RNG seed (workers derive child seeds automatically).
    pin_memory      Pin CPU tensors for faster H2D transfer; set False when
                    debugging on CPU.
    """
    dataset = ShardedNpyDataset(
        bucket=bucket,
        gcp_prefix=gcp_prefix,
        chunk_size=chunk_size,
        shuffle_shards=shuffle_shards,
        shuffle_chunks=shuffle_chunks,
        seed=seed,
        demo_mode=demo_mode
    )
    logger.info(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory and torch.cuda.is_available(),
        # IterableDataset owns its own shuffling; no sampler needed.
        shuffle=False,
    )