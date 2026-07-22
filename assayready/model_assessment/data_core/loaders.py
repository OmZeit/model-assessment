import functools
import hashlib
import json
import logging
import os
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .config import PAD_TOKEN_ID
except ImportError:
    PAD_TOKEN_ID = 4  # Default to N if config missing
from .tokenizer import DnaTokenizer

LOGGER = logging.getLogger(__name__)


def _seed_worker(worker_id):
    import numpy as np, random, torch
    wseed = torch.initial_seed() % 2**32
    np.random.seed(wseed)
    random.seed(wseed)


g = torch.Generator()
g.manual_seed(42)


def set_loader_seed(seed: int):
    g.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _derived_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


class RoundRobinLoader:
    """
    Iterates over multiple DataLoaders in a round-robin fashion.
    """
    def __init__(self, loaders: List[DataLoader], names: List[str] = None, steps_per_loader: List[int] = None):
        self.loaders = list(loaders)
        self.names = list(names) if names is not None else [f"Loader_{i}" for i in range(len(loaders))]
        if len(self.names) != len(self.loaders):
            raise ValueError("names must contain exactly one entry per loader")
        self.steps_per_loader = list(steps_per_loader) if steps_per_loader is not None else None
        if self.steps_per_loader is not None:
            if len(self.steps_per_loader) != len(self.loaders):
                raise ValueError("steps_per_loader must contain exactly one entry per loader")
            if any(int(steps) < 0 for steps in self.steps_per_loader):
                raise ValueError("steps_per_loader values must be non-negative")
            if any(int(steps) > len(loader) for steps, loader in zip(self.steps_per_loader, self.loaders)):
                raise ValueError("steps_per_loader cannot exceed the corresponding loader length")
            self.steps_per_loader = [int(steps) for steps in self.steps_per_loader]
        self.total_len = len(self)

    def set_epoch(self, epoch: int):
        """Set epoch for all underlying datasets."""
        for loader in self.loaders:
            dataset = getattr(loader, "dataset", None)
            if hasattr(dataset, 'set_epoch'):
                dataset.set_epoch(epoch)
            for sampler in (getattr(loader, "batch_sampler", None), getattr(loader, "sampler", None)):
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

    def __len__(self):
        if self.steps_per_loader is not None:
            return sum(self.steps_per_loader)
        return sum(len(loader) for loader in self.loaders)

    def __iter__(self):
        return RoundRobinIterator(self.loaders, remaining=self.steps_per_loader)


class RoundRobinIterator:
    def __init__(self, loaders: List[DataLoader], remaining: List[int] | None = None):
        self.iters = [iter(l) for l in loaders]
        self.remaining = list(remaining) if remaining is not None else [len(l) for l in loaders]
        self.ptr = 0

    def __iter__(self):
        return self

    def __next__(self):
        if sum(self.remaining) <= 0:
            raise StopIteration

        attempts = 0
        max_attempts = len(self.iters)

        while attempts < max_attempts:
            i = self.ptr
            self.ptr = (self.ptr + 1) % len(self.iters)

            if self.remaining[i] <= 0:
                attempts += 1
                continue

            try:
                batch = next(self.iters[i])
                self.remaining[i] -= 1
                return batch
            except StopIteration:
                # Mark this loader exhausted for the rest of the epoch.
                self.remaining[i] = 0
                attempts += 1
                continue

        # All loaders are totally empty
        raise StopIteration


class SmartBatchSampler(torch.utils.data.Sampler):
    """
    Sampler that groups sequences by length to minimize padding.
    """
    def __init__(self, data_source, batch_size, shuffle=True, seed: int = 42):
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0
        self.skip_batches = 0

    def set_skip(self, n: int):
        self.skip_batches = max(0, int(n))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        # Get lengths
        if hasattr(self.data_source, 'get_lengths'):
            lengths = self.data_source.get_lengths()
        else:
            # Fallback: iterate (slow)
            LOGGER.warning("[SmartBatchSampler] data_source has no get_lengths, iterating...")
            lengths = [len(self.data_source[i]['input_ids']) for i in range(len(self.data_source))]

        # Sort indices by length
        indices = torch.argsort(torch.tensor(lengths))

        # Create batches
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size].tolist()
            batches.append(batch)

        # Shuffle batches
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)

        # [Fast Resume] Skip first N batches
        if self.skip_batches > 0:
            LOGGER.info("[SmartBatchSampler] Fast-skipping %s batches.", self.skip_batches)
            batches = batches[self.skip_batches:]

        # Yield indices
        for batch in batches:
            yield batch

    def __len__(self):
        total = (len(self.data_source) + self.batch_size - 1) // self.batch_size
        return max(0, total - self.skip_batches)


def count_chunks_exact(seq_or_len, max_length: int, stride: int, min_chunk_length: int) -> int:
    """
    Count exactly how many chunks will be generated from a sequence.
    Can accept either a sequence string or its length.
    """
    if isinstance(seq_or_len, str):
        seq_len = len(seq_or_len)
    else:
        seq_len = int(seq_or_len)

    if seq_len < min_chunk_length:
        return 0
    if seq_len <= max_length:
        return 1

    count = 0
    for s in range(0, seq_len - min_chunk_length + 1, max(1, stride)):
        chunk_len = min(max_length, seq_len - s)
        if chunk_len >= min_chunk_length:
            count += 1
        if s + max_length >= seq_len:
            break
    return count


def scan_fasta_lengths(path: str) -> List[Tuple[int, int, int, int]]:
    """
    Scan FASTA file and return list of (index, start_offset, end_offset, length).
    Uses caching to avoid re-scanning large files.
    """
    # Cache path: fasta_path + ".index.json"
    cache_path = path + ".index.json"

    # Check cache
    if os.path.exists(cache_path):
        try:
            # Check modification times
            if os.path.getmtime(cache_path) >= os.path.getmtime(path):
                import json
                with open(cache_path, "r") as f:
                    LOGGER.info("[DataPrep] Loading cached offsets from %s", cache_path)
                    return json.load(f)
        except Exception as e:
            LOGGER.warning("[DataPrep] Cache load failed: %s. Re-scanning.", e)

    LOGGER.info("[DataPrep] Scanning %s for offsets (this may take a while)...", path)
    offsets = []
    idx = 0

    with open(path, "rb") as f:
        offset = 0
        current_seq_start = -1
        current_seq_len = 0

        for line in f:
            line_len = len(line)
            if line.startswith(b">"):
                if current_seq_start != -1:
                    # Previous sequence ends here
                    offsets.append((idx, current_seq_start, offset, current_seq_len))
                    idx += 1

                current_seq_start = offset + line_len
                current_seq_len = 0
            else:
                if current_seq_start != -1:
                    stripped = line.strip()
                    current_seq_len += len(stripped)

            offset += line_len

        if current_seq_start != -1:
            offsets.append((idx, current_seq_start, offset, current_seq_len))

    # Save cache
    try:
        import json
        with open(cache_path, "w") as f:
            json.dump(offsets, f)
        LOGGER.info("[DataPrep] Saved cache to %s", cache_path)
    except Exception as e:
        LOGGER.warning("[DataPrep] Failed to save cache: %s", e)

    return offsets


def _path_keys(path: str) -> Set[str]:
    abs_path = os.path.abspath(path)
    return {
        path,
        os.path.normpath(path),
        abs_path,
        os.path.normcase(abs_path),
        os.path.basename(path),
    }


def load_split_manifest(path: str, files: Sequence[str]) -> Dict[str, Dict[str, Set[int]]]:
    """Load JSONL split assignments keyed by input FASTA path/basename."""
    file_keys: Dict[str, str] = {}
    for file_path in files:
        canonical = os.path.normcase(os.path.abspath(file_path))
        for key in _path_keys(file_path):
            file_keys[key] = canonical

    assignments: Dict[str, Dict[str, Set[int]]] = {
        os.path.normcase(os.path.abspath(file_path)): {"train": set(), "eval": set()}
        for file_path in files
    }
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            split = str(row.get("split", "")).strip().lower()
            if split in {"val", "valid", "validation", "test"}:
                split = "eval"
            if split not in {"train", "eval"}:
                raise ValueError(f"{path}:{line_no} has unsupported split {row.get('split')!r}")
            if "seq_index" not in row:
                raise ValueError(f"{path}:{line_no} is missing seq_index")
            seq_index = int(row["seq_index"])
            if seq_index < 0:
                raise ValueError(f"{path}:{line_no} has negative seq_index {seq_index}")
            row_path = str(row.get("path") or row.get("fasta") or row.get("file") or "")
            candidates = _path_keys(row_path) if row_path else set()
            canonical = None
            for key in candidates:
                if key in file_keys:
                    canonical = file_keys[key]
                    break
            if canonical is None and row_path:
                # Common case: manifest path is repo-relative, training path is absolute.
                for file_path in files:
                    if os.path.basename(file_path) == os.path.basename(row_path):
                        canonical = os.path.normcase(os.path.abspath(file_path))
                        break
            if canonical is None:
                continue
            other_split = "eval" if split == "train" else "train"
            if seq_index in assignments[canonical][other_split]:
                raise ValueError(
                    f"{path}:{line_no} assigns seq_index {seq_index} to both train and eval"
                )
            assignments[canonical][split].add(seq_index)
    return assignments


def split_indices_for_path(
    split_manifest: Optional[Dict[str, Dict[str, Set[int]]]],
    path: str,
) -> Optional[Dict[str, Set[int]]]:
    if split_manifest is None:
        return None
    return split_manifest.get(os.path.normcase(os.path.abspath(path)))


class NpyDataset(Dataset):
    """
    Dataset that loads pre-tokenized BPE data from .npy files.
    """
    def __init__(self, fa_path, max_length, stride, min_chunk_length, indices=None, provide_rc=False):
        self.fa_path = fa_path
        self.max_length = int(max_length)
        self.stride = int(stride)
        self.min_chunk_length = min_chunk_length
        self.provide_rc = provide_rc
        self.name_root = os.path.basename(fa_path)

        # 1. Load token offsets. Prefer the BPE preprocessor offsets when present;
        # .index.json stores raw FASTA base offsets and is not valid for .npy token arrays.
        index_path = fa_path + "_offsets.json"
        if not os.path.exists(index_path):
            index_path = fa_path + ".index.json"
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Missing index for {fa_path}")

        with open(index_path, 'r') as f:
            self.all_seq_offsets = json.load(f)

        if indices is not None:
            self.seq_offsets = [self.all_seq_offsets[i] for i in indices]
        else:
            self.seq_offsets = self.all_seq_offsets

        # 2. Locate NPY files
        # New pattern: {filename}_input_ids.npy (uint16)
        self.input_ids_path = fa_path + "_input_ids.npy"

        if not os.path.exists(self.input_ids_path):
            # Try fallback to naming without extra underscore if file exists
            if os.path.exists(fa_path + "input_ids.npy"):
                self.input_ids_path = fa_path + "input_ids.npy"
            else:
                raise FileNotFoundError(f"Missing .npy: {self.input_ids_path}")

        self.kmer_ids_path = fa_path + "_kmer_ids.npy"
        self.has_kmer_ids = os.path.exists(self.kmer_ids_path)

        # 3. Load Mode (RAM vs Memmap)
        try:
            import psutil
            mem = psutil.virtual_memory()
            file_size = os.path.getsize(self.input_ids_path)
            total_size = file_size

            # BPE tokens are stored as uint16 (2 bytes)
            total_tokens = sum(l for _, _, _, l in self.all_seq_offsets)

            if mem.available > total_size * 2:
                LOGGER.info("[NpyDataset] %s: Loading BPE uint16 into RAM...", self.name_root)
                self.input_ids_data = np.fromfile(self.input_ids_path, dtype='uint16').reshape((total_tokens,))
                if self.has_kmer_ids:
                    self.kmer_ids_data = np.fromfile(self.kmer_ids_path, dtype='uint16').reshape((total_tokens,))
            else:
                LOGGER.info("[NpyDataset] %s: Mapping BPE uint16...", self.name_root)
                self.input_ids_data = np.memmap(self.input_ids_path, dtype='uint16', mode='r', shape=(total_tokens,))
                if self.has_kmer_ids:
                    self.kmer_ids_data = np.memmap(self.kmer_ids_path, dtype='uint16', mode='r', shape=(total_tokens,))

        except Exception as e:
            LOGGER.warning("[NpyDataset] Using fallback memmap due to: %s", e)
            total_tokens = sum(l for _, _, _, l in self.all_seq_offsets)
            self.input_ids_data = np.memmap(self.input_ids_path, dtype='uint16', mode='r', shape=(total_tokens,))
            if self.has_kmer_ids:
                self.kmer_ids_data = np.memmap(self.kmer_ids_path, dtype='uint16', mode='r', shape=(total_tokens,))

        # 4. Pre-calculate chunk coordinates
        self.chunk_coords = []
        for i, (_, _, _, seq_len) in enumerate(self.seq_offsets):
            if seq_len < self.min_chunk_length:
                continue
            if seq_len <= self.max_length:
                self.chunk_coords.append((i, 0, seq_len))
            else:
                for s in range(0, seq_len - self.min_chunk_length + 1, max(1, self.stride)):
                    chunk_end = min(s + self.max_length, seq_len)
                    chunk_len = chunk_end - s
                    if chunk_len >= self.min_chunk_length:
                        self.chunk_coords.append((i, s, chunk_end))
                    if s + self.max_length >= seq_len:
                        break

        lengths = [l for _, _, _, l in self.all_seq_offsets]
        self.seq_starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))

    def __len__(self):
        return len(self.chunk_coords)

    def __getitem__(self, idx):
        seq_idx, start_local, end_local = self.chunk_coords[idx]
        original_idx, _, _, _ = self.seq_offsets[seq_idx]

        global_start = self.seq_starts[original_idx] + start_local
        global_end = self.seq_starts[original_idx] + end_local

        input_ids_np = np.array(self.input_ids_data[global_start:global_end], copy=True).astype(np.int64)
        attention_mask_np = (input_ids_np != PAD_TOKEN_ID).astype(np.int64)

        item = {
            "input_ids": torch.tensor(input_ids_np, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_np, dtype=torch.long),
        }

        if self.has_kmer_ids:
            kmer_ids_np = np.array(self.kmer_ids_data[global_start:global_end], copy=True).astype(np.int64)
            item["kmer_ids"] = torch.tensor(kmer_ids_np, dtype=torch.long)

        # Handle RC Consistency (Optional: check if RC .npy exists)
        if self.provide_rc:
             # If RC is stored as a separate file, load it here.
             # Otherwise, the train loop can compute it via tokenizer if needed.
             # Based on user's snippet, we check for a separate file:
             rc_path = self.fa_path + "_input_ids_rc.npy"
             if os.path.exists(rc_path):
                 # This logic mirrors input_ids but from RC file
                 # We'd need rc_seq_starts etc. to be precise.
                 # For now, we assume standard augmentation if RC file isn't found.
                 pass

        if not hasattr(self, 'file_path') or self.file_path is None:
             self.file_path = self.fa_path

        if self.file_path:
             item["domain_name"] = os.path.basename(self.file_path)

        return item


def build_loaders_balanced(
    files: Sequence[str],
    tokenizer: DnaTokenizer,
    *,
    max_length: int,
    stride: int,
    train_split_ratio: float,
    batch_size: int,
    num_workers: int,
    drop_last: bool,
    use_reverse_prob: float,
    provide_rc: bool = False,
    frameshift_prob: float = 0.0,

    max_chunks_per_file: Optional[int] = None,
    cap_by_file: Optional[dict] = None,
    split_manifest: Optional[str] = None,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    use_smart_batching: bool = False,
    seed: int = 42,
):
    """Build balanced lazy loaders with sequence-level train/eval splits."""
    from .dataset import DatasetImpl, collate

    if not (0.0 < train_split_ratio < 1.0):
        raise ValueError(f"train_split_ratio must be in (0,1), got {train_split_ratio}")
    seed = int(seed)

    names = []
    train_loaders = []
    pooled_eval_seqs = []
    first_train_seq = None
    manifest_splits = load_split_manifest(split_manifest, files) if split_manifest else None
    if split_manifest:
        LOGGER.info("[DataPrep] Using split manifest: %s", split_manifest)

    kept_total_windows = 0

    for path in files:
        if not os.path.exists(path):
            LOGGER.warning("[DataPrep] File not found: %s", path)
            continue

        # Scan file for lengths (fast, low memory)
        # Returns list of (index, start_offset, end_offset, length)
        seq_offsets = scan_fasta_lengths(path)
        if not seq_offsets:
            LOGGER.warning("[DataPrep] No sequences found in %s", path)
            continue

        # Filter by min_chunk_length
        min_chunk_length = max(64, max_length // 4)
        valid_items = [(i, l) for i, _, _, l in seq_offsets if l >= min_chunk_length]

        if not valid_items:
            LOGGER.warning("[DataPrep] No sequences >= %sbp in %s", min_chunk_length, path)
            continue

        manifest_for_path = split_indices_for_path(manifest_splits, path)
        if manifest_splits is not None:
            if manifest_for_path is None:
                raise ValueError(f"Split manifest has no assignments for {path}")
            valid_by_index = {i: l for i, l in valid_items}
            assigned_indices = manifest_for_path["train"] | manifest_for_path["eval"]
            out_of_range = sorted(index for index in assigned_indices if index >= len(seq_offsets))
            if out_of_range:
                raise ValueError(
                    f"Split manifest has out-of-range sequence indices for {path}: {out_of_range[:10]}"
                )
            train_items = [(i, valid_by_index[i]) for i in sorted(manifest_for_path["train"]) if i in valid_by_index]
            eval_items = [(i, valid_by_index[i]) for i in sorted(manifest_for_path["eval"]) if i in valid_by_index]
            if not train_items:
                raise ValueError(f"Split manifest produced no training sequences for {path}")
            if not eval_items:
                LOGGER.warning("[DataPrep] Split manifest produced no eval sequences for %s", path)
            total_windows = sum(count_chunks_exact(l, max_length, stride, min_chunk_length) for _, l in valid_items)
        else:
            # Split per file by exact chunk count
            # seq_windows = [((original_index, length), count) ...]
            seq_windows = [((i, l), count_chunks_exact(l, max_length, stride, min_chunk_length))
                           for i, l in valid_items]
            seq_windows = [(item, w) for item, w in seq_windows if w > 0]

            if not seq_windows:
                LOGGER.warning("[DataPrep] No valid chunks >= %sbp in %s", min_chunk_length, path)
                continue

            total_windows = sum(w for _, w in seq_windows)

            # Avoid a row-order split while preserving exact reproducibility.
            split_rng = random.Random(_derived_seed(seed, f"split:{os.path.abspath(path)}"))
            split_rng.shuffle(seq_windows)

            # Split by cumulative window count with better boundary handling
            target_train_windows = int(total_windows * train_split_ratio)
            train_items = [] # List of (original_index, length)
            eval_items = []  # List of (original_index, length)
            cumulative = 0

            for (item, wc) in seq_windows:
                if cumulative + wc <= target_train_windows:
                    # Entire sequence fits in training
                    train_items.append(item)
                    cumulative += wc
                elif cumulative < target_train_windows:
                    # Sequence straddles boundary - assign to whichever is closer
                    remaining_train = target_train_windows - cumulative
                    if remaining_train > wc / 2:
                        # More than half needed for train
                        train_items.append(item)
                        cumulative += wc
                    else:
                        # Send to eval
                        eval_items.append(item)
                else:
                    # Already met training quota
                    eval_items.append(item)

            # Ensure at least one sequence in each split
            if not train_items and valid_items:
                train_items = [valid_items[0]]
                eval_items = valid_items[1:] if len(valid_items) > 1 else []

        # Split per file by exact chunk count
        # seq_windows = [((original_index, length), count) ...]

        # Handle cap_by_file with warnings
        basename = os.path.basename(path)
        effective_cap = None

        if cap_by_file:
            if basename in cap_by_file:
                effective_cap = cap_by_file[basename]
            else:
                LOGGER.warning("[DataPrep] No cap specified for %s in cap_by_file", basename)
                if '*' in cap_by_file:
                    effective_cap = cap_by_file['*']
                    LOGGER.info("[DataPrep] Using default cap: %s", effective_cap)
                else:
                    effective_cap = max_chunks_per_file
                    LOGGER.info("[DataPrep] Using max_chunks_per_file: %s", effective_cap)
        else:
            effective_cap = max_chunks_per_file

        file_kept_windows = 0

        if effective_cap and effective_cap > 0:
            # Recalculate windows for train_items
            train_windows = [((i, l), count_chunks_exact(l, max_length, stride, min_chunk_length))
                           for i, l in train_items]
            train_windows = [(item, w) for item, w in train_windows if w > 0]
            train_windows.sort(key=lambda x: -x[1])  # Prioritize longer sequences

            kept = []
            running = 0
            for (item, c) in train_windows:
                if running >= effective_cap:
                    break
                kept.append(item)
                running += c

            if not kept and train_items:
                kept = [train_items[0]]  # Keep at least one

            LOGGER.info(
                "[DataPrep] %s: windows %s -> %s (capped at %s)",
                basename,
                sum(count_chunks_exact(l, max_length, stride, min_chunk_length) for i, l in train_items),
                running,
                effective_cap,
            )
            train_items = kept
            file_kept_windows = running
        else:
            file_kept_windows = sum(count_chunks_exact(l, max_length, stride, min_chunk_length)
                                  for i, l in train_items)
            LOGGER.info(
                "[DataPrep] %s: train_seqs=%s, eval_seqs=%s, windows~%s",
                basename,
                len(train_items),
                len(eval_items),
                file_kept_windows,
            )

        kept_total_windows += file_kept_windows

        # Extract indices for DatasetImpl
        train_indices = [i for i, l in train_items]

        npy_path = path + "_input_ids.npy"

        ds_train = None
        raw_sequence_required = provide_rc or use_reverse_prob > 0.0 or frameshift_prob > 0.0
        if getattr(tokenizer, "tokenizer_mode", "bpe") == "base":
            raw_sequence_required = True
        if os.path.exists(npy_path) and not raw_sequence_required:
             LOGGER.info("[DataPrep] %s: Found optimized .npy files. Using NpyDataset.", basename)
             try:
                 ds_train = NpyDataset(
                     path, max_length, stride, min_chunk_length, indices=train_indices, provide_rc=provide_rc
                 )
             except Exception as e:
                 LOGGER.warning("[DataPrep] Error initializing NpyDataset: %s. Falling back to standard loader.", e)
                 ds_train = None
        elif os.path.exists(npy_path) and raw_sequence_required:
             reasons = []
             if provide_rc:
                 reasons.append("RC pairs")
             if use_reverse_prob > 0.0:
                 reasons.append("reverse-complement augmentation")
             if frameshift_prob > 0.0:
                 reasons.append("pre-BPE frameshift augmentation")
             reason_text = ", ".join(reasons)
             LOGGER.info("[DataPrep] %s: .npy found, but %s requested; using FASTA loader.", basename, reason_text)

        if ds_train is None:
            # Fallback to standard DatasetImpl (FASTA text)
            try:
                import psutil
                mem = psutil.virtual_memory()
                file_size = os.path.getsize(path)
                # Threshold: If available RAM is > 1.5x file size, load into memory
                should_lazy = file_size * 1.5 > mem.available
                if not should_lazy:
                     LOGGER.info(
                         "[DataPrep] %s: %.1fMB fits in RAM (%.1fGB free). Loading into memory for speed.",
                         basename,
                         file_size / 1e6,
                         mem.available / 1e9,
                     )
                else:
                     LOGGER.info(
                         "[DataPrep] %s: %.1fMB too large for RAM. Using lazy loading.",
                         basename,
                         file_size / 1e6,
                     )
            except ImportError:
                should_lazy = True

            ds_train = DatasetImpl(
                tokenizer, path, max_length, stride,
                use_reverse_prob=use_reverse_prob,
                provide_rc=provide_rc,
                frameshift_prob=frameshift_prob,
                min_chunk_length=min_chunk_length,
                indices=train_indices,
                lazy_load=should_lazy,
                precomputed_offsets=seq_offsets,
                seed=_derived_seed(seed, f"dataset:{os.path.abspath(path)}"),
            )

        if len(ds_train) == 0:
            LOGGER.warning("[DataPrep] No valid chunks from %s, skipping", path)
            continue

        # Smart Batching Logic
        batch_sampler = None
        shuffle = True
        if use_smart_batching:
            LOGGER.info("[DataPrep] Enabling Smart Batching for %s", basename)
            batch_sampler = SmartBatchSampler(
                ds_train,
                batch_size,
                shuffle=True,
                seed=_derived_seed(seed, f"smart-batch:{os.path.abspath(path)}"),
            )
            shuffle = False # batch_sampler handles shuffling

        loader_generator = _torch_generator(_derived_seed(seed, f"loader:{os.path.abspath(path)}"))

        dl_train = DataLoader(
            ds_train,
            shuffle=shuffle,
            batch_sampler=batch_sampler,
            drop_last=drop_last if batch_sampler is None else False, # batch_sampler handles drop_last logic internally or we accept partial
            batch_size=1 if batch_sampler is not None else batch_size, # batch_size is ignored if batch_sampler is provided
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=functools.partial(collate, max_length=max_length),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            worker_init_fn=_seed_worker,
            generator=loader_generator,
        )
        train_loaders.append(dl_train)
        names.append(basename)

        # Collect eval sequences
        # Collect eval sequences
        if eval_items:
            # OPTIMIZED: Use seek() instead of linear scan
            eval_indices_set = set(i for i, l in eval_items)
            LOGGER.info("[DataPrep] Loading %s eval sequences via random access...", len(eval_indices_set))

            # Map original index to offset info
            # seq_offsets is list of (idx, start, end, len)
            # We assume seq_offsets is sorted by idx or access by index if it matches?
            # seq_offsets comes from scan_fasta_lengths which appends in order.
            # So seq_offsets[i] corresponds to sequence i.

            with open(path, "rb") as f:
                for idx in sorted(eval_indices_set):
                    # Retrieve offset info
                    if idx < len(seq_offsets):
                        _, start_offset, end_offset, _ = seq_offsets[idx]
                        f.seek(start_offset)
                        raw_bytes = f.read(end_offset - start_offset)
                        # Clean newlines
                        seq = raw_bytes.replace(b"\n", b"").replace(b"\r", b"").decode("utf-8")
                        pooled_eval_seqs.append(seq)
                    else:
                        LOGGER.error("[DataPrep] Index %s out of bounds for %s", idx, path)

        elif first_train_seq is None and train_items:
            # Need to get one train sequence for fallback
            # OPTIMIZED: Use seek() here too
            idx_to_load = train_items[0][0]
            if idx_to_load < len(seq_offsets):
                _, start_offset, end_offset, _ = seq_offsets[idx_to_load]
                with open(path, "rb") as f:
                    f.seek(start_offset)
                    raw_bytes = f.read(end_offset - start_offset)
                    first_train_seq = raw_bytes.replace(b"\n", b"").replace(b"\r", b"").decode("utf-8")

    # Ensure we have something for evaluation
    if not pooled_eval_seqs:
        if manifest_splits is not None:
            raise ValueError("Split manifest produced no evaluation sequences")
        if first_train_seq is not None:
            pooled_eval_seqs = [first_train_seq]
        else:
            raise ValueError("No sequences available for evaluation!")

    if manifest_splits is None:
        # Trim eval set to match desired ratio for legacy ratio-based splitting.
        r = float(train_split_ratio)
        desired_eval_windows = max(1, int(round(kept_total_windows * (1.0 - r) / r)))

        eval_accum = 0
        trimmed_eval = []

        random.Random(_derived_seed(seed, "pooled-eval")).shuffle(pooled_eval_seqs)

        for s in pooled_eval_seqs:
            w = count_chunks_exact(len(s), max_length, stride, min_chunk_length)
            if w <= 0:
                continue
            if eval_accum >= desired_eval_windows:
                break
            trimmed_eval.append(s)
            eval_accum += w

        pooled_eval_seqs = trimmed_eval

        LOGGER.info(
            "[DataPrep] Adjusted eval windows to %s (target: %s) to match %.1f%% train split ratio "
            "(kept train windows: %s)",
            eval_accum,
            desired_eval_windows,
            r * 100.0,
            kept_total_windows,
        )
    else:
        LOGGER.info("[DataPrep] Preserving eval sequences from split manifest: %s", len(pooled_eval_seqs))

    # Build evaluation loader
    ds_eval = DatasetImpl(
        tokenizer, pooled_eval_seqs, max_length, stride,
        use_reverse_prob=0.0,
        frameshift_prob=0.0,
        min_chunk_length=min_chunk_length,
        seed=_derived_seed(seed, "eval-dataset"),
    )

    # Eval usually fits in memory (via pooled_eval_seqs), but let's check NPY too if using entire file?
    # Actually, below we create ds_eval from pooled_eval_seqs which are STRINGS.
    # So we don't need NPY for eval unless we want to avoid loading strings.
    # But pooled_eval_seqs is already loaded. So let's leave eval as is for safety.

    eval_loader = DataLoader(
        ds_eval,
        shuffle=False,
        drop_last=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=_torch_generator(_derived_seed(seed, "eval-loader")),
    )

    LOGGER.info(
        "[DataPrep] FINAL: windows~%s, files=%s, cap=%s, train_chunks=%s, eval_chunks=%s",
        kept_total_windows,
        len(names),
        max_chunks_per_file,
        sum(len(dl.dataset) for dl in train_loaders),
        len(ds_eval),
    )

    # Create round-robin loader
    if not train_loaders:
        raise ValueError("No training data available!")

    train_loader = RoundRobinLoader(train_loaders, names=names)

    # Final verification
    kept_train = sum(len(dl.dataset) for dl in train_loaders)
    kept_eval = len(ds_eval)
    tot = kept_train + kept_eval if (kept_train + kept_eval) > 0 else 1

    LOGGER.info(
        "[DataPrep] KEPT: train=%s (%.1f%%), eval=%s (%.1f%%), target=%.1f%%",
        kept_train,
        kept_train / tot * 100.0,
        kept_eval,
        kept_eval / tot * 100.0,
        train_split_ratio * 100.0,
    )

    return train_loader, eval_loader
