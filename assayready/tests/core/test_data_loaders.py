from __future__ import annotations

from pathlib import Path

import torch
import pytest

from model_assessment.data_core.dataset import DatasetImpl
from model_assessment.data_core.loaders import (
    RoundRobinLoader,
    SmartBatchSampler,
    build_loaders_balanced,
    load_split_manifest,
    scan_fasta_lengths,
)


class _Loader:
    def __init__(self, rows: list[int]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)


def test_round_robin_loader_stops_after_total_len() -> None:
    loader = RoundRobinLoader([_Loader([1, 2]), _Loader([10])])
    rows = list(iter(loader))
    assert rows == [1, 10, 2]
    assert len(rows) == len(loader)


def test_round_robin_loader_can_iterate_multiple_times() -> None:
    loader = RoundRobinLoader([_Loader([1, 2]), _Loader([10])])
    rows1 = list(loader)
    rows2 = list(loader)
    assert rows1 == [1, 10, 2]
    assert rows2 == [1, 10, 2]


def test_round_robin_loader_honors_per_loader_limits() -> None:
    loader = RoundRobinLoader(
        [_Loader([1, 2, 3]), _Loader([10, 11])],
        steps_per_loader=[2, 1],
    )
    assert len(loader) == 3
    assert list(loader) == [1, 10, 2]


class _Tokenizer:
    tokenizer_mode = "base"

    def encode(self, sequence: str, max_length: int):
        values = [5 + "ACGT".index(base) for base in sequence[:max_length]]
        return {
            "input_ids": torch.tensor(values, dtype=torch.long),
            "attention_mask": torch.ones(len(values), dtype=torch.long),
        }


def test_lazy_dataset_accepts_legacy_three_tuple_offsets(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fa"
    fasta.write_text(">seq\nACGTACGT\n", encoding="utf-8")
    four_tuple = scan_fasta_lengths(str(fasta))[0]
    dataset = DatasetImpl(
        _Tokenizer(),
        str(fasta),
        max_length=8,
        stride=8,
        min_chunk_length=1,
        lazy_load=True,
        precomputed_offsets=[four_tuple[1:]],
    )

    assert dataset[0]["input_ids"].tolist() == [5, 6, 7, 8, 5, 6, 7, 8]


class _LengthDataset:
    def __len__(self) -> int:
        return 12

    def get_lengths(self) -> list[int]:
        return list(range(12))


def test_smart_batch_sampler_seed_and_epoch_are_deterministic() -> None:
    left = SmartBatchSampler(_LengthDataset(), batch_size=2, seed=17)
    right = SmartBatchSampler(_LengthDataset(), batch_size=2, seed=17)
    assert list(left) == list(right)

    left.set_epoch(3)
    epoch_three = list(left)
    left.set_epoch(3)
    assert list(left) == epoch_three


def test_balanced_loader_split_uses_the_explicit_seed(tmp_path: Path) -> None:
    alphabet = "ACGT"

    def suffix(number: int) -> str:
        chars = []
        for _ in range(4):
            chars.append(alphabet[number % 4])
            number //= 4
        return "".join(chars)

    fasta = tmp_path / "seeded.fa"
    fasta.write_text(
        "".join(f">seq-{idx}\n{'ACGT' * 15}{suffix(idx)}\n" for idx in range(12)),
        encoding="utf-8",
    )

    def train_indices(seed: int) -> list[int]:
        train, _ = build_loaders_balanced(
            [str(fasta)],
            _Tokenizer(),
            max_length=64,
            stride=64,
            train_split_ratio=0.75,
            batch_size=2,
            num_workers=0,
            drop_last=False,
            use_reverse_prob=0.0,
            pin_memory=False,
            use_smart_batching=True,
            seed=seed,
        )
        return train.loaders[0].dataset.indices

    assert train_indices(7) == train_indices(7)
    assert train_indices(7) != train_indices(8)


def test_split_manifest_rejects_cross_split_duplicates(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fa"
    fasta.write_text(">seq\nACGT\n", encoding="utf-8")
    manifest = tmp_path / "split.jsonl"
    manifest.write_text(
        "\n".join(
            [
                '{"path": "tiny.fa", "seq_index": 0, "split": "train"}',
                '{"path": "tiny.fa", "seq_index": 0, "split": "test"}',
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="both train and eval"):
        load_split_manifest(str(manifest), [str(fasta)])
