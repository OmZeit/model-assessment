import random
import time
from typing import Any

import pytest

from model_assessment.ml_core.specialization import UnionFind, _union_homologous_kmer_sets, leakage_safe_split


def _generate_dummy_data(n: int, seq_len: int = 100) -> list[dict[str, Any]]:
    rows = []
    bases = ["A", "C", "G", "T"]
    for i in range(n):
        seq = "".join(random.choices(bases, k=seq_len))
        rows.append({"id": f"seq_{i}", "sequence": seq, "group": f"G{i % 5}"})
    return rows


@pytest.mark.performance
@pytest.mark.parametrize("n", [100, 500, 1000, 2000])
def test_leakage_safe_split_performance(n: int) -> None:
    random.seed(42)
    rows = _generate_dummy_data(n)

    start_time = time.perf_counter()
    splits, diagnostics = leakage_safe_split(
        rows,
        sequence_col="sequence",
        group_cols=["group"],
        seed=13,
    )
    elapsed = time.perf_counter() - start_time

    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == n

    assert diagnostics["homology_search_strategy"] == "exact_jaccard_ppjoin"
    assert diagnostics["homology_candidate_pairs"] <= diagnostics["homology_possible_pairs"] * 0.02
    assert elapsed < 15.0, f"Splitting {n} rows took too long: {elapsed:.2f}s"


@pytest.mark.performance
def test_dense_homology_join_avoids_redundant_exact_checks() -> None:
    num_rows = 500
    shared = {f"shared_{index}" for index in range(100)}
    token_sets = [shared | {f"unique_{index}"} for index in range(num_rows)]
    union_find = UnionFind(num_rows)

    start_time = time.perf_counter()
    diagnostics = _union_homologous_kmer_sets(
        token_sets,
        threshold=0.9,
        union_find=union_find,
    )
    elapsed = time.perf_counter() - start_time

    assert len({union_find.find(index) for index in range(num_rows)}) == 1
    assert diagnostics["homology_candidate_pairs"] == num_rows - 1
    assert diagnostics["homology_component_skipped_pairs"] > 100_000
    assert elapsed < 10.0, f"Dense homology join took too long: {elapsed:.2f}s"
