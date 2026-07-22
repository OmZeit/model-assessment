from __future__ import annotations

from itertools import combinations
from pathlib import Path

from model_assessment.ml_core.specialization import (
    UnionFind,
    _union_homologous_kmer_sets,
    ingest_assay_tables,
    jaccard_similarity,
    leakage_safe_split,
    write_table,
)


def test_write_table_sanitizes_csv_formula_prefixes(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    write_table(
        path,
        [
            {"id": "a", "sequence": "=HYPERLINK(\"http://example.com\")"},
            {"id": "b", "sequence": "+SUM(1,2)"},
            {"id": "c", "sequence": "@cmd"},
        ],
        fieldnames=["id", "sequence"],
    )
    text = path.read_text(encoding="utf-8")
    assert "'=HYPERLINK" in text
    assert "'+SUM" in text
    assert "'@cmd" in text


def test_leakage_safe_split_warns_when_clusters_are_insufficient() -> None:
    rows = [
        {"sequence": "ACGTACGTACGT", "target": 1.0},
        {"sequence": "ACGTACGTACGA", "target": 1.1},
    ]
    splits, diagnostics = leakage_safe_split(rows, sequence_col="sequence", homology_threshold=0.0)
    assert len(splits["train"]) == 2
    assert any("fewer than 3" in warning for warning in diagnostics["warnings"])


def test_homology_ppjoin_matches_exhaustive_jaccard_components() -> None:
    universe = tuple("abcde")
    token_sets = [
        {universe[index] for index in range(len(universe)) if mask & (1 << index)}
        for mask in range(1 << len(universe))
    ]
    saw_position_pruning = False

    def partition(union_find: UnionFind) -> list[list[int]]:
        components: dict[int, list[int]] = {}
        for index in range(len(token_sets)):
            components.setdefault(union_find.find(index), []).append(index)
        return sorted(sorted(component) for component in components.values())

    for threshold in (0.01, 0.25, 0.5, 0.75, 0.9, 1.0):
        brute_force = UnionFind(len(token_sets))
        for left, right in combinations(range(len(token_sets)), 2):
            if jaccard_similarity(token_sets[left], token_sets[right]) >= threshold:
                brute_force.union(left, right)

        indexed = UnionFind(len(token_sets))
        diagnostics = _union_homologous_kmer_sets(
            token_sets,
            threshold=threshold,
            union_find=indexed,
        )
        saw_position_pruning |= diagnostics["homology_position_pruned_pairs"] > 0

        assert partition(indexed) == partition(brute_force)

    assert saw_position_pruning


def test_homology_prefix_index_preserves_empty_set_clusters() -> None:
    token_sets = [set(), set(), {"A"}, set()]
    union_find = UnionFind(len(token_sets))

    diagnostics = _union_homologous_kmer_sets(token_sets, threshold=0.9, union_find=union_find)

    assert union_find.find(0) == union_find.find(1) == union_find.find(3)
    assert union_find.find(0) != union_find.find(2)
    assert diagnostics["homology_duplicate_rows"] == 2
    assert diagnostics["homology_candidate_pairs"] == 0


def test_homology_ppjoin_skips_redundant_dense_component_checks() -> None:
    shared = {f"shared_{index}" for index in range(20)}
    token_sets = [shared | {f"unique_{index}"} for index in range(20)]
    union_find = UnionFind(len(token_sets))

    diagnostics = _union_homologous_kmer_sets(token_sets, threshold=0.9, union_find=union_find)

    assert len({union_find.find(index) for index in range(len(token_sets))}) == 1
    assert diagnostics["homology_component_skipped_pairs"] > 0
    assert diagnostics["homology_candidate_pairs"] < diagnostics["homology_prefix_candidates"]


def test_ingest_assay_tables_keeps_zero_numeric_targets(tmp_path: Path) -> None:
    assay = tmp_path / "assay.csv"
    write_table(
        assay,
        [
            {"sequence": "ACGT", "target": 0},
            {"sequence": "TGCA", "target": 1},
        ],
        fieldnames=["sequence", "target"],
    )

    rows, audit, failures = ingest_assay_tables(
        [assay],
        project="zero_target_test",
        task_type="regression",
        sequence_col="sequence",
        target_col="target",
        require_target=True,
    )

    assert len(rows) == 2
    assert audit.missing_target_rows == 0
    assert failures == []


def test_normalize_class_label_handles_integer_floats() -> None:
    from model_assessment.ml_core.specialization import _normalize_class_label
    assert _normalize_class_label(1.0) == "1"
    assert _normalize_class_label("1.0") == "1"
    assert _normalize_class_label(0.0) == "0"
    assert _normalize_class_label("0.0") == "0"
    assert _normalize_class_label("abc") == "abc"
    assert _normalize_class_label(None) is None
    assert _normalize_class_label("nan") is None
