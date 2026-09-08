from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from model_assessment.ml_core.specialization import (
    UnionFind,
    _union_homologous_kmer_sets,
    greedy_diverse_rank,
    ingest_assay_tables,
    jaccard_similarity,
    leakage_safe_split,
    register_similarity_policy,
    registered_similarity_policies,
    run_baselines,
    sequence_similarity,
    unregister_similarity_policy,
    write_table,
)


def _cluster_by_id(splits: dict[str, list[dict]]) -> dict[str, str]:
    return {
        str(row["id"]): str(row["leakage_cluster"])
        for split_rows in splits.values()
        for row in split_rows
    }


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
    assert diagnostics["split_cluster_counts"] == {"train": 2, "val": 0, "test": 0}
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


def test_exact_reverse_complement_policy_clusters_reverse_complements() -> None:
    rows = [
        {"id": "forward", "sequence": "AAGTCCGA"},
        {"id": "reverse", "sequence": "TCGGACTT"},
        {"id": "other", "sequence": "AAAAAAAA"},
    ]

    splits, diagnostics = leakage_safe_split(
        rows,
        sequence_col="sequence",
        val_fraction=0.0,
        test_fraction=0.0,
        homology_threshold=1.0,
        similarity_policy="exact_reverse_complement",
    )
    clusters = _cluster_by_id(splits)

    assert clusters["forward"] == clusters["reverse"]
    assert clusters["forward"] != clusters["other"]
    assert diagnostics["similarity_policy"] == "exact_reverse_complement"
    assert diagnostics["similarity_threshold"] == 1.0
    assert diagnostics["homology_search_strategy"] == "deterministic_pairwise_exact_reverse_complement"


def test_edit_distance_and_kmer_policies_produce_distinct_components() -> None:
    rows = [
        {"id": "left", "sequence": "AAAACCCC"},
        {"id": "right", "sequence": "CCCCAAAA"},
    ]
    kmer_similarity = sequence_similarity(
        rows[0]["sequence"], rows[1]["sequence"], policy="kmer_jaccard", k=3
    )
    edit_similarity = sequence_similarity(
        rows[0]["sequence"], rows[1]["sequence"], policy="edit_distance", k=3
    )
    assert kmer_similarity > 0.30
    assert edit_similarity < 0.30

    kmer_splits, kmer_diagnostics = leakage_safe_split(
        rows,
        sequence_col="sequence",
        val_fraction=0.0,
        test_fraction=0.0,
        homology_threshold=0.30,
        homology_k=3,
        similarity_policy="minhash_kmer",
    )
    edit_splits, edit_diagnostics = leakage_safe_split(
        rows,
        sequence_col="sequence",
        val_fraction=0.0,
        test_fraction=0.0,
        homology_threshold=0.30,
        homology_k=3,
        similarity_policy="edit_distance",
    )

    assert len(set(_cluster_by_id(kmer_splits).values())) == 1
    assert len(set(_cluster_by_id(edit_splits).values())) == 2
    assert kmer_diagnostics["similarity_policy"] == "canonical_kmer_jaccard"
    assert kmer_diagnostics["similarity_policy_requested"] == "minhash_kmer"
    assert edit_diagnostics["similarity_policy"] == "edit_distance"


def test_position_aware_motif_policy_distinguishes_motif_location() -> None:
    left = "ACGAAAAAA"
    moved = "AAAAAAACG"
    kmer_similarity = sequence_similarity(left, moved, policy="canonical_kmer_jaccard", k=3)
    position_similarity = sequence_similarity(left, moved, policy="position_aware_motif", k=3)

    assert sequence_similarity(left, left, policy="position_aware_motif", k=3) == 1.0
    assert position_similarity < kmer_similarity

    rows = [{"id": "left", "sequence": left}, {"id": "moved", "sequence": moved}]
    kmer_splits, _ = leakage_safe_split(
        rows,
        sequence_col="sequence",
        val_fraction=0.0,
        test_fraction=0.0,
        homology_threshold=0.30,
        homology_k=3,
        similarity_policy="canonical_kmer_jaccard",
    )
    position_splits, _ = leakage_safe_split(
        rows,
        sequence_col="sequence",
        val_fraction=0.0,
        test_fraction=0.0,
        homology_threshold=0.30,
        homology_k=3,
        similarity_policy="position_aware_motif",
    )
    assert len(set(_cluster_by_id(kmer_splits).values())) == 1
    assert len(set(_cluster_by_id(position_splits).values())) == 2


def test_customer_family_labels_uses_only_declared_groups() -> None:
    rows = [
        {"id": "family_a_1", "family": "A"},
        {"id": "family_a_2", "family": "A"},
        {"id": "family_b", "family": "B"},
    ]
    splits, diagnostics = leakage_safe_split(
        rows,
        sequence_col="sequence",
        group_cols=["family"],
        val_fraction=0.0,
        test_fraction=0.0,
        similarity_policy="customer_family_labels",
    )
    clusters = _cluster_by_id(splits)

    assert clusters["family_a_1"] == clusters["family_a_2"]
    assert clusters["family_a_1"] != clusters["family_b"]
    assert diagnostics["similarity_policy"] == "customer_family_labels"
    assert diagnostics["homology_search_strategy"] == "customer_family_labels_only"

    with pytest.raises(ValueError, match="requires at least one group_cols"):
        leakage_safe_split(
            rows,
            sequence_col="sequence",
            val_fraction=0.0,
            test_fraction=0.0,
            similarity_policy="customer_family_labels",
        )


def test_registered_similarity_plugin_is_available_to_split_and_ranking() -> None:
    policy = "test_shared_prefix"
    register_similarity_policy(policy, lambda left, right: float(left[:2] == right[:2]))
    try:
        assert policy in registered_similarity_policies()
        assert sequence_similarity("AACCCC", "AAGGGG", policy=policy) == 1.0

        rows = [
            {"id": "first", "sequence": "AACCCC"},
            {"id": "second", "sequence": "AAGGGG"},
            {"id": "third", "sequence": "TTCCCC"},
        ]
        splits, diagnostics = leakage_safe_split(
            rows,
            sequence_col="sequence",
            val_fraction=0.0,
            test_fraction=0.0,
            homology_threshold=1.0,
            similarity_policy=policy,
        )
        clusters = _cluster_by_id(splits)
        assert clusters["first"] == clusters["second"]
        assert clusters["first"] != clusters["third"]
        assert diagnostics["similarity_policy"] == policy

        ranked = greedy_diverse_rank(
            rows,
            np.asarray([3.0, 2.0, 1.0]),
            sequence_col="sequence",
            method=policy,
            diversity_penalty=0.2,
            top_k=3,
        )
        assert {row["diversity_policy"] for row in ranked} == {policy}
    finally:
        assert unregister_similarity_policy(policy) is True
    assert policy not in registered_similarity_policies()


def test_unknown_similarity_and_diversity_policies_are_rejected() -> None:
    rows = [{"id": "one", "sequence": "AAAA"}, {"id": "two", "sequence": "CCCC"}]
    with pytest.raises(ValueError, match="Unknown similarity_policy"):
        leakage_safe_split(rows, sequence_col="sequence", similarity_policy="not_registered")
    with pytest.raises(ValueError, match="Unsupported diversity method"):
        greedy_diverse_rank(
            rows,
            np.asarray([1.0, 0.0]),
            sequence_col="sequence",
            method="not_registered",
        )


def test_default_diversity_alias_records_actual_kmer_cosine_policy() -> None:
    rows = [{"sequence": "AAAA"}, {"sequence": "CCCC"}]
    ranked = greedy_diverse_rank(
        rows,
        np.asarray([1.0, 0.5]),
        sequence_col="sequence",
        method="greedy_embedding_cosine",
        top_k=2,
    )

    assert {row["diversity_policy"] for row in ranked} == {"kmer_cosine"}
    assert {row["diversity_policy_requested"] for row in ranked} == {"greedy_embedding_cosine"}


def test_run_baselines_optionally_returns_aligned_plain_float_predictions() -> None:
    train_rows = [
        {"sequence": "ACGTACGT" + base, "target": float(index)}
        for index, base in enumerate("ACGTAC")
    ]
    test_rows = [
        {"sequence": "TGCATGCA" + base, "target": float(index)}
        for index, base in enumerate("GT")
    ]

    skipped = run_baselines(
        [],
        test_rows,
        task_type="regression",
        sequence_col="sequence",
        target_col="target",
    )
    assert all("test_predictions" not in metrics for metrics in skipped.values())

    baselines = run_baselines(
        train_rows,
        test_rows,
        task_type="regression",
        sequence_col="sequence",
        target_col="target",
        include_predictions=True,
    )
    for metrics in baselines.values():
        assert len(metrics["test_predictions"]) == len(test_rows)
        assert all(type(value) is float for value in metrics["test_predictions"])


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
