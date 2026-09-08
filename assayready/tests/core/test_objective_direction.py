from __future__ import annotations

import numpy as np
from model_assessment.cli import (
    _claim_gate,
    _observed_incumbent,
    _rank_external_predictions,
    _ranking_audit,
)


def test_rank_external_predictions_minimization() -> None:
    # 2 rows: sequence 1 has prediction 10.0, uncertainty 2.0; sequence 2 has prediction 8.0, uncertainty 1.0
    # For minimization, acquisition score is prediction - beta * uncertainty (LCB)
    # With beta = 1.0:
    # row1 LCB = 10.0 - 2.0 = 8.0
    # row2 LCB = 8.0 - 1.0 = 7.0
    # Diverse ranker selects variants to minimize LCB, i.e., rank by maximizing negated LCB:
    # row1 negated LCB = -8.0
    # row2 negated LCB = -7.0
    # Therefore, row2 (-7.0) is ranked first, row1 (-8.0) is ranked second.
    rows = [
        {"seq": "AAAA", "prediction": 10.0, "uncertainty": 2.0},
        {"seq": "TTTT", "prediction": 8.0, "uncertainty": 1.0},
    ]

    ranked = _rank_external_predictions(
        rows,
        sequence_col="seq",
        prediction_col="prediction",
        uncertainty_col="uncertainty",
        uncertainty_type="predictive_standard_deviation",
        beta=1.0,
        diversity_method="greedy_embedding_cosine",
        diversity_penalty=0.0,
        top_k=2,
        objective_direction="minimize",
    )

    assert len(ranked) == 2
    # First is TTTT (rank 1)
    assert ranked[0]["seq"] == "TTTT"
    assert ranked[0]["rank"] == 1
    # Check that acquisition_score is the display LCB score (not negated)
    assert ranked[0]["acquisition_score"] == 7.0
    assert ranked[1]["acquisition_score"] == 8.0


def test_candidate_prioritization_check_in_claim_gate() -> None:
    evaluation_manifest = {
        "claim_thresholds": {
            "min_test_rows": 2,
            "min_num_clusters": 1,
            "min_lift_delta": 0.0,
        },
        "objective_direction": "maximize",
        "training_independence": "internally_controlled",
        "constraints_verified": True,
    }

    # Case 1: Pass
    gate_pass = _claim_gate(
        task_type="regression",
        split_diagnostics={"split_sizes": {"test": 5}, "num_clusters": 2},
        warnings=[],
        best_baseline_metric=0.1,
        model_metric=0.5, # lift is 0.4 >= 0.0
        uncertainty_audit=None,
        ranked_candidates=10,
        evaluation_manifest=evaluation_manifest,
    )
    assert gate_pass["candidate_prioritization"]["ok"] is True
    assert "recommended" not in gate_pass

    # Case 2: Fail due to baseline beating model
    gate_fail_lift = _claim_gate(
        task_type="regression",
        split_diagnostics={"split_sizes": {"test": 5}, "num_clusters": 2},
        warnings=[],
        best_baseline_metric=0.6,
        model_metric=0.5, # lift is -0.1 < 0.0
        uncertainty_audit=None,
        ranked_candidates=10,
        evaluation_manifest=evaluation_manifest,
    )
    assert gate_fail_lift["candidate_prioritization"]["ok"] is False


def test_metric_lift_stays_higher_is_better_for_minimization_targets() -> None:
    gate = _claim_gate(
        task_type="regression",
        split_diagnostics={"split_sizes": {"test": 5}, "num_clusters": 2},
        warnings=[],
        best_baseline_metric=0.1,
        model_metric=0.5,
        uncertainty_audit=None,
        ranked_candidates=2,
        evaluation_manifest={
            "claim_thresholds": {"min_test_rows": 2, "min_num_clusters": 1, "min_lift_delta": 0.0},
            "objective_direction": "minimize",
            "training_independence": "internally_controlled",
            "constraints_verified": True,
        },
    )
    assert gate["lift_claim"]["ok"] is True


def test_external_evidence_check_requires_verified_holdout() -> None:
    gate = _claim_gate(
        task_type="regression",
        split_diagnostics={"split_sizes": {"test": 5}, "num_clusters": 2},
        warnings=[],
        best_baseline_metric=0.1,
        model_metric=0.5,
        uncertainty_audit=None,
        ranked_candidates=2,
        evaluation_manifest={
            "claim_thresholds": {"min_test_rows": 2, "min_num_clusters": 1, "min_lift_delta": 0.0},
            "training_independence": "unverified",
            "constraints_verified": True,
        },
        external_predictions=True,
    )
    assert gate["leakage_controlled"]["ok"] is False
    assert gate["candidate_prioritization"]["ok"] is False


def test_missing_uncertainty_uses_explicit_prediction_only_policy() -> None:
    ranked = _rank_external_predictions(
        [
            {"seq": "AAAA", "prediction": 1.0, "uncertainty": None},
            {"seq": "CCCC", "prediction": 0.5, "uncertainty": 0.2},
        ],
        sequence_col="seq",
        prediction_col="prediction",
        uncertainty_col="uncertainty",
        uncertainty_type="predictive_standard_deviation",
        beta=1.0,
        diversity_method="kmer_cosine",
        diversity_penalty=0.0,
        top_k=2,
    )
    assert {row["acquisition_policy"] for row in ranked} == {"prediction_only"}
    missing = next(row for row in ranked if row["seq"] == "AAAA")
    assert missing["uncertainty"] is None
    assert "missing_or_invalid" in missing["uncertainty_status"]


def test_minimization_ranking_audit_uses_low_values_as_best() -> None:
    result = _ranking_audit(
        [
            {"target": 10.0, "prediction": 9.0},
            {"target": 1.0, "prediction": 2.0},
            {"target": 5.0, "prediction": 6.0},
        ],
        task_type="regression",
        target_col="target",
        prediction_col="prediction",
        top_k=1,
        positive_label=None,
        objective_direction="minimize",
    )
    assert result["best_true_item_rank_by_prediction"] == 1
    assert result["top_k_target_mean"] == 1.0
    assert result["objective_direction"] == "minimize"


def test_minimization_incumbent_is_lowest_observed_target() -> None:
    rows = [{"target": 7.0}, {"target": 2.0}, {"target": 4.0}]
    assert (
        _observed_incumbent(
            rows,
            target_col="target",
            task_type="regression",
            objective_direction="minimize",
        )
        == 2.0
    )
