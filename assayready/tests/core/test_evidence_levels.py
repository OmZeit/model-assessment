from __future__ import annotations

from model_assessment.cli import (
    _derive_assurance_level,
    _derive_evidence_level,
    _paired_lift_bootstrap,
    _threshold_sensitivity_analysis,
)


def _passing_checks() -> dict:
    return {
        "leakage_controlled": {"ok": True, "reasons": []},
        "lift_claim": {"ok": True, "reasons": []},
    }


def _stable_sensitivity() -> dict:
    return {
        "stable_under_stricter_thresholds": True,
        "conclusion_changes_under_stricter_thresholds": False,
    }


def test_assurance_levels_are_derived_from_execution_and_artifacts() -> None:
    self_declared = _derive_assurance_level(
        {"training_independence": "unverified", "provenance": {}},
        external_predictions=True,
    )
    assert self_declared["key"] == "self_declared"

    hashes = {
        "training_data_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "code_sha256": "d" * 64,
    }
    provenance = _derive_assurance_level(
        {"training_independence": "verified_holdout", "provenance": hashes},
        external_predictions=True,
    )
    assert provenance["key"] == "provenance_verified"

    controlled_provenance = _derive_assurance_level(
        {
            "training_independence": "verified_holdout",
            "provenance": {
                **hashes,
                "artifact_verification": "controlled_execution_manifest_sha256:" + "e" * 64,
            },
        },
        external_predictions=True,
    )
    assert controlled_provenance["key"] == "provenance_verified"
    assert any("controlled-execution" in item for item in controlled_provenance["basis"])
    assert not any("did not execute" in item for item in controlled_provenance["limitations"])

    controlled = _derive_assurance_level(
        {"training_independence": "internally_controlled", "provenance": {}},
        external_predictions=False,
    )
    assert controlled["key"] == "controlled_evaluation"

    prospective = _derive_assurance_level(
        {
            "training_independence": "verified_holdout",
            "provenance": hashes,
            "prospective_protocol": {"artifact_hashes_verified": True},
        },
        external_predictions=True,
    )
    assert prospective["key"] == "prospective_evaluation"


def test_evidence_levels_are_capped_by_assurance() -> None:
    checks = _passing_checks()
    stable = _stable_sensitivity()

    insufficient = _derive_evidence_level(
        claim_gate=checks,
        assurance_level={"key": "controlled_evaluation", "limitations": []},
        threshold_sensitivity=stable,
        metric=None,
        evaluation_manifest={"training_independence": "internally_controlled"},
    )
    assert insufficient["key"] == "insufficient_evidence"

    descriptive = _derive_evidence_level(
        claim_gate=checks,
        assurance_level={"key": "self_declared", "limitations": []},
        threshold_sensitivity=stable,
        metric=0.5,
        evaluation_manifest={"training_independence": "unverified"},
    )
    assert descriptive["key"] == "descriptive_audit_only"

    retrospective = _derive_evidence_level(
        claim_gate=checks,
        assurance_level={"key": "controlled_evaluation", "limitations": []},
        threshold_sensitivity=stable,
        metric=0.5,
        evaluation_manifest={"training_independence": "internally_controlled"},
    )
    assert retrospective["key"] == "retrospectively_credible"

    locked = _derive_evidence_level(
        claim_gate=checks,
        assurance_level={"key": "provenance_verified", "limitations": []},
        threshold_sensitivity=stable,
        metric=0.5,
        evaluation_manifest={"training_independence": "verified_holdout"},
    )
    assert locked["key"] == "locked_holdout_credible"

    prospective = _derive_evidence_level(
        claim_gate=checks,
        assurance_level={"key": "prospective_evaluation", "limitations": []},
        threshold_sensitivity=stable,
        metric=0.5,
        evaluation_manifest={"training_independence": "verified_holdout"},
    )
    assert prospective["key"] == "prospectively_demonstrated"


def test_unevaluated_similarity_sensitivity_caps_evidence_at_descriptive() -> None:
    result = _derive_evidence_level(
        claim_gate=_passing_checks(),
        assurance_level={"key": "controlled_evaluation", "limitations": []},
        threshold_sensitivity=_stable_sensitivity(),
        metric=0.5,
        evaluation_manifest={"training_independence": "internally_controlled"},
        similarity_sensitivity={"status": "not_evaluated"},
    )
    assert result["key"] == "descriptive_audit_only"
    assert any("not evaluated" in item for item in result["limitations"])


def test_threshold_sensitivity_reports_a_stricter_conclusion_change() -> None:
    result = _threshold_sensitivity_analysis(
        task_type="regression",
        split_diagnostics={"split_sizes": {"test": 60}, "num_clusters": 6},
        warnings=[],
        best_baseline_metric=0.10,
        model_metric=0.20,
        uncertainty_audit=None,
        ranked_candidates=4,
        evaluation_manifest={
            "claim_thresholds": {
                "min_test_rows": 20,
                "min_num_clusters": 3,
                "min_lift_delta": 0.0,
            },
            "training_independence": "internally_controlled",
            "constraints_verified": True,
        },
        cross_split_violations=[],
        model_metric_ci=None,
        lift_delta_ci=None,
        external_predictions=False,
    )
    assert result["profiles"][0]["retrospective_criteria_met"] is True
    assert result["profiles"][-1]["retrospective_criteria_met"] is False
    assert result["conclusion_changes_under_stricter_thresholds"] is True


def test_claim_gate_counts_clusters_in_test_split_only() -> None:
    result = _threshold_sensitivity_analysis(
        task_type="regression",
        split_diagnostics={
            "split_sizes": {"test": 60},
            "num_clusters": 12,
            "split_cluster_counts": {"train": 11, "test": 1},
        },
        warnings=[],
        best_baseline_metric=0.1,
        model_metric=0.5,
        uncertainty_audit=None,
        ranked_candidates=0,
        evaluation_manifest={
            "claim_thresholds": {"min_test_rows": 20, "min_num_clusters": 3, "min_lift_delta": 0.0},
            "training_independence": "internally_controlled",
            "constraints_verified": False,
        },
        cross_split_violations=[],
        model_metric_ci=None,
        lift_delta_ci=None,
        external_predictions=False,
    )
    assert result["profiles"][0]["retrospective_criteria_met"] is False
    assert "clusters 1 < required 3" in result["profiles"][0]["failed_checks"]["leakage_controlled"]


def test_paired_lift_bootstrap_uses_aligned_clustered_predictions() -> None:
    rows = [
        {
            "target": float(index),
            "prediction": float(index) + (0.05 if index % 2 else -0.05),
            "leakage_cluster": f"cluster_{index // 2}",
        }
        for index in range(8)
    ]
    result = _paired_lift_bootstrap(
        rows,
        baseline_predictions=[3.5] * len(rows),
        task_type="regression",
        target_col="target",
        prediction_col="prediction",
        positive_label=None,
        baseline_name="constant",
        n_resamples=80,
        seed=7,
    )
    assert result["status"] == "ok"
    assert result["bootstrap_unit"] == "leakage_cluster"
    assert result["method"] == "paired_bootstrap_model_minus_baseline"
    assert result["interval"][0] > 0
