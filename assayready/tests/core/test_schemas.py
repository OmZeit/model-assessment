from __future__ import annotations

import pytest

from model_assessment.schemas import inferred_manifest_for_args, validate_evaluation_manifest


def _config() -> dict:
    return {
        "evaluation": {
            "schema_version": 1,
            "semantics": {
                "objective_direction": "minimize",
                "units": "micromolar",
                "uncertainty_type": "predictive_standard_deviation",
            },
            "constraints_verified": True,
            "biological_constraints": ["passes synthesis review"],
            "provenance": {
                "model_identifier": "model-v1",
                "data_identifier": "dataset-v1",
                "split_identifier": "locked-test-v1",
                "code_revision": "abc123",
                "duration_seconds": 1.2,
                "artifact_verification": "sha256",
                "training_independence": "verified_holdout",
                "training_data_sha256": "a" * 64,
                "model_sha256": "b" * 64,
                "split_sha256": "c" * 64,
                "code_sha256": "d" * 64,
                "dependency_versions": {"numpy": "2.4"},
            },
        }
    }


def test_manifest_preserves_claim_critical_semantics() -> None:
    manifest = validate_evaluation_manifest(_config(), task_type="regression").to_dict()
    assert manifest["objective_direction"] == "minimize"
    assert manifest["training_independence"] == "verified_holdout"
    assert manifest["constraints_verified"] is True


def test_classification_manifest_requires_positive_label() -> None:
    with pytest.raises(ValueError, match="positive_label"):
        validate_evaluation_manifest(_config(), task_type="classification")


def test_inferred_manifest_is_explicitly_self_declared_and_uses_development_thresholds() -> None:
    manifest = inferred_manifest_for_args(task_type="regression").to_dict()
    assert manifest["training_independence"] == "unverified"
    assert manifest["constraints_verified"] is False
    assert manifest["threshold_policy"]["source"] == "development_default"
    assert manifest["prospective_protocol"] is None


def test_verified_holdout_requires_code_hash() -> None:
    config = _config()
    del config["evaluation"]["provenance"]["code_sha256"]
    with pytest.raises(ValueError, match="code_sha256"):
        validate_evaluation_manifest(config, task_type="regression")


def test_prospective_protocol_requires_selection_before_measurement() -> None:
    config = _config()
    config["evaluation"]["prospective_protocol"] = {
        "protocol_identifier": "study-v1",
        "candidate_selection_timestamp": "2026-08-05T12:00:00Z",
        "measurement_timestamp": "2026-08-04T12:00:00Z",
        "selection_manifest_path": "selected.csv",
        "selection_manifest_sha256": "e" * 64,
        "outcome_data_sha256": "f" * 64,
    }
    with pytest.raises(ValueError, match="must precede"):
        validate_evaluation_manifest(config, task_type="regression")


def test_empirical_threshold_policy_requires_assay_context() -> None:
    config = _config()
    config["evaluation"]["threshold_policy"] = {
        "source": "empirically_validated",
        "policy_name": "promoter-v1",
        "rationale": "Historical assay study",
    }
    with pytest.raises(ValueError, match="must document"):
        validate_evaluation_manifest(config, task_type="regression")
