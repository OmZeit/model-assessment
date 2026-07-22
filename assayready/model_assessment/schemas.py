from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


EVALUATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvaluationManifest:
    schema_version: int
    task_type: str
    positive_label: str | None
    objective_direction: str
    units: str
    uncertainty_type: str
    training_independence: str
    constraints_verified: bool
    biological_constraints: list[str]
    provenance: dict[str, Any]
    claim_thresholds: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "positive_label": self.positive_label,
            "objective_direction": self.objective_direction,
            "units": self.units,
            "uncertainty_type": self.uncertainty_type,
            "training_independence": self.training_independence,
            "constraints_verified": self.constraints_verified,
            "biological_constraints": list(self.biological_constraints),
            "provenance": dict(self.provenance),
            "claim_thresholds": dict(self.claim_thresholds),
        }


def _as_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object.")
    return value


def _as_nonempty_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required and must be non-empty.")
    return text


def _as_float(value: Any, *, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric.") from exc
    return out


def _as_int(value: Any, *, field: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer.") from exc
    return out


def _normalize_positive_label(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return text or None


def _default_thresholds() -> dict[str, Any]:
    return {
        "min_test_rows": 20,
        "min_num_clusters": 3,
        "min_lift_delta": 0.0,
        "min_uncertainty_spearman": 0.20,
        "max_calibration_gap_ratio": 1.0,
    }


def validate_evaluation_manifest(config: Mapping[str, Any], *, task_type: str) -> EvaluationManifest:
    evaluation = _as_mapping(config.get("evaluation"), field="evaluation")
    schema_version = _as_int(evaluation.get("schema_version"), field="evaluation.schema_version")
    if schema_version != EVALUATION_SCHEMA_VERSION:
        raise ValueError(
            f"evaluation.schema_version must be {EVALUATION_SCHEMA_VERSION} for this release; got {schema_version}."
        )

    semantics = _as_mapping(evaluation.get("semantics"), field="evaluation.semantics")
    objective_direction = _as_nonempty_text(
        semantics.get("objective_direction"), field="evaluation.semantics.objective_direction"
    ).lower()
    if objective_direction not in {"maximize", "minimize"}:
        raise ValueError("evaluation.semantics.objective_direction must be 'maximize' or 'minimize'.")

    units = _as_nonempty_text(semantics.get("units"), field="evaluation.semantics.units")
    uncertainty_type = _as_nonempty_text(
        semantics.get("uncertainty_type"), field="evaluation.semantics.uncertainty_type"
    ).lower()
    if uncertainty_type not in {
        "none",
        "predicted_absolute_error",
        "predictive_standard_deviation",
        "predictive_interval",
        "variance",
        "entropy",
        "score",
        "aleatoric",
        "epistemic",
        "hybrid",
    }:
        raise ValueError(
            "evaluation.semantics.uncertainty_type is not supported; use a concrete representation such as "
            "'predicted_absolute_error', 'predictive_standard_deviation', 'predictive_interval', or 'none'."
        )

    positive_label = _normalize_positive_label(semantics.get("positive_label"))
    if task_type == "classification" and positive_label is None:
        raise ValueError("evaluation.semantics.positive_label is required for classification tasks.")

    constraints = evaluation.get("biological_constraints")
    if not isinstance(constraints, list) or not constraints:
        raise ValueError("evaluation.biological_constraints must be a non-empty list of constraint statements.")
    biological_constraints = [_as_nonempty_text(item, field="evaluation.biological_constraints[]") for item in constraints]

    provenance = _as_mapping(evaluation.get("provenance"), field="evaluation.provenance")
    training_independence = str(provenance.get("training_independence") or "unverified").strip().lower()
    if training_independence not in {"verified_holdout", "internally_controlled", "unverified"}:
        raise ValueError(
            "evaluation.provenance.training_independence must be 'verified_holdout', "
            "'internally_controlled', or 'unverified'."
        )
    constraints_verified = evaluation.get("constraints_verified", False)
    if not isinstance(constraints_verified, bool):
        raise ValueError("evaluation.constraints_verified must be true or false.")
    normalized_provenance = {
        "model_identifier": _as_nonempty_text(provenance.get("model_identifier"), field="evaluation.provenance.model_identifier"),
        "data_identifier": _as_nonempty_text(provenance.get("data_identifier"), field="evaluation.provenance.data_identifier"),
        "split_identifier": _as_nonempty_text(provenance.get("split_identifier"), field="evaluation.provenance.split_identifier"),
        "code_revision": _as_nonempty_text(provenance.get("code_revision"), field="evaluation.provenance.code_revision"),
        "duration_seconds": _as_float(provenance.get("duration_seconds"), field="evaluation.provenance.duration_seconds"),
        "artifact_verification": _as_nonempty_text(
            provenance.get("artifact_verification"), field="evaluation.provenance.artifact_verification"
        ),
    }
    for hash_field in ["training_data_sha256", "model_sha256", "split_sha256"]:
        value = str(provenance.get(hash_field) or "").strip().lower()
        if value and (len(value) != 64 or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError(f"evaluation.provenance.{hash_field} must be a 64-character SHA-256 digest.")
        normalized_provenance[hash_field] = value
    if training_independence == "verified_holdout":
        missing_hashes = [
            field for field in ["training_data_sha256", "model_sha256", "split_sha256"] if not normalized_provenance[field]
        ]
        if missing_hashes:
            raise ValueError(
                "verified_holdout provenance requires SHA-256 values for: " + ", ".join(missing_hashes)
            )
    dependency_versions = _as_mapping(
        provenance.get("dependency_versions"), field="evaluation.provenance.dependency_versions"
    )
    if not dependency_versions:
        raise ValueError("evaluation.provenance.dependency_versions must include at least one dependency pin.")
    normalized_provenance["dependency_versions"] = {
        _as_nonempty_text(name, field="evaluation.provenance.dependency_versions key"): _as_nonempty_text(
            version, field=f"evaluation.provenance.dependency_versions[{name}]"
        )
        for name, version in dependency_versions.items()
    }

    thresholds = dict(_default_thresholds())
    custom_thresholds = evaluation.get("claim_thresholds") or {}
    if custom_thresholds:
        custom_mapping = _as_mapping(custom_thresholds, field="evaluation.claim_thresholds")
        for key in ["min_test_rows", "min_num_clusters"]:
            if key in custom_mapping:
                thresholds[key] = _as_int(custom_mapping[key], field=f"evaluation.claim_thresholds.{key}")
        for key in ["min_lift_delta", "min_uncertainty_spearman", "max_calibration_gap_ratio"]:
            if key in custom_mapping:
                thresholds[key] = _as_float(custom_mapping[key], field=f"evaluation.claim_thresholds.{key}")

    return EvaluationManifest(
        schema_version=schema_version,
        task_type=task_type,
        positive_label=positive_label,
        objective_direction=objective_direction,
        units=units,
        uncertainty_type=uncertainty_type,
        training_independence=training_independence,
        constraints_verified=constraints_verified,
        biological_constraints=biological_constraints,
        provenance=normalized_provenance,
        claim_thresholds=thresholds,
    )


def inferred_manifest_for_args(*, task_type: str, positive_label: str | None = None) -> EvaluationManifest:
    return EvaluationManifest(
        schema_version=EVALUATION_SCHEMA_VERSION,
        task_type=task_type,
        positive_label=_normalize_positive_label(positive_label),
        objective_direction="maximize",
        units="unspecified",
        uncertainty_type="none",
        training_independence="unverified",
        constraints_verified=False,
        biological_constraints=["unspecified"],
        provenance={
            "model_identifier": "cli_args_unspecified",
            "data_identifier": "cli_args_unspecified",
            "split_identifier": "cli_args_unspecified",
            "code_revision": "unknown",
            "duration_seconds": 0.0,
            "artifact_verification": "not_provided",
            "dependency_versions": {"model_assessment": "0.1.0"},
        },
        claim_thresholds=_default_thresholds(),
    )
