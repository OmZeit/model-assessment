from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
    threshold_policy: dict[str, Any]
    prospective_protocol: dict[str, Any] | None

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
            "threshold_policy": dict(self.threshold_policy),
            "prospective_protocol": (
                dict(self.prospective_protocol) if self.prospective_protocol is not None else None
            ),
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


def _default_threshold_policy() -> dict[str, Any]:
    return {
        "source": "development_default",
        "policy_name": "assayready_development_defaults_v1",
        "rationale": (
            "Generic development defaults for sensitivity analysis; not empirically validated "
            "as universal biological credibility standards."
        ),
        "assay_type": "unspecified",
        "measurement_noise": "unspecified",
        "effect_size": "unspecified",
        "class_imbalance": "unspecified",
        "decision_consequence": "unspecified",
    }


def _parse_iso_timestamp(value: Any, *, field: str) -> str:
    text = _as_nonempty_text(value, field=field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp.") from exc
    if parsed.utcoffset() is None:
        raise ValueError(f"{field} must include an explicit UTC offset or Z suffix.")
    return text


def _validate_sha256(value: Any, *, field: str, required: bool = False) -> str:
    digest = str(value or "").strip().lower()
    if required and not digest:
        raise ValueError(f"{field} is required.")
    if digest and (len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)):
        raise ValueError(f"{field} must be a 64-character SHA-256 digest.")
    return digest


def _validate_threshold_policy(evaluation: Mapping[str, Any], *, custom_thresholds: bool) -> dict[str, Any]:
    raw = evaluation.get("threshold_policy")
    if raw is None:
        policy = _default_threshold_policy()
        if custom_thresholds:
            policy.update(
                {
                    "source": "customer_configured",
                    "policy_name": "customer_configured_thresholds",
                    "rationale": "Customer-supplied values; empirical assay validation was not documented.",
                }
            )
        return policy

    configured = _as_mapping(raw, field="evaluation.threshold_policy")
    source = _as_nonempty_text(configured.get("source"), field="evaluation.threshold_policy.source").lower()
    if source not in {"development_default", "customer_configured", "empirically_validated"}:
        raise ValueError(
            "evaluation.threshold_policy.source must be 'development_default', "
            "'customer_configured', or 'empirically_validated'."
        )
    normalized = {
        "source": source,
        "policy_name": _as_nonempty_text(
            configured.get("policy_name"), field="evaluation.threshold_policy.policy_name"
        ),
        "rationale": _as_nonempty_text(
            configured.get("rationale"), field="evaluation.threshold_policy.rationale"
        ),
    }
    for field_name in [
        "assay_type",
        "measurement_noise",
        "effect_size",
        "class_imbalance",
        "decision_consequence",
    ]:
        normalized[field_name] = _as_nonempty_text(
            configured.get(field_name, "unspecified"),
            field=f"evaluation.threshold_policy.{field_name}",
        )
    if source == "empirically_validated":
        unspecified = [
            name
            for name in [
                "assay_type",
                "measurement_noise",
                "effect_size",
                "class_imbalance",
                "decision_consequence",
            ]
            if normalized[name].lower() == "unspecified"
        ]
        if unspecified:
            raise ValueError(
                "An empirically_validated threshold policy must document: " + ", ".join(unspecified)
            )
    return normalized


def _validate_prospective_protocol(evaluation: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = evaluation.get("prospective_protocol")
    if raw is None:
        return None
    protocol = _as_mapping(raw, field="evaluation.prospective_protocol")
    selected_at = _parse_iso_timestamp(
        protocol.get("candidate_selection_timestamp"),
        field="evaluation.prospective_protocol.candidate_selection_timestamp",
    )
    measured_at = _parse_iso_timestamp(
        protocol.get("measurement_timestamp"),
        field="evaluation.prospective_protocol.measurement_timestamp",
    )
    selected_time = datetime.fromisoformat(selected_at.replace("Z", "+00:00"))
    measured_time = datetime.fromisoformat(measured_at.replace("Z", "+00:00"))
    if selected_time >= measured_time:
        raise ValueError(
            "evaluation.prospective_protocol.candidate_selection_timestamp must precede measurement_timestamp."
        )
    return {
        "protocol_identifier": _as_nonempty_text(
            protocol.get("protocol_identifier"),
            field="evaluation.prospective_protocol.protocol_identifier",
        ),
        "candidate_selection_timestamp": selected_at,
        "measurement_timestamp": measured_at,
        "selection_manifest_path": _as_nonempty_text(
            protocol.get("selection_manifest_path"),
            field="evaluation.prospective_protocol.selection_manifest_path",
        ),
        "selection_manifest_sha256": _validate_sha256(
            protocol.get("selection_manifest_sha256"),
            field="evaluation.prospective_protocol.selection_manifest_sha256",
            required=True,
        ),
        "outcome_data_sha256": _validate_sha256(
            protocol.get("outcome_data_sha256"),
            field="evaluation.prospective_protocol.outcome_data_sha256",
            required=True,
        ),
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
    for hash_field in ["training_data_sha256", "model_sha256", "split_sha256", "code_sha256"]:
        normalized_provenance[hash_field] = _validate_sha256(
            provenance.get(hash_field), field=f"evaluation.provenance.{hash_field}"
        )
    if training_independence == "verified_holdout":
        missing_hashes = [
            field
            for field in ["training_data_sha256", "model_sha256", "split_sha256", "code_sha256"]
            if not normalized_provenance[field]
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
    if thresholds["min_test_rows"] < 1 or thresholds["min_num_clusters"] < 1:
        raise ValueError("evaluation claim thresholds require at least one test row and one cluster.")
    if not -1.0 <= thresholds["min_uncertainty_spearman"] <= 1.0:
        raise ValueError("evaluation.claim_thresholds.min_uncertainty_spearman must be in [-1, 1].")
    if thresholds["max_calibration_gap_ratio"] < 0:
        raise ValueError("evaluation.claim_thresholds.max_calibration_gap_ratio must be non-negative.")

    threshold_policy = _validate_threshold_policy(evaluation, custom_thresholds=bool(custom_thresholds))
    prospective_protocol = _validate_prospective_protocol(evaluation)

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
        threshold_policy=threshold_policy,
        prospective_protocol=prospective_protocol,
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
            "training_data_sha256": "",
            "model_sha256": "",
            "split_sha256": "",
            "code_sha256": "",
            "dependency_versions": {"assayready": "0.1.0"},
        },
        claim_thresholds=_default_thresholds(),
        threshold_policy=_default_threshold_policy(),
        prospective_protocol=None,
    )
