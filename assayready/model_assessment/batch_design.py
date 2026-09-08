from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any


PLATE_LAYOUTS = {
    24: ("ABCD", 6),
    48: ("ABCDEF", 8),
    96: ("ABCDEFGH", 12),
    384: ("ABCDEFGHIJKLMNOP", 24),
}


@dataclass(frozen=True)
class BatchDesignPolicy:
    plate_size: int = 96
    controls: dict[str, int] = field(
        default_factory=lambda: {"positive_control": 2, "negative_control": 2, "blank_control": 2}
    )
    candidate_replicates: int = 1
    family_column: str | None = None
    max_per_family: int | None = None
    cost_column: str | None = None
    max_total_cost: float | None = None
    required_columns: tuple[str, ...] = ("sequence",)
    exclude_edge_wells: bool = False
    seed: int = 13

    def __post_init__(self) -> None:
        if self.plate_size not in PLATE_LAYOUTS:
            raise ValueError("plate_size must be one of 24, 48, 96, or 384.")
        if self.candidate_replicates < 1:
            raise ValueError("candidate_replicates must be at least one.")
        if any(int(count) < 0 for count in self.controls.values()):
            raise ValueError("control counts must be non-negative.")
        if self.max_per_family is not None and self.max_per_family < 1:
            raise ValueError("max_per_family must be positive when supplied.")
        if self.max_total_cost is not None and self.max_total_cost < 0:
            raise ValueError("max_total_cost must be non-negative when supplied.")


def well_names(plate_size: int, *, exclude_edges: bool = False) -> list[str]:
    rows, columns = PLATE_LAYOUTS[plate_size]
    names = []
    for row_index, row in enumerate(rows):
        for column in range(1, columns + 1):
            if exclude_edges and (
                row_index in {0, len(rows) - 1} or column in {1, columns}
            ):
                continue
            names.append(f"{row}{column}")
    return names


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _candidate_id(row: dict[str, Any], index: int) -> str:
    for key in ("candidate_id", "design_id", "sequence_id", "id", "sequence_hash"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    sequence = str(row.get("sequence") or "")
    if sequence:
        return hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]
    return f"candidate-{index + 1}"


def _ordered_candidates(
    candidates: list[dict[str, Any]], objective_weights: dict[str, float] | None
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in candidates]
    weights = objective_weights or {}
    for index, row in enumerate(rows):
        row["candidate_id"] = _candidate_id(row, index)
        if weights:
            row["batch_objective_score"] = sum(
                _number(row.get(column)) * float(weight) for column, weight in weights.items()
            )
        elif row.get("diversified_acquisition_score") is not None:
            row["batch_objective_score"] = _number(row.get("diversified_acquisition_score"))
        elif row.get("acquisition_score") is not None:
            row["batch_objective_score"] = _number(row.get("acquisition_score"))
        else:
            row["batch_objective_score"] = -_number(row.get("rank"), index + 1)
    return sorted(rows, key=lambda row: row["batch_objective_score"], reverse=True)


def design_batch(
    candidates: list[dict[str, Any]],
    *,
    policy: BatchDesignPolicy,
    objective_weights: dict[str, float] | None = None,
    scientist_approval: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic, constrained draft plate design.

    A non-empty scientist approval identifier is required before the manifest
    reports the plan as approved for execution.
    """
    violations: list[str] = []
    for column in policy.required_columns:
        if any(not str(row.get(column) or "").strip() for row in candidates):
            violations.append(f"required candidate column is missing or blank: {column}")

    wells = well_names(policy.plate_size, exclude_edges=policy.exclude_edge_wells)
    control_slots = sum(int(count) for count in policy.controls.values())
    if control_slots > len(wells):
        raise ValueError("Controls exceed usable plate capacity.")
    candidate_slots = (len(wells) - control_slots) // policy.candidate_replicates
    ordered = _ordered_candidates(candidates, objective_weights)
    selected: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    total_cost = 0.0
    for row in ordered:
        if len(selected) >= candidate_slots:
            break
        family = str(row.get(policy.family_column) or "unspecified") if policy.family_column else ""
        if policy.max_per_family is not None and family_counts.get(family, 0) >= policy.max_per_family:
            continue
        cost = _number(row.get(policy.cost_column)) if policy.cost_column else 0.0
        proposed_cost = total_cost + cost * policy.candidate_replicates
        if policy.max_total_cost is not None and proposed_cost > policy.max_total_cost:
            continue
        selected.append(row)
        family_counts[family] = family_counts.get(family, 0) + 1
        total_cost = proposed_cost

    if not selected:
        violations.append("no candidates satisfied the configured constraints")
    if len(selected) < min(candidate_slots, len(candidates)):
        violations.append("some plate capacity was unused because candidates failed quotas or budget constraints")

    assignments: list[dict[str, Any]] = []
    for role, count in policy.controls.items():
        for replicate in range(1, int(count) + 1):
            assignments.append(
                {"role": role, "candidate_id": "", "replicate": replicate, "sequence": ""}
            )
    for row in selected:
        for replicate in range(1, policy.candidate_replicates + 1):
            assignments.append(
                {
                    **row,
                    "role": "candidate",
                    "candidate_id": row["candidate_id"],
                    "replicate": replicate,
                }
            )

    rng = random.Random(policy.seed)
    rng.shuffle(wells)
    rng.shuffle(assignments)
    for well, assignment in zip(wells, assignments):
        assignment["well"] = well
        assignment["approved_for_execution"] = bool(scientist_approval) and not violations

    return {
        "schema_version": 1,
        "status": "approved" if scientist_approval and not violations else "draft",
        "approved_for_execution": bool(scientist_approval) and not violations,
        "scientist_approval": scientist_approval,
        "policy": {
            "plate_size": policy.plate_size,
            "controls": dict(policy.controls),
            "candidate_replicates": policy.candidate_replicates,
            "family_column": policy.family_column,
            "max_per_family": policy.max_per_family,
            "cost_column": policy.cost_column,
            "max_total_cost": policy.max_total_cost,
            "required_columns": list(policy.required_columns),
            "exclude_edge_wells": policy.exclude_edge_wells,
            "seed": policy.seed,
        },
        "objective_weights": dict(objective_weights or {}),
        "selected_candidates": len(selected),
        "total_candidate_cost": total_cost,
        "violations": violations,
        "assignments": assignments,
    }
