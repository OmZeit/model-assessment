from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from model_assessment import ui
from model_assessment.run_store import list_artifacts, record_run, verify_run_artifacts


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_client_plate_artifacts_refresh_manifest_and_run_index(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "run"
    artifact_dir.mkdir()
    readiness_path = artifact_dir / "readiness_report.md"
    readiness_path.write_text("# Readiness\n", encoding="utf-8")
    candidate_path = artifact_dir / "candidate_prioritization.csv"
    pd.DataFrame(
        [
            {"rank": 1, "sequence": "ACGTACGT", "predicted_activity": 0.8},
            {"rank": 2, "sequence": "ACGTTCGT", "predicted_activity": 0.7},
        ]
    ).to_csv(candidate_path, index=False)

    execution_id = "fixed-execution-id"
    execution_manifest = {
        "schema_version": 1,
        "execution_id": execution_id,
        "workflow": "internal",
        "status": "completed",
        "artifacts": [_artifact_record(readiness_path), _artifact_record(candidate_path)],
    }
    (artifact_dir / "execution_manifest.json").write_text(
        json.dumps(execution_manifest), encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "run_id": execution_id,
        "project": "postprocessing-integrity",
        "artifact_dir": str(artifact_dir),
        "params": {},
        "audit": {"accepted_rows": 2},
        "benchmark_report": {"verdict": "Review required."},
        "ranked_candidates": 2,
        "artifacts": [
            "readiness_report.md",
            "candidate_prioritization.csv",
            "execution_manifest.json",
        ],
        "execution_manifest": execution_manifest,
    }

    updated = ui._add_client_plate_plan_artifacts(
        summary, plate_size=24, control_wells=2, plate_seed=13
    )

    persisted_manifest = json.loads(
        (artifact_dir / "execution_manifest.json").read_text(encoding="utf-8")
    )
    assert persisted_manifest["execution_id"] == execution_id
    assert updated["execution_manifest"] == persisted_manifest
    records_by_name = {
        Path(record["path"]).name: record for record in persisted_manifest["artifacts"]
    }
    assert {
        "readiness_report.md",
        "candidate_prioritization.csv",
        "simulation_report.md",
        "draft_well_layout.csv",
        "plate_plan.csv",
    } == set(records_by_name)
    for name, record in records_by_name.items():
        path = artifact_dir / name
        assert record["size_bytes"] == path.stat().st_size
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_run_artifacts(updated)["ok"] is True

    db_path = tmp_path / "runs.sqlite"
    run_id = record_run(
        updated,
        workflow="internal",
        report_name="readiness_report.md",
        db_path=db_path,
    )
    indexed_names = {record["name"] for record in list_artifacts(run_id, db_path=db_path)}
    assert {"simulation_report.md", "draft_well_layout.csv", "plate_plan.csv"} <= indexed_names


def test_refresh_execution_inventory_rejects_artifact_escape(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "run"
    artifact_dir.mkdir()
    execution_manifest = {"execution_id": "run-id", "artifacts": []}
    (artifact_dir / "execution_manifest.json").write_text(
        json.dumps(execution_manifest), encoding="utf-8"
    )
    summary = {
        "artifact_dir": str(artifact_dir),
        "artifacts": ["../outside.txt"],
        "execution_manifest": execution_manifest,
    }

    try:
        ui._refresh_execution_artifact_inventory(summary)
    except ValueError as exc:
        assert "outside the run directory" in str(exc)
    else:
        raise AssertionError("Expected an escaping artifact path to be rejected.")
