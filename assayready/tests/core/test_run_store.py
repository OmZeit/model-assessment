from __future__ import annotations

from pathlib import Path

from model_assessment.cli import _artifact_dir, main
from model_assessment.run_store import RUN_DB_ENV, get_run, record_run, verify_run_artifacts


def test_artifact_dir_preserves_an_existing_run(tmp_path: Path) -> None:
    configured = tmp_path / "audit-output"
    first = _artifact_dir("client/project", configured)
    assert first.parent == configured.resolve()
    (first / "existing.txt").write_text("first run\n", encoding="utf-8")

    second = _artifact_dir("client/project", configured)
    assert second.parent == configured.resolve()
    assert second != first
    assert (first / "existing.txt").read_text(encoding="utf-8") == "first run\n"


def test_default_artifact_dir_uses_canonical_output_root_not_cwd(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "canonical-output"
    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()
    monkeypatch.setenv("ASSAYREADY_OUTPUT_ROOT", str(output_root))
    monkeypatch.chdir(unrelated_cwd)

    artifact_dir = _artifact_dir("Client Project", None)

    assert artifact_dir.parent == (output_root / "Client-Project").resolve()


def test_record_run_uses_execution_identity_and_provenance(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    summary = {
        "schema_version": 1,
        "run_id": "execution-123",
        "project": "demo",
        "artifact_dir": str(artifact_dir),
        "params": {"seed": 13},
        "audit": {"accepted_rows": 4},
        "split_diagnostics": {"split_sizes": {"test": 1}},
        "benchmark_report": {
            "verdict": "audit only",
            "primary_metric_name": "r2",
            "task_head_primary_metric": 0.1,
            "best_simple_baseline": {"name": "mean", "primary_metric": 0.0},
            "warnings": [],
        },
        "execution_manifest": {
            "status": "completed",
            "started_at": "2026-07-11T01:00:00+00:00",
            "completed_at": "2026-07-11T01:00:02+00:00",
            "duration_seconds": 2.0,
        },
        "artifacts": [],
    }
    db_path = tmp_path / "runs.sqlite"

    assert record_run(summary, workflow="internal", db_path=db_path) == "execution-123"
    stored = get_run("execution-123", db_path=db_path)
    assert stored is not None
    assert stored["status"] == "completed"
    assert stored["duration_seconds"] == 2.0
    assert "started_at" in stored


def test_verify_run_artifacts_detects_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "report.json"
    artifact.write_text('{"ok": true}\n', encoding="utf-8")
    import hashlib

    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    summary = {
        "execution_manifest": {
            "artifacts": [
                {"path": str(artifact), "size_bytes": artifact.stat().st_size, "sha256": expected}
            ]
        }
    }
    assert verify_run_artifacts(summary)["ok"] is True
    artifact.write_text("tampered\n", encoding="utf-8")
    verification = verify_run_artifacts(summary)
    assert verification["ok"] is False
    assert verification["errors"]


def test_runs_cli_lists_records(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "runs.sqlite"
    monkeypatch.setenv(RUN_DB_ENV, str(db_path))
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    record_run(
        {
            "run_id": "run-list-1",
            "project": "demo",
            "artifact_dir": str(artifact_dir),
            "audit": {"accepted_rows": 1},
            "split_diagnostics": {"split_sizes": {"test": 0}},
            "benchmark_report": {"verdict": "audit only", "warnings": []},
            "artifacts": [],
        },
        workflow="internal",
        db_path=db_path,
    )
    assert main(["runs", "list", "--json"]) == 0
    assert "run-list-1" in capsys.readouterr().out
