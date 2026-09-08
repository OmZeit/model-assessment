import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

def test_golden_run_exploratory(tmp_path: Path) -> None:
    # 1. Locate the dummy public dataset
    project_root = Path(__file__).resolve().parent.parent.parent / "model_assessment"
    assay_csv = project_root / "examples" / "public_dream_promoter_predictions.csv"
    
    # 2. Run the CLI in "run" (exploratory) mode
    env = {
        **os.environ,
        "PYTHONPATH": str(project_root.parent),
        "ASSAYREADY_RUN_DB": str(tmp_path / "runs.sqlite"),
    }
    cmd = [
        sys.executable,
        "-m",
        "model_assessment.cli",
        "run",
        "--assay",
        str(assay_csv),
        "--output-dir",
        str(tmp_path / "golden_output"),
        "--sequence-col", "sequence",
        "--target-col", "measured_activity",
        "--project", "test_exploratory_run",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    assert result.returncode == 0, f"Command failed: {result.stderr}"
    
    # 3. Verify output artifacts
    base_output_dir = tmp_path / "golden_output"
    assert base_output_dir.exists(), "Output directory was not created."
    
    # Find the run token subdirectory
    subdirs = [d for d in base_output_dir.iterdir() if d.is_dir()]
    assert len(subdirs) == 1, f"Expected exactly one run directory, found {len(subdirs)}"
    output_dir = subdirs[0]

    actual_files = [f.name for f in output_dir.iterdir() if f.is_file()]
    expected_files = [
        "execution_manifest.json",
        "run_summary.json",
        "data_audit_report.json",
        "data_audit_report.md",
        "benchmark_report.json",
        "benchmark_report.md",
        "readiness_report.md",
        "model_card.md",
        "candidate_prioritization.csv",
        "baseline_predictions.csv",
        "similarity_sensitivity.json",
    ]
    for expected in expected_files:
        assert (output_dir / expected).exists(), f"Missing artifact: {expected}. Actual files: {actual_files}"
    
    with (output_dir / "run_summary.json").open() as f:
        run_data = json.load(f)
        
    assert run_data["project"] == "test_exploratory_run"
    assert "benchmark_report" in run_data
    assert "artifact_dir" in run_data
    assert run_data["benchmark_report"]["assurance_level"]["key"] == "controlled_evaluation"
    assert "Recommended" not in run_data["benchmark_report"]["verdict"]
    
    with (output_dir / "execution_manifest.json").open() as f:
        manifest_data = json.load(f)
        
    assert "completed_at" in manifest_data
    assert "duration_seconds" in manifest_data
    assert "dependencies" in manifest_data
    assert "artifacts" in manifest_data
