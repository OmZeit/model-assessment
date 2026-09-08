import json
import os
from pathlib import Path
from unittest import mock
import subprocess
import sys
import pytest

def test_golden_public_dream_audit(tmp_path: Path) -> None:
    # 1. Load the original config
    project_root = Path(__file__).resolve().parent.parent.parent / "model_assessment"
    config_path = project_root / "examples" / "public_dream_promoter_audit.json"
    
    with config_path.open() as f:
        config_data = json.load(f)
        
    # 2. Modify paths to point to absolute locations and temp output
    config_data["output_dir"] = str(tmp_path / "golden_output")
    # Resolve assay and candidate files relative to the examples directory
    config_data["assay_files"] = [str(project_root / "examples" / f) for f in config_data["assay_files"]]
    config_data["candidate_files"] = [str(project_root / "examples" / f) for f in config_data["candidate_files"]]
    
    temp_config_path = tmp_path / "temp_config.json"
    with temp_config_path.open("w") as f:
        json.dump(config_data, f)
        
    # 3. Run the CLI
    env = {
        **os.environ,
        "PYTHONPATH": str(project_root.parent),
        "ASSAYREADY_RUN_DB": str(tmp_path / "runs.sqlite"),
    }
    cmd = [
        sys.executable,
        "-m",
        "model_assessment.cli",
        "audit-predictions",
        "--config",
        str(temp_config_path)
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    assert result.returncode == 0, f"Command failed: {result.stderr}"
    
    # 4. Verify output artifacts
    base_output_dir = tmp_path / "golden_output"
    assert base_output_dir.exists(), "Output directory was not created."
    
    # Find the run token subdirectory
    subdirs = [d for d in base_output_dir.iterdir() if d.is_dir()]
    assert len(subdirs) == 1, f"Expected exactly one run directory, found {len(subdirs)}"
    output_dir = subdirs[0]

    actual_files = [f.name for f in output_dir.iterdir() if f.is_file()]
    expected_files = [
        "execution_manifest.json",
        "prediction_audit_summary.json",
        "ranked_candidates.csv",
        "prediction_audit_report.md",
        "prediction_audit_report.json",
        "candidate_prioritization.csv",
        "baseline_predictions.csv",
        "similarity_sensitivity.json",
        "data_audit_report.json",
        "data_audit_report.md",
    ]
    for expected in expected_files:
        assert (output_dir / expected).exists(), f"Missing artifact: {expected}. Actual files: {actual_files}"
    
    with (output_dir / "prediction_audit_summary.json").open() as f:
        run_data = json.load(f)
        
    assert "prediction_audit" in run_data
    assert run_data["project"] == "public_dream_promoter_audit"
    report = run_data["prediction_audit"]
    assert report["assurance_level"]["key"] == "self_declared"
    assert report["evidence_level"]["key"] in {"descriptive_audit_only", "insufficient_evidence"}
    assert "Recommended" not in report["verdict"]
    assert "threshold_sensitivity" in report
    assert "similarity_sensitivity" in report
