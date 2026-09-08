from __future__ import annotations

from pathlib import Path

import pytest

from model_assessment.cli import _doctor_checks


def test_doctor_reports_public_demo_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASSAYREADY_OUTPUT_ROOT", str(tmp_path / "outputs"))
    checks = {item["name"]: item for item in _doctor_checks()}
    assert "public demo" in checks
    assert checks["public demo"]["required"] is True
    assert checks["public demo"]["ok"] is True
    assert "runtime imports" in checks
    assert "optional foundation-model imports" in checks
