from __future__ import annotations

from model_assessment.cli import _doctor_checks


def test_doctor_reports_public_demo_files() -> None:
    checks = {item["name"]: item for item in _doctor_checks()}
    assert "public demo" in checks
    assert checks["public demo"]["required"] is True
    assert checks["public demo"]["ok"] is True
    assert "runtime imports" in checks
