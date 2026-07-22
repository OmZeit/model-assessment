from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import pytest
from dash import html

from model_assessment import ui


def _has_component(component: Any, target_type: type) -> bool:
    if isinstance(component, target_type):
        return True
    children = getattr(component, "children", None)
    if children is None:
        return False
    if not isinstance(children, (list, tuple)):
        children = [children]
    return any(_has_component(child, target_type) for child in children)


def test_decode_upload_rejects_payload_over_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ui.UPLOAD_LIMIT_ENV, "0.000001")
    payload = base64.b64encode(b"too big").decode("ascii")
    with pytest.raises(ValueError, match="too large"):
        ui._decode_upload(f"data:text/csv;base64,{payload}")


def test_safe_output_dir_accepts_default_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ui.OUTPUT_ROOT_ENV, str(tmp_path / "outputs" / "assayready"))
    resolved = ui._safe_output_dir("outputs/assayready/client_audit")
    assert resolved == (tmp_path / "outputs" / "assayready" / "client_audit").resolve()


def test_safe_output_dir_rejects_external_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ui.OUTPUT_ROOT_ENV, str(tmp_path / "outputs" / "assayready"))
    external = tmp_path.parent / "outside"
    with pytest.raises(ValueError, match="outputs/assayready"):
        ui._safe_output_dir(str(external))


def test_safe_artifact_path_requires_indexed_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_dir = tmp_path / "outputs" / "assayready" / "run_1"
    artifact_dir.mkdir(parents=True)
    allowed = artifact_dir / "prediction_audit_report.md"
    allowed.write_text("# report\n", encoding="utf-8")
    blocked = artifact_dir / "uploaded_assay.csv"
    blocked.write_text("secret\n", encoding="utf-8")

    monkeypatch.setattr(
        ui,
        "list_runs",
        lambda limit=1000: [
            {
                "run_id": "run-1",
                "artifact_dir": str(artifact_dir),
                "report_name": allowed.name,
                "summary_path": str(artifact_dir / "prediction_audit_summary.json"),
            }
        ],
    )
    monkeypatch.setattr(ui, "list_artifacts", lambda run_id: [{"path": str(allowed)}])

    assert ui._safe_artifact_path(str(allowed)) == allowed.resolve()
    with pytest.raises(ValueError, match="indexed"):
        ui._safe_artifact_path(str(blocked))


def test_failure_panel_hides_traceback_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ui.SHOW_TRACEBACKS_ENV, raising=False)
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        panel = ui._failure_panel("Analysis failed", exc, debug=False)
    assert not _has_component(panel, html.Pre)


def test_remote_host_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="Refusing to bind"):
        ui.run_server(host="0.0.0.0", port=8050, allow_remote=False)


def test_remote_mode_flag_is_unsupported() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        ui.run_server(host="0.0.0.0", port=8050, allow_remote=True)


def test_remote_debug_combination_blocked() -> None:
    # Any remote binding or allow_remote flag combined with debug mode should be blocked.
    with pytest.raises(ValueError, match="remote"):
        ui.run_server(host="127.0.0.1", port=8050, debug=True, allow_remote=True)


def test_resolve_upload_reference_security(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Safe path inside search root (config_dir)
    safe_file = config_dir / "safe.csv"
    safe_file.write_text("data", encoding="utf-8")

    resolved = ui._resolve_upload_reference(str(safe_file), config_dir=config_dir)
    assert resolved == safe_file.resolve()

    # Unsafe path outside allowed search roots, but inside the test sandbox
    unsafe_dir = tmp_path / "unsafe_dir"
    unsafe_dir.mkdir()
    unsafe_file = unsafe_dir / "unsafe_system_file.csv"
    unsafe_file.write_text("secrets", encoding="utf-8")

    with pytest.raises(PermissionError, match="Access denied"):
        ui._resolve_upload_reference(str(unsafe_file), config_dir=config_dir)


def test_upload_filename_preserves_supported_extension() -> None:
    assert ui._safe_upload_filename("Prediction Audit.csv") == "prediction_audit.csv"
    assert ui._safe_upload_filename("../unsafe.exe") == "unsafe"


def test_resolve_upload_reference_rejects_prefix_collision(tmp_path: Path) -> None:
    config_dir = tmp_path / "assayready"
    config_dir.mkdir()
    sibling = tmp_path / "assayready-hacked"
    sibling.mkdir()
    outside = sibling / "data.csv"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(PermissionError, match="Access denied"):
        ui._resolve_upload_reference(str(outside), config_dir=config_dir)


def test_cleanup_stale_uploads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "uploads"
    stale = root / "assay_old"
    stale.mkdir(parents=True)
    monkeypatch.setattr(ui, "_upload_root", lambda: root)
    monkeypatch.setenv(ui.UPLOAD_RETENTION_HOURS_ENV, "24")

    ui._cleanup_stale_uploads(now=time.time() + 48 * 3600)
    assert not stale.exists()


def test_package_assets_resolve_inside_import_package() -> None:
    assert (ui._package_dir() / "assets" / "assayready.css").is_file()
