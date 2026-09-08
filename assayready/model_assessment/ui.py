from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.metadata
import importlib.util
import io
import json
import os
import random
import re
import shutil
import secrets
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import pandas as pd
import dash_ag_grid as dag
import plotly.io as pio
from dash import MATCH, Dash, Input, Output, State, callback_context, dcc, html, no_update
from dash.exceptions import PreventUpdate

try:
    from .cli import _load_config, _prediction_params_from_config, run_assayready, run_prediction_audit
    from .foundation_scoring import ModelScoringError, inspect_foundation_scorer, load_verified_foundation_scorer
    from .open_models import DNABERT2_REPO, DNABERT2_REVISION, download_dnabert2, ensure_dnabert2_scaffold, verify_manifest_file_records, verify_pinned_dnabert2_snapshot
    from .run_store import (
        default_db_path,
        get_run_summary,
        index_existing_runs,
        list_artifacts,
        list_candidates,
        list_runs,
        record_run,
    )
    from .schemas import inferred_manifest_for_args, validate_evaluation_manifest
    from .visualization import (
        benchmark_metric_figure,
        classification_discrimination_figure,
        plate_layout_figure,
        ranking_diagnostics_figure,
        regression_fit_figure,
        regression_slice_summary,
        residual_figure,
        simulation_constraints_figure,
        simulation_generation_figure,
        simulation_landscape_figure,
        split_composition_figure,
        uncertainty_figure,
    )
except ImportError:
    from model_assessment.cli import _load_config, _prediction_params_from_config, run_assayready, run_prediction_audit
    from model_assessment.foundation_scoring import ModelScoringError, inspect_foundation_scorer, load_verified_foundation_scorer
    from model_assessment.open_models import DNABERT2_REPO, DNABERT2_REVISION, download_dnabert2, ensure_dnabert2_scaffold, verify_manifest_file_records, verify_pinned_dnabert2_snapshot
    from model_assessment.run_store import (
        default_db_path,
        get_run_summary,
        index_existing_runs,
        list_artifacts,
        list_candidates,
        list_runs,
        record_run,
    )
    from model_assessment.schemas import inferred_manifest_for_args, validate_evaluation_manifest
    from model_assessment.visualization import (
        benchmark_metric_figure,
        classification_discrimination_figure,
        plate_layout_figure,
        ranking_diagnostics_figure,
        regression_fit_figure,
        regression_slice_summary,
        residual_figure,
        simulation_constraints_figure,
        simulation_generation_figure,
        simulation_landscape_figure,
        split_composition_figure,
        uncertainty_figure,
    )


NAV_SECTIONS = [
    (
        "ANALYZE",
        [
            ("Analyze", "/analyze"),
            ("Design Sandbox", "/simulations"),
        ],
    ),
    (
        "EVIDENCE",
        [
            ("Runs", "/runs"),
            ("Benchmarks", "/benchmarks"),
            ("Reports", "/reports"),
            ("Candidates", "/candidates"),
        ],
    ),
    (
        "SYSTEM",
        [
            ("Settings", "/settings"),
        ],
    ),
]

NAV_ITEMS = [item for _, section in NAV_SECTIONS for item in section]

NAV_ICONS = {
    "/analyze": "✦",
    "/benchmarks": "⌁",
    "/runs": "◷",
    "/candidates": "◇",
    "/reports": "▤",
    "/simulations": "⌘",
    "/settings": "⚙",
}

PAGE_PANEL_TITLES = {
    "/analyze": "Analysis canvas",
    "/benchmarks": "Benchmark library",
    "/runs": "Run history",
    "/candidates": "Candidate explorer",
    "/reports": "Evidence reports",
    "/simulations": "Design sandbox",
    "/settings": "Workspace settings",
}


def _run_href(path: str, run_id: Any | None) -> str:
    """Build an internal route that keeps the selected indexed run."""
    normalized = str(run_id or "").strip()
    if not normalized:
        return path
    return f"{path}?{urlencode({'run_id': normalized})}"


def _requested_run_id(search: str | None) -> str | None:
    values = parse_qs(str(search or "").lstrip("?"), keep_blank_values=False).get("run_id") or []
    normalized = str(values[0]).strip() if values else ""
    return normalized or None


def _linked_run_selection(
    pathname: str | None,
    search: str | None,
    expected_path: str,
    options: list[dict[str, Any]] | None,
) -> str | None:
    if pathname != expected_path:
        return None
    requested = _requested_run_id(search)
    allowed = {str(option.get("value")) for option in (options or []) if option.get("value") is not None}
    return requested if requested in allowed else None

DEMO_SCORER = "__demo_heuristic__"
CUSTOM_SCORER = "__custom_local_bundle__"
MODEL_ROOTS_ENV = "ASSAYREADY_MODEL_ROOTS"
_OUTPUT_PICKER_SECRET = secrets.token_bytes(32)

SIMULATION_VISUAL_LABELS = {
    "landscape": "Score and uncertainty landscape",
    "generation": "Mode-specific sequence view",
    "constraints": "Sequence constraint distributions",
    "plate": "Draft plate layout",
}

SIMULATION_VISUAL_PRESETS = {
    "compact": ["landscape", "generation"],
    "recommended": ["landscape", "generation", "constraints"],
    "full": ["landscape", "generation", "constraints", "plate"],
}


COLORS = {
    "ink": "#edf1f5",
    "muted": "#9ba6b2",
    "line": "#353c46",
    "soft": "#101318",
    "panel": "#20252c",
    "accent": "#327f69",
    "accent_dark": "#1f6957",
    "warn": "#9a3412",
    "danger": "#991b1b",
}

PAGE_STYLE = {
    "minHeight": "100vh",
    "background": COLORS["soft"],
    "color": COLORS["ink"],
    "fontFamily": "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
}

SIDEBAR_STYLE = {}

CONTENT_STYLE = {}

CARD_STYLE = {
    "background": COLORS["panel"],
    "border": f"1px solid {COLORS['line']}",
    "borderRadius": "6px",
    "boxShadow": "0 1px 2px rgba(0, 0, 0, 0.18)",
    "padding": "18px",
}

INPUT_STYLE = {
    "width": "100%",
    "boxSizing": "border-box",
    "border": f"1px solid {COLORS['line']}",
    "borderRadius": "6px",
    "padding": "9px 10px",
    "fontSize": "14px",
    "background": "#171b20",
    "color": COLORS["ink"],
}

BUTTON_STYLE = {
    "border": 0,
    "borderRadius": "6px",
    "background": COLORS["accent"],
    "color": "white",
    "fontWeight": 700,
    "padding": "10px 14px",
    "cursor": "pointer",
}

TABLE_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
CONFIG_EXTENSIONS = {".json", ".yaml", ".yml"}

SECONDARY_BUTTON_STYLE = {
    **BUTTON_STYLE,
    "background": "#2a3038",
    "color": COLORS["ink"],
    "border": f"1px solid {COLORS['line']}",
}


def _value_or_default(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    return value


UPLOAD_LIMIT_ENV = "ASSAYREADY_UPLOAD_LIMIT_MB"
SHOW_TRACEBACKS_ENV = "ASSAYREADY_SHOW_TRACEBACKS"
ALLOW_EXTERNAL_OUTPUTS_ENV = "ASSAYREADY_ALLOW_EXTERNAL_OUTPUTS"
CONFIG_DATA_ROOTS_ENV = "ASSAYREADY_CONFIG_DATA_ROOTS"
OUTPUT_ROOT_ENV = "ASSAYREADY_OUTPUT_ROOT"
UPLOAD_RETENTION_HOURS_ENV = "ASSAYREADY_UPLOAD_RETENTION_HOURS"
MAX_TABLE_RENDER_ROWS_ENV = "ASSAYREADY_MAX_TABLE_RENDER_ROWS"


def _output_root() -> Path:
    configured = os.environ.get(OUTPUT_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    project_root = _package_dir().parents[1]
    if (project_root / "pyproject.toml").is_file():
        return (project_root / "outputs" / "assayready").resolve()
    return (Path.home() / ".assayready" / "outputs").resolve()


def _safe_output_dir(raw: str, *, explicitly_selected: bool = False) -> Path:
    allowed_root = _output_root()
    cleaned = str(raw or "").strip()
    if not cleaned:
        return allowed_root
    candidate = Path(cleaned).expanduser()
    if candidate.is_absolute():
        target = candidate.resolve()
    else:
        parts = candidate.parts
        if len(parts) >= 2 and tuple(part.lower() for part in parts[:2]) == ("outputs", "assayready"):
            candidate = Path(*parts[2:])
        target = (allowed_root / candidate).resolve()
    allow_external = os.environ.get(ALLOW_EXTERNAL_OUTPUTS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
    if not (allow_external or explicitly_selected) and not (
        target == allowed_root or target.is_relative_to(allowed_root)
    ):
        raise ValueError(
            f"Paths must reside under outputs/assayready (configured root: {allowed_root}). To permit other local folders, set "
            f"{ALLOW_EXTERNAL_OUTPUTS_ENV}=1 before launching AssayReady."
        )
    if target.exists() and not target.is_dir():
        raise ValueError(f"The selected output path is not a folder: {target}")
    return target


def _output_selection_authorization(path: str | Path) -> dict[str, str]:
    resolved = str(Path(path).expanduser().resolve())
    message = os.path.normcase(resolved).encode("utf-8")
    signature = hmac.new(_OUTPUT_PICKER_SECRET, message, hashlib.sha256).hexdigest()
    return {"path": resolved, "signature": signature}


def _authorized_output_selection(
    target: Path,
    selected_output: dict[str, Any] | None,
) -> bool:
    selected_raw = (selected_output or {}).get("path")
    signature = str((selected_output or {}).get("signature") or "")
    if not str(selected_raw or "").strip() or not signature:
        return False
    selected_target = Path(str(selected_raw)).expanduser().resolve()
    expected = _output_selection_authorization(selected_target)["signature"]
    return selected_target == target.resolve() and hmac.compare_digest(signature, expected)


def _analysis_output_dir(
    raw: str | None,
    project: str | None,
    *,
    selected_output: dict[str, Any] | None = None,
) -> Path:
    """Resolve every UI analysis into the same root displayed by Settings."""
    if str(raw or "").strip():
        raw_target = Path(str(raw)).expanduser().resolve()
        return _safe_output_dir(
            str(raw_target),
            explicitly_selected=_authorized_output_selection(raw_target, selected_output),
        )
    return _safe_output_dir(str(_output_root() / _slug(str(project or "assayready"))))


def _choose_directory(*, title: str, initial_dir: Path) -> str | None:
    """Open the workstation's native folder chooser for this local-only UI."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        root.update_idletasks()
        selected = filedialog.askdirectory(
            parent=root,
            title=title,
            initialdir=str(initial_dir if initial_dir.is_dir() else initial_dir.parent),
            mustexist=True,
        )
        return str(selected) if selected else None
    finally:
        root.destroy()


def _safe_artifact_path(raw: str) -> Path:
    target = Path(raw).resolve()
    all_runs = list_runs(limit=1000)
    for run in all_runs:
        run_id = run.get("run_id")
        run_dir_str = run.get("artifact_dir")
        if run_id and run_dir_str:
            run_dir = Path(run_dir_str).resolve()
            if target == run_dir or target.is_relative_to(run_dir):
                allowed = {Path(art["path"]).resolve() for art in list_artifacts(run_id)}
                report_name = run.get("report_name")
                summary_path = run.get("summary_path")
                if report_name:
                    allowed.add((run_dir / str(report_name)).resolve())
                if summary_path:
                    allowed.add(Path(str(summary_path)).resolve())
                if target in allowed and target.is_file():
                    return target
    raise ValueError(f"Requested file is not an indexed AssayReady artifact: {raw}")


def _evidence_package_bytes(run_id: str) -> tuple[bytes, str]:
    run = next((item for item in list_runs(limit=1000) if str(item.get("run_id")) == str(run_id)), None)
    if not run:
        raise ValueError("The selected run is not indexed.")
    run_dir = Path(str(run.get("artifact_dir") or "")).resolve()
    candidates: list[Path] = []
    for item in list_artifacts(str(run_id)):
        candidates.append(Path(str(item.get("path") or "")))
    for field in ["report_name", "summary_path"]:
        value = run.get(field)
        if value:
            path = Path(str(value))
            candidates.append(path if path.is_absolute() else run_dir / path)
    files: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen or not path.is_file() or not path.is_relative_to(run_dir):
            continue
        seen.add(path)
        files.append(path)
    manifest_files: list[dict[str, Any]] = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files, key=lambda item: str(item.relative_to(run_dir)).lower()):
            relative = path.relative_to(run_dir).as_posix()
            payload = path.read_bytes()
            archive.writestr(relative, payload)
            manifest_files.append({"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
        package_manifest = {
            "schema_version": 1,
            "run_id": str(run_id),
            "project": str(run.get("project") or ""),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "note": "SHA-256 values are integrity fingerprints, not credentials or biological-validation claims.",
            "files": manifest_files,
        }
        archive.writestr("evidence_package_manifest.json", json.dumps(package_manifest, indent=2, sort_keys=True))
    filename = f"{_slug(str(run.get('project') or 'assayready'))}_evidence_package.zip"
    return buffer.getvalue(), filename


def _failure_panel(title: str, exc: Exception, debug: bool = False) -> html.Div:
    import os
    show_tb = os.environ.get(SHOW_TRACEBACKS_ENV, "0") == "1" or debug
    children = [
        _status(f"{title}: {exc}", kind="danger"),
    ]
    if show_tb:
        tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        children.append(
            html.Pre(
                tb_text,
                style={
                    "whiteSpace": "pre-wrap",
                    "fontSize": "12px",
                    "background": "#fff",
                    "padding": "12px",
                    "border": f"1px solid {COLORS['line']}",
                    "borderRadius": "6px",
                },
            )
        )
    return html.Div(children)



def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "", str(value).lower().replace(" ", "_")) or "assayready"


def _safe_upload_filename(value: str) -> str:
    name = Path(str(value).replace("\\", "/")).name
    suffix = Path(name).suffix.lower()
    stem = Path(name).stem
    safe_stem = re.sub(r"[^a-z0-9_-]+", "_", stem.lower()).strip("_") or "upload"
    safe_suffix = suffix if suffix in TABLE_EXTENSIONS | CONFIG_EXTENSIONS else ""
    return f"{safe_stem}{safe_suffix}"


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _installed_version() -> str:
    for distribution in ("assayready", "model-assessment"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "Development checkout"


def _bundled_model_count() -> int:
    root = _package_dir() / "model_bundles"
    if not root.is_dir():
        return 0
    return sum(1 for child in root.iterdir() if child.is_dir() and not child.name.startswith("."))


def _model_search_roots() -> list[Path]:
    configured = [
        Path(item).expanduser().resolve()
        for item in os.environ.get(MODEL_ROOTS_ENV, "").split(os.pathsep)
        if item.strip()
    ]
    project_models = _package_dir().parents[1] / "model_bundles_local"
    user_models = Path.home() / ".assayready" / "models"
    roots: list[Path] = []
    for root in [*configured, project_models, user_models]:
        resolved = root.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def _discover_local_scorers() -> list[dict[str, str]]:
    """Discover structurally complete local bundles without hashing large weights."""
    discovered: list[dict[str, str]] = []
    seen: set[Path] = set()
    for root in _model_search_roots():
        if not root.is_dir():
            continue
        try:
            manifests = list(root.rglob("assayready_model_manifest.json"))
        except OSError:
            continue
        for manifest_path in manifests[:100]:
            model_dir = manifest_path.parent.resolve()
            if model_dir in seen:
                continue
            bundle_root = model_dir.parent if model_dir.name.lower() == "weights" else model_dir.parent
            head_dir = bundle_root / "task_head"
            if not (model_dir / "model.safetensors").is_file():
                continue
            if not (head_dir / "head_manifest.json").is_file() or not (head_dir / "head.npz").is_file():
                continue
            try:
                model_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                head_manifest = json.loads((head_dir / "head_manifest.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            model_name = str(model_manifest.get("model_name") or model_dir.parent.name or model_dir.name)
            target = str(head_manifest.get("target_col") or "assay outcome")
            task = str(head_manifest.get("task_type") or "task")
            discovered.append(
                {
                    "label": f"{model_name} - {target} ({task})",
                    "value": str(model_dir),
                }
            )
            seen.add(model_dir)
    return sorted(discovered, key=lambda item: item["label"].lower())


def _scorer_options() -> list[dict[str, str]]:
    return [
        {"label": "Heuristic scoring (no trained model)", "value": DEMO_SCORER},
        *_discover_local_scorers(),
        {"label": "Custom AssayReady local bundle...", "value": CUSTOM_SCORER},
    ]


def _resolve_scorer_path(selection: str | None, custom_model_path: str | None) -> str | None:
    if not selection or selection == DEMO_SCORER:
        return None
    raw = custom_model_path if selection == CUSTOM_SCORER else selection
    selected = Path(str(raw or "").strip()).expanduser()
    if not str(raw or "").strip():
        raise ModelScoringError("Choose the weights folder for the custom AssayReady bundle.")
    if not selected.is_absolute():
        raise ModelScoringError("The scorer folder must be an absolute local path selected with Browse.")
    return str(selected.resolve())


def _simulation_visual_defaults(mode: str | None, preset: str | None) -> list[str]:
    del mode  # Presets are stable; the generation figure itself changes with the selected mode.
    return list(SIMULATION_VISUAL_PRESETS.get(str(preset or "recommended"), SIMULATION_VISUAL_PRESETS["recommended"]))


def _simulation_visual_options(mode: str | None) -> list[dict[str, str]]:
    generation_label = {
        "Mask-fill": "Masked-position nucleotide composition",
        "Random mutagenesis": "Mutation position and count profile",
        "Diversity sampling": "Observed sequence-spread diagnostics",
    }.get(str(mode), SIMULATION_VISUAL_LABELS["generation"])
    labels = {**SIMULATION_VISUAL_LABELS, "generation": generation_label}
    return [{"label": labels[key], "value": key} for key in ["landscape", "generation", "constraints", "plate"]]


def _example_path(name: str) -> Path:
    return _package_dir() / "examples" / name


def _json_clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_clean(value.item())
        except Exception:
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    _write_text_file(path, json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n")


def _write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _decode_upload(contents: str) -> bytes:
    import os
    if not contents:
        return b""
    limit_mb_str = os.environ.get(UPLOAD_LIMIT_ENV, "10.0")
    try:
        limit_mb = float(limit_mb_str)
    except ValueError:
        limit_mb = 10.0
    approx_size = len(contents) * 0.75
    if approx_size > limit_mb * 1024 * 1024:
        raise ValueError(f"Uploaded file is too large (exceeds limit of {limit_mb} MB).")
    try:
        _, encoded = contents.split(",", 1)
    except ValueError as exc:
        raise ValueError("Upload payload was not in Dash data-url format.") from exc
    return base64.b64decode(encoded)


def _upload_root() -> Path:
    return _output_root() / "_ui_uploads" / "dash"


def _cleanup_stale_uploads(*, now: float | None = None) -> None:
    try:
        retention_hours = max(0.0, float(os.environ.get(UPLOAD_RETENTION_HOURS_ENV, "24")))
    except ValueError:
        retention_hours = 24.0
    root = _upload_root()
    if not root.exists():
        return
    cutoff = (time.time() if now is None else now) - retention_hours * 3600
    for child in root.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child)
        except OSError:
            continue


def _save_upload(contents: str, filename: str, *, kind: str) -> tuple[Path, bytes]:
    blob = _decode_upload(contents)
    _cleanup_stale_uploads()
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    folder = _upload_root() / f"{kind}_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / _safe_upload_filename(filename or f"{kind}.csv")
    path.write_bytes(blob)
    return path, blob


def _read_table_bytes(blob: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    suffix = Path(name).suffix
    stream = io.BytesIO(blob)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(stream)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(stream, sep="\t")
    if suffix in {"", ".csv"}:
        return pd.read_csv(stream)
    if suffix in CONFIG_EXTENSIONS:
        raise ValueError("JSON/YAML uploads must be AssayReady config files that reference CSV/TSV/XLSX assay tables.")
    raise ValueError("Unsupported table upload. Use CSV, TSV, XLSX, or an AssayReady JSON/YAML config.")


def _read_table_path(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Config-referenced data file {path} must be CSV, TSV, or XLSX.")


def _load_config_bytes(blob: bytes, filename: str) -> dict[str, Any]:
    text = blob.decode("utf-8-sig")
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config uploads require PyYAML.") from exc
        config = yaml.safe_load(text) or {}
    else:
        config = json.loads(text)
    if not isinstance(config, dict):
        raise ValueError("Config upload must contain a JSON/YAML object.")
    return config


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _resolve_upload_reference(raw: Any, *, config_dir: Path) -> Path:
    configured_roots = [
        Path(item).expanduser().resolve()
        for item in os.environ.get(CONFIG_DATA_ROOTS_ENV, "").split(os.pathsep)
        if item.strip()
    ]
    search_roots = list(dict.fromkeys([config_dir.resolve(), (_package_dir() / "examples").resolve(), *configured_roots]))
    candidate = Path(str(raw))
    resolved = None
    if candidate.is_absolute():
        if candidate.exists():
            resolved = candidate.resolve()
        else:
            raise FileNotFoundError(f"Config references missing file: {candidate}")
    else:
        attempted = []
        for root in search_roots:
            option = root / candidate
            attempted.append(str(option))
            if option.exists():
                resolved = option.resolve()
                break
        if resolved is None:
            raise FileNotFoundError(f"Config references missing file {raw!r}. Tried: {', '.join(attempted)}")

    # Compare path segments, never string prefixes ("root-hacked" is not inside "root").
    is_safe = any(resolved == root or resolved.is_relative_to(root) for root in search_roots)
    if not is_safe:
        raise PermissionError(f"Access denied: Path {resolved} is outside permitted search roots.")

    return resolved


def _config_table_paths(config: dict[str, Any], *, config_dir: Path) -> tuple[list[Path], list[Path]]:
    ingestion = config.get("ingestion") or {}
    candidates = config.get("candidates") or {}
    assay_refs = _as_list(config.get("assay") or config.get("assay_files") or ingestion.get("raw_files"))
    candidate_refs = _as_list(config.get("candidate_files") or candidates.get("raw_files"))
    if not assay_refs:
        raise ValueError("Config must provide assay, assay_files, or ingestion.raw_files.")
    assay_paths = [_resolve_upload_reference(ref, config_dir=config_dir) for ref in assay_refs]
    candidate_paths = [_resolve_upload_reference(ref, config_dir=config_dir) for ref in candidate_refs]
    return assay_paths, candidate_paths


def _concat_tables(paths: list[Path]) -> pd.DataFrame:
    frames = [_read_table_path(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True, sort=False)


def _config_column_defaults(config: dict[str, Any]) -> dict[str, Any]:
    task = config.get("task") or {}
    split = config.get("split") or {}
    metadata = config.get("metadata") or {}
    return {
        "sequence_col": task.get("sequence_col", config.get("sequence_col")),
        "target_col": task.get("target_col", config.get("target_col")),
        "prediction_col": task.get("prediction_col", config.get("prediction_col")),
        "uncertainty_col": task.get("uncertainty_col", config.get("uncertainty_col")),
        "id_col": task.get("id_col", config.get("id_col")),
        "group_cols": list(split.get("group_cols") or config.get("group_cols") or []),
        "metadata_cols": list(metadata.get("categorical_cols") or config.get("metadata_cols") or []),
    }


def _config_ui_defaults(config: dict[str, Any], *, evaluation_manifest: dict[str, Any] | None) -> dict[str, Any]:
    task = config.get("task") or {}
    split = config.get("split") or {}
    training = config.get("training") or {}
    acquisition = config.get("acquisition") or {}
    generation = config.get("generation") or {}
    ingestion = config.get("ingestion") or {}
    uq = config.get("uq") or {}
    manifest = evaluation_manifest or {}
    provenance = manifest.get("provenance") or {}
    prediction_col = task.get("prediction_col", config.get("prediction_col"))
    return {
        "analysis_type": "prediction" if prediction_col else "local_model",
        "project": str(config.get("project") or ("prediction_audit" if prediction_col else "assayready_project")),
        "task_type": str(task.get("type", config.get("task_type", "regression"))).lower(),
        "top_k": int(acquisition.get("top_k", config.get("top_k", generation.get("plate_size", 96)))),
        "low_n": int(ingestion.get("low_n_threshold", config.get("low_n_threshold", 200))),
        "val_fraction": float(split.get("val_fraction", config.get("val_fraction", 0.15))),
        "test_fraction": float(split.get("test_fraction", config.get("test_fraction", 0.15))),
        "homology_threshold": float(split.get("homology_threshold", config.get("homology_threshold", 0.90))),
        "homology_k": int(split.get("homology_k", config.get("homology_k", 8))),
        "beta": float(acquisition.get("beta", config.get("beta", 1.0))),
        "diversity_penalty": float(acquisition.get("diversity_penalty", config.get("diversity_penalty", 0.2))),
        "seed": int(training.get("seed", split.get("seed", config.get("seed", 13)))),
        "ensemble_size": int(uq.get("ensemble_size", config.get("ensemble_size", 5))),
        "epochs": int(training.get("epochs", config.get("epochs", 80))),
        "learning_rate": float(training.get("learning_rate", config.get("learning_rate", 0.001))),
        "num_proposals": int(generation.get("num_proposals", config.get("num_proposals", 1000))),
        "evaluation_objective": str(manifest.get("objective_direction") or "maximize"),
        "evaluation_units": str(manifest.get("units") or "unspecified"),
        "evaluation_uncertainty_type": str(manifest.get("uncertainty_type") or "none"),
        "evaluation_model_id": str(provenance.get("model_identifier") or ""),
        "evaluation_data_id": str(provenance.get("data_identifier") or ""),
        "runtime": {
            "positive_label": task.get("positive_label", config.get("positive_label", manifest.get("positive_label"))),
            "column_aliases": dict(ingestion.get("column_aliases") or config.get("column_aliases") or {}),
            "similarity_policy": str(split.get("similarity_policy", config.get("similarity_policy", "canonical_kmer_jaccard"))),
            "similarity_sensitivity_policies": list(split.get("sensitivity_policies") or config.get("similarity_sensitivity_policies") or []),
            "similarity_sensitivity_thresholds": list(split.get("sensitivity_thresholds") or config.get("similarity_sensitivity_thresholds") or []),
            "acquisition_method": str(acquisition.get("method", config.get("acquisition_method", "upper_confidence_bound"))),
            "generate_candidates": bool(generation.get("enabled", config.get("generate_candidates", not bool(prediction_col)))),
        },
    }


def _config_preview_frame(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Mirror configured header aliases in the UI while leaving source files immutable."""
    ingestion = config.get("ingestion") or {}
    aliases = ingestion.get("column_aliases") or config.get("column_aliases") or {}
    if not isinstance(aliases, dict) or not aliases:
        return frame

    def clean(value: Any) -> str:
        normalized = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip().lower())
        return re.sub(r"_+", "_", normalized).strip("_")

    alias_lookup: dict[str, str] = {}
    for canonical, values in aliases.items():
        alias_lookup[clean(canonical)] = str(canonical)
        for value in _as_list(values):
            alias_lookup[clean(value)] = str(canonical)
    rename = {column: alias_lookup[clean(column)] for column in frame.columns if clean(column) in alias_lookup}
    if len(set(rename.values())) != len(rename.values()):
        raise ValueError("Configured column aliases map multiple uploaded columns to the same canonical column.")
    return frame.rename(columns=rename)


def _classification_label_options(
    assay_state: dict[str, Any] | None,
    target_col: str | None,
) -> list[dict[str, str]]:
    values = ((assay_state or {}).get("label_values") or {}).get(str(target_col or "")) or []
    return [{"label": str(value), "value": str(value)} for value in values]


def _classification_label_key(value: Any) -> str:
    text = str(value or "").strip()
    try:
        number = float(text)
    except ValueError:
        return text.casefold()
    return str(int(number)) if number.is_integer() else str(number)


def _valid_column(columns: list[str], value: Any, *, optional: bool = False) -> Any:
    if value is None or value == "":
        return None if optional else None
    return value if str(value) in set(columns) else None


def _valid_columns(columns: list[str], values: Any) -> list[str]:
    allowed = set(columns)
    return [str(value) for value in _as_list(values) if str(value) in allowed]


def _first_valid_col(columns: list[str], configured: Any, candidates: list[str], *, optional: bool = False) -> str | None:
    configured_col = _valid_column(columns, configured, optional=optional)
    if configured_col:
        return configured_col
    return _default_col(columns, candidates, optional=optional)


def _default_col(columns: list[str], candidates: list[str], *, optional: bool = False) -> str | None:
    lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    if optional:
        return None
    return columns[0] if columns else None


def _dropdown_options(values: list[str]) -> list[dict[str, str]]:
    return [{"label": value, "value": value} for value in values]


def _metric_text(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _short_hash(value: Any) -> str:
    text = str(value)
    return text[:12] if len(text) > 12 else text


def _as_clean_string(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _load_json_cell(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _report_payload(summary: dict[str, Any]) -> dict[str, Any]:
    if "prediction_audit" in summary:
        return summary["prediction_audit"]
    if "simulation_report" in summary:
        return summary["simulation_report"]
    return summary.get("benchmark_report", {})


def _safe_frame(frame: pd.DataFrame, *, limit: int | None = None) -> pd.DataFrame:
    if limit is not None:
        frame = frame.head(limit)
    out = frame.copy()
    for column in out.columns:
        if out[column].map(lambda item: isinstance(item, (dict, list, tuple))).any():
            out[column] = out[column].map(lambda item: json.dumps(_json_clean(item)) if isinstance(item, (dict, list, tuple)) else item)
    return out.where(pd.notna(out), "")


def _data_table(
    frame: pd.DataFrame,
    *,
    page_size: int = 10,
    height: int | None = None,
    compact: bool = False,
    table_id: str | None = None,
    hidden_columns: set[str] | None = None,
    selectable: bool = False,
    selected_rows: list[dict[str, Any]] | None = None,
) -> Any:
    if frame.empty:
        return html.Div("No rows to display.", style={"color": COLORS["muted"], "padding": "12px 0"})
    try:
        render_limit = max(1, int(os.environ.get(MAX_TABLE_RENDER_ROWS_ENV, "1000")))
    except ValueError:
        render_limit = 1000
    clean = _safe_frame(frame, limit=render_limit)
    columns = []
    hidden_columns = hidden_columns or set()
    full_text_grid = table_id in {"runs-index-table", "candidate-index-table"}
    for column in clean.columns:
        column_name = str(column)
        definition: dict[str, Any] = {
            "headerName": column_name,
            "field": column_name,
            "hide": column_name in hidden_columns,
            "tooltipField": column_name,
        }
        source = frame[column]
        if pd.api.types.is_integer_dtype(source) and not pd.api.types.is_bool_dtype(source):
            definition.update({"type": "numericColumn", "valueFormatter": {"function": "params.value == null || params.value === '' ? '' : Math.trunc(params.value).toLocaleString()"}})
        elif pd.api.types.is_float_dtype(source):
            definition.update({"type": "numericColumn", "valueFormatter": {"function": "params.value == null || params.value === '' ? '' : Number(params.value.toPrecision(5)).toString()"}})
        elif pd.api.types.is_bool_dtype(source):
            definition.update({"valueFormatter": {"function": "params.value === true ? 'Yes' : (params.value === false ? 'No' : '')"}})
        elif full_text_grid:
            definition.update({"wrapText": True, "autoHeight": True})
        if full_text_grid and column_name in {"Project", "Metric", "Sequence", "Primary risk"}:
            definition["minWidth"] = 240 if column_name != "Sequence" else 320
        if table_id == "runs-index-table":
            run_widths = {
                "Updated": 145,
                "Project": 245,
                "Workflow": 135,
                "Recommended use": 215,
                "Evaluation": 170,
                "Evidence": 165,
                "Held-out n": 105,
            }
            definition["minWidth"] = run_widths.get(column_name, definition.get("minWidth", 110))
        if table_id == "candidate-index-table" and column_name == "Sequence":
            definition["cellClass"] = "sequence-table-cell"
        if table_id == "candidate-index-table" and column_name == "Primary risk":
            definition.update(
                {
                    "cellClass": "candidate-risk-cell",
                    "cellClassRules": {
                        "candidate-risk-cell-clear": "params.value === 'None recorded'",
                        "candidate-risk-cell-warning": "params.value !== 'None recorded'",
                    },
                }
            )
        columns.append(definition)
    table_height = height or min(480, max(150, 54 + min(len(clean), page_size) * 36))
    table = dag.AgGrid(
        **({"id": table_id} if table_id else {}),
        rowData=clean.to_dict("records"),
        selectedRows=selected_rows or [],
        columnDefs=columns,
        defaultColDef={
            "sortable": True,
            "filter": not compact,
            "resizable": True,
            "minWidth": 110,
            "flex": 1,
        },
        dashGridOptions={
            "pagination": not compact,
            "paginationPageSize": int(page_size),
            "paginationPageSizeSelector": False,
            "animateRows": False,
            "rowHeight": 34 if compact else 38,
            "headerHeight": 38,
            "tooltipShowDelay": 300,
            **({"rowSelection": {"mode": "singleRow", "checkboxes": False, "headerCheckbox": False, "enableClickSelection": True}} if selectable else {}),
        },
        style={"height": f"{table_height}px", "width": "100%"},
        className=(
            f"ag-theme-quartz assayready-grid"
            f"{' selectable-grid' if selectable else ''}"
            f"{' full-text-grid' if full_text_grid else ''}"
        ),
    )
    if len(frame) > render_limit:
        return html.Div([
            _status(f"Showing the first {render_limit:,} of {len(frame):,} rows to protect browser memory.", kind="warning"),
            table,
        ])
    return table


def _card(children: Any, *, style: dict[str, Any] | None = None, className: str | None = None, **props: Any) -> html.Div:
    merged = dict(CARD_STYLE)
    if style:
        merged.update(style)
    classes = "ui-card" if not className else f"ui-card {className}"
    return html.Div(children, style=merged, className=classes, **props)


def _page_header(title: str, subtitle: str) -> html.Div:
    return html.Div(
        [
            html.H1(title, className="page-title", style={"fontSize": "28px", "margin": "0 0 6px"}),
            html.Div(subtitle, className="page-subtitle", style={"color": COLORS["muted"], "fontSize": "15px", "maxWidth": "880px"}),
        ],
        className="page-heading",
        style={"marginBottom": "20px"},
    )


def _field(label: str, child: Any, help_text: str | None = None) -> html.Div:
    label_children: list[Any] = [html.Span(label)]
    if help_text:
        label_children.append(html.Span("i", className="help-icon", title=help_text, **{"aria-label": help_text}))
    return html.Div(
        [
            html.Label(label_children, className="field-label"),
            child,
        ],
        style={"marginBottom": "14px"},
    )


def _metric(label: str, value: Any, *, note: str | None = None, tone: str = "neutral") -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "0", "color": COLORS["muted"]}),
            html.Div(_metric_text(value), style={"fontSize": "24px", "fontWeight": 800, "marginTop": "3px"}),
            html.Div(note, className="metric-note") if note else None,
        ],
        style={**CARD_STYLE, "padding": "14px"},
        className=f"metric-card metric-card-{tone}",
    )


def _planning_status_strip(*, eligible: int, generated: int, model_backed: bool) -> html.Div:
    scoring_label = "Model-backed scoring" if model_backed else "Heuristic scoring"
    return html.Div(
        [
            html.Strong(f"DRAFT · {scoring_label} · {eligible:,}/{generated:,} candidates eligible"),
            html.Span("Scores are for planning and ranking only—not biological validation."),
        ],
        className="planning-status-strip",
        role="status",
    )


def _generation_funnel(*, requested: int, generated: int, eligible: int) -> html.Section:
    excluded = max(0, generated - eligible)
    steps = [
        (requested, "requested", "requested"),
        (generated, "generated", "generated"),
        (eligible, "eligible", "eligible"),
        (excluded, "excluded", "excluded"),
    ]
    children: list[Any] = []
    for index, (value, label, tone) in enumerate(steps):
        if index:
            children.append(html.Span("→", className="generation-funnel-arrow", **{"aria-hidden": "true"}))
        children.append(
            html.Div(
                [html.Strong(f"{value:,}"), html.Span(label)],
                className=f"generation-funnel-step generation-funnel-{tone}",
            )
        )
    return html.Section(
        [
            html.Div(
                [
                    html.Div("Generation result", className="step-eyebrow"),
                    html.H2(f"{eligible:,} eligible candidates", className="generation-result-title"),
                    html.P("Eligibility reflects the saved sequence constraints.", className="chart-description"),
                ],
                className="generation-result-heading",
            ),
            html.Div(children, className="generation-funnel"),
        ],
        className="generation-result-summary",
        **{"aria-label": f"{requested} requested, {generated} generated, {eligible} eligible, {excluded} excluded"},
    )


def _run_index_stat(label: str, value: Any, *, title: str | None = None) -> html.Div:
    return html.Div(
        [html.Span(label), html.Strong(_metric_text(value))],
        className="run-index-stat",
        title=title,
    )


def _status(message: str, *, kind: str = "info") -> html.Div:
    palette = {
        "info": ("#e8f3f1", COLORS["accent_dark"]),
        "success": ("#e8f5ee", "#166534"),
        "warning": ("#fff7ed", COLORS["warn"]),
        "danger": ("#fef2f2", COLORS["danger"]),
    }
    bg, fg = palette.get(kind, palette["info"])
    return html.Div(
        message,
        style={"background": bg, "color": fg, "padding": "10px 12px", "borderRadius": "6px", "fontWeight": 650},
        className=f"status-message status-{kind}",
    )


def _copyable_command(command: str, description: str) -> html.Div:
    return html.Div(
        [
            html.P(description, className="setup-step-description"),
            html.Div(
                [
                    html.Code(command, className="command-chip"),
                    dcc.Clipboard(content=command, title="Copy command"),
                ],
                className="command-row",
            ),
        ],
        className="setup-step",
    )


def _scorer_setup_guide(*, open_by_default: bool = False) -> html.Details:
    install = 'python -m pip install -e ".[foundation-models]"'
    prepare = "assayready model prepare-dnabert2 --output-dir model_bundles_local/dnabert2"
    train = (
        "python model_bundles_local/dnabert2/train_head.py "
        "--model-dir model_bundles_local/dnabert2/weights "
        "--training outputs/assayready/<project>/<run-id>/processed/train.csv "
        "--output-dir model_bundles_local/dnabert2/task_head "
        "--sequence-col sequence --target-col <measured-column> "
        "--task-type regression --objective-direction maximize"
    )
    steps = [
        (
            "1",
            "Install the local model runtime",
            _copyable_command(
                install,
                "Run from the AssayReady project folder. Prediction-table audits do not require these extras.",
            ),
        ),
        (
            "2",
            "Download the pinned DNABERT-2 backbone",
            _copyable_command(prepare, "This downloads about 470 MB and verifies the pinned model snapshot."),
        ),
        (
            "3",
            "Train an assay-specific head",
            _copyable_command(
                train,
                "Use only a locked training partition—never validation, test, candidate, or future-outcome rows. Replace the placeholders; for classification, also declare the positive label.",
            ),
        ),
        (
            "4",
            "Select and verify",
            html.P(
                "Select the weights folder, select the head folder if it is not the adjacent task_head folder, then click Verify scorer. Verification checks recorded model files, head compatibility, and a real forward pass.",
                className="setup-step-description",
            ),
        ),
        (
            "5",
            "Use a different model",
            html.Div(
                [
                    html.P(
                        "Direct Design Sandbox scoring currently supports the AssayReady DNABERT-2 bundle contract, not arbitrary Hugging Face or pickle models. For another model, generate a table with predictions (and optional uncertainty), then use Analyze > Evaluate model predictions. Containerized models can use the controlled-execution workflow.",
                        className="setup-step-description",
                    ),
                    dcc.Link("Open Analyze", href="/analyze", className="text-action-link"),
                ]
            ),
        ),
    ]
    return html.Details(
        [
            html.Summary("Set up a scorer or use your own model", className="details-summary"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [html.Span(number, className="setup-step-number"), html.Strong(title)],
                                className="setup-step-heading",
                            ),
                            body,
                        ],
                        className="setup-step-block",
                    )
                    for number, title, body in steps
                ],
                className="scorer-setup-guide",
            ),
        ],
        className="settings-details scorer-guide-details",
        open=open_by_default,
    )


def _recommended_model_bundle() -> Path:
    return (_output_root() / "_models" / "dnabert2_117m").resolve()


def _scorer_setup_wizard() -> html.Details:
    """Non-technical install/import/train workflow for directly supported scorers."""
    return html.Details(
        [
            html.Summary("Add or prepare a scorer", className="details-summary"),
            dcc.Store(id="scorer-verification-state"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("1", className="setup-step-number"),
                            html.Div(
                                [
                                    html.Strong("Get the supported frozen model"),
                                    html.P(
                                        "Download AssayReady's authenticated DNABERT-2 snapshot. The revision is fixed, every downloaded file is checked, and an interrupted download can be resumed.",
                                        className="setup-step-description",
                                    ),
                                    html.Div(
                                        [
                                            html.Button("Install modeling support", id="install-model-runtime-button", n_clicks=0, style=SECONDARY_BUTTON_STYLE),
                                            html.Button("Download recommended model", id="install-recommended-model-button", n_clicks=0, style=BUTTON_STYLE),
                                            html.Span("Runtime may use several GB; model download is about 470 MB.", className="model-size-note"),
                                        ],
                                        className="model-wizard-actions",
                                    ),
                                    html.Div(id="recommended-model-install-status", className="inline-action-status"),
                                ]
                            ),
                        ],
                        className="model-wizard-step",
                    ),
                    html.Div(
                        [
                            html.Span("2", className="setup-step-number"),
                            html.Div(
                                [
                                    html.Strong("Teach it your assay"),
                                    html.P(
                                        "A frozen DNA model does not know your assay outcome by itself. Upload a CSV, TSV, or Excel table containing training sequences and measured outcomes; AssayReady will create the small assay-specific head.",
                                        className="setup-step-description",
                                    ),
                                    dcc.Store(id="scorer-training-upload-state"),
                                    dcc.Upload(
                                        id="scorer-training-upload",
                                        children=html.Div([html.Strong("Choose training table"), html.Span(" or drop it here")]),
                                        className="secondary-upload model-training-upload",
                                        accept=".csv,.tsv,.txt,.xlsx,.xls",
                                        multiple=False,
                                    ),
                                    html.Div(id="scorer-training-preview"),
                                    html.Div(
                                        [
                                            _field("DNA sequence column", dcc.Dropdown(id="scorer-training-sequence-col", options=[], value=None, clearable=False)),
                                            _field("Measured outcome column", dcc.Dropdown(id="scorer-training-target-col", options=[], value=None, clearable=False)),
                                            _field(
                                                "Outcome type",
                                                dcc.Dropdown(
                                                    id="scorer-training-task-type",
                                                    options=[
                                                        {"label": "Number (regression)", "value": "regression"},
                                                        {"label": "Two categories (classification)", "value": "classification"},
                                                    ],
                                                    value="regression",
                                                    clearable=False,
                                                ),
                                            ),
                                            _field(
                                                "Preferred direction",
                                                dcc.Dropdown(
                                                    id="scorer-training-objective",
                                                    options=[
                                                        {"label": "Higher is better", "value": "maximize"},
                                                        {"label": "Lower is better", "value": "minimize"},
                                                    ],
                                                    value="maximize",
                                                    clearable=False,
                                                ),
                                            ),
                                            _field(
                                                "Positive category (classification only)",
                                                dcc.Dropdown(id="scorer-training-positive-label", options=[], value=None, placeholder="Choose after selecting the outcome column"),
                                            ),
                                        ],
                                        className="model-training-grid",
                                    ),
                                    dcc.Checklist(
                                        id="scorer-training-confirmation",
                                        options=[
                                            {
                                                "label": "I confirm this table contains training data only—not validation, held-out test, candidate, or future outcome rows.",
                                                "value": "confirmed",
                                            }
                                        ],
                                        value=[],
                                        className="model-training-confirmation",
                                    ),
                                    html.Button("Create and verify scorer", id="train-scorer-button", n_clicks=0, style=BUTTON_STYLE),
                                    html.Div(id="scorer-training-status", className="inline-action-status", **{"aria-live": "polite"}),
                                ]
                            ),
                        ],
                        className="model-wizard-step",
                    ),
                    html.Div(
                        [
                            html.Span("3", className="setup-step-number"),
                            html.Div(
                                [
                                    html.Strong("Already have a scorer?"),
                                    html.P(
                                        "Import an existing AssayReady bundle with the folder buttons below. Verification is required before the scorer can run.",
                                        className="setup-step-description",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(_field("Model weights folder", dcc.Input(id="custom-model-path", type="text", placeholder="Choose the folder containing assayready_model_manifest.json", readOnly=True, style=INPUT_STYLE)), className="folder-path-field"),
                                            html.Button("Browse model...", id="browse-custom-model-button", n_clicks=0, style=SECONDARY_BUTTON_STYLE),
                                        ],
                                        className="folder-picker-row",
                                    ),
                                    html.Div(id="custom-model-picker-status", className="inline-action-status"),
                                    html.Div(
                                        [
                                            html.Div(_field("Assay head folder", dcc.Input(id="custom-head-path", type="text", placeholder="Auto-detects an adjacent task_head folder", readOnly=True, style=INPUT_STYLE)), className="folder-path-field"),
                                            html.Button("Browse head...", id="browse-custom-head-button", n_clicks=0, style=SECONDARY_BUTTON_STYLE),
                                        ],
                                        className="folder-picker-row",
                                    ),
                                    html.Div(id="custom-head-picker-status", className="inline-action-status"),
                                    html.Button("Verify imported scorer", id="verify-scorer-button", n_clicks=0, style=SECONDARY_BUTTON_STYLE),
                                ]
                            ),
                        ],
                        className="model-wizard-step",
                    ),
                    html.Div(
                        [
                            html.Strong("Want a different online model?"),
                            html.P(
                                "Frozen weights alone are not a compatible scorer. AssayReady currently runs only the authenticated DNABERT-2 adapter directly. For another model, export sequence IDs, predictions, and optional uncertainty from that model, then choose Analyze \u2192 Evaluate model predictions. This avoids executing arbitrary internet code inside the local UI.",
                                className="setup-step-description",
                            ),
                            dcc.Link("Open prediction evaluation", href="/analyze", className="text-action-link"),
                        ],
                        className="model-compatibility-note",
                    ),
                ],
                className="model-setup-wizard",
            ),
        ],
        className="settings-details model-setup-details",
    )


def _readiness_item(label: str, ready: bool) -> html.Div:
    return html.Div(
        [
            html.Span("✓ Ready" if ready else "○ Pending", className=f"ready-pill {'ready' if ready else 'pending'}"),
            html.Span(label),
        ],
        className="ready-item",
    )


def _readiness_panel(*, assay_ready: bool, columns_ready: bool, prediction_ready: bool, analysis_type: str) -> html.Div:
    items = [
        _readiness_item("Assay table loaded", assay_ready),
        _readiness_item("Required columns valid", columns_ready),
    ]
    if analysis_type == "prediction":
        items.append(_readiness_item("Prediction column selected", prediction_ready))
    return html.Div(items, className="readiness-panel")


def _artifact_options(summary: dict[str, Any], report_name: str) -> list[dict[str, str]]:
    artifact_dir = Path(str(summary.get("artifact_dir") or ""))
    names = [report_name, "run_summary.json", "prediction_audit_summary.json", "simulation_summary.json"]
    names.extend(str(item) for item in summary.get("artifacts") or [])
    seen: set[str] = set()
    options: list[dict[str, str]] = []
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        path = artifact_dir / name
        if path.exists() and path.is_file():
            options.append({"label": name, "value": str(path)})
    return options


def _store_result(summary: dict[str, Any], report_name: str, workflow: str) -> None:
    try:
        summary["run_id"] = record_run(summary, workflow=workflow, report_name=report_name)
    except Exception as exc:
        summary.setdefault("run_store_warning", f"Run completed, but local run indexing failed: {exc}")


def _drop_empty_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for column in frame.columns:
        values = frame[column]
        if values.notna().any() and values.astype(str).str.strip().ne("").any():
            keep.append(column)
    return frame[keep]


def _prepare_ranked_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _drop_empty_columns(frame.copy())
    if "sequence_hash" in frame.columns:
        frame["sequence_hash"] = frame["sequence_hash"].map(_short_hash)
    id_columns = [column for column in ["sequence_id", "construct_id", "variant_id", "id"] if column in frame.columns]
    display_ids = pd.Series([""] * len(frame), index=frame.index, dtype="object")
    for column in id_columns:
        values = frame[column].map(_as_clean_string)
        display_ids = display_ids.mask(display_ids.eq(""), values)
    if "sequence_hash" in frame.columns:
        hash_values = frame["sequence_hash"].map(_as_clean_string).map(lambda value: f"hash:{value}" if value else "")
        display_ids = display_ids.mask(display_ids.eq(""), hash_values)
    if "sequence" in frame.columns:
        sequence_values = frame["sequence"].map(_as_clean_string).str.slice(0, 18)
        display_ids = display_ids.mask(display_ids.eq(""), sequence_values)
    frame["display_id"] = display_ids
    return frame


def _candidate_prioritization_path(artifact_dir: Path) -> Path | None:
    """Prefer the claim-safe artifact name while retaining legacy run support."""
    for name in ["candidate_prioritization.csv", "ranked_candidates.csv"]:
        path = artifact_dir / name
        if path.is_file():
            return path
    return None


def _load_candidate_explanations(artifact_dir: Path) -> list[dict[str, Any]]:
    json_path = artifact_dir / "candidate_explanations.json"
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            rows = payload.get("candidates") if isinstance(payload, dict) else []
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        except json.JSONDecodeError:
            pass
    csv_path = artifact_dir / "candidate_explanations.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path).to_dict("records")
    return []


def _explanation_map(artifact_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in _load_candidate_explanations(artifact_dir):
        reason = str(item.get("ranking_reason") or "")
        if not reason:
            continue
        for key in [item.get("rank"), item.get("display_id"), item.get("sequence_hash")]:
            if key is not None and str(key):
                out[str(key)] = reason
    return out


def _find_sequence_column(frame: pd.DataFrame, summary: dict[str, Any]) -> str | None:
    params = summary.get("params") or {}
    for column in ["sequence", str(params.get("sequence_col") or ""), "dna", "promoter_sequence", "insert_sequence"]:
        if column and column in frame.columns:
            return column
    return None


def _prediction_column(frame: pd.DataFrame) -> str | None:
    for column in ["assay_prediction", "prediction", "model_prediction", "predicted_activity", "predicted_value"]:
        if column in frame.columns:
            return column
    return None


def _uncertainty_column(frame: pd.DataFrame) -> str | None:
    for column in ["assay_uncertainty", "uncertainty", "model_uncertainty"]:
        if column in frame.columns:
            return column
    return None


def _top_candidates(summary: dict[str, Any], *, top_n: int = 10) -> pd.DataFrame:
    artifact_dir = Path(str(summary["artifact_dir"]))
    prioritization_path = _candidate_prioritization_path(artifact_dir)
    if prioritization_path is None:
        return pd.DataFrame()
    prioritized = _prepare_ranked_frame(pd.read_csv(prioritization_path))
    if prioritized.empty:
        return prioritized
    sequence_col = _find_sequence_column(prioritized, summary)
    prediction_col = _prediction_column(prioritized)
    uncertainty_col = _uncertainty_column(prioritized)
    explanations = _explanation_map(artifact_dir)
    rows: list[dict[str, Any]] = []
    for _, row in prioritized.head(top_n).iterrows():
        rank = row.get("rank", "")
        display_id = str(row.get("display_id") or "")
        sequence_hash = str(row.get("sequence_hash") or "")
        reason = explanations.get(str(rank)) or explanations.get(display_id) or explanations.get(sequence_hash)
        rows.append(
            {
                "rank": rank,
                "id": display_id,
                "sequence": row.get(sequence_col, "") if sequence_col else "",
                "prediction": row.get(prediction_col, "") if prediction_col else "",
                "uncertainty": row.get(uncertainty_col, "") if uncertainty_col else "",
                "uncertainty_status": row.get("uncertainty_status", ""),
                "acquisition_policy": row.get("acquisition_policy", ""),
                "acquisition_score": row.get("acquisition_score", ""),
                "prioritization_status": row.get("prioritization_status", ""),
                "evidence_level": row.get("evidence_level", ""),
                "diversity_cluster": row.get("diversity_cluster", ""),
                "diversity_policy": row.get("diversity_policy", ""),
                "max_similarity_to_previous_selection": row.get(
                    "max_similarity_to_previous_selection", ""
                ),
                "why_this_rank": reason or "Prioritized by the configured acquisition score.",
            }
        )
    return pd.DataFrame(rows)


def _friendly_metric_name(value: Any, *, fallback: str = "Primary metric") -> str:
    text = str(value or "").strip().lower()
    labels = {
        "r2": "R-squared",
        "r_squared": "R-squared",
        "mae": "MAE",
        "rmse": "RMSE",
        "spearman": "Spearman correlation",
        "auroc": "AUROC",
        "average_precision": "Average precision",
        "balanced_accuracy": "Balanced accuracy",
    }
    return labels.get(text, _human_label(text, fallback=fallback) if text else fallback)


def _declared_min_test_rows(report: dict[str, Any]) -> int:
    profiles = ((report.get("threshold_sensitivity") or {}).get("profiles") or [])
    declared = next((item for item in profiles if str((item or {}).get("name") or "").lower() == "declared"), None)
    value = ((declared or {}).get("thresholds") or {}).get("min_test_rows", 20)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 20


def _uncertainty_semantics(report: dict[str, Any], summary: dict[str, Any] | None = None) -> str:
    manifest = report.get("evaluation_manifest") or (summary or {}).get("evaluation_manifest") or {}
    audit = report.get("uncertainty_audit") or {}
    return str(audit.get("declared_type") or manifest.get("uncertainty_type") or "none").strip().lower()


def _candidate_eligibility(summary: dict[str, Any]) -> tuple[bool, str, list[str]]:
    report = _report_payload(summary)
    scope = _row_scope(summary)
    test_rows = int(scope.get("test") or 0)
    minimum = _declared_min_test_rows(report)
    gate = report.get("claim_gate") or {}
    reasons: list[str] = []
    if test_rows < minimum:
        reasons.append(f"only {test_rows} held-out rows were available; the declared minimum is {minimum}")
    for key, label in [
        ("leakage_controlled", "leakage/independence checks did not pass"),
        ("lift_claim", "model lift did not pass"),
        ("candidate_constraints", "biological and synthesis constraints were not verified"),
    ]:
        if not bool((gate.get(key) or {}).get("ok")):
            reasons.append(label)
    if _uncertainty_semantics(report, summary) in {"", "none", "undeclared", "unspecified"}:
        reasons.append("uncertainty semantics were not declared")
    reasons = list(dict.fromkeys(reasons))
    eligible = not reasons
    if eligible:
        return True, "Policy-supported retrospective ordering", []
    return False, "Exploratory ordering only - not supported for wet-lab selection", reasons


def _row_scope(summary: dict[str, Any]) -> dict[str, int | None]:
    report = _report_payload(summary)
    split_sizes = (summary.get("split_diagnostics") or {}).get("split_sizes") or {}
    accepted = summary.get("valid_prediction_rows")
    if accepted is None:
        accepted = (summary.get("audit") or {}).get("accepted_rows")
    test_rows = (report.get("test_metrics") or {}).get("num_rows")
    if test_rows is None:
        test_rows = split_sizes.get("test")
    return {
        "accepted": int(accepted) if accepted is not None else None,
        "train": int(split_sizes["train"]) if split_sizes.get("train") is not None else None,
        "validation": int(split_sizes["val"]) if split_sizes.get("val") is not None else None,
        "test": int(test_rows) if test_rows is not None else None,
    }


def _summary_metrics(summary: dict[str, Any]) -> list[dict[str, Any]]:
    report = _report_payload(summary)
    best = report.get("best_simple_baseline") or {}
    scope = _row_scope(summary)
    metric_name = _friendly_metric_name(report.get("primary_metric_name"))
    test_rows = scope["test"] or 0
    minimum = _declared_min_test_rows(report)
    low_n = bool(test_rows < minimum)
    if "prediction_audit" in summary:
        uncertainty = report.get("uncertainty_audit") or {}
        uncertainty_value = uncertainty.get("uncertainty_abs_error_spearman")
        uncertainty_type = _uncertainty_semantics(report, summary)
        uncertainty_undeclared = uncertainty_type in {"", "none", "undeclared", "unspecified"}
        uncertainty_unreliable = low_n or uncertainty_undeclared
        if uncertainty_undeclared:
            uncertainty_label = "Not evaluated"
            uncertainty_note = f"Observed {_metric_text(uncertainty_value)}; semantics not declared"
        elif low_n:
            uncertainty_label = "Not reliable"
            uncertainty_note = f"Observed {_metric_text(uncertainty_value)}; n={test_rows}"
        else:
            uncertainty_label = uncertainty_value
            uncertainty_note = "Higher positive values indicate more useful error ordering"
        return [
            {"label": "Valid uploaded rows", "value": scope["accepted"]},
            {"label": "Held-out test rows", "value": scope["test"], "note": f"Declared minimum: {minimum}", "tone": "warning" if low_n else "neutral"},
            {
                "label": "Reliability" if low_n else f"Held-out {metric_name}",
                "value": "Insufficient evidence" if low_n else (report.get("test_metrics") or {}).get("primary_metric"),
                "note": f"Observed {metric_name}: {_metric_text((report.get('test_metrics') or {}).get('primary_metric'))} · only {test_rows} held-out rows" if low_n else "Higher is better" if metric_name in {"R-squared", "AUROC", "Average precision", "Balanced accuracy"} else None,
                "tone": "warning" if low_n else "neutral",
            },
            {
                "label": "Uncertainty/error Spearman",
                "value": uncertainty_label,
                "note": uncertainty_note,
                "tone": "warning" if uncertainty_unreliable else "neutral",
            },
        ]
    model_metric = report.get("task_head_primary_metric") or report.get("primary_metric")
    return [
        {"label": "Valid uploaded rows", "value": scope["accepted"]},
        {"label": "Held-out test rows", "value": scope["test"], "note": f"Declared minimum: {minimum}", "tone": "warning" if low_n else "neutral"},
        {
            "label": "Reliability" if low_n else f"Held-out {metric_name}",
            "value": "Insufficient evidence" if low_n else model_metric,
            "note": f"Observed {metric_name}: {_metric_text(model_metric)} · only {test_rows} held-out rows" if low_n else None,
            "tone": "warning" if low_n else "neutral",
        },
        {"label": "Ranked candidates", "value": summary.get("ranked_candidates")},
    ]


GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "displayModeBar": "hover",
    "showTips": False,
    "modeBarButtonsToRemove": ["zoomIn2d", "zoomOut2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "svg", "filename": "assayready_figure", "scale": 1},
}


def _selected_row_keys(event: dict[str, Any] | None) -> list[str]:
    """Return stable row keys from a Plotly selection/click payload."""
    if not isinstance(event, dict):
        return []
    keys: list[str] = []
    seen: set[str] = set()
    for point in event.get("points") or []:
        if not isinstance(point, dict):
            continue
        curve_number = point.get("curveNumber")
        if curve_number is not None and str(curve_number) != "0":
            continue
        customdata = point.get("customdata")
        if not isinstance(customdata, (list, tuple)) or len(customdata) < 2:
            continue
        key = str(customdata[0])
        if key.startswith("row-") and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _observation_value(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.5g}"


def _inspector_metric(label: str, value: Any) -> html.Div:
    return html.Div(
        [
            html.Div(label, className="inspector-metric-label"),
            html.Div(_observation_value(value), className="inspector-metric-value"),
        ],
        className="inspector-metric",
    )


def _observation_inspector(payload: dict[str, Any] | None, row_key: str | None) -> html.Div:
    """Render a scientifically scoped view of one persisted held-out row."""
    payload = payload or {}
    columns = payload.get("columns") or {}
    row_key_col = str(columns.get("row_key") or "row_key")
    row = next(
        (
            item
            for item in payload.get("rows") or []
            if isinstance(item, dict) and str(item.get(row_key_col)) == str(row_key)
        ),
        None,
    )
    if row is None:
        return html.Div(
            [
                html.Div("Click an observation", className="inspector-empty-title"),
                html.P("Choose a point in a held-out row chart to inspect its sequence, metadata, and model error."),
            ],
            className="inspector-empty",
        )

    id_col = columns.get("id")
    target_col = str(columns.get("target") or "target")
    prediction_col = str(columns.get("prediction") or "prediction")
    uncertainty_col = columns.get("uncertainty")
    sequence_col = columns.get("sequence")
    metadata_cols = [str(column) for column in columns.get("metadata") or []]
    measured = row.get(target_col)
    predicted = row.get(prediction_col)
    task_type = str(payload.get("task_type") or "regression").lower()
    residual: float | None = None
    try:
        residual = float(predicted) - float(measured)
    except (TypeError, ValueError):
        pass

    display_id = row.get(str(id_col)) if id_col else row.get(row_key_col)
    metrics = [
        _inspector_metric("Observed" if task_type == "classification" else "Measured", measured),
        _inspector_metric("Model score" if task_type == "classification" else "Predicted", predicted),
    ]
    if task_type == "regression":
        metrics.append(_inspector_metric("Residual", residual))
    if uncertainty_col and str(uncertainty_col) in row:
        metrics.append(_inspector_metric("Uncertainty", row.get(str(uncertainty_col))))

    if task_type == "regression" and residual is not None:
        if residual > 0:
            interpretation = f"The model over-predicted this observation by {_observation_value(abs(residual))}."
        elif residual < 0:
            interpretation = f"The model under-predicted this observation by {_observation_value(abs(residual))}."
        else:
            interpretation = "The prediction matches the persisted measurement for this observation."
    else:
        interpretation = "The model score is shown exactly as supplied; probability semantics are not assumed."

    metadata = [
        html.Div(
            [html.Dt(column.replace("_", " ").title()), html.Dd(_observation_value(row.get(column)))],
            className="inspector-metadata-item",
        )
        for column in metadata_cols
        if column in row and row.get(column) not in (None, "")
    ]
    sequence = row.get(str(sequence_col)) if sequence_col else None
    return html.Div(
        [
            html.Div("Held-out observation", className="inspector-eyebrow"),
            html.H3(str(display_id or row_key), className="inspector-title"),
            html.Div(metrics, className="inspector-metric-grid"),
            html.P(interpretation, className="inspector-interpretation"),
            html.Div(
                [html.Div("DNA sequence", className="inspector-section-label"), html.Code(str(sequence), className="inspector-sequence")],
                className="inspector-section",
            ) if sequence not in (None, "") else None,
            html.Div(
                [html.Div("Metadata", className="inspector-section-label"), html.Dl(metadata, className="inspector-metadata")],
                className="inspector-section",
            ) if metadata else None,
        ],
        className="inspector-content",
    )


def _graph_card(
    figure: Any,
    description: str,
    *,
    graph_id: dict[str, str] | str | None = None,
    interactive: bool = False,
    class_name: str | None = None,
) -> html.Div:
    graph_kwargs: dict[str, Any] = {
        "figure": figure,
        "config": {**GRAPH_CONFIG, "displayModeBar": True} if interactive else GRAPH_CONFIG,
        "className": f"result-graph{' row-explorer-graph' if interactive else ''}",
    }
    if graph_id is not None:
        graph_kwargs["id"] = graph_id
    return _card(
        [
            html.P(description, className="chart-description"),
            dcc.Graph(**graph_kwargs),
            html.Div("Drag to select · click a point to inspect · double-click to reset", className="chart-interaction-hint") if interactive else None,
        ],
        className=" ".join(
            item
            for item in [
                "visualization-card",
                "interactive-chart-card" if interactive else None,
                class_name,
            ]
            if item
        ),
    )


def _benchmark_evaluation_frame(summary: dict[str, Any], *, limit: int = 5000) -> tuple[pd.DataFrame, dict[str, str | None], str | None]:
    artifact_dir = Path(str(summary.get("artifact_dir") or "")).resolve()
    params = summary.get("params") or {}
    if "prediction_audit" in summary:
        path = artifact_dir / "processed" / "test.csv"
        columns = {
            "target": str(params.get("target_col") or ""),
            "prediction": str(params.get("prediction_col") or ""),
            "uncertainty": str(params.get("uncertainty_col") or "") or None,
            "id": str(params.get("id_col") or "") or None,
        }
    else:
        path = artifact_dir / "predictions.csv"
        columns = {"target": "target", "prediction": "prediction", "uncertainty": "uncertainty", "id": "id"}
    if not path.is_file() or not (path.resolve() == artifact_dir or path.resolve().is_relative_to(artifact_dir)):
        return pd.DataFrame(), columns, "The row-level held-out artifact was not found."
    try:
        frame = pd.read_csv(path, nrows=limit + 1)
    except Exception as exc:
        return pd.DataFrame(), columns, f"The held-out artifact could not be read: {exc}"
    if "prediction_audit" not in summary:
        companion_path = artifact_dir / "processed" / "test.csv"
        try:
            companion = pd.read_csv(companion_path, nrows=limit + 1) if companion_path.is_file() else pd.DataFrame()
        except Exception:
            companion = pd.DataFrame()
        if len(companion) >= len(frame) and not frame.empty:
            companion = companion.head(len(frame)).reset_index(drop=True)
            primary = frame.reset_index(drop=True)
            aligned = False
            if "sequence_hash" in primary and "sequence_hash" in companion:
                aligned = primary["sequence_hash"].astype(str).equals(companion["sequence_hash"].astype(str))
            target_source = str(params.get("target_col") or "")
            if not aligned and target_source in companion and "target" in primary:
                left = pd.to_numeric(primary["target"], errors="coerce")
                right = pd.to_numeric(companion[target_source], errors="coerce")
                aligned = bool((left.eq(right) | (left.isna() & right.isna())).all())
            if aligned:
                for column in companion.columns:
                    if column not in primary.columns:
                        primary[column] = companion[column]
                frame = primary
    note = None
    if len(frame) > limit:
        frame = frame.head(limit)
        note = f"Figures show the first {limit:,} persisted held-out rows; report metrics still use the complete evaluation set."
    return frame, columns, note


def _classification_text_items(value: Any) -> list[str]:
    return [str(item).strip() for item in _as_list(value) if str(item).strip()]


def _evidence_classification_panel(report: dict[str, Any]) -> html.Div:
    assurance = report.get("assurance_level") or {}
    evidence = report.get("evidence_level") or {}
    if not assurance:
        assurance = {
            "label": "Evidence not reported",
            "basis": [],
            "limitations": ["This run predates explicit assurance-level reporting."],
        }
    if not evidence:
        evidence = {
            "label": "Evaluation not reported",
            "support": [],
            "limitations": ["This run predates explicit evidence-level reporting."],
        }

    def level_card(
        payload: dict[str, Any],
        *,
        eyebrow: str,
        support_key: str,
        support_heading: str,
        exclude_limitations: set[str] | None = None,
    ) -> html.Div:
        support = _classification_text_items(payload.get(support_key))
        limitations = [
            item for item in _classification_text_items(payload.get("limitations"))
            if item not in (exclude_limitations or set())
        ]
        empty_support = (
            "This run supports description and inspection only; the policy checks below explain what prevents a stronger claim."
            if "descriptive" in str(payload.get("label") or "").lower()
            else "No positive support was recorded for this classification."
        )
        displayed_label = (
            _display_assurance_label(payload.get("label") or payload.get("key"))
            if eyebrow == "Evidence"
            else _display_evaluation_label(payload.get("label") or payload.get("key"))
        )
        return html.Div(
            [
                html.Div(eyebrow, className="evidence-level-eyebrow"),
                html.H3(displayed_label, className="evidence-level-label"),
                html.Div(
                    [
                        html.Strong(support_heading),
                        html.Ul([html.Li(item) for item in support])
                        if support
                        else html.P(empty_support, className="evidence-empty"),
                    ],
                    className="evidence-basis",
                ),
                html.Div(
                    [
                        html.Strong("Limitations"),
                        html.Ul([html.Li(item) for item in limitations])
                        if limitations
                        else html.P("No limitation was recorded; review the underlying policy checks.", className="evidence-empty"),
                    ],
                    className="evidence-limitations",
                ),
            ],
            className="evidence-level-item",
        )

    return html.Section(
        [
            html.Div(
                [
                    html.H2("Evidence classification", className="section-heading"),
                    html.P(
                        "Evidence describes provenance; evaluation describes the analysis performed. Recommended use is reported separately in each run summary.",
                        className="chart-description",
                    ),
                ],
                className="evidence-classification-heading",
            ),
            html.Div(
                [
                    level_card(assurance, eyebrow="Evidence", support_key="basis", support_heading="Basis"),
                    level_card(
                        evidence,
                        eyebrow="Evaluation",
                        support_key="support",
                        support_heading="Support",
                        exclude_limitations=set(_classification_text_items(assurance.get("limitations"))),
                    ),
                ],
                className="evidence-level-grid",
            ),
        ],
        className="evidence-classification-panel",
    )


def _threshold_sensitivity_panel(report: dict[str, Any]) -> html.Div | None:
    sensitivity = report.get("threshold_sensitivity") or {}
    profiles = [item for item in sensitivity.get("profiles") or [] if isinstance(item, dict)]
    if not sensitivity and not profiles:
        return None

    rows: list[Any] = []
    threshold_columns = [
        ("min_test_rows", "Min. test rows"),
        ("min_num_clusters", "Min. groups"),
        ("min_lift_delta", "Min. lift"),
        ("min_uncertainty_spearman", "Min. uncertainty correlation"),
        ("max_calibration_gap_ratio", "Max. calibration-gap ratio"),
    ]
    for profile in profiles:
        thresholds = profile.get("thresholds") or {}
        supported = bool(profile.get("retrospective_criteria_met"))
        rows.append(
            html.Tr(
                [
                    html.Th(str(profile.get("name") or "profile"), scope="row"),
                    *[html.Td(_metric_text(thresholds.get(key))) for key, _ in threshold_columns],
                    html.Td(_status_badge("Passed declared policy" if supported else "Did not pass", tone="success" if supported else "warning")),
                ]
            )
        )

    changes = bool(sensitivity.get("conclusion_changes_under_stricter_thresholds"))
    stable = bool(sensitivity.get("stable_under_stricter_thresholds"))
    if changes:
        conclusion = "The retrospective conclusion changes under at least one stricter threshold profile."
    elif stable:
        conclusion = "The retrospective criteria remain supported under the included stricter threshold profiles."
    else:
        conclusion = "The declared retrospective criteria are not supported; stricter profiles do not strengthen the conclusion."

    return _card(
        [
            html.H3("Threshold sensitivity", className="section-heading"),
            html.P(str(sensitivity.get("interpretation") or "Threshold profiles are diagnostic only."), className="chart-description"),
            html.P(conclusion, className="sensitivity-conclusion"),
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Profile"),
                                *[
                                    html.Th(label, title="A calibration-gap ratio compares observed calibration error with the configured tolerance." if key == "max_calibration_gap_ratio" else None)
                                    for key, label in threshold_columns
                                ],
                                html.Th("Retrospective policy"),
                            ]
                        )
                    ),
                    html.Tbody(rows),
                ],
                className="sensitivity-table threshold-table",
            )
            if rows
            else html.P("No sensitivity profiles were recorded.", className="evidence-empty"),
        ],
        className="threshold-sensitivity-card",
    )


def _similarity_sensitivity_panel(report: dict[str, Any]) -> html.Div | None:
    sensitivity = report.get("similarity_sensitivity") or {}
    if not sensitivity:
        return None
    status = str(sensitivity.get("status") or "").lower()
    if status != "evaluated":
        return _card(
            [
                html.H3("Similarity sensitivity", className="section-heading"),
                html.P(
                    str(
                        sensitivity.get("reason")
                        or sensitivity.get("interpretation")
                        or "Alternate similarity settings were not evaluated."
                    ),
                    className="chart-description",
                ),
                html.P(
                    "Not evaluated; this report makes no claim that the conclusion is stable across similarity definitions.",
                    className="sensitivity-conclusion",
                ),
            ],
            className="similarity-sensitivity-card",
        )
    changes = bool(sensitivity.get("conclusion_changes_across_similarity_settings"))
    membership = bool(sensitivity.get("test_membership_changes_across_similarity_settings"))
    settings = [item for item in sensitivity.get("settings") or [] if isinstance(item, dict)]
    rows = [
        html.Tr(
            [
                html.Th(_human_label(item.get("similarity_policy_requested") or "policy"), scope="row"),
                html.Td(str(item.get("similarity_threshold"))),
                html.Td(str(item.get("test_clusters", item.get("num_clusters")))),
                html.Td(_status_badge("Passed" if item.get("retrospective_criteria_met") else "Did not pass", tone="success" if item.get("retrospective_criteria_met") else "warning")),
            ]
        )
        for item in settings
    ]
    return _card(
        [
            html.H3("Similarity sensitivity", className="section-heading"),
            html.P(
                str(sensitivity.get("interpretation") or sensitivity.get("reason") or ""),
                className="chart-description",
            ),
            html.P(
                f"Test membership changed: {'Yes' if membership else 'No'}. Technical retrospective conclusion changed: {'Yes' if changes else 'No'}.",
                className="sensitivity-conclusion",
            ),
            html.Table(
                [
                    html.Thead(html.Tr([html.Th("Similarity policy"), html.Th("Threshold"), html.Th("Test groups"), html.Th("Declared policy")])),
                    html.Tbody(rows),
                ],
                className="sensitivity-table",
            )
            if rows
            else html.P("Alternate settings were not evaluated for this workflow.", className="evidence-empty"),
        ],
        className="similarity-sensitivity-card",
    )


def _claim_gate_panel(report: dict[str, Any]) -> html.Div:
    gate = report.get("claim_gate") or {}
    labels = [
        ("leakage_controlled", "Leakage / independence"),
        ("lift_claim", "Model lift"),
        ("uncertainty_usable", "Uncertainty usefulness"),
        ("candidate_constraints", "Candidate constraints"),
        ("candidate_prioritization", "Candidate prioritization"),
    ]
    items: list[Any] = []
    for key, label in labels:
        detail = gate.get(key) or (gate.get("recommended") if key == "candidate_prioritization" else {}) or {}
        ok = bool(detail.get("ok"))
        reasons = [_readable_reason(reason) for reason in detail.get("reasons") or []]
        state = "supported" if ok else "not-supported"
        items.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("✓ PASSED DECLARED POLICY" if ok else "× DID NOT PASS", className=f"gate-badge {state}"),
                            html.Strong(label),
                        ],
                        className="gate-heading",
                    ),
                    html.Ul([html.Li(reason) for reason in reasons]) if reasons else html.Div("Passed the configured retrospective checks; this is not biological validation.", className="gate-reason"),
                ],
                className=f"gate-item {state}",
            )
        )
    return _card(
        [
            html.H3("Policy checks", className="section-heading"),
            html.P("These checks compare the run with the declared policy; they are not validation badges.", className="chart-description"),
            html.Div(items, className="claim-gate-grid"),
        ],
        className="claim-gate-card",
    )


def _provenance_panel(report: dict[str, Any], summary: dict[str, Any]) -> html.Div:
    manifest = report.get("evaluation_manifest") or summary.get("evaluation_manifest") or {}
    provenance = manifest.get("provenance") or {}
    values = [
        ("Objective", _human_label(manifest.get("objective_direction"), fallback="Not provided")),
        ("Units", _human_label(manifest.get("units"), fallback="Not provided")),
        ("Uncertainty", _human_label(manifest.get("uncertainty_type"), fallback="Not declared")),
        ("Training independence", _human_label(manifest.get("training_independence"), fallback="Unverified")),
        ("Data ID", _provenance_identifier(provenance.get("data_identifier"), kind="data")),
        ("Model ID", _provenance_identifier(provenance.get("model_identifier"), kind="model")),
    ]
    return _card(
        [
            html.H3("Evaluation context", className="section-heading"),
            html.Dl([html.Div([html.Dt(label), html.Dd(value)], className="provenance-item") for label, value in values], className="provenance-grid"),
        ]
    )


def _benchmark_scope(summary: dict[str, Any], surface: str) -> str:
    run_hint = summary.get("run_id") or Path(str(summary.get("artifact_dir") or "run")).name or "run"
    return f"{surface}:{run_hint}"


def _benchmark_metadata_columns(
    frame: pd.DataFrame,
    summary: dict[str, Any],
    *,
    excluded: set[str],
) -> tuple[list[str], list[str]]:
    params = summary.get("params") or {}
    configured_metadata = [str(value) for value in params.get("metadata_cols") or []]
    configured_groups = [str(value) for value in params.get("group_cols") or []]
    by_lower = {str(column).lower(): str(column) for column in frame.columns}
    common = [
        by_lower[name]
        for name in ["organism", "species", "strain", "batch", "family", "source", "replicate", "plate", "leakage_cluster"]
        if name in by_lower
    ]
    candidates = list(dict.fromkeys(configured_metadata + configured_groups + common))
    metadata = [
        column
        for column in candidates
        if column in frame.columns
        and column not in excluded
        and column != "_source_file"
        and not column.startswith("_")
    ]
    slice_cols = [
        column
        for column in dict.fromkeys(configured_metadata + configured_groups + common)
        if column in metadata and 2 <= frame[column].nunique(dropna=True) <= 50
    ]
    return metadata, slice_cols


def _benchmark_row_payload(
    frame: pd.DataFrame,
    summary: dict[str, Any],
    report: dict[str, Any],
    *,
    target_col: str,
    prediction_col: str,
    uncertainty_col: str | None,
    id_col: str | None,
    task_type: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    data = frame.reset_index(drop=True).copy()
    data["_row_key"] = [f"row-{index}" for index in range(len(data))]
    valid = pd.to_numeric(data[prediction_col], errors="coerce").notna() & data[target_col].notna()
    if task_type == "regression":
        valid &= pd.to_numeric(data[target_col], errors="coerce").notna()
    data = data.loc[valid].reset_index(drop=True)
    id_col = id_col if id_col and id_col in data.columns else None
    uncertainty_col = uncertainty_col if uncertainty_col and uncertainty_col in data.columns else None
    sequence_col = _find_sequence_column(data, summary)
    excluded = {target_col, prediction_col, "_row_key"}
    excluded.update(value for value in [id_col, uncertainty_col, sequence_col] if value)
    metadata, slice_cols = _benchmark_metadata_columns(data, summary, excluded=excluded)
    plot_columns = list(
        dict.fromkeys(
            [
                "_row_key",
                target_col,
                prediction_col,
                *([id_col] if id_col else []),
                *([uncertainty_col] if uncertainty_col else []),
                *metadata,
                *slice_cols,
            ]
        )
    )
    inspector_columns = list(dict.fromkeys([*plot_columns, *([sequence_col] if sequence_col else [])]))
    stored = _safe_frame(data[plot_columns])
    payload = {
        "rows": stored.to_dict("records"),
        "columns": {
            "row_key": "_row_key",
            "id": id_col,
            "target": target_col,
            "prediction": prediction_col,
            "uncertainty": uncertainty_col,
            "sequence": sequence_col,
            "metadata": metadata,
        },
        "slice_cols": slice_cols,
        "task_type": task_type,
        "uncertainty_audit": report.get("uncertainty_audit") or {},
    }
    inspector_payload = {**payload, "rows": _safe_frame(data[inspector_columns]).to_dict("records")}
    return data, payload, inspector_payload


def _payload_frame(payload: dict[str, Any] | None) -> pd.DataFrame:
    return pd.DataFrame((payload or {}).get("rows") or [])


def _payload_selected_frame(payload: dict[str, Any] | None, selected_row_keys: list[str] | None) -> pd.DataFrame:
    frame = _payload_frame(payload)
    selected = {str(value) for value in (selected_row_keys or [])}
    row_key_col = str(((payload or {}).get("columns") or {}).get("row_key") or "_row_key")
    if selected and row_key_col in frame:
        return frame[frame[row_key_col].astype(str).isin(selected)].copy()
    return frame


def _payload_regression_figure(payload: dict[str, Any], selected_row_keys: list[str] | None, *, residual: bool) -> Any:
    columns = payload.get("columns") or {}
    builder = residual_figure if residual else regression_fit_figure
    return builder(
        _payload_frame(payload),
        target_col=str(columns.get("target") or "target"),
        prediction_col=str(columns.get("prediction") or "prediction"),
        id_col=columns.get("id"),
        hover_cols=list(columns.get("metadata") or []),
        row_key_col=str(columns.get("row_key") or "_row_key"),
        selected_row_keys=selected_row_keys,
    )


def _payload_uncertainty_figure(payload: dict[str, Any], selected_row_keys: list[str] | None) -> Any:
    columns = payload.get("columns") or {}
    return uncertainty_figure(
        _payload_frame(payload),
        target_col=str(columns.get("target") or "target"),
        prediction_col=str(columns.get("prediction") or "prediction"),
        uncertainty_col=str(columns.get("uncertainty") or "uncertainty"),
        audit=payload.get("uncertainty_audit") or {},
        id_col=columns.get("id"),
        hover_cols=list(columns.get("metadata") or []),
        row_key_col=str(columns.get("row_key") or "_row_key"),
        selected_row_keys=selected_row_keys,
    )


def _payload_slice_table(payload: dict[str, Any], selected_row_keys: list[str] | None) -> Any:
    columns = payload.get("columns") or {}
    frame = _payload_selected_frame(payload, selected_row_keys)
    slice_frame = regression_slice_summary(
        frame,
        target_col=str(columns.get("target") or "target"),
        prediction_col=str(columns.get("prediction") or "prediction"),
        slice_cols=[str(value) for value in payload.get("slice_cols") or []],
    )
    if slice_frame.empty:
        return _status("No eligible subgroup columns were available for this cohort.")
    return _data_table(slice_frame, page_size=12, height=380)


def _selection_count(payload: dict[str, Any] | None, selected_row_keys: list[str] | None) -> html.Span:
    total = len((payload or {}).get("rows") or [])
    selected = len(set(str(value) for value in (selected_row_keys or [])))
    if selected:
        return html.Span(f"{selected:,} of {total:,} rows selected", className="selection-count-pill active")
    return html.Span(f"All {total:,} held-out rows", className="selection-count-pill")


def _format_interval(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "Not available"
    try:
        low, high = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return "Not available"
    return f"[{low:.3g}, {high:.3g}]"


def _diagnostic_finding(
    frame: pd.DataFrame,
    *,
    target_col: str,
    prediction_col: str,
    report: dict[str, Any],
) -> str | None:
    if frame.empty or target_col not in frame or prediction_col not in frame:
        return None
    measured = pd.to_numeric(frame[target_col], errors="coerce")
    predicted = pd.to_numeric(frame[prediction_col], errors="coerce")
    valid = measured.notna() & predicted.notna()
    if int(valid.sum()) < 3:
        return "Too few held-out observations are available for a stable diagnostic pattern."
    measured = measured[valid]
    predicted = predicted[valid]
    measured_sd = float(measured.std(ddof=1))
    predicted_sd = float(predicted.std(ddof=1))
    spread_ratio = predicted_sd / measured_sd if measured_sd > 0 else float("nan")
    bias = float((predicted - measured).mean())
    parts: list[str] = []
    if pd.notna(spread_ratio) and spread_ratio < 0.6:
        parts.append(f"Predictions span only {spread_ratio:.0%} of the measured variation, indicating strong compression toward the mean")
    elif pd.notna(spread_ratio) and spread_ratio > 1.6:
        parts.append(f"Predictions vary {spread_ratio:.1f} times more than measurements, indicating a scale mismatch")
    if abs(bias) > 0.25 * max(measured_sd, 1e-12):
        parts.append(f"the mean prediction-minus-measurement bias is {bias:.3g}")
    rho = ((report.get("uncertainty_audit") or {}).get("uncertainty_abs_error_spearman"))
    if rho is not None:
        try:
            rho_value = float(rho)
            if rho_value <= 0:
                parts.append(f"reported uncertainty does not positively order observed error (Spearman {rho_value:.3f})")
            elif rho_value < 0.2:
                parts.append(f"reported uncertainty has weak error-ordering usefulness (Spearman {rho_value:.3f})")
        except (TypeError, ValueError):
            pass
    if not parts:
        return "No single dominant scale, bias, or uncertainty-ordering warning was detected; inspect the plots and subgroup results."
    return ". ".join(part[0].upper() + part[1:] for part in parts) + "."


def _benchmark_section(summary: dict[str, Any], *, standalone: bool = False, surface: str = "benchmark") -> Any:
    report = _report_payload(summary)
    if not report.get("baselines"):
        if report.get("assurance_level") or report.get("evidence_level"):
            return html.Section(
                [
                    _status(
                        "No fitted baselines are available. The evidence classification and its limitations remain visible below.",
                        kind="warning",
                    ),
                    _evidence_classification_panel(report),
                    _threshold_sensitivity_panel(report),
                    _similarity_sensitivity_panel(report),
                    _claim_gate_panel(report),
                    _provenance_panel(report, summary),
                ],
                className="benchmark-section evidence-only-benchmark",
            )
        return _status("This run does not contain fitted benchmark baselines.") if standalone else None
    frame, columns, frame_note = _benchmark_evaluation_frame(summary)
    target_col = str(columns.get("target") or "")
    prediction_col = str(columns.get("prediction") or "")
    uncertainty_col = str(columns.get("uncertainty") or "")
    id_col = str(columns.get("id") or "") or None
    task_type = str(report.get("task_type") or (summary.get("params") or {}).get("task_type") or "regression").lower()
    split = summary.get("split_diagnostics") or {}
    claim_gate = report.get("claim_gate") or {}
    leakage_ok = bool((claim_gate.get("leakage_controlled") or {}).get("ok"))
    test_metrics = report.get("test_metrics") or {}
    lift = claim_gate.get("lift_claim") or {}
    model_ci = lift.get("model_metric_ci")
    model_ci_text = _format_interval(model_ci)
    test_rows = int(test_metrics.get("num_rows") or (split.get("split_sizes") or {}).get("test") or 0)
    minimum_test_rows = _declared_min_test_rows(report)
    low_n = test_rows < minimum_test_rows
    metric_name = _friendly_metric_name(report.get("primary_metric_name"))
    full_groups = split.get("num_clusters")
    test_groups = (split.get("split_cluster_counts") or {}).get("test")
    uncertainty_undeclared = _uncertainty_semantics(report, summary) in {"", "none", "undeclared", "unspecified"}
    scope = _benchmark_scope(summary, surface)
    run_charts: list[Any] = [
        _graph_card(
            benchmark_metric_figure(report),
            "Actual baselines were fitted on the persisted train split and evaluated on the held-out split. The interval applies to the model only.",
            class_name="evidence-overview-chart",
        ),
        _graph_card(
            split_composition_figure(split),
            "Group and sequence-similarity boundaries determine these splits; few clusters make performance estimates unstable.",
            class_name="evidence-overview-chart split-composition-card",
        ),
    ]
    required = {target_col, prediction_col}
    row_explorer: Any = None
    diagnostic_finding: str | None = None
    if not frame.empty and required.issubset(frame.columns):
        if low_n:
            diagnostic_finding = (
                f"Only {test_rows} held-out rows are available; diagnostic patterns are unstable below the declared minimum of {minimum_test_rows}. "
                "Use these plots for inspection only."
            )
        elif task_type == "regression":
            diagnostic_finding = _diagnostic_finding(
                frame,
                target_col=target_col,
                prediction_col=prediction_col,
                report=report,
            )
        interactive_frame, row_payload, inspector_payload = _benchmark_row_payload(
            frame,
            summary,
            report,
            target_col=target_col,
            prediction_col=prediction_col,
            uncertainty_col=uncertainty_col or None,
            id_col=id_col,
            task_type=task_type,
        )
        if task_type == "classification":
            run_charts.append(
                _graph_card(
                    classification_discrimination_figure(
                        interactive_frame,
                        target_col=target_col,
                        prediction_col=prediction_col,
                        positive_label=report.get("positive_label"),
                    ),
                    "ROC and precision–recall use score ordering only. Calibration is intentionally omitted unless probability semantics are declared.",
                )
            )
        if task_type == "ranking":
            objective_direction = str(
                (report.get("evaluation_manifest") or {}).get("objective_direction")
                or (summary.get("params") or {}).get("objective_direction")
                or "maximize"
            )
            run_charts.append(
                _graph_card(
                    ranking_diagnostics_figure(
                        interactive_frame,
                        target_col=target_col,
                        prediction_col=prediction_col,
                        objective_direction=objective_direction,
                    ),
                    "Rank agreement and top-k recovery use the declared objective direction. They describe held-out ordering, not prospective utility.",
                )
            )
        row_charts: list[Any] = []
        if task_type == "regression":
            row_charts.extend(
                [
                    _graph_card(
                        _payload_regression_figure(row_payload, [], residual=False),
                        "Each point is a persisted held-out observation; the diagonal is perfect agreement.",
                        graph_id={"type": "benchmark-fit-graph", "index": scope},
                        interactive=True,
                    ),
                    _graph_card(
                        _payload_regression_figure(row_payload, [], residual=True),
                        "Residuals are prediction minus measurement. Structure around zero can reveal bias, scale errors, or outliers.",
                        graph_id={"type": "benchmark-residual-graph", "index": scope},
                        interactive=True,
                    ),
                ]
            )
        if task_type == "regression" and uncertainty_col and uncertainty_col in frame.columns:
            row_charts.append(
                _graph_card(
                    _payload_uncertainty_figure(row_payload, []),
                    "This is an ordering/usefulness diagnostic unless uncertainty was explicitly declared as predicted absolute error.",
                    graph_id={"type": "benchmark-uncertainty-graph", "index": scope},
                    interactive=True,
                )
            )
        elif task_type == "classification" and uncertainty_col and uncertainty_col in frame.columns:
            row_explorer = _status(
                "Row-level classification uncertainty is not plotted until prediction values are declared as probabilities, logits, or ranking scores.",
                kind="info",
            )
        if row_charts:
            slice_card = None
            if task_type == "regression":
                slice_card = _card(
                    [
                        html.H3("Descriptive subgroup errors", className="section-heading"),
                        html.P(
                            "Slices are exploratory and are not multiplicity-adjusted. The table recomputes for selected rows; n < 5 remains flagged as unstable.",
                            className="chart-description",
                        ),
                        html.Div(
                            _payload_slice_table(row_payload, []),
                            id={"type": "benchmark-slice-container", "index": scope},
                        ),
                    ],
                    className="selected-slice-card",
                )
            row_explorer = html.Div(
                [
                    dcc.Store(id={"type": "benchmark-row-store", "index": scope}, data=row_payload),
                    dcc.Store(id={"type": "benchmark-inspector-store", "index": scope}, data=inspector_payload),
                    dcc.Store(id={"type": "benchmark-inspector-a11y", "index": scope}, data={"open": False}),
                    dcc.Store(id={"type": "benchmark-selection-store", "index": scope}, data=[]),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Linked row explorer", className="row-explorer-eyebrow"),
                                    html.H3("Explore held-out observations", className="row-explorer-title"),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        _selection_count(row_payload, []),
                                        id={"type": "benchmark-selection-count", "index": scope},
                                    ),
                                    html.Button(
                                        "Clear selection",
                                        id={"type": "benchmark-clear-selection", "index": scope},
                                        n_clicks=0,
                                        disabled=True,
                                        className="selection-clear-button",
                                    ),
                                ],
                                className="benchmark-selection-toolbar",
                            ),
                        ],
                        className="row-explorer-header",
                    ),
                    html.Div(row_charts, className="visualization-grid row-explorer-grid"),
                    slice_card,
                    html.Aside(
                        [
                            html.Div(
                                [
                                    html.Strong("Observation inspector"),
                                    html.Button(
                                        "Close",
                                        id={"type": "benchmark-inspector-close", "index": scope},
                                        n_clicks=0,
                                        className="inspector-close-button",
                                        **{"aria-label": "Close observation inspector"},
                                    ),
                                ],
                                className="inspector-header",
                            ),
                            html.Div(
                                _observation_inspector(inspector_payload, None),
                                id={"type": "benchmark-inspector-body", "index": scope},
                                className="observation-inspector-body",
                            ),
                        ],
                        id={"type": "benchmark-inspector", "index": scope},
                        className="observation-inspector",
                        role="dialog",
                        tabIndex=-1,
                        **{
                            "aria-label": "Held-out observation details",
                            "aria-hidden": "true",
                            "aria-modal": "false",
                        },
                    ),
                ],
                className="row-explorer",
            )

    banner_kind = "info" if leakage_ok else "warning"
    title = "Retrospective benchmark evidence" if leakage_ok else "Retrospective benchmark — audit only"
    overview = html.Div(
        [
            _methodology_fields(
                _run_display_metadata(
                    {
                        "workflow": "prediction" if "prediction_audit" in summary else "internal",
                        "summary_json": summary,
                    }
                )
            ),
            _status(str(report.get("verdict") or "Retrospective benchmark completed."), kind=banner_kind),
            _status(
                f"Insufficient held-out data for a reliable performance estimate: n={test_rows}; the declared minimum is {minimum_test_rows}. Raw metrics remain available as technical diagnostics.",
                kind="warning",
            ) if low_n else None,
            html.Div(
                [
                    _metric("Held-out test rows", test_rows, note=f"Declared minimum: {minimum_test_rows}", tone="warning" if low_n else "neutral"),
                    _metric(
                        "Similarity groups",
                        full_groups,
                        note=f"Across {int(split.get('num_rows') or 0):,} accepted rows; {test_groups if test_groups is not None else 'n/a'} represented in held-out data",
                    ),
                    _metric(f"{metric_name} 95% interval", "Not reliable" if low_n else model_ci_text, note=f"Observed interval {model_ci_text}" if low_n else "Model interval only", tone="warning" if low_n else "neutral"),
                    _metric(
                        f"Model lift in {metric_name}",
                        "Not reliable" if low_n else lift.get("delta"),
                        note=f"Observed {_metric_text(lift.get('delta'))}; model minus best fitted baseline" if low_n else "Model minus best fitted baseline",
                        tone="warning" if low_n else "neutral",
                    ),
                ],
                className="metric-grid benchmark-metrics",
            ),
            _status(frame_note, kind="info") if frame_note else None,
            html.Div(run_charts, className="visualization-grid run-evidence-grid"),
        ],
        className="benchmark-tab-content",
    )
    diagnostics = html.Div(
        [
            _status(
                diagnostic_finding,
                kind="warning" if diagnostic_finding and ("compression" in diagnostic_finding.lower() or "does not positively" in diagnostic_finding.lower()) else "info",
            ) if diagnostic_finding else None,
            _status(
                "Uncertainty semantics were not declared. The uncertainty plot is descriptive and does not establish calibrated error estimates.",
                kind="warning",
            ) if uncertainty_undeclared and uncertainty_col and uncertainty_col in frame.columns else None,
            row_explorer or _status("Row-level diagnostics are unavailable for this run."),
        ],
        className="benchmark-tab-content",
    )
    evidence_content = html.Div(
        [_evidence_classification_panel(report), _claim_gate_panel(report)],
        className="benchmark-tab-content",
    )
    technical = html.Div(
        [
            _threshold_sensitivity_panel(report),
            _similarity_sensitivity_panel(report),
            _provenance_panel(report, summary),
            _card(
                [
                    html.H3("Statistical cautions", className="section-heading"),
                    html.Ul(
                        [
                            html.Li("Bootstrap units are recorded in the report; leakage-cluster resampling is used when at least two clusters are available."),
                            html.Li("The lift interval is a paired model-minus-baseline bootstrap using aligned baseline predictions."),
                            html.Li("External predictions require independently verified training and locked-holdout provenance for claim-grade conclusions."),
                            html.Li("Every supported similarity policy is a proxy, not proof of biological independence."),
                        ]
                    ),
                ],
                className="warning-card",
                style={"marginTop": "14px"},
            ),
        ],
        className="benchmark-tab-content",
    )
    return html.Section(
        [
            html.Div(
                [
                    html.H2(title, style={"margin": "0 0 6px"}),
                    html.P(
                        "These panels use the persisted evaluation rows and fitted baselines from this run. They are not a prospective biological validation or a universal leaderboard.",
                        style={"margin": 0, "color": COLORS["muted"]},
                    ),
                ],
                className="benchmark-heading",
            ),
            dcc.Tabs(
                [
                    dcc.Tab(label="Overview", value="overview", children=overview),
                    dcc.Tab(label="Diagnostics", value="diagnostics", children=diagnostics),
                    dcc.Tab(label="Evidence", value="evidence", children=evidence_content),
                    dcc.Tab(label="Technical details", value="technical", children=technical),
                ],
                id={"type": "benchmark-tabs", "index": scope},
                value="overview",
                persistence=True,
                persistence_type="session",
                className="benchmark-tabs",
            ),
        ],
        className="benchmark-section",
    )


def _result_summary(summary: dict[str, Any], report_name: str, *, surface: str = "analysis") -> html.Div:
    report = _report_payload(summary)
    top = _top_candidates(summary)
    run_id = summary.get("run_id")
    is_simulation = "simulation_report" in summary
    artifact_opts = _artifact_options(summary, report_name)
    verdict_kind = "warning" if "simulation_report" in summary else "info"
    benchmark = _benchmark_section(summary, surface=surface)
    candidate_body = _data_table(_candidate_result_frame(top), page_size=10) if not top.empty else _status("No candidate-prioritization rows were produced for this run.")
    candidate_ok, candidate_label, candidate_reasons = _candidate_eligibility(summary)
    compact_reason_labels = {
        "leakage/independence checks did not pass": "independence checks",
        "model lift did not pass": "model lift",
        "biological and synthesis constraints were not verified": "synthesis constraints",
        "uncertainty semantics were not declared": "uncertainty semantics",
    }
    compact_reasons = []
    for reason in candidate_reasons:
        heldout_match = re.fullmatch(r"only (\d+) held-out rows were available; the declared minimum is (\d+)", reason)
        compact_reasons.append(
            f"held-out n={heldout_match.group(1)} < {heldout_match.group(2)}"
            if heldout_match
            else compact_reason_labels.get(reason, reason)
        )
    candidate_status = html.Div(
        [
            html.Strong("Candidate use: " + ("policy-supported retrospective ordering" if candidate_ok else "exploratory only")),
            html.Span("Blocked by " + " \u00b7 ".join(compact_reasons) if compact_reasons else candidate_label),
        ],
        className=f"decision-guidance {'success' if candidate_ok else 'warning'}",
    ) if not top.empty else None
    if surface == "runs":
        run_metrics = [metric for metric in _summary_metrics(summary) if metric["label"] != "Held-out test rows"]
        return html.Div(
            [
                html.Div(
                    [_metric(**metric) for metric in run_metrics],
                    className="metric-grid metric-grid-three" if len(run_metrics) == 3 else "metric-grid",
                ),
                candidate_status,
                html.Div(
                    [
                        dcc.Link("Open report", href=_run_href("/reports", run_id), className="primary-action-link"),
                        dcc.Link("Candidates", href=_run_href("/candidates", run_id), className="text-action-link") if not top.empty and not is_simulation else None,
                        dcc.Link("Benchmark diagnostics", href=_run_href("/benchmarks", run_id), className="text-action-link") if not is_simulation else None,
                    ],
                    className="run-detail-actions",
                ),
                html.P(
                    "The complete diagnostics, policy checks, provenance, and downloads are available on Reports.",
                    className="chart-description",
                ),
            ],
            className="compact-run-result",
        )
    if surface == "analysis":
        benchmark = html.Details(
            [
                html.Summary("View diagnostics on this page", className="result-details-summary"),
                benchmark,
            ],
            className="result-details",
        )
        candidate_body = html.Details(
            [
                html.Summary(f"Candidate prioritization ({len(top):,} rows)", className="result-details-summary"),
                html.Div(candidate_body, style={"marginTop": "12px"}),
            ],
            className="result-details",
        )
    else:
        candidate_body = html.Div(
            [html.H3("Candidate prioritization", style={"margin": "18px 0 10px"}), candidate_body]
        )
    return html.Div(
        [
            html.Div(
                [
                    html.Div([html.Div("Completed evidence package", className="step-eyebrow"), html.H2("Latest completed run", className="step-title")]),
                    dcc.Link("Open run history", href="/runs", className="text-action-link"),
                ],
                className="results-heading",
            ) if surface == "analysis" else None,
            _status(str(report.get("verdict") or "Run completed."), kind=verdict_kind) if surface == "analysis" else None,
            html.Div(
                [
                    dcc.Link("Open report", href=_run_href("/reports", run_id), className="primary-action-link"),
                    dcc.Link("Candidates", href=_run_href("/candidates", run_id), className="text-action-link") if not top.empty else None,
                    dcc.Link("Benchmark diagnostics", href=_run_href("/benchmarks", run_id), className="text-action-link"),
                ],
                className="run-detail-actions",
            ) if surface == "analysis" else None,
            html.Div([_metric(**metric) for metric in _summary_metrics(summary)], className="metric-grid"),
            candidate_status,
            benchmark,
            candidate_body,
            html.Details(
                [
                    html.Summary("Downloads and local files", className="result-details-summary"),
                    html.Div(
                        [
                            dcc.Dropdown(id="analysis-artifact-select", options=artifact_opts, value=artifact_opts[0]["value"] if artifact_opts else None, clearable=False),
                            html.Button("Download selected artifact", id="analysis-download-button", n_clicks=0, style={**BUTTON_STYLE, "marginTop": "10px"}),
                            html.Details(
                                [html.Summary("Local artifact location", className="details-summary"), html.Code(str(summary.get("artifact_dir") or ""), className="path-value")],
                                className="maintenance-details",
                            ),
                            dcc.Download(id="analysis-download"),
                        ],
                        className="artifact-download-panel",
                    ),
                ],
                className="result-details",
            ),
            _status(str(summary.get("run_store_warning")), kind="warning") if summary.get("run_store_warning") else None,
        ],
        className="analysis-results-section" if surface == "analysis" else None,
        style={"marginTop": "18px"},
    )


def _well_names(plate_size: int) -> list[str]:
    layouts = {
        24: ("ABCD", 6),
        48: ("ABCDEF", 8),
        96: ("ABCDEFGH", 12),
        384: ("ABCDEFGHIJKLMNOP", 24),
    }
    rows, cols = layouts.get(int(plate_size), ("ABCDEFGH", max(1, int(plate_size) // 8)))
    wells: list[str] = []
    for row in rows:
        for col in range(1, cols + 1):
            wells.append(f"{row}{col}")
            if len(wells) >= int(plate_size):
                return wells
    return wells


def _control_anchor_wells(wells: list[str], count: int) -> list[str]:
    """Choose stable, spatially separated control anchors before filling extras."""
    if count <= 0 or not wells:
        return []
    parsed = [(well, re.sub(r"[^A-Za-z]", "", well), int(re.sub(r"[^0-9]", "", well))) for well in wells]
    rows = list(dict.fromkeys(row for _, row, _ in parsed))
    cols = sorted({col for _, _, col in parsed})
    anchors = [
        f"{rows[0]}{cols[0]}",
        f"{rows[0]}{cols[-1]}",
        f"{rows[-1]}{cols[0]}",
        f"{rows[-1]}{cols[-1]}",
    ]
    middle_row = rows[len(rows) // 2]
    middle_col = cols[len(cols) // 2]
    anchors.extend([f"{middle_row}{cols[0]}", f"{middle_row}{cols[-1]}", f"{rows[0]}{middle_col}", f"{rows[-1]}{middle_col}"])
    ordered = [well for well in dict.fromkeys(anchors) if well in wells]
    if len(ordered) < count:
        stride = max(1, len(wells) // max(1, count - len(ordered)))
        ordered.extend(well for well in wells[::stride] if well not in ordered)
    return ordered[:count]


def _plate_role_selection(frame: pd.DataFrame, slots: int) -> list[tuple[str, dict[str, Any]]]:
    if slots <= 0 or frame.empty:
        return []
    eligible = frame[frame.get("passes_basic_filters", True).astype(bool)].copy() if "passes_basic_filters" in frame.columns else frame.copy()
    if eligible.empty:
        eligible = frame.copy()
    # Canonical candidate-prioritization exports already encode objective
    # direction, uncertainty semantics, and diversity. Preserve that order in
    # the unapproved well preview instead of re-ranking raw predictions.
    if "rank" in eligible.columns:
        ranked = eligible.assign(
            __rank=pd.to_numeric(eligible["rank"], errors="coerce")
        ).sort_values(["__rank"], na_position="last")
        return [
            ("prioritized_candidate", row.drop(labels=["__rank"]).to_dict())
            for _, row in ranked.head(slots).iterrows()
        ]
    selected_ids: set[Any] = set()
    selections: list[tuple[str, dict[str, Any]]] = []

    def add_rows(role: str, rows: pd.DataFrame, limit: int) -> None:
        for _, row in rows.iterrows():
            if len(selections) >= slots or limit <= 0:
                break
            key = row.get("design_id", row.get("sequence", len(selections)))
            if key in selected_ids:
                continue
            selected_ids.add(key)
            selections.append((role, row.to_dict()))
            limit -= 1

    exploitation_n = max(1, int(round(slots * 0.60)))
    exploration_n = max(1, int(round(slots * 0.25))) if slots >= 4 else 0
    score_col = "assay_prediction" if "assay_prediction" in eligible.columns else "acquisition_score"
    add_rows("exploit_top_prediction", eligible.sort_values(score_col, ascending=False), exploitation_n)
    uncertainty_col = "assay_uncertainty" if "assay_uncertainty" in eligible.columns else "model_uncertainty"
    if uncertainty_col in eligible.columns:
        add_rows("explore_high_uncertainty", eligible.sort_values(uncertainty_col, ascending=False), exploration_n)
    remaining = eligible[~eligible.apply(lambda row: row.get("design_id", row.get("sequence")) in selected_ids, axis=1)].copy()
    if not remaining.empty:
        sort_cols = [column for column in ["gc_fraction", "rank"] if column in remaining.columns]
        add_rows("diversity_representative", remaining.sort_values(sort_cols or [score_col]), slots - len(selections))
    if len(selections) < slots and "rank" in eligible.columns:
        add_rows("backup_ranked_candidate", eligible.sort_values("rank"), slots - len(selections))
    return selections[:slots]


def _build_plate_plan(frame: pd.DataFrame, *, plate_size: int, control_wells: int, seed: int) -> pd.DataFrame:
    wells = _well_names(int(plate_size))
    rng = random.Random(int(seed))
    control_count = max(0, min(int(control_wells), len(wells)))
    control_well_names = _control_anchor_wells(wells, control_count)
    candidate_wells = [well for well in wells if well not in set(control_well_names)]
    rng.shuffle(candidate_wells)
    selections = _plate_role_selection(frame, len(candidate_wells))
    rows: list[dict[str, Any]] = []
    for idx, (role, candidate) in enumerate(selections):
        well = candidate_wells[idx]
        rows.append(
            {
                "well": well,
                "plate_row": re.sub(r"[^A-Za-z]", "", well),
                "plate_column": re.sub(r"[^0-9]", "", well),
                "role": role,
                "design_id": candidate.get("design_id", ""),
                "rank": candidate.get("rank", ""),
                "sequence": candidate.get("sequence", ""),
                "assay_prediction": candidate.get("assay_prediction", candidate.get("predicted_activity", "")),
                "assay_uncertainty": candidate.get("assay_uncertainty", candidate.get("model_uncertainty", "")),
                "acquisition_score": candidate.get("acquisition_score", ""),
                "mlm_plausibility": candidate.get("mlm_plausibility", ""),
                "gc_fraction": candidate.get("gc_fraction", ""),
                "max_homopolymer": candidate.get("max_homopolymer", ""),
                "approved_for_execution": False,
                "layout_limitations": "controls, replicates, quotas, batch balance, and position effects not enforced",
                "notes": candidate.get("why_this_rank", candidate.get("score_source", "")),
            }
        )
    control_roles = ["positive_control", "negative_control", "blank_control", "process_control"]
    for idx, well in enumerate(control_well_names):
        rows.append(
            {
                "well": well,
                "plate_row": re.sub(r"[^A-Za-z]", "", well),
                "plate_column": re.sub(r"[^0-9]", "", well),
                "role": control_roles[idx % len(control_roles)],
                "design_id": "",
                "rank": "",
                "sequence": "",
                "assay_prediction": "",
                "assay_uncertainty": "",
                "acquisition_score": "",
                "mlm_plausibility": "",
                "gc_fraction": "",
                "max_homopolymer": "",
                "approved_for_execution": False,
                "layout_limitations": "control identity and placement require scientist approval",
                "notes": "Reserved control well. Fill with lab-specific control sequence or reagent.",
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_row_sort"] = out["plate_row"].map(lambda item: "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(str(item)) if str(item) else 999)
        out["_col_sort"] = pd.to_numeric(out["plate_column"], errors="coerce").fillna(999)
        out = out.sort_values(["_row_sort", "_col_sort"]).drop(columns=["_row_sort", "_col_sort"]).reset_index(drop=True)
    return out


def _prepare_plate_source(ranked: pd.DataFrame, summary: dict[str, Any]) -> pd.DataFrame:
    frame = ranked.copy()
    sequence_col = _find_sequence_column(frame, summary)
    if sequence_col and sequence_col != "sequence":
        frame["sequence"] = frame[sequence_col]
    elif "sequence" not in frame.columns:
        frame["sequence"] = ""
    if "design_id" not in frame.columns:
        id_series = pd.Series([""] * len(frame), index=frame.index, dtype="object")
        for column in ["sequence_id", "construct_id", "variant_id", "id", "sequence_hash"]:
            if column in frame.columns:
                id_series = id_series.mask(id_series.eq(""), frame[column].map(_as_clean_string))
        if "rank" in frame.columns:
            id_series = id_series.mask(id_series.eq(""), frame["rank"].map(lambda value: f"rank_{value}" if str(value) else ""))
        frame["design_id"] = id_series
    prediction_col = _prediction_column(frame)
    uncertainty_col = _uncertainty_column(frame)
    if prediction_col and "assay_prediction" not in frame.columns:
        frame["assay_prediction"] = frame[prediction_col]
    if uncertainty_col and "assay_uncertainty" not in frame.columns:
        frame["assay_uncertainty"] = frame[uncertainty_col]
    if "passes_basic_filters" not in frame.columns:
        frame["passes_basic_filters"] = True
    if "gc_fraction" not in frame.columns:
        frame["gc_fraction"] = frame["sequence"].map(_gc_fraction)
    if "max_homopolymer" not in frame.columns:
        frame["max_homopolymer"] = frame["sequence"].map(_max_homopolymer)
    return frame


def _client_design_report_markdown(summary: dict[str, Any], plate_plan: pd.DataFrame) -> str:
    report = _report_payload(summary)
    best = report.get("best_simple_baseline") or {}
    lines = [
        f"# Candidate Prioritization & Draft Well Layout: {summary.get('project')}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or "Run completed."),
        "",
        "## Key Metrics",
        "",
        f"- Valid assay rows: {summary.get('audit', {}).get('accepted_rows')}",
        f"- Model metric: {report.get('task_head_primary_metric') or report.get('primary_metric')}",
        f"- Best simple baseline: {best.get('primary_metric')}",
        f"- Candidate-prioritization rows: {summary.get('ranked_candidates')}",
        f"- Draft layout wells: {len(plate_plan)}",
        "- Approved for execution: false",
        "",
        "## Next Action",
        "",
        "Review the prioritization with a scientist. The draft well layout does not enforce controls, replicates, family quotas, batch balancing, range coverage, or position-effect policy.",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(f"- {artifact}" for artifact in summary.get("artifacts", []))
    lines.append("")
    return "\n".join(lines)


def _refresh_execution_artifact_inventory(summary: dict[str, Any]) -> None:
    """Refresh artifact hashes without replacing the core execution identity."""

    artifact_dir = Path(str(summary["artifact_dir"])).expanduser().resolve()
    manifest_path = artifact_dir / "execution_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Execution manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Execution manifest is unreadable: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Execution manifest must be a JSON object: {manifest_path}")

    summary_manifest = summary.get("execution_manifest") or {}
    disk_execution_id = str(manifest.get("execution_id") or "")
    summary_execution_id = str(summary_manifest.get("execution_id") or "")
    if not disk_execution_id or disk_execution_id != summary_execution_id:
        raise ValueError("Execution manifest identity does not match the completed run summary.")

    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw_name in summary.get("artifacts") or []:
        relative_path = Path(str(raw_name))
        if relative_path.is_absolute():
            raise ValueError(f"Artifact names must be relative to the run directory: {raw_name}")
        path = (artifact_dir / relative_path).resolve()
        try:
            path.relative_to(artifact_dir)
        except ValueError as exc:
            raise ValueError(f"Artifact resolves outside the run directory: {raw_name}") from exc
        if path == manifest_path or path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Declared run artifact is missing: {path}")
        seen.add(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )

    manifest["artifacts"] = records
    manifest["artifact_inventory_updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    summary["execution_manifest"] = manifest
    _write_json_file(manifest_path, manifest)


def _add_client_plate_plan_artifacts(summary: dict[str, Any], *, plate_size: int, control_wells: int, plate_seed: int) -> dict[str, Any]:
    artifact_dir = Path(str(summary["artifact_dir"]))
    ranked_path = _candidate_prioritization_path(artifact_dir)
    if ranked_path is None:
        return summary
    ranked = pd.read_csv(ranked_path)
    plate_source = _prepare_plate_source(ranked, summary)
    plate_plan = _build_plate_plan(plate_source, plate_size=plate_size, control_wells=control_wells, seed=plate_seed)
    plate_csv = plate_plan.to_csv(index=False)
    _write_text_file(artifact_dir / "draft_well_layout.csv", plate_csv)
    _write_text_file(artifact_dir / "plate_plan.csv", plate_csv)  # Legacy compatibility export.
    artifacts = list(summary.get("artifacts") or [])
    for artifact in ["simulation_report.md", "draft_well_layout.csv", "plate_plan.csv"]:
        if artifact not in artifacts:
            artifacts.append(artifact)
    summary["artifacts"] = artifacts
    summary.setdefault("params", {})
    summary["params"]["plate_size"] = int(plate_size)
    summary["params"]["control_wells"] = int(control_wells)
    summary["params"]["plate_seed"] = int(plate_seed)
    summary["params"]["approved_for_execution"] = False
    _write_text_file(artifact_dir / "simulation_report.md", _client_design_report_markdown(summary, plate_plan))
    _refresh_execution_artifact_inventory(summary)
    _write_json_file(artifact_dir / "run_summary.json", summary)
    return summary


def _normalize_dna(sequence: str) -> str:
    return "".join(base for base in sequence.upper() if base in {"A", "C", "G", "T"})


def _gc_fraction(sequence: str) -> float:
    seq = _normalize_dna(sequence)
    if not seq:
        return 0.0
    return float((seq.count("G") + seq.count("C")) / len(seq))


def _sequence_input_summary(sequence: str | None, mode: str | None) -> html.Div:
    compact = "".join(str(sequence or "").upper().replace("[MASK]", "N").split())
    invalid = sorted(set(compact) - set("ACGTN"))
    fixed = "".join(base for base in compact if base in "ACGT")
    masked = compact.count("N")
    if not compact:
        status, tone = "Enter a sequence to continue", "warning"
    elif invalid:
        status, tone = "Invalid symbols: " + ", ".join(invalid), "warning"
    elif mode == "Mask-fill" and masked == 0:
        status, tone = "Add at least one N or [MASK] position", "warning"
    else:
        status, tone = f"Ready for {str(mode or 'generation').lower()}", "ready"
    gc_text = f"{_gc_fraction(fixed):.1%}" if fixed else "n/a"
    return html.Div(
        [
            html.Div([html.Span("Length"), html.Strong(f"{len(compact):,} nt")]),
            html.Div([html.Span("Masked"), html.Strong(f"{masked:,}")]),
            html.Div([html.Span("Fixed-base GC"), html.Strong(gc_text)]),
            html.Div([html.Span("Input check"), html.Strong(status, className=f"sequence-check-{tone}")]),
        ],
        className="sequence-input-summary",
        role="status",
        **{"aria-live": "polite"},
    )


def _max_homopolymer(sequence: str) -> int:
    seq = _normalize_dna(sequence)
    if not seq:
        return 0
    longest = current = 1
    for idx in range(1, len(seq)):
        if seq[idx] == seq[idx - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def _fill_masked_sequence(sequence: str, rng: random.Random, mode: str, mutation_rate: float) -> str:
    raw = sequence.upper().replace("[MASK]", "N")
    bases = "ACGT"
    out: list[str] = []
    for base in raw:
        if base == "N":
            out.append(rng.choice(bases))
        elif base in bases:
            if mode != "Mask-fill" and rng.random() < mutation_rate:
                out.append(rng.choice([candidate for candidate in bases if candidate != base]))
            else:
                out.append(base)
    return "".join(out)


def _normalized_hamming(left: str, right: str) -> float:
    if len(left) != len(right) or not left:
        return 1.0
    return sum(a != b for a, b in zip(left, right, strict=True)) / len(left)


def _select_diverse_sequences(pool: list[str], *, parent: str, count: int) -> list[str]:
    """Deterministic farthest-first selection over normalized Hamming distance."""
    remaining = list(dict.fromkeys(pool))
    if len(remaining) <= count:
        return remaining
    selected: list[str] = []
    first = max(remaining, key=lambda sequence: (_normalized_hamming(sequence, parent), sequence))
    selected.append(first)
    remaining.remove(first)
    while remaining and len(selected) < count:
        next_sequence = max(
            remaining,
            key=lambda sequence: (
                min(_normalized_hamming(sequence, chosen) for chosen in selected),
                _normalized_hamming(sequence, parent),
                sequence,
            ),
        )
        selected.append(next_sequence)
        remaining.remove(next_sequence)
    return selected


def _acquisition_utility(
    predictions: Any,
    uncertainties: Any,
    *,
    exploration_weight: float,
    objective_direction: str,
) -> pd.Series:
    """Return a single descending-sort utility for maximize and minimize heads."""
    direction = str(objective_direction or "").strip().lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError("objective_direction must be 'maximize' or 'minimize'.")
    prediction_values = pd.to_numeric(pd.Series(predictions), errors="coerce")
    uncertainty_values = pd.to_numeric(pd.Series(uncertainties), errors="coerce")
    if len(prediction_values) != len(uncertainty_values):
        raise ValueError("predictions and uncertainties must have the same length.")
    if prediction_values.isna().any() or uncertainty_values.isna().any():
        raise ValueError("predictions and uncertainties must be finite numeric values.")
    objective_sign = -1.0 if direction == "minimize" else 1.0
    return objective_sign * prediction_values + float(exploration_weight) * uncertainty_values


def _parse_sequence_motifs(value: str | None) -> list[str]:
    motifs = [item for item in re.split(r"[\s,;]+", str(value or "").upper()) if item]
    invalid = [motif for motif in motifs if set(motif) - set("ACGT")]
    if invalid:
        raise ValueError("Forbidden motifs must contain only A, C, G, and T: " + ", ".join(invalid))
    return list(dict.fromkeys(motifs))


def _reverse_complement(sequence: str) -> str:
    return str(sequence).translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _heuristic_sequence_scores(
    sequence: str,
    target_gc: tuple[float, float],
    max_homopolymer: int,
    rng: random.Random,
    *,
    forbidden_motifs: list[str] | tuple[str, ...] | None = None,
    check_reverse_complements: bool = True,
) -> dict[str, Any]:
    gc = _gc_fraction(sequence)
    homopolymer = _max_homopolymer(sequence)
    center = (target_gc[0] + target_gc[1]) / 2.0
    gc_penalty = min(1.0, abs(gc - center) / max(0.01, target_gc[1] - target_gc[0]))
    homopolymer_penalty = max(0.0, homopolymer - max_homopolymer) / max(1.0, max_homopolymer)
    novelty = rng.uniform(0.05, 0.45)
    plausibility = max(0.0, min(1.0, 0.88 - 0.38 * gc_penalty - 0.30 * homopolymer_penalty + rng.uniform(-0.06, 0.06)))
    uncertainty = max(0.02, min(0.45, 0.06 + novelty * 0.45 + rng.uniform(0.0, 0.05)))
    predicted_activity = max(0.0, 8.0 + 5.0 * plausibility + rng.gauss(0.0, 0.35))
    acquisition = predicted_activity + uncertainty
    failures = []
    if gc < target_gc[0]:
        failures.append("GC below minimum")
    if gc > target_gc[1]:
        failures.append("GC above maximum")
    if homopolymer > max_homopolymer:
        failures.append("Homopolymer exceeds limit")
    motif_hits: list[str] = []
    for motif in forbidden_motifs or []:
        normalized = str(motif).upper()
        if normalized and normalized in sequence:
            motif_hits.append(normalized)
        reverse = _reverse_complement(normalized)
        if check_reverse_complements and reverse != normalized and reverse in sequence:
            motif_hits.append(f"{normalized} reverse complement")
    motif_hits = list(dict.fromkeys(motif_hits))
    if motif_hits:
        failures.append("Forbidden motif: " + ", ".join(motif_hits))
    return {
        "gc_fraction": gc,
        "max_homopolymer": homopolymer,
        "mlm_plausibility": plausibility,
        "model_uncertainty": uncertainty,
        "predicted_activity": predicted_activity,
        "acquisition_score": acquisition,
        "novelty_proxy": novelty,
        "passes_basic_filters": not failures,
        "filter_result": "Eligible under sequence constraints" if not failures else "; ".join(failures),
        "forbidden_motif_hits": ", ".join(motif_hits),
        "scoring_backend": "demo_heuristic_scorer",
    }


def _simulation_preview_frame(
    frame: pd.DataFrame,
    *,
    gc_low: float,
    gc_high: float,
    max_homopolymer: int,
) -> pd.DataFrame:
    """Re-evaluate sequence constraints for an explicitly non-persistent UI view."""
    preview = frame.copy()
    gc_values = pd.to_numeric(
        preview.get("gc_fraction", pd.Series([float("nan")] * len(preview), index=preview.index)),
        errors="coerce",
    )
    homopolymer_values = pd.to_numeric(
        preview.get("max_homopolymer", pd.Series([float("nan")] * len(preview), index=preview.index)),
        errors="coerce",
    )
    failures: list[str] = []
    passes: list[bool] = []
    for gc_value, homopolymer_value in zip(gc_values, homopolymer_values, strict=True):
        row_failures: list[str] = []
        if pd.isna(gc_value):
            row_failures.append("GC unavailable")
        elif float(gc_value) < float(gc_low):
            row_failures.append("GC below minimum")
        elif float(gc_value) > float(gc_high):
            row_failures.append("GC above maximum")
        if pd.isna(homopolymer_value):
            row_failures.append("Homopolymer unavailable")
        elif int(homopolymer_value) > int(max_homopolymer):
            row_failures.append("Homopolymer exceeds limit")
        passes.append(not row_failures)
        failures.append("Eligible under sequence constraints" if not row_failures else "; ".join(row_failures))
    preview["passes_preview_filters"] = passes
    preview["preview_filter_result"] = failures
    return preview


def _simulation_candidate_id(event: dict[str, Any] | None) -> str | None:
    """Extract one candidate ID from a Plotly point or AG Grid cell event."""
    if not isinstance(event, dict):
        return None
    row = event.get("data")
    if isinstance(row, dict) and str(row.get("design_id") or "").strip():
        return str(row["design_id"])
    points = event.get("points") or []
    if not points or not isinstance(points[0], dict):
        return None
    customdata = points[0].get("customdata")
    if isinstance(customdata, (list, tuple)) and customdata:
        candidate_id = str(customdata[0]).strip()
        return candidate_id or None
    return None


def _sequence_changes(parent: str, candidate: str, *, limit: int = 40) -> tuple[list[str], int]:
    normalized_parent = "".join(str(parent or "").upper().replace("[MASK]", "N").split())
    normalized_candidate = "".join(str(candidate or "").upper().split())
    changes = [
        f"{index + 1}: {left}\u2192{right}"
        for index, (left, right) in enumerate(zip(normalized_parent, normalized_candidate, strict=False))
        if left != right
    ]
    if len(normalized_parent) != len(normalized_candidate):
        changes.append(f"Length: {len(normalized_parent)}\u2192{len(normalized_candidate)} nt")
    return changes[:limit], len(changes)


def _simulation_candidate_inspector(
    payload: dict[str, Any] | None,
    candidate_id: str | None,
    *,
    gc_low: float | None = None,
    gc_high: float | None = None,
    max_homopolymer: int | None = None,
) -> html.Div:
    payload = payload or {}
    rows = [row for row in payload.get("rows") or [] if isinstance(row, dict)]
    candidate = next((row for row in rows if str(row.get("design_id")) == str(candidate_id)), None)
    if candidate is None:
        return html.Div(
            [
                html.Div("Inspect a candidate", className="inspector-empty-title"),
                html.P("Click a point or a candidate-table row to inspect its sequence, score, constraints, and saved plate location."),
            ],
            className="inspector-empty",
        )

    preview_frame = _simulation_preview_frame(
        pd.DataFrame([candidate]),
        gc_low=float(gc_low if gc_low is not None else payload.get("gc_low", 0.0)),
        gc_high=float(gc_high if gc_high is not None else payload.get("gc_high", 1.0)),
        max_homopolymer=int(max_homopolymer if max_homopolymer is not None else payload.get("max_homopolymer", 20)),
    )
    preview = preview_frame.iloc[0].to_dict()
    parent = str(payload.get("parent_sequence") or "")
    sequence = str(candidate.get("sequence") or "")
    changes, total_changes = _sequence_changes(parent, sequence)
    plate_row = next(
        (
            row
            for row in payload.get("plate_rows") or []
            if isinstance(row, dict) and str(row.get("design_id") or "") == str(candidate_id)
        ),
        None,
    )
    prediction = candidate.get("predicted_activity", candidate.get("assay_prediction"))
    uncertainty = candidate.get("model_uncertainty", candidate.get("assay_uncertainty"))
    metrics = [
        _inspector_metric("Rank", candidate.get("rank")),
        _inspector_metric("Prediction", prediction),
        _inspector_metric("Uncertainty", uncertainty),
        _inspector_metric("Acquisition", candidate.get("acquisition_score")),
        _inspector_metric("GC fraction", candidate.get("gc_fraction")),
        _inspector_metric("Longest run", candidate.get("max_homopolymer")),
    ]
    preview_passes = bool(preview.get("passes_preview_filters"))
    change_items = [html.Li(change) for change in changes]
    if total_changes > len(changes):
        change_items.append(html.Li(f"{total_changes - len(changes)} additional changes are available in the candidate CSV."))
    provenance = payload.get("scorer_provenance") or {}
    scorer_caption = provenance.get("caption") or candidate.get("scoring_backend") or payload.get("model_name")
    return html.Div(
        [
            html.Div("Design Sandbox candidate", className="inspector-eyebrow"),
            html.H3(str(candidate_id), className="inspector-title"),
            html.Div(metrics, className="inspector-metric-grid"),
            html.Div(
                [
                    _status_badge("✓ Eligible" if preview_passes else "× Excluded", tone="success" if preview_passes else "warning"),
                    html.Span(str(preview.get("preview_filter_result"))),
                ],
                className="candidate-preview-verdict",
            ),
            html.Div(
                [
                    html.Div("Candidate sequence", className="inspector-section-label"),
                    html.Code(sequence, className="inspector-sequence"),
                ],
                className="inspector-section",
            ),
            html.Div(
                [
                    html.Div(f"Changes from parent ({total_changes})", className="inspector-section-label"),
                    html.Ul(change_items, className="candidate-change-list") if change_items else html.P("No sequence changes."),
                ],
                className="inspector-section",
            ),
            html.Div(
                [
                    html.Div("Saved planning context", className="inspector-section-label"),
                    html.Dl(
                        [
                            html.Div([html.Dt("Draft well"), html.Dd(str(plate_row.get("well")) if plate_row else "Not plated")], className="inspector-metadata-item"),
                            html.Div([html.Dt("Ranking objective"), html.Dd(str(candidate.get("objective_direction") or payload.get("objective_direction") or "maximize"))], className="inspector-metadata-item"),
                            html.Div([html.Dt("Scorer"), html.Dd(_human_label(scorer_caption, fallback="Not recorded"))], className="inspector-metadata-item"),
                            html.Div([html.Dt("Run-time filter"), html.Dd(str(candidate.get("filter_result") or "Not recorded"))], className="inspector-metadata-item"),
                        ],
                        className="inspector-metadata",
                    ),
                    html.P("Preview limits only change this screen. Saved rankings, artifacts, and plate assignments remain unchanged.", className="chart-description"),
                ],
                className="inspector-section",
            ),
        ],
        className="inspector-content",
    )


def _sequence_hash(sequence: Any) -> str:
    import hashlib

    return hashlib.sha256(str(sequence or "").upper().encode("utf-8")).hexdigest()


def _public_simulation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return _safe_frame(frame.drop(columns=[column for column in ["embedding"] if column in frame.columns]))


def _candidate_explanations_from_simulation(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, row in frame.iterrows():
        risk_text = str(row.get("risk_flags", "") or "")
        risk_flags = [item.strip() for item in risk_text.split(",") if item.strip()]
        rows.append(
            {
                "rank": int(row.get("rank", len(rows) + 1)),
                "display_id": str(row.get("design_id") or ""),
                "sequence_hash": _sequence_hash(row.get("sequence")),
                "sequence": str(row.get("sequence") or ""),
                "prediction": row.get("assay_prediction", row.get("predicted_activity")),
                "uncertainty": row.get("assay_uncertainty", row.get("model_uncertainty")),
                "acquisition_score": row.get("acquisition_score"),
                "diversified_acquisition_score": row.get("assay_acquisition_score", row.get("acquisition_score")),
                "diversity_cluster": row.get("diversity_cluster", ""),
                "training_distribution_status": row.get("scoring_backend", ""),
                "nearest_train_id": "",
                "nearest_train_similarity": "",
                "nearest_train_edit_distance": "",
                "risk_flags": risk_flags,
                "ranking_reason": row.get("why_this_rank", row.get("score_source", "")),
            }
        )
    return rows


def _simulation_report_markdown(summary: dict[str, Any], top_frame: pd.DataFrame, plate_plan: pd.DataFrame) -> str:
    report = summary["simulation_report"]
    provenance = report.get("scorer_provenance") or {}
    lines = [
        f"# Simulation Report: {summary['project']}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Model Evidence",
        "",
        f"- Scoring backend: {report.get('scoring_backend')}",
        f"- Primary metric: {report.get('primary_metric_name')}: {report.get('primary_metric')}",
        f"- Candidate-prioritization rows: {summary.get('ranked_candidates')}",
        f"- Draft layout wells: {len(plate_plan)}",
        "- Approved for execution: false",
        "",
        "## Scorer Provenance",
        "",
    ]
    if provenance:
        lines.extend(
            [
                f"- Model: {provenance.get('model_name')}",
                f"- Repository: {provenance.get('model_repository')}",
                f"- Revision: {provenance.get('model_revision')}",
                f"- Model weights SHA-256: {provenance.get('model_weights_sha256')}",
                f"- Assay head SHA-256: {provenance.get('head_sha256')}",
                f"- Head type: {provenance.get('head_type')}",
                f"- Task/target: {provenance.get('task_type')} / {provenance.get('target_col')}",
                f"- Ensemble members: {provenance.get('ensemble_size')}",
                f"- Head training rows: {provenance.get('training_rows')}",
                f"- Head training data SHA-256: {provenance.get('training_sha256')}",
                f"- Execution device: {provenance.get('device')}",
                "",
            ]
        )
    else:
        lines.extend(["- No trained model was used; this run used heuristic scoring.", ""])
    warnings = [str(item) for item in report.get("warnings") or [] if str(item).strip()]
    if warnings:
        lines.extend(["## Warnings", "", *[f"- {item}" for item in warnings], ""])
    lines.extend([
        "## Candidate Prioritization",
        "",
        "```text",
        top_frame.head(10).to_string(index=False),
        "```",
        "",
    ])
    if not plate_plan.empty:
        lines.extend([
            "## Draft Well Layout",
            "",
            "`draft_well_layout.csv` is an unapproved planning preview. It does not enforce replicates, family quotas, batch balance, range coverage, cost, or plate-position effects; reserved controls must be specified by a scientist.",
            "",
        ])
    lines.extend(["## Artifacts", ""])
    lines.extend(f"- {artifact}" for artifact in summary.get("artifacts", []))
    lines.append("")
    return "\n".join(lines)


def _save_simulation_artifacts(
    *,
    project: str,
    frame: pd.DataFrame,
    plate_plan: pd.DataFrame,
    params: dict[str, Any],
    scoring_caption: str,
    figures: dict[str, Any] | None = None,
    scorer_provenance: dict[str, Any] | None = None,
    generation_warnings: list[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_dir = _output_root() / "simulations" / f"{_slug(project)}_{timestamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    public_frame = _public_simulation_frame(frame)
    explanations = _candidate_explanations_from_simulation(public_frame)
    public_frame.to_csv(artifact_dir / "candidate_prioritization.csv", index=False)
    public_frame.to_csv(artifact_dir / "ranked_candidates.csv", index=False)  # Legacy compatibility export.
    public_frame.to_csv(artifact_dir / "simulated_candidates.csv", index=False)
    pd.DataFrame(explanations).to_csv(artifact_dir / "candidate_explanations.csv", index=False)
    _write_json_file(artifact_dir / "candidate_explanations.json", {"candidates": explanations})
    artifacts = [
        "simulation_report.md",
        "simulation_summary.json",
        "candidate_prioritization.csv",
        "ranked_candidates.csv",
        "simulated_candidates.csv",
        "candidate_explanations.csv",
        "candidate_explanations.json",
    ]
    if not plate_plan.empty:
        plate_plan.to_csv(artifact_dir / "draft_well_layout.csv", index=False)
        plate_plan.to_csv(artifact_dir / "plate_plan.csv", index=False)  # Legacy compatibility export.
        artifacts.extend(["draft_well_layout.csv", "plate_plan.csv"])

    selected_figures = figures or {}
    if selected_figures:
        visual_parts: list[str] = []
        for index, (key, figure) in enumerate(selected_figures.items()):
            filename = f"visual_{_slug(key)}.plotly.json"
            pio.write_json(figure, artifact_dir / filename, pretty=True, remove_uids=True)
            artifacts.append(filename)
            visual_parts.append(
                pio.to_html(
                    figure,
                    full_html=False,
                    include_plotlyjs=True if index == 0 else False,
                    config=GRAPH_CONFIG,
                )
            )
        visual_report = (
            "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>AssayReady visual report</title><style>body{font-family:Inter,system-ui,sans-serif;margin:24px;"
            "color:#e8eef5;background:#0d1117}.visual{max-width:1200px;margin:0 auto 20px;background:#141920;border:1px solid #303a46;"
            "border-radius:8px;padding:10px}</style></head><body><main>"
            + "".join(f"<section class='visual'>{part}</section>" for part in visual_parts)
            + "</main></body></html>"
        )
        (artifact_dir / "visual_report.html").write_text(visual_report, encoding="utf-8")
        artifacts.append("visual_report.html")
    summary = {
        "project": project,
        "artifact_dir": str(artifact_dir.resolve()),
        "ranked_candidates": int(len(public_frame)),
        "artifacts": artifacts,
        "params": {**params, "approved_for_execution": False},
        "simulation_report": {
            "verdict": (
                "Model-backed scoring completed. These remain planning candidates and require wet-lab validation."
                if scorer_provenance
                else "Heuristic scoring completed. Treat these as draft planning candidates until model and wet-lab validation."
            ),
            "primary_metric_name": "model_acquisition_score" if scorer_provenance else "heuristic_acquisition_score",
            "primary_metric": float(public_frame["acquisition_score"].max()) if "acquisition_score" in public_frame and not public_frame.empty else None,
            "best_simple_baseline": {},
            "scoring_backend": scoring_caption,
            "scorer_provenance": scorer_provenance or {},
            "evidence_level": "synthetic_planning_only",
            "warnings": (generation_warnings or []) + (
                ["Predictions come from a verified local model/head pair but are not wet-lab measurements or validation evidence."]
                if scorer_provenance
                else ["Activity and uncertainty are deterministic heuristic proxies, not measured or trained-model outputs."]
            ),
        },
    }
    (artifact_dir / "simulation_report.md").write_text(_simulation_report_markdown(summary, public_frame, plate_plan), encoding="utf-8")
    _write_json_file(artifact_dir / "simulation_summary.json", summary)
    try:
        record_run(summary, workflow="simulation", report_name="simulation_report.md")
    except Exception as exc:
        summary["run_store_warning"] = f"Simulation artifacts were saved, but local run indexing failed: {exc}"
    return artifact_dir, summary


HUMAN_LABELS = {
    "prediction": "Prediction audit",
    "internal": "Local model",
    "simulation": "Synthetic sandbox",
    "canonical_kmer_jaccard": "Canonical k-mer similarity",
    "exact_reverse_complement": "Exact/reverse-complement",
    "edit_distance": "Edit-distance similarity",
    "position_aware_motif": "Position-aware motif",
    "customer_family_labels": "Customer family labels",
    "near_training_distribution": "Near training distribution",
    "sparse_training_neighborhood": "Sparse training neighborhood",
    "outside_training_distribution": "Outside training distribution",
    "self_declared": "Self-declared",
    "provenance_verified": "Provenance recorded",
    "controlled": "Controlled evaluation",
    "descriptive_audit_only": "Descriptive audit",
    "synthetic_planning_only": "Planning analysis",
}

FRIENDLY_PROJECT_NAMES = {
    "gse135464_gpd_verified_holdout": "GSE135464 constitutive promoter benchmark",
    "dnabert2_dream_verified_holdout": "DREAM promoter benchmark - DNABERT-2",
    "dnabert2_dream_validation_holdout": "DREAM promoter validation - DNABERT-2",
    "dnabert2_dream_validation_heldout": "DREAM promoter validation - DNABERT-2",
    "dnabert2_dream_candidate_ranking": "DREAM candidate ranking - DNABERT-2",
    "public_dream_promoter_audit": "Public DREAM promoter audit",
    "assayready_prediction_audit": "Prediction audit",
    "assayready_design_rank": "Local assay model",
    "dash_sequence_simulation": "Design Sandbox draft",
    "design_sandbox": "Design Sandbox draft",
}


def _human_label(value: Any, *, fallback: str = "Not provided") -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "unspecified", "ui_unspecified", "null"}:
        return fallback
    return HUMAN_LABELS.get(text, text.replace("_", " ").strip().title())


def _display_assurance_label(value: Any, *, fallback: str = "Evidence not reported") -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "unspecified", "ui_unspecified", "null"}:
        return fallback
    label = HUMAN_LABELS.get(text, text)
    normalized = label.lower().replace("_", "-")
    if "provenance-verified" in normalized or "provenance verified" in normalized:
        return "Provenance recorded"
    if "provenance-recorded" in normalized or "provenance recorded" in normalized:
        return "Provenance recorded"
    if "self-declared" in normalized or "self declared" in normalized:
        return "Self-declared"
    return label


def _display_evaluation_label(value: Any, *, fallback: str = "Evaluation not reported") -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "unspecified", "ui_unspecified", "null"}:
        return fallback
    normalized = text.lower().replace("_", " ").replace("-", " ")
    if "descriptive" in normalized or "audit only" in normalized:
        return "Descriptive audit"
    if "controlled" in normalized:
        return "Controlled evaluation"
    if "synthetic" in normalized or "planning" in normalized:
        return "Planning analysis"
    return _human_label(text, fallback=fallback) if "_" in text or "-" in text else text


def _friendly_project_name(value: Any) -> str:
    text = str(value or "").strip()
    return FRIENDLY_PROJECT_NAMES.get(text.lower(), _human_label(text, fallback="Untitled run"))


def _friendly_data_identifier(value: Any) -> str:
    text = str(value or "").strip()
    lower = text.lower()
    if not text or lower in {"none", "unspecified", "ui_unspecified", "null"}:
        return "Not provided"
    if "gse135464" in lower:
        return "NCBI GEO GSE135464 - GPD constitutive promoter test sample"
    if "dream" in lower and "promoter" in lower:
        return "Random Promoter DREAM Challenge dataset"
    if lower.endswith(".csv"):
        return Path(text).name
    return _human_label(text)


def _provenance_identifier(value: Any, *, kind: str) -> Any:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "unspecified", "ui_unspecified", "null"}:
        return "Not provided"
    match = re.search(r"(?i)(?:@?sha256:)([0-9a-f]{8,64})", text)
    if not match:
        return _friendly_data_identifier(text) if kind == "data" else _human_label(text)
    digest = match.group(1).lower()
    prefix = text[: match.start()].strip(" -@:_")
    if kind == "model" and "dnabert2" in prefix.lower():
        prefix = "DNABERT-2"
    elif not prefix:
        prefix = "Artifact"
    short = f"{prefix} - {digest[:8]}..."
    return html.Details(
        [
            html.Summary(short, className="fingerprint-summary", title="Expand to inspect the full SHA-256 provenance fingerprint"),
            html.Div(
                [
                    html.Code(f"sha256:{digest}", className="fingerprint-full"),
                    dcc.Clipboard(content=f"sha256:{digest}", title="Copy full SHA-256 fingerprint", className="fingerprint-copy"),
                ],
                className="fingerprint-details",
            ),
        ],
        className="fingerprint-disclosure",
    )


def _readable_reason(value: Any) -> str:
    text = str(value or "").strip()
    replacements = {
        "uncertainty_type is 'none'": "Uncertainty semantics were not declared",
        "split quality warning flagged as DO NOT TRUST": "Split quality did not meet the selected policy",
        "model-training independence is unverified": "Independence from model-training data is unverified",
        "candidate biological/synthesis constraints were not verified": "Biological and synthesis constraints were not verified",
    }
    if text in replacements:
        return replacements[text]
    spearman = re.search(r"uncertainty/error spearman\s+([-+0-9.eE]+)\s*<\s*required\s+([-+0-9.eE]+)", text, re.IGNORECASE)
    if spearman:
        return f"Uncertainty/error Spearman correlation {float(spearman.group(1)):.3f} was below the required {float(spearman.group(2)):.3f}"
    return text.replace("_", " ")


def _run_record_summary(run: dict[str, Any]) -> dict[str, Any]:
    payload = _load_json_cell(run.get("summary_json"), {})
    return payload if isinstance(payload, dict) else {}


def _run_display_metadata(run: dict[str, Any]) -> dict[str, Any]:
    summary = _run_record_summary(run)
    report = _report_payload(summary)
    assurance = report.get("assurance_level") or run.get("assurance_level") or {}
    evidence = report.get("evidence_level") or run.get("evidence_level") or {}
    if isinstance(assurance, str):
        assurance = {"key": assurance, "label": _human_label(assurance)}
    if isinstance(evidence, str):
        evidence = {"key": evidence, "label": _human_label(evidence)}
    manifest = report.get("evaluation_manifest") or summary.get("evaluation_manifest") or {}
    provenance = manifest.get("provenance") or {}
    params = summary.get("params") or {}
    assurance_key = str(assurance.get("key") or "unreported")
    assurance_label = _display_assurance_label(assurance.get("label") or assurance_key, fallback="Not reported")
    evidence_key = str(evidence.get("key") or report.get("evidence_level") or "unreported")
    evidence_label = str(evidence.get("label") or _human_label(evidence_key, fallback="Not reported"))
    assurance_text = f"{assurance_key} {assurance_label}".lower()
    evidence_text = f"{evidence_key} {evidence_label}".lower()
    if "verified" in assurance_text or "provenance-recorded" in assurance_text or "provenance recorded" in assurance_text:
        provenance_label = "Provenance recorded"
    elif "self" in assurance_text:
        provenance_label = "Self-declared"
    elif "synthetic" in evidence_text or str(run.get("workflow") or "").lower() == "simulation":
        provenance_label = "Planning output"
    else:
        provenance_label = "Not reported"
    if "controlled" in evidence_text:
        evaluation_label = "Controlled evaluation"
    elif "descriptive" in evidence_text or "audit" in evidence_text:
        evaluation_label = "Descriptive audit"
    elif "synthetic" in evidence_text or str(run.get("workflow") or "").lower() == "simulation":
        evaluation_label = "Planning analysis"
    else:
        evaluation_label = "Not reported"
    eligible, _eligibility_label, _reasons = _candidate_eligibility(summary)
    recommended_use = (
        "Exploratory only"
        if str(run.get("workflow") or "").lower() == "simulation" or not eligible
        else "Policy-supported ordering"
    )
    return {
        "assurance_key": assurance_key,
        "assurance": assurance_label,
        "evidence_key": evidence_key,
        "evidence": evidence_label,
        "provenance": provenance_label,
        "evaluation": evaluation_label,
        "recommended_use": recommended_use,
        "data": _friendly_data_identifier(provenance.get("data_identifier") or params.get("data_id") or params.get("assay_files")),
        "model": _provenance_identifier(provenance.get("model_identifier"), kind="model"),
        "workflow": _human_label(run.get("workflow")),
        "verified": "verified" in assurance_key.lower() or "verified" in assurance_label.lower(),
    }


def _friendly_timestamp(value: Any) -> str:
    text = str(value or "")
    if not text:
        return "Not recorded"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text[:16].replace("T", " ")
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone()
        zone = parsed.tzname() or "local"
        return f"{parsed:%Y-%m-%d %H:%M} {zone}"
    return f"{parsed:%Y-%m-%d %H:%M}"


def _compact_timestamp(value: Any) -> str:
    """Format run-list timestamps compactly while preserving local time."""
    text = str(value or "")
    if not text:
        return "Not recorded"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text[:16].replace("T", " ")
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone()
    clock = parsed.strftime("%I:%M %p").lstrip("0")
    return f"{parsed:%b} {parsed.day}, {clock}"


def _status_badge(label: str, *, tone: str = "neutral") -> html.Span:
    return html.Span(label, className=f"product-badge {tone}")


def _methodology_fields(metadata: dict[str, Any], *, compact: bool = False) -> html.Dl:
    values = [
        ("Evidence", metadata.get("provenance") or "Not reported"),
        ("Evaluation", metadata.get("evaluation") or "Not reported"),
        ("Recommended use", metadata.get("recommended_use") or "Exploratory only"),
    ]
    return html.Dl(
        [html.Div([html.Dt(label), html.Dd(str(value))]) for label, value in values],
        className=f"methodology-fields{' methodology-fields-compact' if compact else ''}",
    )


def _run_summary_header(summary: dict[str, Any], *, run: dict[str, Any] | None = None) -> html.Div:
    report = _report_payload(summary)
    metadata_run = dict(run or {})
    metadata_run["summary_json"] = summary
    methodology = _run_display_metadata(metadata_run)
    params = summary.get("params") or {}
    manifest = report.get("evaluation_manifest") or summary.get("evaluation_manifest") or {}
    provenance = manifest.get("provenance") or {}
    split = summary.get("split_diagnostics") or {}
    test_rows = (report.get("test_metrics") or {}).get("num_rows") or (split.get("split_sizes") or {}).get("test")
    values = [
        ("Dataset", _friendly_data_identifier(provenance.get("data_identifier") or params.get("data_id") or (run or {}).get("project"))),
        ("Model", _provenance_identifier(provenance.get("model_identifier"), kind="model")),
        ("Held-out test rows", _metric_text(test_rows)),
        ("Updated", _friendly_timestamp((run or {}).get("updated_at"))),
    ]
    return _card(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Run evidence", className="step-eyebrow"),
                            html.H2(_friendly_project_name((run or {}).get("project") or summary.get("project") or "AssayReady run"), className="run-summary-title"),
                            html.Div(str((run or {}).get("project") or summary.get("project") or ""), className="run-internal-id"),
                        ]
                    ),
                    _methodology_fields(methodology, compact=True),
                ],
                className="run-summary-heading",
            ),
            html.Dl(
                [html.Div([html.Dt(label), html.Dd(value)], className="run-summary-fact") for label, value in values],
                className="run-summary-facts",
            ),
        ],
        className="run-summary-card",
    )


def _runs_display_frame(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        metadata = _run_display_metadata(run)
        rows.append(
            {
                "Updated": _compact_timestamp(run.get("updated_at")),
                "Project": _friendly_project_name(run.get("project")),
                "Recommended use": metadata["recommended_use"],
                "Evaluation": metadata["evaluation"],
                "Evidence": metadata["provenance"],
                "Held-out n": run.get("test_rows"),
                "Run ID": run.get("run_id"),
            }
        )
    return pd.DataFrame(rows)


def _candidate_result_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    renamed = frame.rename(
        columns={
            "rank": "Rank",
            "id": "Candidate",
            "sequence": "Sequence",
            "prediction": "Prediction",
            "uncertainty": "Uncertainty",
            "why_this_rank": "Ranking rationale",
        }
    ).copy()
    if "Ranking rationale" in renamed:
        renamed["Ranking rationale"] = renamed["Ranking rationale"].map(_readable_reason)
    visible = ["Rank", "Candidate", "Sequence", "Prediction", "Uncertainty", "Ranking rationale"]
    return renamed[[column for column in visible if column in renamed.columns]]


def _run_label(run: dict[str, Any]) -> str:
    metadata = _run_display_metadata(run)
    return f"{_friendly_timestamp(run.get('updated_at'))} | {_friendly_project_name(run.get('project'))} | {metadata['assurance']} | {metadata['evidence']}"


def _benchmark_run_priority(run: dict[str, Any]) -> tuple[int, int, float, str]:
    metadata = _run_display_metadata(run)
    project = str(run.get("project") or "").lower()
    public_reference = int(metadata["verified"] and any(token in project for token in ["gse135464", "dream"]))
    verified = int(metadata["verified"])
    rows = int(run.get("test_rows") or 0)
    return public_reference, verified, float(rows), str(run.get("updated_at") or "")


def _deduplicate_run_views(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for run in runs:
        metadata = _run_display_metadata(run)
        key = (
            run.get("project"),
            run.get("workflow"),
            run.get("primary_metric"),
            run.get("best_baseline_metric"),
            run.get("test_rows"),
            run.get("ranked_candidates"),
            run.get("accepted_rows"),
            str(metadata.get("assurance")),
            str(metadata.get("evidence")),
            str(metadata.get("data")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(run)
    return unique
def _workflow_progress_items(active_step: int) -> list[Any]:
    steps = [
        "Select objective",
        "Load data",
        "Map columns",
        "Review and run",
    ]
    items = []
    for i, label in enumerate(steps, 1):
        classes = ["progress-item"]
        if i < active_step:
            classes.append("completed")
        elif i == active_step:
            classes.append("active")
        else:
            classes.append("pending")

        attrs = {"aria-current": "step"} if i == active_step else {}
        items.append(
            html.Li(
                [
                    html.Span("✓" if i < active_step else str(i), className="step-number"),
                    html.Span(label, className="step-label"),
                ],
                className=" ".join(classes),
                **attrs
            )
        )
    return items


def _analysis_step_heading(
    number: int,
    title: str,
    *,
    summary_id: str,
    initial_summary: str,
    edit_id: str | None = None,
    edit_label: str = "Edit",
) -> html.Div:
    action = (
        html.Button(edit_label, id=edit_id, n_clicks=0, className="step-edit-action")
        if edit_id
        else None
    )
    return html.Div(
        [
            html.Div(
                [
                    html.Span(str(number), className="analysis-step-number", **{"aria-hidden": "true"}),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2(title, className="step-title"),
                                    html.Span(className="step-state-label", **{"aria-hidden": "true"}),
                                ],
                                className="analysis-step-title-row",
                            ),
                            html.Div(initial_summary, id=summary_id, className="analysis-step-summary"),
                        ],
                        className="analysis-step-copy",
                    ),
                ],
                className="analysis-step-identity",
            ),
            action,
        ],
        className="analysis-step-header",
    )


def _analysis_stage_class(step: int, active_step: int, *extra: str) -> str:
    state = "completed-stage" if step < active_step else "active-stage" if step == active_step else "locked-stage"
    return " ".join(item for item in ["workflow-stage-card", state, *extra] if item)


def _analysis_preview_from_state(assay_state: dict[str, Any] | None) -> Any:
    if not assay_state:
        return None
    filename = str(assay_state.get("filename") or "Uploaded assay table")
    columns = [str(column) for column in assay_state.get("columns") or []]
    rows = int(assay_state.get("rows") or 0)
    preview_rows = assay_state.get("preview_rows") or []
    preview_frame = pd.DataFrame(preview_rows, columns=columns) if preview_rows else pd.DataFrame(columns=columns)
    return html.Div(
        [
            html.Div(
                [
                    html.Div([html.Span("File"), html.Strong(filename)]),
                    html.Div([html.Span("Rows"), html.Strong(f"{rows:,}")]),
                    html.Div([html.Span("Columns"), html.Strong(str(len(columns)))]),
                    html.Div([html.Span("Parsing"), html.Strong("Complete", className="parse-success")]),
                ],
                className="upload-facts",
            ),
            html.Div(
                [html.Span(column, className="detected-column") for column in columns],
                className="detected-columns",
                **{"aria-label": "Detected columns"},
            ),
            html.Div(
                [
                    html.Div("Data preview", className="data-preview-title"),
                    _data_table(preview_frame, page_size=5, height=210, compact=True),
                ],
                className="mapping-preview-table",
            ) if not preview_frame.empty else None,
        ],
        className="mapping-data-preview",
    )


def _analysis_blocker_message(
    *,
    assay_ready: bool,
    columns_ready: bool,
    prediction_ready: bool,
    classification_ready: bool,
    mappings_confirmed: bool,
    analysis_type: str,
) -> html.Div:
    if not assay_ready:
        title, detail, tone = "Upload data to continue", "Your file will be parsed locally and checked before mapping.", "blocked"
    elif not columns_ready:
        title, detail, tone = "Verify required column mappings", "Choose the sequence and measured-outcome columns.", "blocked"
    elif analysis_type == "prediction" and not prediction_ready:
        title, detail, tone = "Select the model prediction column", "Prediction evaluation needs measured and predicted values.", "blocked"
    elif not classification_ready:
        title, detail, tone = "Choose the positive class", "Binary classification needs exactly two labels and an explicit positive class.", "blocked"
    elif not mappings_confirmed:
        title, detail, tone = "Confirm mappings to continue", "Check the suggested roles against the data preview, then confirm them.", "blocked"
    else:
        title, detail, tone = "New analysis ready", "Run the current setup to create a new evidence package.", "ready"
    return html.Div(
        [html.Strong(title), html.Span(detail)],
        className=f"run-path-message {tone}",
    )


def _analysis_help_content(
    assay_state: dict[str, Any] | None,
    analysis_type: str,
    *,
    sequence_col: str | None = None,
    target_col: str | None = None,
    prediction_col: str | None = None,
    task_type: str | None = None,
    positive_label: str | None = None,
) -> list[Any]:
    columns = [str(column) for column in (assay_state or {}).get("columns") or []]
    available = set(columns)
    required = [
        ("Sequence", "DNA or protein sequence"),
        ("Measured outcome", "Experimental target"),
    ]
    if analysis_type == "prediction":
        required.append(("Model prediction", "Existing model output"))

    warnings: list[str] = []
    if not assay_state:
        warnings.append("Upload an assay table to begin validation.")
    else:
        if sequence_col not in available:
            warnings.append("Map the sequence column.")
        if target_col not in available:
            warnings.append("Map the measured-outcome column.")
        if analysis_type == "prediction" and prediction_col not in available:
            warnings.append("Map the model-prediction column.")
        if task_type == "classification":
            labels = _classification_label_options(assay_state, target_col)
            if len(labels) != 2 or str(positive_label or "") not in {item["value"] for item in labels}:
                warnings.append("Choose the positive class for binary classification.")

    evidence = (
        [
            "Held-out model performance",
            "Lift against simple baselines",
            "Leakage and independence checks",
            "Uncertainty usefulness and claim gates",
        ]
        if analysis_type == "prediction"
        else [
            "Leakage-aware train, validation, and test splits",
            "Held-out local-model performance",
            "Baseline comparison and uncertainty",
            "Candidate ranking with evidence limits",
        ]
    )
    upload_summary = (
        html.Div(
            [
                html.Strong(str(assay_state.get("filename") or "Uploaded table")),
                html.Span(f"{int(assay_state.get('rows') or 0):,} rows · {len(columns)} detected columns · Parsed"),
            ],
            className="context-upload-summary loaded",
        )
        if assay_state
        else html.Div(
            [html.Strong("No file uploaded"), html.Span("CSV, TSV, XLSX, JSON, and YAML are supported.")],
            className="context-upload-summary",
        )
    )
    return [
        html.Div(
            [
                html.Div("Required columns", className="context-card-title"),
                html.Div(
                    [
                        html.Div([html.Strong(label), html.Span(description)])
                        for label, description in required
                    ],
                    className="context-required-list",
                ),
            ],
            className="context-card",
        ),
        html.Div(
            [html.Div("Uploaded file", className="context-card-title"), upload_summary],
            className="context-card",
        ),
        html.Div(
            [
                html.Div("Validation", className="context-card-title"),
                html.Ul(
                    [html.Li(item) for item in warnings] if warnings else [html.Li("No blockers detected in the current mappings.", className="validation-ok")],
                    className="context-validation-list",
                ),
            ],
            className=f"context-card {'has-warning' if warnings else 'is-valid'}",
        ),
        html.Div(
            [
                html.Div("Sample schema", className="context-card-title"),
                html.Pre(
                    "sequence,measured_outcome,model_prediction\nACGTACGT,1.42,1.35"
                    if analysis_type == "prediction"
                    else "sequence,measured_outcome\nACGTACGT,1.42",
                    className="sample-schema",
                ),
            ],
            className="context-card",
        ),
        html.Div(
            [
                html.Div("Evidence this analysis produces", className="context-card-title"),
                html.Ul([html.Li(item) for item in evidence], className="context-evidence-list"),
            ],
            className="context-card evidence-context-card",
        ),
    ]


def _analysis_run_rail_state(summary: dict[str, Any]) -> dict[str, Any]:
    report = _report_payload(summary)
    assurance = report.get("assurance_level") or {}
    evidence = report.get("evidence_level") or {}
    if isinstance(assurance, str):
        assurance = {"label": _human_label(assurance)}
    if isinstance(evidence, str):
        evidence = {"label": _human_label(evidence)}
    gates = report.get("claim_gate") or {}
    gate_items = [value for value in gates.values() if isinstance(value, dict) and "ok" in value]
    return {
        "status": "complete",
        "project": str(summary.get("project") or "Analysis"),
        "assurance": str(assurance.get("label") or assurance.get("key") or "Not reported"),
        "evidence": str(evidence.get("label") or evidence.get("key") or "Not reported"),
        "metrics": [
            {"label": str(item.get("label") or "Metric"), "value": _metric_text(item.get("value"))}
            for item in _summary_metrics(summary)[:3]
        ],
        "gates_passed": sum(1 for item in gate_items if bool(item.get("ok"))),
        "gates_total": len(gate_items),
    }


def _analysis_results_help(run_state: dict[str, Any]) -> list[Any]:
    status = str(run_state.get("status") or "")
    if status != "complete":
        return [
            html.Div(
                [
                    html.Div("Analysis needs attention", className="context-card-title"),
                    html.P(str(run_state.get("message") or "The analysis did not complete.")),
                ],
                className="context-card has-warning",
            )
        ]
    gates_total = int(run_state.get("gates_total") or 0)
    gates_passed = int(run_state.get("gates_passed") or 0)
    gate_summary = f"{gates_passed} of {gates_total} passed" if gates_total else "Not reported"
    return [
        html.Div(
            [html.Span("✓", className="evidence-complete-icon"), html.Div([html.Strong("Analysis complete"), html.Span(str(run_state.get("project") or "Evidence package created"))])],
            className="evidence-complete-banner",
        ),
        html.Div(
            [
                html.Div("Evidence classification", className="context-card-title"),
                html.Div([html.Span("Assurance"), html.Strong(str(run_state.get("assurance") or "Not reported"))], className="evidence-result-row"),
                html.Div([html.Span("Evidence"), html.Strong(str(run_state.get("evidence") or "Not reported"))], className="evidence-result-row"),
                html.Div([html.Span("Claim gates"), html.Strong(gate_summary)], className="evidence-result-row"),
            ],
            className="context-card",
        ),
        html.Div(
            [
                html.Div("Key results", className="context-card-title"),
                html.Div(
                    [html.Div([html.Span(item["label"]), html.Strong(item["value"])]) for item in run_state.get("metrics") or []],
                    className="rail-result-metrics",
                ),
            ],
            className="context-card",
        ),
        html.P("Full diagnostics and downloads are available in the analysis canvas.", className="context-results-note"),
    ]


def _workspace_page_help(pathname: str | None) -> list[Any]:
    route = "/analyze" if pathname in {None, "/"} else str(pathname)
    if route == "/simulations":
        return [
            html.Div(
                [
                    html.Div("Design sandbox", className="context-card-title"),
                    html.P("Generate synthetic variants, check exact sequence constraints, and preview planning diagnostics."),
                ],
                className="context-card",
            ),
            html.Div(
                [
                    html.Div("Before generating", className="context-card-title"),
                    html.Ul(
                        [
                            html.Li("Confirm the generation mode and variable positions."),
                            html.Li("Check the live length, mask, and GC summary."),
                            html.Li("Verify a trained scorer before interpreting model-backed scores."),
                        ],
                        className="context-evidence-list",
                    ),
                ],
                className="context-card",
            ),
            html.Div(
                [
                    html.Div("Evidence limit", className="context-card-title"),
                    html.P("Sandbox output is planning material, not validation or benchmark evidence."),
                ],
                className="context-card has-warning",
            ),
        ]
    if route == "/runs":
        return [
            html.Div(
                [
                    html.Div("Run history", className="context-card-title"),
                    html.P("Select a table row to inspect its evidence summary alongside the index."),
                ],
                className="context-card",
            ),
            html.Div(
                [
                    html.Div("Compare carefully", className="context-card-title"),
                    html.Ul(
                        [
                            html.Li("Metric names and values are separated in the table."),
                            html.Li("Compare values only when workflows, metrics, and units match."),
                            html.Li("Open Reports for policy checks, provenance, and downloads."),
                        ],
                        className="context-evidence-list",
                    ),
                ],
                className="context-card",
            ),
        ]
    return [
        html.Div(
            [
                html.Div("Workspace guide", className="context-card-title"),
                html.P("Use the current page for focused review; complete evidence remains available in Reports."),
            ],
            className="context-card",
        )
    ]


def page_analyze() -> html.Div:
    return html.Div(
        [
            _page_header(
                "Analyze",
                "Choose what you want to evaluate, load measured assay data, and review the evidence package.",
            ),
            dcc.Store(id="assay-upload-state"),
            dcc.Store(id="candidate-upload-state"),
            dcc.Store(id="output-dir-authorization"),
            dcc.Store(id="analysis-run-rail-state"),
            html.Ol(
                _workflow_progress_items(2),
                id="analysis-progress",
                className="workflow-progress analysis-progress-overview",
                **{"aria-label": "Analysis workflow"},
            ),
            _card(
                [
                    _analysis_step_heading(
                        1,
                        "Choose objective",
                        summary_id="objective-step-summary",
                        initial_summary="Evaluate model predictions",
                        edit_id="edit-objective-button",
                        edit_label="Change",
                    ),
                    dcc.RadioItems(
                        id="analysis-type",
                        options=[
                            {
                                "label": html.Div(
                                    [
                                        html.Strong("Evaluate model predictions"),
                                        html.Span("Check held-out predictions, baselines, leakage controls, and uncertainty."),
                                        html.Span("Selected", className="selection-label"),
                                    ]
                                ),
                                "value": "prediction",
                            },
                            {
                                "label": html.Div(
                                    [
                                        html.Strong("Train a local assay model"),
                                        html.Span("Fit and evaluate a local sequence model from measured outcomes."),
                                        html.Span("Selected", className="selection-label"),
                                    ]
                                ),
                                "value": "local_model",
                            },
                        ],
                        value="prediction",
                        className="workflow-choice-grid",
                        labelClassName="workflow-choice",
                        inputClassName="workflow-choice-input",
                    ),
                    html.Div(id="analysis-type-guidance", className="workflow-guidance"),
                ],
                id="objective-card",
                className=_analysis_stage_class(1, 2),
            ),
            _card(
                [
                    _analysis_step_heading(
                        2,
                        "Upload assay data",
                        summary_id="data-step-summary",
                        initial_summary="Upload data to continue",
                        edit_id="edit-data-button",
                        edit_label="Replace",
                    ),
                    html.P("Load measured sequences and outcomes from CSV, TSV, XLSX, or an AssayReady configuration.", className="step-description"),
                    dcc.Upload(
                        id="assay-upload",
                        children=html.Div(
                            [
                                html.Span("↑", className="upload-icon", **{"aria-hidden": "true"}),
                                html.Strong("Upload assay data"),
                                html.Span("Drop a file here or choose one from this computer"),
                            ]
                        ),
                        className="primary-upload",
                        multiple=False,
                    ),
                    html.Div(
                        [
                            html.Button("Try with sample data", id="load-prediction-example-button", n_clicks=0, className="sample-data-button"),
                            dcc.Link("Browse public benchmarks", href="/benchmarks", className="text-action-link"),
                        ],
                        className="example-actions",
                    ),
                    html.P(
                        "Sample data is a quick self-declared audit. Public benchmarks include recorded provenance.",
                        className="example-note",
                    ),
                    html.Div(id="assay-preview", style={"marginTop": "14px"}),
                ],
                id="data-card",
                className=_analysis_stage_class(2, 2),
            ),
            _card(
                [
                    _analysis_step_heading(
                        3,
                        "Verify column mappings",
                        summary_id="columns-step-summary",
                        initial_summary="Available after upload",
                        edit_id="edit-columns-button",
                        edit_label="Edit",
                    ),
                    html.P("AssayReady suggests mappings from column names. Review them before running.", className="step-description"),
                    html.Div("Load an assay table to enable column mapping.", id="columns-empty-hint", className="empty-hint"),
                    html.Div(id="mapping-data-preview"),
                    html.Div(
                        [
                            _field("Sequence", dcc.Dropdown(id="sequence-col", options=[]), "DNA or protein sequence used by the model."),
                            _field("Measured outcome", dcc.Dropdown(id="target-col", options=[]), "Experimentally measured target used for evaluation."),
                            html.Div(
                                _field("Model prediction", dcc.Dropdown(id="prediction-col", options=[]), "Existing model output to compare with the measured outcome."),
                                id="prediction-field-wrapper",
                            ),
                            html.Div(
                                _field("Uncertainty (optional)", dcc.Dropdown(id="uncertainty-col", options=[], clearable=True), "A declared error estimate, predictive deviation, variance, or score."),
                                id="uncertainty-field-wrapper",
                            ),
                            _field("Row identifier (optional)", dcc.Dropdown(id="id-col", options=[], clearable=True)),
                            _field(
                                "Outcome type",
                                dcc.Dropdown(
                                    id="task-type",
                                    options=[{"label": item.title(), "value": item} for item in ["regression", "classification", "ranking"]],
                                    value="regression",
                                    clearable=False,
                                ),
                            ),
                            html.Div(
                                _field(
                                    "Positive class",
                                    dcc.Dropdown(
                                        id="positive-label",
                                        options=[],
                                        value=None,
                                        clearable=False,
                                        placeholder="Choose the class treated as positive",
                                    ),
                                    "Required for binary classification; it controls probability, ROC/PR, and classification-metric interpretation.",
                                ),
                                id="positive-label-wrapper",
                                style={"display": "none"},
                            ),
                            _field(
                                "Leakage grouping",
                                dcc.Dropdown(id="group-cols", options=[], multi=True),
                                "Rows sharing any selected value stay together. Avoid low-cardinality descriptive fields unless leave-group-out testing is intentional.",
                            ),
                            _field(
                                "Descriptive metadata",
                                dcc.Dropdown(id="metadata-cols", options=[], multi=True),
                                "Columns used for exploratory subgroup summaries, not split boundaries.",
                            ),
                        ],
                        className="column-mapping-grid",
                    ),
                    html.Div(
                        [
                            html.Span("Check each required role against the preview above."),
                            html.Button(
                                "Confirm mappings",
                                id="confirm-mappings-button",
                                n_clicks=0,
                                className="confirm-mappings-button",
                                disabled=True,
                            ),
                        ],
                        className="mapping-confirm-action",
                    ),
                ],
                id="columns-card",
                className=_analysis_stage_class(3, 2, "columns-card", "muted-card"),
            ),
            _card(
                [
                    _analysis_step_heading(
                        4,
                        "Review and run",
                        summary_id="review-step-summary",
                        initial_summary="Available after column mapping",
                    ),
                    html.Div(
                        [
                            _field("Project name", dcc.Input(id="project-name", value="assayready_prediction_audit", style=INPUT_STYLE)),
                            html.Div(id="review-workflow-summary", className="review-workflow-summary"),
                        ],
                        className="review-grid",
                    ),
                    html.Details(
                        [
                            html.Summary("Optional candidate ranking", className="details-summary"),
                            html.P("Add an unmeasured candidate pool only when you want ranked nominations in the output package.", className="chart-description"),
                            dcc.Upload(id="candidate-upload", children=html.Div(["Select candidate CSV/TSV/XLSX"]), className="secondary-upload"),
                            html.Div(id="candidate-preview", style={"marginTop": "12px"}),
                            _field(
                                "Maximum ranked candidates",
                                dcc.Input(id="top-k", type="number", min=1, max=384, step=1, value=96, style=INPUT_STYLE),
                            ),
                        ],
                        className="settings-details",
                    ),
                    html.Details(
                        [
                            html.Summary("Advanced analysis settings", className="details-summary"),
                            html.P("Adjust split behavior, model training, provenance, and storage only when the defaults do not fit this analysis.", className="chart-description"),
                            html.Section(
                                [
                                    html.H3("Split and ranking", className="advanced-settings-title"),
                                    html.Div(
                                        [
                                            _field("Low-N warning threshold", dcc.Input(id="low-n", type="number", min=0, step=25, value=200, style=INPUT_STYLE)),
                                            _field("Validation fraction", dcc.Input(id="val-fraction", type="number", min=0, max=0.4, step=0.01, value=0.15, style=INPUT_STYLE)),
                                            _field("Test fraction", dcc.Input(id="test-fraction", type="number", min=0, max=0.4, step=0.01, value=0.15, style=INPUT_STYLE)),
                                            _field("Similarity threshold", dcc.Input(id="homology-threshold", type="number", min=0, max=1, step=0.01, value=0.9, style=INPUT_STYLE)),
                                            _field("Similarity k-mer size", dcc.Input(id="homology-k", type="number", min=2, max=12, step=1, value=8, style=INPUT_STYLE)),
                                            _field("Uncertainty weight", dcc.Input(id="beta", type="number", min=0, max=10, step=0.1, value=1.0, style=INPUT_STYLE)),
                                            _field("Diversity penalty", dcc.Input(id="diversity-penalty", type="number", min=0, max=2, step=0.05, value=0.2, style=INPUT_STYLE)),
                                            _field("Random seed", dcc.Input(id="seed", type="number", min=0, step=1, value=13, style=INPUT_STYLE)),
                                        ],
                                        className="settings-grid",
                                    ),
                                ],
                                className="advanced-settings-group",
                            ),
                            html.Div(
                                [
                                    html.H3("Local model", className="advanced-settings-title"),
                                    html.P("These controls apply only when training a local assay model.", className="chart-description"),
                                    html.Div(
                                        [
                                            _field("Ensemble size", dcc.Input(id="ensemble-size", type="number", min=1, max=16, step=1, value=5, style=INPUT_STYLE)),
                                            _field("Epochs", dcc.Input(id="epochs", type="number", min=1, max=500, step=5, value=80, style=INPUT_STYLE)),
                                            _field("Learning rate", dcc.Input(id="learning-rate", type="number", min=0.00001, max=0.1, step=0.0001, value=0.001, style=INPUT_STYLE)),
                                            _field("Generated candidate pool", dcc.Input(id="num-proposals", type="number", min=96, max=5000, step=96, value=1000, style=INPUT_STYLE)),
                                            _field("Draft layout size", dcc.Dropdown(id="plate-size", options=[{"label": str(item), "value": item} for item in [24, 48, 96, 384]], value=96, clearable=False)),
                                            _field("Control placeholders", dcc.Input(id="control-wells", type="number", min=0, step=1, value=8, style=INPUT_STYLE)),
                                            _field("Layout seed", dcc.Input(id="plate-seed", type="number", min=0, step=1, value=13, style=INPUT_STYLE)),
                                        ],
                                        className="settings-grid",
                                    ),
                                ],
                                id="local-model-options",
                                className="advanced-settings-group",
                                style={"display": "none"},
                            ),
                            html.Section(
                                [
                                    html.H3("Evaluation context and provenance", className="advanced-settings-title"),
                                    html.P("Declarations improve interpretation but do not independently verify a holdout.", className="chart-description"),
                                    html.Div(
                                        [
                                            _field("Objective", dcc.Dropdown(id="evaluation-objective", options=[{"label": "Maximize", "value": "maximize"}, {"label": "Minimize", "value": "minimize"}], value="maximize", clearable=False)),
                                            _field("Measurement units", dcc.Input(id="evaluation-units", value="unspecified", style=INPUT_STYLE)),
                                            _field(
                                                "Uncertainty representation",
                                                dcc.Dropdown(
                                                    id="evaluation-uncertainty-type",
                                                    options=[
                                                        {"label": "None / undeclared", "value": "none"},
                                                        {"label": "Predicted absolute error", "value": "predicted_absolute_error"},
                                                        {"label": "Predictive standard deviation", "value": "predictive_standard_deviation"},
                                                        {"label": "Variance", "value": "variance"},
                                                        {"label": "Generic score", "value": "score"},
                                                    ],
                                                    value="none",
                                                    clearable=False,
                                                ),
                                            ),
                                            _field("Model identifier", dcc.Input(id="evaluation-model-id", value="", placeholder="Model name/version", style=INPUT_STYLE)),
                                            _field("Data identifier", dcc.Input(id="evaluation-data-id", value="", placeholder="Dataset/version", style=INPUT_STYLE)),
                                        ],
                                        className="settings-grid",
                                    ),
                                ],
                                className="advanced-settings-group",
                            ),
                            html.Section(
                                [
                                    html.H3("Storage", className="advanced-settings-title"),
                                    html.P("AssayReady creates a unique run folder inside the selected base folder.", className="chart-description"),
                                    html.Div(
                                        [
                                            html.Div(
                                                _field(
                                                    "Output base folder",
                                                    dcc.Input(
                                                        id="output-dir",
                                                        value="",
                                                        placeholder=f"Default: {_output_root()}",
                                                        readOnly=True,
                                                        style=INPUT_STYLE,
                                                    ),
                                                ),
                                                className="folder-path-field",
                                            ),
                                            html.Button("Choose folder...", id="browse-output-dir-button", n_clicks=0, style=SECONDARY_BUTTON_STYLE),
                                            html.Button("Use default", id="reset-output-dir-button", n_clicks=0, style=SECONDARY_BUTTON_STYLE),
                                        ],
                                        className="folder-picker-row",
                                    ),
                                    html.Div(id="output-dir-status", className="inline-action-status"),
                                ],
                                className="advanced-settings-group",
                            ),
                        ],
                        className="settings-details advanced-analysis-settings",
                    ),
                ],
                id="review-card",
                className=_analysis_stage_class(4, 2),
            ),
            html.Div(
                [
                    html.Button("Run analysis", id="run-analysis-button", n_clicks=0, style={**SECONDARY_BUTTON_STYLE, "cursor": "not-allowed", "opacity": 0.62}, disabled=True),
                    html.Div(id="analysis-ready-state", className="run-ready-state"),
                ],
                className="run-bar sticky-run-bar",
            ),
            html.Div(id="analysis-stale-state"),
            dcc.Loading(html.Div(id="analysis-result"), type="circle"),
        ],
        className="analyze-page",
    )


def _runs_content(status: Any | None = None) -> html.Div:
    runs = _deduplicate_run_views(list_runs(limit=500))
    options = [{"label": _run_label(run), "value": str(run["run_id"])} for run in runs]
    provenance_backed = sum(1 for run in runs if _run_display_metadata(run)["verified"])
    workflow_options = sorted({_run_display_metadata(run)["workflow"] for run in runs})
    evaluation_options = sorted({_run_display_metadata(run)["evaluation"] for run in runs})
    use_options = sorted({_run_display_metadata(run)["recommended_use"] for run in runs})
    return html.Div(
        [
            status,
            dcc.Store(id="runs-index-store", data=runs),
            html.Div(
                [
                    _run_index_stat("Runs", len(runs), title="Unique runs in the local index"),
                    _run_index_stat("Provenance-backed", provenance_backed, title="Runs with verified provenance"),
                    _run_index_stat("Projects", len({run.get("project") for run in runs}), title="Distinct projects in the index"),
                ],
                className="run-index-stats",
            ) if runs else None,
            dcc.Dropdown(id="runs-run-select", options=options, value=options[0]["value"] if options else None, clearable=False, style={"display": "none"}) if options else None,
            html.Div(
                [
                    _card(
                        [
                            html.H2("Run history", className="section-heading"),
                            html.P("Select a run for a compact summary, then open Reports for complete diagnostics and downloads.", className="chart-description"),
                            html.Div(
                                [
                                    _field("Search", dcc.Input(id="runs-search", type="text", placeholder="Project or workflow", debounce=True, style=INPUT_STYLE)),
                                    _field("Workflow", dcc.Dropdown(id="runs-workflow-filter", options=[{"label": "All workflows", "value": "all"}] + [{"label": value, "value": value} for value in workflow_options], value="all", clearable=False)),
                                    _field("Evaluation", dcc.Dropdown(id="runs-assurance-filter", options=[{"label": "All evaluations", "value": "all"}] + [{"label": value, "value": value} for value in evaluation_options], value="all", clearable=False)),
                                    _field("Recommended use", dcc.Dropdown(id="runs-evidence-filter", options=[{"label": "All recommended uses", "value": "all"}] + [{"label": value, "value": value} for value in use_options], value="all", clearable=False)),
                                ],
                                className="runs-filter-grid",
                            ),
                            html.Div(
                                _data_table(
                                    _runs_display_frame(runs),
                                    page_size=12,
                                    height=480,
                                    table_id="runs-index-table",
                                    hidden_columns={"Run ID"},
                                    selectable=True,
                                    selected_rows=[_runs_display_frame(runs).iloc[0].to_dict()] if runs else None,
                                ) if runs else _status("No runs are indexed yet. Run an analysis first."),
                                id="runs-table-container",
                            ),
                        ],
                        className="runs-master-card",
                    ),
                    html.Aside(
                        html.Div(
                            id="runs-run-detail",
                            children=_status("Select a run to inspect its evidence summary.") if not runs else None,
                        ),
                        className="runs-evidence-drawer",
                        **{"aria-label": "Selected run evidence"},
                    ),
                ],
                className="runs-master-detail",
            ),
        ]
    )


def page_runs() -> html.Div:
    return html.Div(
        [
            _page_header("Runs", "Local experiment history for audits, model checks, candidate prioritizations, and generated reports."),
            html.Div(
                [
                    html.Button("Refresh runs", id="refresh-runs-button", n_clicks=0, style=SECONDARY_BUTTON_STYLE),
                    html.Span("Times are shown in your local timezone.", className="page-action-note"),
                ],
                className="page-actions runs-page-actions",
            ),
            html.Details(
                [
                    html.Summary("Run-index maintenance", className="details-summary"),
                    html.P(f"Index database: {default_db_path()}", className="chart-description"),
                    html.Button("Index existing artifacts", id="index-runs-button", n_clicks=0, style=SECONDARY_BUTTON_STYLE),
                ],
                className="maintenance-details",
            ),
            html.Div(id="runs-content", children=_runs_content(), style={"marginTop": "16px"}),
        ],
        className="workspace-page",
    )


def _candidate_display_frame(candidates: list[dict[str, Any]], *, include_context: bool = False) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame()
    frame = pd.DataFrame(candidates)
    risk_lists = frame.get("risk_flags_json", pd.Series([[]] * len(frame))).map(lambda value: _load_json_cell(value, []))
    frame["Risk count"] = risk_lists.map(len)
    frame["Primary risk"] = risk_lists.map(
        lambda values: _human_label(values[0]) + (f" +{len(values) - 1} more" if len(values) > 1 else "") if values else "None recorded"
    )
    frame["Sequence"] = frame.get("sequence", pd.Series([""] * len(frame))).map(str)
    frame["Source"] = frame["workflow"].map(_human_label)
    frame["Distribution"] = frame["training_distribution_status"].map(_human_label)
    frame["Project"] = frame["project"].map(_friendly_project_name)
    frame = frame.rename(
        columns={
            "rank": "Rank",
            "display_id": "Candidate",
            "prediction": "Prediction",
            "uncertainty": "Uncertainty",
            "acquisition_score": "Acquisition score",
            "run_id": "Run ID",
        }
    )
    visible = (["Project", "Source"] if include_context else []) + ["Rank", "Candidate", "Sequence", "Prediction", "Uncertainty", "Distribution", "Risk count", "Primary risk", "Run ID"]
    return frame[[column for column in visible if column in frame.columns]]


def _candidate_grid_height(row_count: int) -> int:
    return min(520, max(190, 90 + max(1, int(row_count)) * 39))


def _candidate_run_status(run_id: str | None) -> Any:
    if not run_id or run_id == "all":
        return _status(
            "Candidates are grouped by run. Ranks, predictions, and uncertainty values are not comparable across projects or measurement units.",
            kind="warning",
        )
    summary = get_run_summary(str(run_id)) or {}
    eligible, label, reasons = _candidate_eligibility(summary)
    detail = "; ".join(reasons)
    return _status(label + (f". {detail}." if detail else "."), kind="success" if eligible else "warning")


def _candidate_navigation_snapshot() -> tuple[list[dict[str, Any]], list[dict[str, str]], str]:
    runs = _deduplicate_run_views(list_runs(limit=500))
    allowed_run_ids = {str(run.get("run_id")) for run in runs}
    candidates = [row for row in list_candidates(limit=5000) if str(row.get("run_id")) in allowed_run_ids]
    candidate_keys: set[tuple[Any, ...]] = set()
    candidates = [
        row for row in candidates
        if not (
            (row.get("run_id"), row.get("display_id"), row.get("sequence_hash")) in candidate_keys
            or candidate_keys.add((row.get("run_id"), row.get("display_id"), row.get("sequence_hash")))
        )
    ]
    candidate_run_ids = {str(row.get("run_id")) for row in candidates}
    candidate_runs = [run for run in runs if str(run.get("run_id")) in candidate_run_ids]
    model_runs = [run for run in candidate_runs if str(run.get("workflow")) != "simulation"]
    default_run_id = str((model_runs or candidate_runs)[0].get("run_id")) if (model_runs or candidate_runs) else "all"
    run_options = [
        {"label": f"{_friendly_project_name(run.get('project'))} - {_friendly_timestamp(run.get('updated_at'))}", "value": str(run.get("run_id"))}
        for run in candidate_runs
    ]
    return candidates, run_options, default_run_id


def _report_navigation_snapshot() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    runs = _deduplicate_run_views(list_runs(limit=500))
    return runs, [{"label": _run_label(run), "value": str(run["run_id"])} for run in runs]


def _benchmark_library_label(run: dict[str, Any]) -> html.Div:
    metadata = _run_display_metadata(run)
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Verified public reference" if metadata["verified"] else "Indexed benchmark", className="step-eyebrow"),
                    html.H3(_friendly_project_name(run.get("project")), className="benchmark-card-title"),
                    html.Div(str(run.get("project") or ""), className="benchmark-card-id"),
                ],
                className="benchmark-library-heading",
            ),
            _methodology_fields(metadata, compact=True),
            html.Div(
                f"{int(run.get('test_rows') or 0):,} held-out rows · updated {_compact_timestamp(run.get('updated_at'))}",
                className="benchmark-card-meta",
            ),
        ],
        className="benchmark-library-label",
    )


def _benchmark_navigation_snapshot() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_runs = [run for run in _deduplicate_run_views(list_runs(limit=500)) if str(run.get("workflow")) != "simulation"]
    runs = sorted(all_runs, key=_benchmark_run_priority, reverse=True)
    return runs, [{"label": _benchmark_library_label(run), "value": str(run["run_id"])} for run in runs]


def _preferred_run_value(
    search: str | None,
    options: list[dict[str, Any]],
    current: Any | None,
    fallback: Any | None,
) -> Any | None:
    allowed = {str(option.get("value")) for option in options if option.get("value") is not None}
    requested = _requested_run_id(search)
    if requested in allowed:
        return requested
    normalized_current = str(current) if current is not None else None
    if normalized_current in allowed:
        return normalized_current
    normalized_fallback = str(fallback) if fallback is not None else None
    return normalized_fallback if normalized_fallback in allowed else None


def page_candidates() -> html.Div:
    candidates, run_options, default_run_id = _candidate_navigation_snapshot()
    model_candidates = [row for row in candidates if str(row.get("workflow")) != "simulation"]
    initial_candidates = [row for row in model_candidates if str(row.get("run_id")) == default_run_id]
    body = html.Div(
        [
            _candidate_run_status(default_run_id),
            _data_table(_candidate_display_frame(initial_candidates), page_size=15, height=_candidate_grid_height(len(initial_candidates)), table_id="candidate-index-table", hidden_columns={"Run ID"}, selectable=True),
        ],
        className="candidate-grid-stack",
    ) if initial_candidates else _status("No model or audit candidates are indexed yet.")
    return html.Div(
        [
            _page_header("Candidates", "Review ranked sequences by source, evidence context, distribution status, and risk."),
            _card(
                [
                    html.Div(
                        [
                            _field("Candidate run", dcc.Dropdown(id="candidate-project-filter", options=run_options + [{"label": "All runs (grouped; no global comparison)", "value": "all"}], value=default_run_id, clearable=False)),
                            _field("Source", dcc.Dropdown(id="candidate-source-filter", options=[{"label": "Model and audit candidates", "value": "model"}, {"label": "Synthetic sandbox", "value": "simulation"}, {"label": "All sources", "value": "all"}], value="model", clearable=False)),
                            _field("Risk", dcc.Dropdown(id="candidate-risk-filter", options=[{"label": "All risk states", "value": "all"}, {"label": "No recorded risks", "value": "none"}, {"label": "Has review flags", "value": "flagged"}], value="all", clearable=False)),
                        ],
                        className="candidate-filter-grid",
                    ),
                    dcc.Store(id="candidate-index-store", data=candidates),
                    html.Div(id="candidate-table-container", children=body),
                    html.Div(id="candidate-detail", children=_status("Click any candidate row to inspect its complete sequence, risks, neighborhood, and ranking context."), className="candidate-detail-panel"),
                ]
            ),
        ],
        className="workspace-page",
    )


def page_reports() -> html.Div:
    _runs, options = _report_navigation_snapshot()
    return html.Div(
        [
            _page_header("Reports", "Review a run's executive evidence summary, diagnostics, technical report, and downloadable artifacts."),
            _field("Report run", dcc.Dropdown(id="reports-run-select", options=options, value=options[0]["value"] if options else None, clearable=False)),
            html.Div(id="report-detail"),
            dcc.Download(id="report-download"),
            dcc.Download(id="report-package-download"),
        ],
        className="workspace-page",
    )


def _is_non_audited_sandbox_run(summary: dict[str, Any], run: dict[str, Any]) -> bool:
    """Return whether an indexed run is planning output without audit evidence."""
    has_audit = "prediction_audit" in summary or "benchmark_report" in summary
    is_sandbox = "simulation_report" in summary or str(run.get("workflow") or "").lower() == "simulation"
    return bool(is_sandbox and not has_audit)


def _sandbox_report_empty_state(summary: dict[str, Any], run: dict[str, Any]) -> html.Div:
    run_id = str(run.get("run_id") or summary.get("run_id") or "")
    return html.Div(
        [
            _run_summary_header(summary, run=run),
            _card(
                [
                    html.Div("Sandbox draft", className="step-eyebrow"),
                    html.H2("No audit report is available for this draft", className="report-empty-title"),
                    html.P(
                        "This run contains planning output. It was not evaluated against held-out assay measurements, so it has no fitted-baseline comparison, leakage assessment, or audit policy decision.",
                        className="report-empty-description",
                    ),
                    html.Ul(
                        [
                            html.Li("Return to Design Sandbox to review or regenerate the draft."),
                            html.Li("Use Analyze with measured outcomes and model predictions to create an auditable evidence report."),
                        ],
                        className="report-empty-actions-list",
                    ),
                    html.Div(
                        [
                            dcc.Link("Open Design Sandbox", href="/simulations", className="primary-action-link"),
                            dcc.Link("View draft candidates", href=_run_href("/candidates", run_id), className="text-action-link"),
                            dcc.Link("Start an audit", href="/analyze", className="text-action-link"),
                        ],
                        className="run-detail-actions",
                    ),
                ],
                className="report-empty-state",
                role="status",
            ),
        ]
    )


def page_benchmarks() -> html.Div:
    runs, options = _benchmark_navigation_snapshot()
    return html.Div(
        [
            _page_header(
                "Benchmarks",
                "Inspect real held-out measurements, fitted sequence baselines, leakage-aware splits, uncertainty behavior, and evidence gates.",
            ),
            _card(
                [
                    html.H2("Benchmark library", className="section-heading"),
                    html.P("Select a benchmark to inspect its evidence and diagnostics.", className="chart-description"),
                    dcc.RadioItems(
                        id="benchmark-run-select",
                        options=options,
                        value=options[0]["value"] if options else None,
                        className="benchmark-library-list",
                        labelClassName="benchmark-library-option",
                        inputClassName="benchmark-library-input",
                    ),
                    _status("No benchmark runs are indexed yet. Run an analysis or the public reference benchmark first.") if not runs else None,
                ],
                className="benchmark-library",
            ),
            _card(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2("Bundled DREAM smoke test", style={"fontSize": "19px", "margin": "0 0 6px"}),
                                    html.P(
                                        "Run the bundled 400-row workflow check. It verifies the interface and provenance path, but is not the flagship benchmark.",
                                        style={"margin": 0, "color": COLORS["muted"]},
                                    ),
                                ]
                            ),
                            html.Button("Run smoke test", id="run-public-benchmark-button", n_clicks=0, style=BUTTON_STYLE),
                        ],
                        className="benchmark-launch-row",
                    ),
                    html.Details(
                        [
                            html.Summary("Dataset and interpretation", className="details-summary"),
                            dcc.Markdown(
                                "Source: Random Promoter DREAM Challenge public data (see the packaged provenance note). "
                                "Predictions come from a small local demo baseline; the run still enforces leakage and claim gates."
                            ),
                        ],
                        className="settings-details",
                    ),
                ]
            ),
            dcc.Loading(html.Div(id="public-benchmark-result", style={"marginTop": "16px"}), type="circle"),
            dcc.Loading(html.Div(id="benchmark-detail", style={"marginTop": "16px"}), type="circle"),
        ],
        className="workspace-page",
    )


def page_simulations() -> html.Div:
    scorer_options = _scorer_options()
    return html.Div(
        [
            _page_header("Design Sandbox", "Explore synthetic DNA variants, apply sequence constraints, and build a well-layout preview."),
            _card(
                [
                    html.H2("Sequence generation", className="section-heading"),
                    html.Div(
                        [
                            html.Button(
                                "Generate draft designs",
                                id="run-simulation-button",
                                n_clicks=0,
                                style=BUTTON_STYLE,
                                className="simulation-primary-action",
                            ),
                            html.Div(
                                [
                                    html.Strong("Up to 96 candidates"),
                                    html.Span("Heuristic scoring selected"),
                                ],
                                id="simulation-action-summary",
                                className="simulation-action-summary",
                            ),
                        ],
                        className="simulation-action-bar",
                    ),
                    html.Div(
                        [
                            _field(
                                "Generation mode",
                                dcc.Dropdown(
                                    id="sim-mode",
                                    options=[{"label": item, "value": item} for item in ["Mask-fill", "Random mutagenesis", "Diversity sampling"]],
                                    value="Mask-fill",
                                    clearable=False,
                                ),
                            ),
                            _field(
                                "Scorer",
                                dcc.Dropdown(
                                    id="sim-model",
                                    options=scorer_options,
                                    value=DEMO_SCORER,
                                    clearable=False,
                                ),
                                "Only discovered, structurally complete AssayReady bundles are listed. Verify a trained scorer before generating model-backed output.",
                            ),
                            _field("Candidate count", dcc.Input(id="sim-num-candidates", type="number", min=1, max=1000, step=1, value=96, style=INPUT_STYLE)),
                            _field("Uncertainty exploration weight", dcc.Input(id="sim-context", type="number", min=0, max=8, step=0.5, value=2.0, style=INPUT_STYLE), "Utility aligns the prediction with the declared maximize/minimize objective, then adds this weight × uncertainty. It is a planning preference, not a calibrated biological parameter."),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(0, 1fr))", "gap": "14px"},
                        className="simulation-settings-grid",
                    ),
                    html.Div(id="simulation-mode-guidance", className="workflow-guidance"),
                    _field("Parent or masked sequence", dcc.Textarea(id="sim-parent", value="TATAATNNNNGGTTTT", style={**INPUT_STYLE, "minHeight": "92px"})),
                    html.Div(
                        _sequence_input_summary("TATAATNNNNGGTTTT", "Mask-fill"),
                        id="simulation-sequence-summary",
                    ),
                    html.H3("Sequence constraints", className="sandbox-section-title"),
                    html.Div(
                        [
                            _field("Mutation rate", dcc.Input(id="sim-mutation-rate", type="number", min=0, max=0.5, step=0.01, value=0.08, style=INPUT_STYLE)),
                            _field("Minimum GC fraction", dcc.Input(id="sim-gc-low", type="number", min=0, max=1, step=0.01, value=0.30, style=INPUT_STYLE)),
                            _field("Maximum GC fraction", dcc.Input(id="sim-gc-high", type="number", min=0, max=1, step=0.01, value=0.70, style=INPUT_STYLE)),
                            _field("Max homopolymer", dcc.Input(id="sim-max-homopolymer", type="number", min=1, max=20, step=1, value=6, style=INPUT_STYLE)),
                            _field("Seed", dcc.Input(id="sim-seed", type="number", min=0, step=1, value=13, style=INPUT_STYLE)),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "repeat(5, minmax(0, 1fr))", "gap": "14px"},
                        className="simulation-settings-grid",
                    ),
                    html.Details(
                        [
                            html.Summary("Motif and cloning constraints", className="details-summary"),
                            html.P(
                                "These are exact sequence checks, not biological-effect predictions. Use rules required by your cloning or synthesis workflow.",
                                className="chart-description",
                            ),
                            html.Div(
                                [
                                    _field(
                                        "Rule preset",
                                        dcc.Dropdown(
                                            id="sim-constraint-preset",
                                            options=[
                                                {"label": "Basic GC and homopolymer only", "value": "basic"},
                                                {"label": "Golden Gate: avoid BsaI and BsmBI sites", "value": "golden_gate"},
                                                {"label": "Custom forbidden motifs", "value": "custom"},
                                            ],
                                            value="basic",
                                            clearable=False,
                                        ),
                                    ),
                                    _field(
                                        "Forbidden DNA motifs",
                                        dcc.Input(id="sim-forbidden-motifs", type="text", value="", placeholder="Example: GGTCTC, CGTCTC", style=INPUT_STYLE),
                                        "Separate exact A/C/G/T motifs with commas or spaces.",
                                    ),
                                ],
                                className="motif-constraint-grid",
                            ),
                            dcc.Checklist(
                                id="sim-check-reverse-complements",
                                options=[{"label": "Check each forbidden motif on both DNA strands", "value": "both_strands"}],
                                value=["both_strands"],
                                className="visual-choice-list",
                            ),
                        ],
                        className="settings-details",
                    ),
                    html.Div(id="model-installation-panel", style={"marginTop": "14px"}),
                    _scorer_setup_wizard(),
                    html.Details(
                        [
                            html.Summary("Choose result visuals", className="details-summary"),
                            html.P("The mode selects a relevant recommended set. Keep it compact, show all diagnostics, or adjust individual visuals below.", className="chart-description"),
                            html.Div(
                                [
                                    _field(
                                        "Visual preset",
                                        dcc.Dropdown(
                                            id="sim-visual-preset",
                                            options=[
                                                {"label": "Recommended for this mode", "value": "recommended"},
                                                {"label": "Compact", "value": "compact"},
                                                {"label": "Full diagnostics", "value": "full"},
                                            ],
                                            value="recommended",
                                            clearable=False,
                                        ),
                                    ),
                                    _field(
                                        "Included visuals",
                                        dcc.Checklist(
                                            id="sim-visuals",
                                            options=_simulation_visual_options("Mask-fill"),
                                            value=_simulation_visual_defaults("Mask-fill", "recommended"),
                                            className="visual-choice-list",
                                        ),
                                    ),
                                ],
                                className="visual-settings-grid",
                            ),
                            html.Div(id="simulation-visual-guidance", className="workflow-guidance"),
                        ],
                        className="settings-details",
                    ),
                    html.Details(
                        [
                            html.Summary("Draft plate layout and advanced settings", className="details-summary"),
                            html.Div(
                                [
                                    _field("Project", dcc.Input(id="sim-project", value="design_sandbox", style=INPUT_STYLE)),
                                    _field("Plate size", dcc.Dropdown(id="sim-plate-size", options=[{"label": str(item), "value": item} for item in [24, 96, 384]], value=96, clearable=False)),
                                    _field("Reserved control wells", dcc.Input(id="sim-control-wells", type="number", min=0, step=1, value=4, style=INPUT_STYLE), "Reserves positions only. Control identities require scientist approval."),
                                    _field("Layout seed", dcc.Input(id="sim-plate-seed", type="number", min=0, step=1, value=13, style=INPUT_STYLE)),
                                ],
                                style={"display": "grid", "gridTemplateColumns": "2fr 1fr 1fr 1fr", "gap": "14px"},
                                className="simulation-settings-grid",
                            ),
                        ],
                        className="settings-details",
                    ),
                    html.Div(
                        [
                            html.Progress(className="simulation-indeterminate-progress"),
                            html.Div(
                                [
                                    html.Strong("Building the draft run"),
                                    html.Span(" Generating candidates, checking constraints, scoring, ranking, and saving artifacts. Progress is indeterminate while the scorer is busy."),
                                ]
                            ),
                        ],
                        id="simulation-running-status",
                        className="simulation-running-status",
                        hidden=True,
                        role="status",
                        **{"aria-live": "polite"},
                    ),
                ]
            ),
            dcc.Loading(html.Div(id="simulation-result", style={"marginTop": "18px"}), type="circle"),
        ],
        className="workspace-page",
    )


def page_settings() -> html.Div:
    output_root = _output_root()
    upload_root = _upload_root()
    discovered_scorers = _discover_local_scorers()
    model_runtime_modules = [
        "torch",
        "transformers",
        "tokenizers",
        "einops",
        "huggingface_hub",
        "safetensors",
    ]
    missing_model_runtime = [name for name in model_runtime_modules if importlib.util.find_spec(name) is None]
    retention = os.environ.get(UPLOAD_RETENTION_HOURS_ENV, "24")
    upload_limit = os.environ.get(UPLOAD_LIMIT_ENV, "10")
    return html.Div(
        [
            _page_header("Settings", "Review local storage, runtime configuration, evidence defaults, privacy, and deployment readiness."),
            _card(
                [
                    html.H2("Local storage", className="section-heading"),
                    html.Dl(
                        [
                            html.Div([html.Dt("Storage mode"), html.Dd("Local workstation")], className="settings-fact"),
                            html.Div([html.Dt("Upload retention"), html.Dd(f"{retention} hours")], className="settings-fact"),
                            html.Div([html.Dt("Upload limit"), html.Dd(f"{upload_limit} MB")], className="settings-fact"),
                        ],
                        className="settings-facts-grid",
                    ),
                    html.Details(
                        [
                            html.Summary("Reveal local storage paths", className="details-summary"),
                            *[
                                html.Div(
                                    [
                                        html.Strong(label),
                                        html.Code(str(path), className="path-value"),
                                        dcc.Clipboard(content=str(path), title=f"Copy {label.lower()}", className="path-copy"),
                                    ],
                                    className="revealed-path",
                                )
                                for label, path in [("Output root", output_root), ("Upload cache", upload_root), ("Run index", default_db_path())]
                            ],
                        ],
                        className="maintenance-details",
                    ),
                    html.Button("Clear upload cache", id="clear-upload-cache-button", n_clicks=0, style=SECONDARY_BUTTON_STYLE),
                    html.Div(
                        [
                            html.Span("Delete cached uploads? Saved runs and reports will remain."),
                            html.Button("Confirm clear", id="confirm-clear-upload-cache-button", n_clicks=0, style=BUTTON_STYLE),
                            html.Button("Cancel", id="cancel-clear-upload-cache-button", n_clicks=0, style=SECONDARY_BUTTON_STYLE),
                        ],
                        id="clear-upload-cache-confirmation",
                        className="destructive-confirmation",
                        style={"display": "none"},
                    ),
                    html.Div(id="clear-upload-cache-status", className="inline-action-status"),
                ],
                className="settings-section-card",
            ),
            _card(
                [
                    html.H2("Evaluation defaults (read only)", className="section-heading"),
                    html.Div(
                        [
                            _status_badge("Similarity: canonical k-mer"),
                            _status_badge("Validation: 15%"),
                            _status_badge("Test: 15%"),
                            _status_badge("Seed: 13"),
                        ],
                        className="product-badge-row",
                    ),
                    html.P("These are development defaults and not universal biological credibility standards. Assay-specific policies should be supplied in configuration.", className="chart-description"),
                ],
                className="settings-section-card",
                style={"marginTop": "14px"},
            ),
            _card(
                [
                    html.H2("Privacy and deployment", className="section-heading"),
                    html.P("AssayReady does not send assay data to a hosted service. Uploaded files, outputs, and run metadata stay on this machine unless you move or expose them."),
                    html.P("For shared deployments, put Dash behind authenticated HTTPS, restrict filesystem access, and keep uploads size-limited at the reverse proxy."),
                    html.Ul(
                        [
                            html.Li("Local-only host binding is recommended for this workstation deployment."),
                            html.Li("Authentication and HTTPS are required before shared-network exposure."),
                            html.Li("Model containers should remain network-disabled and digest-pinned."),
                        ],
                        className="security-checklist",
                    ),
                ],
                className="settings-section-card",
                style={"marginTop": "14px"},
            ),
            _card(
                [
                    html.H2("Runtime and models", className="section-heading"),
                    html.P("These are lightweight availability checks. The command-line doctor checks the general runtime; use Verify scorer in Design Sandbox for model-file integrity, head compatibility, device loading, and a forward-pass health check.", className="chart-description"),
                    html.Dl(
                        [
                            html.Div([html.Dt("Python"), html.Dd(sys.version.split()[0])], className="settings-fact"),
                            html.Div([html.Dt("AssayReady version"), html.Dd(_installed_version())], className="settings-fact"),
                            html.Div([html.Dt("Docker CLI"), html.Dd("Detected" if shutil.which("docker") else "Not detected")], className="settings-fact"),
                            html.Div([html.Dt("NVIDIA tools"), html.Dd("Detected" if shutil.which("nvidia-smi") else "Not detected")], className="settings-fact"),
                            html.Div([html.Dt("Bundled model definitions"), html.Dd(str(_bundled_model_count()))], className="settings-fact"),
                            html.Div([html.Dt("Structurally complete scorer bundles"), html.Dd(str(len(discovered_scorers)))], className="settings-fact"),
                            html.Div([html.Dt("Model runtime imports"), html.Dd("Ready" if not missing_model_runtime else "Missing: " + ", ".join(missing_model_runtime))], className="settings-fact"),
                        ],
                        className="settings-facts-grid",
                    ),
                    html.Div(
                        [
                            html.Code("assayready doctor", className="command-chip"),
                            dcc.Clipboard(content="assayready doctor", title="Copy doctor command"),
                        ],
                        className="command-row",
                    ),
                    _scorer_setup_guide(open_by_default=not bool(discovered_scorers)),
                ],
                className="settings-section-card",
                style={"marginTop": "14px"},
            ),
        ],
        className="workspace-page settings-page",
    )


def _sidebar_links(pathname: str | None) -> list[Any]:
    path = pathname or "/analyze"
    elements: list[Any] = []
    for section_label, items in NAV_SECTIONS:
        elements.append(html.Div(section_label, className="nav-section-label"))
        group_links: list[Any] = []
        for label, href in items:
            active = path == href or (path == "/" and href == "/analyze")
            group_links.append(
                dcc.Link(
                    [
                        html.Span(NAV_ICONS.get(href, "•"), className="nav-icon", **{"aria-hidden": "true"}),
                        html.Span(label, className="nav-label"),
                    ],
                    href=href,
                    className=f"nav-link {'active' if active else ''}",
                )
            )
        elements.append(html.Div(group_links, className="nav-section-group"))
    return elements


def _sidebar() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Workspace", className="panel-title"),
                        ],
                        className="panel-title-wrap",
                    ),
                    html.Span("7 tools", className="panel-count"),
                ],
                className="panel-header",
            ),
            html.Div(
                [
                    dcc.Link(
                        [html.Span("+", className="action-plus"), html.Span("NEW ANALYSIS")],
                        href="/analyze",
                        className="side-primary-action",
                    ),
                    html.Nav(id="sidebar-links", children=_sidebar_links("/analyze"), **{"aria-label": "Workspace navigation"}),
                ],
                className="sidebar-body",
            ),
            html.Div(
                [
                    html.Div("LOCAL WORKSPACE", className="side-status-label"),
                    html.Div(
                        [html.Span(className="status-dot", **{"aria-hidden": "true"}), html.Span("Data stays on this machine")],
                        className="side-status",
                    ),
                    html.P("Runs, reports, and uploaded assay files remain in your configured local storage."),
                ],
                className="sidebar-footer",
            ),
        ],
        style=SIDEBAR_STYLE,
        className="app-sidebar",
    )


def _topbar() -> html.Header:
    return html.Header(
        [
            dcc.Link(
                [
                    html.Span("AR", className="brand-mark", **{"aria-hidden": "true"}),
                    html.Div(
                        [
                            html.Div("AssayReady", className="topbar-title"),
                            html.Div("Bio-Sequence Assurance Console", className="topbar-subtitle"),
                        ]
                    ),
                ],
                href="/analyze",
                className="topbar-brand",
            ),
            html.Div(
                [
                    html.Details(
                        [
                            html.Summary(
                                [
                                    html.Span(className="pulse-dot", **{"aria-hidden": "true"}),
                                    html.Span("Local engine ready", className="topbar-telemetry-text"),
                                ],
                                className="engine-summary",
                            ),
                            html.Div(
                                [
                                    html.Div([html.Span("Runtime"), html.Strong("PyTorch · Pixi")]),
                                    html.Div([html.Span("Run store"), html.Strong("SQLite · local")]),
                                    html.Div([html.Span("Holdouts"), html.Strong("Leakage-aware")]),
                                ],
                                className="engine-popover",
                            ),
                        ],
                        className="topbar-engine-details",
                    ),
                ],
                className="topbar-center",
            ),
            html.Nav(
                [
                    html.Div(
                        [html.Span(className="pulse-dot", **{"aria-hidden": "true"}), html.Span("127.0.0.1:8050", className="topbar-env-text")],
                        className="topbar-env-badge",
                    ),
                    dcc.Link([html.Span("📖", **{"aria-hidden": "true"}), "Docs"], href="/benchmarks", className="top-action", title="Documentation and verified benchmarks"),
                    dcc.Link([html.Span("⚙", **{"aria-hidden": "true"}), "Settings"], href="/settings", className="top-action", title="Workspace settings"),
                ],
                className="topbar-actions",
                **{"aria-label": "Quick navigation"},
            ),
        ],
        className="app-topbar",
    )


def _toolkit_panel() -> html.Aside:
    return html.Aside(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Analysis guide", className="panel-title"),
                        ],
                        className="panel-title-wrap",
                    ),
                    html.Div(
                        [
                            html.Span("Context aware", className="panel-count"),
                            html.Button(
                                "×",
                                id="close-toolkit-button",
                                n_clicks=0,
                                className="toolkit-close-button",
                                title="Close analysis guide",
                                **{"aria-label": "Close analysis guide"},
                            ),
                        ],
                        className="toolkit-header-actions",
                    ),
                ],
                className="panel-header",
            ),
            html.Div(
                [
                    html.Div(
                        _analysis_help_content(None, "prediction"),
                        id="toolkit-evidence-container",
                        className="toolkit-context-content",
                    ),
                    dcc.Link("Open run history", href="/runs", className="toolkit-bottom-action"),
                ],
                className="toolkit-body",
            ),
        ],
        id="toolkit-panel",
        className="toolkit-panel is-collapsed",
        **{"aria-label": "AssayReady inspector and evidence panel"},
    )


def _app_shell() -> html.Div:
    return html.Div(
        [
            dcc.Location(id="location"),
            dcc.Store(id="toolkit-open-state", data=False),
            _topbar(),
            html.Div(
                [
                    _sidebar(),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div("Analysis canvas", id="workspace-panel-title", className="panel-title"),
                                        ],
                                        className="panel-title-wrap",
                                    ),
                                    html.Div(
                                        [
                                            html.Span("Local Engine", className="canvas-status"),
                                            html.Button(
                                                [html.Span("◇", **{"aria-hidden": "true"}), html.Span("Guide")],
                                                id="toggle-toolkit-button",
                                                n_clicks=0,
                                                className="toolkit-toggle-button",
                                                title="Open the contextual analysis guide",
                                                **{"aria-expanded": "false", "aria-controls": "toolkit-panel"},
                                            ),
                                            html.Span("⋮", className="panel-header-icon", **{"aria-hidden": "true"}),
                                        ],
                                        className="canvas-header-actions",
                                    ),
                                ],
                                className="panel-header workspace-panel-header",
                            ),
                            html.Main(
                                [
                                    html.Div(page_analyze(), id="page-analyze", style={"display": "block"}),
                                    html.Div(page_benchmarks(), id="page-benchmarks", style={"display": "none"}),
                                    html.Div(page_runs(), id="page-runs", style={"display": "none"}),
                                    html.Div(page_candidates(), id="page-candidates", style={"display": "none"}),
                                    html.Div(page_reports(), id="page-reports", style={"display": "none"}),
                                    html.Div(page_simulations(), id="page-simulations", style={"display": "none"}),
                                    html.Div(page_settings(), id="page-settings", style={"display": "none"}),
                                    html.Div(_page_header("Not Found", "Choose a page from the workspace navigation."), id="page-not-found", style={"display": "none"}),
                                ],
                                id="page-content",
                                style=CONTENT_STYLE,
                                className="app-content",
                            ),
                        ],
                        className="workspace-panel",
                    ),
                    _toolkit_panel(),
                ],
                id="workspace-frame",
                className="workspace-frame toolkit-collapsed",
            ),
        ],
        style=PAGE_STYLE,
        className="app-shell",
    )


def create_app() -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True, title="AssayReady", assets_folder=str(_package_dir() / "assets"))
    _install_ui_request_guards(app)
    app.layout = _app_shell

    @app.callback(
        Output("sidebar-links", "children"),
        Output("workspace-panel-title", "children"),
        Input("location", "pathname"),
    )
    def render_sidebar_links(pathname: str | None) -> tuple[list[Any], str]:
        path = pathname or "/analyze"
        route = "/analyze" if path == "/" else path
        return _sidebar_links(pathname), PAGE_PANEL_TITLES.get(route, "Assay workspace")

    @app.callback(
        Output("toolkit-open-state", "data"),
        Input("toggle-toolkit-button", "n_clicks"),
        Input("close-toolkit-button", "n_clicks"),
        Input("location", "pathname"),
        State("toolkit-open-state", "data"),
        prevent_initial_call=True,
    )
    def update_toolkit_open_state(
        _toggle_clicks: int,
        _close_clicks: int,
        _pathname: str | None,
        is_open: bool | None,
    ) -> bool:
        if callback_context.triggered_id == "toggle-toolkit-button":
            return not bool(is_open)
        return False

    @app.callback(
        Output("workspace-frame", "className"),
        Output("toolkit-panel", "className"),
        Output("toggle-toolkit-button", "aria-expanded"),
        Output("toggle-toolkit-button", "title"),
        Input("toolkit-open-state", "data"),
    )
    def render_toolkit_state(is_open: bool | None) -> tuple[str, str, str, str]:
        if is_open:
            return (
                "workspace-frame toolkit-open",
                "toolkit-panel is-open",
                "true",
                "Close the contextual analysis guide",
            )
        return (
            "workspace-frame toolkit-collapsed",
            "toolkit-panel is-collapsed",
            "false",
            "Open the contextual analysis guide",
        )

    @app.callback(
        Output("page-analyze", "style"),
        Output("page-benchmarks", "style"),
        Output("page-runs", "style"),
        Output("page-candidates", "style"),
        Output("page-reports", "style"),
        Output("page-simulations", "style"),
        Output("page-settings", "style"),
        Output("page-not-found", "style"),
        Input("location", "pathname"),
    )
    def render_page(pathname: str | None) -> tuple[dict[str, str], ...]:
        path = pathname or "/analyze"
        route = "/analyze" if path == "/" else path
        known = ["/analyze", "/benchmarks", "/runs", "/candidates", "/reports", "/simulations", "/settings"]
        visible = route if route in known else "/not-found"
        return tuple({"display": "block" if item == visible else "none"} for item in [*known, "/not-found"])

    @app.callback(
        Output("candidate-project-filter", "options"),
        Output("candidate-project-filter", "value"),
        Output("candidate-index-store", "data"),
        Output("reports-run-select", "options"),
        Output("reports-run-select", "value"),
        Output("benchmark-run-select", "options"),
        Output("benchmark-run-select", "value"),
        Input("location", "pathname"),
        Input("location", "search"),
        State("candidate-project-filter", "value"),
        State("reports-run-select", "value"),
        State("benchmark-run-select", "value"),
    )
    def refresh_linked_run_pages(
        pathname: str | None,
        search: str | None,
        current_candidate: Any | None,
        current_report: Any | None,
        current_benchmark: Any | None,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        route = "/analyze" if pathname == "/" else pathname
        unchanged = (no_update,) * 7
        if route == "/candidates":
            candidates, run_options, default_run_id = _candidate_navigation_snapshot()
            options = run_options + [{"label": "All runs (grouped; no global comparison)", "value": "all"}]
            selected = _preferred_run_value(search, options, current_candidate, default_run_id)
            return options, selected, candidates, *unchanged[3:]
        if route == "/reports":
            _runs, options = _report_navigation_snapshot()
            selected = _preferred_run_value(search, options, current_report, options[0]["value"] if options else None)
            return *unchanged[:3], options, selected, *unchanged[5:]
        if route == "/benchmarks":
            _runs, options = _benchmark_navigation_snapshot()
            selected = _preferred_run_value(search, options, current_benchmark, options[0]["value"] if options else None)
            return *unchanged[:5], options, selected
        return unchanged

    @app.callback(
        Output("clear-upload-cache-confirmation", "style"),
        Input("clear-upload-cache-button", "n_clicks"),
        Input("confirm-clear-upload-cache-button", "n_clicks"),
        Input("cancel-clear-upload-cache-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_upload_cache_confirmation(_request: int, _confirm: int, _cancel: int) -> dict[str, str]:
        if callback_context.triggered_id == "clear-upload-cache-button":
            return {"display": "flex"}
        return {"display": "none"}

    @app.callback(
        Output("clear-upload-cache-status", "children"),
        Input("confirm-clear-upload-cache-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_upload_cache(n_clicks: int) -> Any:
        if not n_clicks:
            raise PreventUpdate
        root = _upload_root().resolve()
        allowed = _output_root().resolve()
        if root == allowed or not root.is_relative_to(allowed):
            return _status("Upload cache path failed the storage safety check.", kind="danger")
        try:
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True, exist_ok=True)
            return _status("Upload cache cleared. Run artifacts and indexed reports were not removed.", kind="success")
        except OSError as exc:
            return _status(f"Could not clear upload cache: {exc}", kind="danger")

    @app.callback(
        Output("output-dir", "value"),
        Output("output-dir-status", "children"),
        Output("output-dir-authorization", "data"),
        Input("browse-output-dir-button", "n_clicks"),
        Input("reset-output-dir-button", "n_clicks"),
        State("output-dir", "value"),
        prevent_initial_call=True,
    )
    def choose_output_directory(
        _browse_clicks: int,
        _reset_clicks: int,
        current_value: str | None,
    ) -> tuple[Any, Any, Any]:
        if callback_context.triggered_id == "reset-output-dir-button":
            return "", _status(f"Using the default output root: {_output_root()}", kind="info"), None
        initial = Path(str(current_value)).resolve() if str(current_value or "").strip() else _output_root()
        try:
            selected = _choose_directory(title="Choose AssayReady output folder", initial_dir=initial)
            if not selected:
                return no_update, _status("Folder selection canceled; the previous output setting was kept."), no_update
            resolved = _safe_output_dir(selected, explicitly_selected=True)
            return (
                str(resolved),
                _status(f"Runs will be saved in unique subfolders under {resolved}.", kind="success"),
                _output_selection_authorization(resolved),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return no_update, _status(f"Could not use that output folder: {exc}", kind="warning"), no_update

    @app.callback(
        Output({"type": "benchmark-selection-store", "index": MATCH}, "data"),
        Input({"type": "benchmark-fit-graph", "index": MATCH}, "selectedData", allow_optional=True),
        Input({"type": "benchmark-residual-graph", "index": MATCH}, "selectedData", allow_optional=True),
        Input({"type": "benchmark-uncertainty-graph", "index": MATCH}, "selectedData", allow_optional=True),
        Input({"type": "benchmark-clear-selection", "index": MATCH}, "n_clicks"),
        prevent_initial_call=True,
    )
    def sync_benchmark_selection(
        fit_selection: dict[str, Any] | None,
        residual_selection: dict[str, Any] | None,
        uncertainty_selection: dict[str, Any] | None,
        _clear_clicks: int,
    ) -> list[str]:
        triggered = callback_context.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        component_type = triggered.get("type")
        if component_type == "benchmark-clear-selection":
            return []
        event_by_type = {
            "benchmark-fit-graph": fit_selection,
            "benchmark-residual-graph": residual_selection,
            "benchmark-uncertainty-graph": uncertainty_selection,
        }
        if component_type not in event_by_type:
            raise PreventUpdate
        event = event_by_type[component_type]
        keys = _selected_row_keys(event)
        if isinstance(event, dict) and (event.get("points") or []) and not keys:
            raise PreventUpdate
        return keys

    @app.callback(
        Output({"type": "benchmark-fit-graph", "index": MATCH}, "figure"),
        Input({"type": "benchmark-selection-store", "index": MATCH}, "data"),
        State({"type": "benchmark-row-store", "index": MATCH}, "data"),
    )
    def update_fit_selection(selected_row_keys: list[str] | None, payload: dict[str, Any] | None) -> Any:
        if not payload:
            raise PreventUpdate
        return _payload_regression_figure(payload, selected_row_keys, residual=False)

    @app.callback(
        Output({"type": "benchmark-residual-graph", "index": MATCH}, "figure"),
        Input({"type": "benchmark-selection-store", "index": MATCH}, "data"),
        State({"type": "benchmark-row-store", "index": MATCH}, "data"),
    )
    def update_residual_selection(selected_row_keys: list[str] | None, payload: dict[str, Any] | None) -> Any:
        if not payload:
            raise PreventUpdate
        return _payload_regression_figure(payload, selected_row_keys, residual=True)

    @app.callback(
        Output({"type": "benchmark-uncertainty-graph", "index": MATCH}, "figure"),
        Input({"type": "benchmark-selection-store", "index": MATCH}, "data"),
        State({"type": "benchmark-row-store", "index": MATCH}, "data"),
    )
    def update_uncertainty_selection(selected_row_keys: list[str] | None, payload: dict[str, Any] | None) -> Any:
        if not payload:
            raise PreventUpdate
        return _payload_uncertainty_figure(payload, selected_row_keys)

    @app.callback(
        Output({"type": "benchmark-slice-container", "index": MATCH}, "children"),
        Input({"type": "benchmark-selection-store", "index": MATCH}, "data"),
        State({"type": "benchmark-row-store", "index": MATCH}, "data"),
    )
    def update_selected_slices(selected_row_keys: list[str] | None, payload: dict[str, Any] | None) -> Any:
        if not payload:
            raise PreventUpdate
        return _payload_slice_table(payload, selected_row_keys)

    @app.callback(
        Output({"type": "benchmark-selection-count", "index": MATCH}, "children"),
        Output({"type": "benchmark-clear-selection", "index": MATCH}, "disabled"),
        Input({"type": "benchmark-selection-store", "index": MATCH}, "data"),
        State({"type": "benchmark-row-store", "index": MATCH}, "data"),
    )
    def update_selection_count(selected_row_keys: list[str] | None, payload: dict[str, Any] | None) -> tuple[Any, bool]:
        return _selection_count(payload, selected_row_keys), not bool(selected_row_keys)

    @app.callback(
        Output({"type": "benchmark-inspector-body", "index": MATCH}, "children"),
        Output({"type": "benchmark-inspector", "index": MATCH}, "className"),
        Output({"type": "benchmark-inspector", "index": MATCH}, "aria-hidden"),
        Input({"type": "benchmark-fit-graph", "index": MATCH}, "clickData", allow_optional=True),
        Input({"type": "benchmark-residual-graph", "index": MATCH}, "clickData", allow_optional=True),
        Input({"type": "benchmark-uncertainty-graph", "index": MATCH}, "clickData", allow_optional=True),
        Input({"type": "benchmark-inspector-close", "index": MATCH}, "n_clicks"),
        State({"type": "benchmark-inspector-store", "index": MATCH}, "data"),
        prevent_initial_call=True,
    )
    def inspect_benchmark_row(
        fit_click: dict[str, Any] | None,
        residual_click: dict[str, Any] | None,
        uncertainty_click: dict[str, Any] | None,
        _close_clicks: int,
        payload: dict[str, Any] | None,
    ) -> tuple[Any, str, str]:
        triggered = callback_context.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        component_type = triggered.get("type")
        if component_type == "benchmark-inspector-close":
            return _observation_inspector(payload, None), "observation-inspector", "true"
        event_by_type = {
            "benchmark-fit-graph": fit_click,
            "benchmark-residual-graph": residual_click,
            "benchmark-uncertainty-graph": uncertainty_click,
        }
        keys = _selected_row_keys(event_by_type.get(str(component_type)))
        if not keys:
            raise PreventUpdate
        return _observation_inspector(payload, keys[0]), "observation-inspector is-open", "false"

    app.clientside_callback(
        r"""
        function(className) {
            const isOpen = String(className || '').split(/\s+/).includes('is-open');
            const state = window.__assayreadyInspectorA11y || {trigger: null, handler: null};
            window.__assayreadyInspectorA11y = state;
            if (isOpen) {
                if (!state.trigger || !document.body.contains(state.trigger)) {
                    state.trigger = document.activeElement;
                }
                if (!state.handler) {
                    state.handler = function(event) {
                        if (event.key === 'Escape') {
                            const closeButton = document.querySelector('.observation-inspector.is-open .inspector-close-button');
                            if (closeButton) {
                                event.preventDefault();
                                closeButton.click();
                            }
                        }
                    };
                    document.addEventListener('keydown', state.handler);
                }
                window.setTimeout(function() {
                    const closeButton = document.querySelector('.observation-inspector.is-open .inspector-close-button');
                    if (closeButton) closeButton.focus();
                }, 0);
            } else {
                if (state.handler) {
                    document.removeEventListener('keydown', state.handler);
                    state.handler = null;
                }
                const trigger = state.trigger;
                state.trigger = null;
                window.setTimeout(function() {
                    if (trigger && document.body.contains(trigger) && typeof trigger.focus === 'function') trigger.focus();
                }, 0);
            }
            return {open: isOpen, changed: Date.now()};
        }
        """,
        Output({"type": "benchmark-inspector-a11y", "index": MATCH}, "data"),
        Input({"type": "benchmark-inspector", "index": MATCH}, "className"),
        prevent_initial_call=True,
    )

    @app.callback(Output("benchmark-detail", "children"), Input("benchmark-run-select", "value"), Input("benchmark-run-select", "options"))
    def show_benchmark(run_id: str | None, _options: list[dict[str, Any]] | None) -> Any:
        if not run_id:
            return _status("No benchmark runs are indexed yet. Run an analysis or the public reference benchmark first.")
        summary = get_run_summary(str(run_id))
        if not summary:
            return _status("Could not load the selected benchmark summary.", kind="warning")
        return _benchmark_section(summary, standalone=True, surface="indexed-benchmark")

    @app.callback(
        Output("public-benchmark-result", "children"),
        Input("run-public-benchmark-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def run_public_benchmark(n_clicks: int) -> Any:
        if not n_clicks:
            raise PreventUpdate
        try:
            config_path = _example_path("public_dream_promoter_audit.json")
            params = _prediction_params_from_config(_load_config(config_path), config_dir=config_path.parent)
            params["output_dir"] = _output_root() / "public_dream_promoter_audit"
            summary = run_prediction_audit(**params)
            _store_result(summary, "prediction_audit_report.md", "prediction")
            return html.Div(
                [
                    _status("Public DREAM reference benchmark completed and was added to the run index.", kind="success"),
                    _benchmark_section(summary, standalone=True, surface="public-benchmark"),
                ]
            )
        except Exception as exc:
            return _failure_panel("Public benchmark failed", exc)

    @app.callback(
        Output("assay-upload-state", "data"),
        Output("assay-preview", "children"),
        Output("sequence-col", "options"),
        Output("sequence-col", "value"),
        Output("target-col", "options"),
        Output("target-col", "value"),
        Output("prediction-col", "options"),
        Output("prediction-col", "value"),
        Output("uncertainty-col", "options"),
        Output("uncertainty-col", "value"),
        Output("id-col", "options"),
        Output("id-col", "value"),
        Output("group-cols", "options"),
        Output("group-cols", "value"),
        Output("metadata-cols", "options"),
        Output("metadata-cols", "value"),
        Output("candidate-upload-state", "data"),
        Output("candidate-preview", "children"),
        Input("assay-upload", "contents"),
        Input("candidate-upload", "contents"),
        Input("load-prediction-example-button", "n_clicks"),
        State("assay-upload", "filename"),
        State("candidate-upload", "filename"),
        prevent_initial_call=True,
    )
    def load_analysis_inputs(
        assay_contents: str | None,
        candidate_contents: str | None,
        example_clicks: int,
        assay_filename: str | None,
        candidate_filename: str | None,
    ) -> tuple[Any, ...]:
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""

        def assay_outputs(
            path: Path,
            filename: str,
            frame: pd.DataFrame,
            *,
            paths: list[Path] | None = None,
            defaults: dict[str, Any] | None = None,
            ui_config: dict[str, Any] | None = None,
            source_message: str | None = None,
            evaluation_manifest: dict[str, Any] | None = None,
        ) -> tuple[Any, ...]:
            columns = [str(column) for column in frame.columns]
            options = _dropdown_options(columns)
            data_paths = paths or [path]
            defaults = defaults or {}
            state = {
                "path": str(data_paths[0]),
                "paths": [str(item) for item in data_paths],
                "filename": filename,
                "columns": columns,
                "rows": int(len(frame)),
                "preview_rows": json.loads(frame.head(5).to_json(orient="records", date_format="iso")),
                "cardinalities": {str(column): int(frame[column].nunique(dropna=True)) for column in frame.columns},
                "label_values": {
                    str(column): sorted(
                        {str(value) for value in frame[column].dropna().tolist()},
                        key=str.casefold,
                    )
                    for column in frame.columns
                    if int(frame[column].nunique(dropna=True)) <= 20
                },
                "evaluation_manifest": evaluation_manifest,
                "ui_config": ui_config,
            }
            group_defaults = _valid_columns(columns, defaults.get("group_cols"))
            metadata_defaults = _valid_columns(columns, defaults.get("metadata_cols"))
            preview = html.Div(
                [
                    html.Div(
                        [
                            html.Strong(filename),
                            html.Span(f"{len(frame):,} rows"),
                            html.Span(f"{len(columns):,} detected columns"),
                            html.Span("Parsing complete", className="parse-success"),
                        ],
                        className="inline-upload-summary",
                    ),
                    _status(source_message, kind="info") if source_message else None,
                    html.Div(
                        [
                            html.Div("Preview first 5 rows", className="data-preview-title"),
                            html.Div(_data_table(frame.head(5), page_size=5, height=190, compact=True), style={"marginTop": "10px"}),
                        ],
                        className="data-preview-details",
                    ),
                ]
            )
            return (
                state,
                preview,
                options,
                _first_valid_col(columns, defaults.get("sequence_col"), ["sequence", "dna", "promoter_sequence", "insert_sequence"]),
                options,
                _first_valid_col(columns, defaults.get("target_col"), ["measured_value", "measured_expression", "expression", "activity", "fluorescence", "target"]),
                options,
                _first_valid_col(columns, defaults.get("prediction_col"), ["model_prediction", "prediction", "predicted_value", "score"]),
                options,
                _first_valid_col(columns, defaults.get("uncertainty_col"), ["model_uncertainty", "uncertainty", "std", "variance"], optional=True),
                options,
                _first_valid_col(columns, defaults.get("id_col"), ["sequence_id", "construct_id", "variant_id", "id"], optional=True),
                options,
                group_defaults or [column for column in ["batch", "construct_family"] if column in columns],
                options,
                metadata_defaults or [column for column in ["batch", "organism", "construct_type", "assay"] if column in columns],
            )

        def candidate_outputs(path: Path, filename: str, frame: pd.DataFrame, *, paths: list[Path] | None = None) -> tuple[Any, Any]:
            columns = [str(column) for column in frame.columns]
            data_paths = paths or [path]
            state = {
                "path": str(data_paths[0]),
                "paths": [str(item) for item in data_paths],
                "filename": filename,
                "columns": columns,
                "rows": int(len(frame)),
            }
            hint = None
            if {"measured_expression", "model_prediction"}.issubset(set(columns)) or "prediction_audit" in filename:
                hint = _status("This looks like a measured prediction-audit table. It belongs in Step 1 as assay data; candidate files are optional ranking pools.", kind="warning")
            return (
                state,
                html.Div(
                    [
                        _status(f"Loaded candidate table {filename}: {len(frame):,} rows.", kind="success"),
                        hint,
                        html.Details(
                            [
                                html.Summary("Preview first 5 candidates", className="details-summary"),
                                html.Div(_data_table(frame.head(5), page_size=5, height=190, compact=True), style={"marginTop": "10px"}),
                            ],
                            className="data-preview-details",
                        ),
                    ]
                ),
            )

        try:
            if triggered == "load-prediction-example-button":
                assay_path = _example_path("prediction_audit.csv")
                candidate_path = _example_path("prediction_candidates.csv")
                assay_frame = pd.read_csv(assay_path)
                candidate_frame = pd.read_csv(candidate_path)
                return (
                    *assay_outputs(assay_path, assay_path.name, assay_frame),
                    *candidate_outputs(candidate_path, candidate_path.name, candidate_frame),
                )

            if triggered == "assay-upload":
                if not assay_contents:
                    raise PreventUpdate
                path, blob = _save_upload(assay_contents, assay_filename or "assay.csv", kind="assay")
                upload_name = assay_filename or path.name
                if Path(upload_name).suffix.lower() in CONFIG_EXTENSIONS:
                    config = _load_config_bytes(blob, upload_name)
                    assay_paths, candidate_paths = _config_table_paths(config, config_dir=path.parent)
                    frame = _config_preview_frame(_concat_tables(assay_paths), config)
                    config_task = config.get("task") or {}
                    config_task_type = str(config_task.get("type", config.get("task_type", "regression"))).lower()
                    evaluation_manifest = validate_evaluation_manifest(config, task_type=config_task_type).to_dict() if config.get("evaluation") else None
                    source_message = f"Loaded config {upload_name}; using assay file(s): {', '.join(item.name for item in assay_paths)}."
                    candidate_state = None
                    candidate_preview = None
                    if candidate_paths:
                        candidate_frame = _concat_tables(candidate_paths)
                        candidate_state, candidate_preview = candidate_outputs(
                            candidate_paths[0],
                            f"{upload_name} candidates",
                            candidate_frame,
                            paths=candidate_paths,
                        )
                    return (
                        *assay_outputs(
                            assay_paths[0],
                            upload_name,
                            frame,
                            paths=assay_paths,
                            defaults=_config_column_defaults(config),
                            ui_config=_config_ui_defaults(config, evaluation_manifest=evaluation_manifest),
                            source_message=source_message,
                            evaluation_manifest=evaluation_manifest,
                        ),
                        candidate_state,
                        candidate_preview,
                    )
                frame = _read_table_bytes(blob, upload_name)
                return (*assay_outputs(path, upload_name, frame), no_update, no_update)

            if triggered == "candidate-upload":
                if not candidate_contents:
                    raise PreventUpdate
                path, blob = _save_upload(candidate_contents, candidate_filename or "candidates.csv", kind="candidates")
                frame = _read_table_bytes(blob, candidate_filename or "")
                return (
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    *candidate_outputs(path, candidate_filename or path.name, frame),
                )
        except Exception as exc:
            empty: list[dict[str, str]] = []
            error = _status(f"Could not read input file: {exc}", kind="danger")
            return None, error, empty, None, empty, None, empty, None, empty, None, empty, None, empty, [], empty, [], None, no_update

        raise PreventUpdate

    @app.callback(
        Output("positive-label", "options"),
        Output("positive-label", "value"),
        Output("positive-label-wrapper", "style"),
        Input("assay-upload-state", "data"),
        Input("target-col", "value"),
        Input("task-type", "value"),
        State("positive-label", "value"),
    )
    def update_positive_label(
        assay_state: dict[str, Any] | None,
        target_col: str | None,
        task_type: str | None,
        current_value: str | None,
    ) -> tuple[list[dict[str, str]], Any, dict[str, str]]:
        if task_type != "classification":
            return [], None, {"display": "none"}
        options = _classification_label_options(assay_state, target_col)
        allowed = {option["value"] for option in options}
        configured = str(
            (((assay_state or {}).get("ui_config") or {}).get("runtime") or {}).get("positive_label")
            or ""
        )
        configured_key = _classification_label_key(configured)
        configured_match = next(
            (
                option["value"]
                for option in options
                if _classification_label_key(option["value"]) == configured_key
            ),
            None,
        )
        selected = current_value if str(current_value or "") in allowed else configured_match
        return options, selected, {"display": "block"}

    @app.callback(
        Output("analysis-type", "value"),
        Output("task-type", "value"),
        Output("top-k", "value"),
        Output("low-n", "value"),
        Output("val-fraction", "value"),
        Output("test-fraction", "value"),
        Output("homology-threshold", "value"),
        Output("homology-k", "value"),
        Output("beta", "value"),
        Output("diversity-penalty", "value"),
        Output("seed", "value"),
        Output("ensemble-size", "value"),
        Output("epochs", "value"),
        Output("learning-rate", "value"),
        Output("num-proposals", "value"),
        Output("evaluation-objective", "value"),
        Output("evaluation-units", "value"),
        Output("evaluation-uncertainty-type", "value"),
        Output("evaluation-model-id", "value"),
        Output("evaluation-data-id", "value"),
        Input("assay-upload-state", "data"),
        prevent_initial_call=True,
    )
    def hydrate_analysis_config(assay_state: dict[str, Any] | None) -> tuple[Any, ...]:
        configured = (assay_state or {}).get("ui_config")
        if not isinstance(configured, dict):
            raise PreventUpdate
        keys = [
            "analysis_type",
            "task_type",
            "top_k",
            "low_n",
            "val_fraction",
            "test_fraction",
            "homology_threshold",
            "homology_k",
            "beta",
            "diversity_penalty",
            "seed",
            "ensemble_size",
            "epochs",
            "learning_rate",
            "num_proposals",
            "evaluation_objective",
            "evaluation_units",
            "evaluation_uncertainty_type",
            "evaluation_model_id",
            "evaluation_data_id",
        ]
        return tuple(configured.get(key, no_update) for key in keys)

    @app.callback(
        Output("project-name", "value"),
        Output("analysis-type-guidance", "children"),
        Output("review-workflow-summary", "children"),
        Output("prediction-field-wrapper", "style"),
        Output("uncertainty-field-wrapper", "style"),
        Output("local-model-options", "style"),
        Output("objective-step-summary", "children"),
        Input("analysis-type", "value"),
        State("assay-upload-state", "data"),
    )
    def update_analysis_workflow(analysis_type: str, assay_state: dict[str, Any] | None) -> tuple[str, Any, Any, dict[str, str], dict[str, str], dict[str, str], str]:
        configured = (assay_state or {}).get("ui_config") or {}
        configured_project = configured.get("project") if configured.get("analysis_type") == analysis_type else None
        if analysis_type == "local_model":
            return (
                str(configured_project or "assayready_design_rank"),
                "Use measured sequences and outcomes. AssayReady will create leakage-aware splits, fit local models, and report held-out performance.",
                html.Div([html.Strong("Local model workflow"), html.Span(" Prediction and uncertainty columns are not required.")]),
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
                "Train a local assay model",
            )

        return (
            str(configured_project or "assayready_prediction_audit"),
            "Use measured outcomes alongside predictions produced by an existing model. Stronger claims require independent training and locked-holdout provenance.",
            html.Div([html.Strong("Existing-model prediction audit"), html.Span(" Ready when measured outcomes and model predictions are mapped.")]),
            {"display": "block"},
            {"display": "block"},
            {"display": "none"},
            "Evaluate model predictions",
        )

    @app.callback(
        Output("run-analysis-button", "disabled"),
        Output("run-analysis-button", "style"),
        Output("analysis-ready-state", "children"),
        Output("objective-card", "className"),
        Output("data-card", "className"),
        Output("columns-card", "className"),
        Output("columns-empty-hint", "style"),
        Output("analysis-progress", "children"),
        Output("review-card", "className"),
        Output("data-step-summary", "children"),
        Output("columns-step-summary", "children"),
        Output("review-step-summary", "children"),
        Output("mapping-data-preview", "children"),
        Output("confirm-mappings-button", "disabled"),
        Input("assay-upload-state", "data"),
        Input("sequence-col", "value"),
        Input("target-col", "value"),
        Input("prediction-col", "value"),
        Input("group-cols", "value"),
        Input("analysis-type", "value"),
        Input("task-type", "value"),
        Input("positive-label", "value"),
        Input("edit-objective-button", "n_clicks"),
        Input("edit-data-button", "n_clicks"),
        Input("edit-columns-button", "n_clicks"),
        Input("confirm-mappings-button", "n_clicks"),
    )
    def update_analysis_readiness(
        assay_state: dict[str, Any] | None,
        sequence_col: str | None,
        target_col: str | None,
        prediction_col: str | None,
        group_cols: list[str] | None,
        analysis_type: str,
        task_type: str,
        positive_label: str | None,
        _edit_objective: int,
        _edit_data: int,
        _edit_columns: int,
        _confirm_mappings: int,
    ) -> tuple[bool, dict[str, Any], Any, str, str, str, dict[str, str], list[Any], str, str, str, str, Any, bool]:
        assay_ready = bool(assay_state and assay_state.get("path"))
        available_columns = set(str(column) for column in (assay_state or {}).get("columns", []))
        columns_ready = bool(sequence_col in available_columns and target_col in available_columns)
        prediction_ready = bool(prediction_col in available_columns) if analysis_type == "prediction" else True
        classification_options = _classification_label_options(assay_state, target_col)
        classification_values = {option["value"] for option in classification_options}
        classification_ready = task_type != "classification" or (
            len(classification_values) == 2 and str(positive_label or "") in classification_values
        )
        mappings_ready = assay_ready and columns_ready and prediction_ready and classification_ready
        mappings_confirmed = callback_context.triggered_id == "confirm-mappings-button" and mappings_ready
        can_run = mappings_ready and mappings_confirmed
        style = dict(BUTTON_STYLE if can_run else SECONDARY_BUTTON_STYLE)
        if not can_run:
            style.update({"cursor": "not-allowed", "opacity": 0.62})
        hint_style = {"display": "none"} if assay_ready else {"display": "block"}
        cardinalities = (assay_state or {}).get("cardinalities") or {}
        low_cardinality_groups = [
            f"{column} ({cardinalities.get(column)} levels)"
            for column in (group_cols or [])
            if isinstance(cardinalities.get(column), int) and cardinalities[column] < 3
        ]
        readiness = _analysis_blocker_message(
            assay_ready=assay_ready,
            columns_ready=columns_ready,
            prediction_ready=prediction_ready,
            classification_ready=classification_ready,
            mappings_confirmed=mappings_confirmed,
            analysis_type=analysis_type,
        )
        if task_type == "classification" and not classification_ready:
            readiness = html.Div(
                [
                    readiness,
                    _status(
                        "Binary classification requires exactly two observed target labels and an explicit positive class.",
                        kind="warning",
                    ),
                ]
            )
        if low_cardinality_groups:
            readiness = html.Div(
                [
                    readiness,
                    _status(
                        "Independence warning: " + ", ".join(low_cardinality_groups) + " may leave too few leakage-safe groups. Move these to metadata unless leave-group-out testing is intentional.",
                        kind="warning",
                    ),
                ]
            )
        current_stage = 4 if mappings_confirmed else 3 if assay_ready else 2
        if callback_context.triggered_id == "edit-objective-button":
            current_stage = 1
        elif callback_context.triggered_id == "edit-data-button":
            current_stage = 2
        elif callback_context.triggered_id == "edit-columns-button" and assay_ready:
            current_stage = 3

        filename = str((assay_state or {}).get("filename") or "")
        rows = int((assay_state or {}).get("rows") or 0)
        data_summary = f"{filename} · {rows:,} rows · parsed" if assay_ready else "Upload data to continue"
        mapped = [value for value in [sequence_col, target_col, prediction_col if analysis_type == "prediction" else None] if value]
        columns_summary = (
            f"{len(mapped)} required roles mapped · {', '.join(mapped)}"
            if mappings_ready
            else "Choose the required column roles" if assay_ready else "Available after upload"
        )
        review_summary = "Ready for final review" if mappings_confirmed else "Available after you confirm mappings"
        return (
            not can_run,
            style,
            readiness,
            f"ui-card {_analysis_stage_class(1, current_stage)}",
            f"ui-card {_analysis_stage_class(2, current_stage)}",
            f"ui-card {_analysis_stage_class(3, current_stage, 'columns-card', *([] if assay_ready else ['muted-card']))}",
            hint_style,
            _workflow_progress_items(current_stage),
            f"ui-card {_analysis_stage_class(4, current_stage)}",
            data_summary,
            columns_summary,
            review_summary,
            _analysis_preview_from_state(assay_state),
            not mappings_ready,
        )

    @app.callback(
        Output("toolkit-evidence-container", "children"),
        Input("assay-upload-state", "data"),
        Input("analysis-type", "value"),
        Input("sequence-col", "value"),
        Input("target-col", "value"),
        Input("prediction-col", "value"),
        Input("task-type", "value"),
        Input("positive-label", "value"),
        Input("analysis-run-rail-state", "data"),
        Input("location", "pathname"),
    )
    def update_analysis_help(
        assay_state: dict[str, Any] | None,
        analysis_type: str,
        sequence_col: str | None,
        target_col: str | None,
        prediction_col: str | None,
        task_type: str,
        positive_label: str | None,
        run_state: dict[str, Any] | None,
        pathname: str | None,
    ) -> list[Any]:
        if pathname not in {None, "/", "/analyze"}:
            return _workspace_page_help(pathname)
        if callback_context.triggered_id == "analysis-run-rail-state" and run_state:
            return _analysis_results_help(run_state)
        return _analysis_help_content(
            assay_state,
            analysis_type,
            sequence_col=sequence_col,
            target_col=target_col,
            prediction_col=prediction_col,
            task_type=task_type,
            positive_label=positive_label,
        )

    @app.callback(
        Output("analysis-stale-state", "children"),
        Input("run-analysis-button", "n_clicks"),
        Input("assay-upload-state", "data"),
        Input("candidate-upload-state", "data"),
        Input("analysis-type", "value"),
        Input("project-name", "value"),
        Input("output-dir", "value"),
        Input("sequence-col", "value"),
        Input("target-col", "value"),
        Input("prediction-col", "value"),
        Input("uncertainty-col", "value"),
        Input("id-col", "value"),
        Input("group-cols", "value"),
        Input("metadata-cols", "value"),
        Input("task-type", "value"),
        Input("positive-label", "value"),
        Input("top-k", "value"),
        Input("low-n", "value"),
        Input("val-fraction", "value"),
        Input("test-fraction", "value"),
        Input("homology-threshold", "value"),
        Input("homology-k", "value"),
        Input("beta", "value"),
        Input("diversity-penalty", "value"),
        Input("seed", "value"),
        Input("ensemble-size", "value"),
        Input("epochs", "value"),
        Input("learning-rate", "value"),
        Input("num-proposals", "value"),
        Input("plate-size", "value"),
        Input("control-wells", "value"),
        Input("plate-seed", "value"),
        Input("evaluation-objective", "value"),
        Input("evaluation-units", "value"),
        Input("evaluation-uncertainty-type", "value"),
        Input("evaluation-model-id", "value"),
        Input("evaluation-data-id", "value"),
        prevent_initial_call=True,
    )
    def mark_analysis_result_stale(n_clicks: int | None, *_values: Any) -> Any:
        if not n_clicks or callback_context.triggered_id == "run-analysis-button":
            return None
        return _status(
            "Inputs changed after this result was generated. Run the analysis again before using the displayed evidence or candidate ordering.",
            kind="warning",
        )

    @app.callback(
        Output("analysis-result", "children"),
        Output("analysis-run-rail-state", "data"),
        Input("run-analysis-button", "n_clicks"),
        State("assay-upload-state", "data"),
        State("candidate-upload-state", "data"),
        State("analysis-type", "value"),
        State("project-name", "value"),
        State("output-dir", "value"),
        State("output-dir-authorization", "data"),
        State("sequence-col", "value"),
        State("target-col", "value"),
        State("prediction-col", "value"),
        State("uncertainty-col", "value"),
        State("id-col", "value"),
        State("group-cols", "value"),
        State("metadata-cols", "value"),
        State("task-type", "value"),
        State("positive-label", "value"),
        State("top-k", "value"),
        State("low-n", "value"),
        State("val-fraction", "value"),
        State("test-fraction", "value"),
        State("homology-threshold", "value"),
        State("homology-k", "value"),
        State("beta", "value"),
        State("diversity-penalty", "value"),
        State("seed", "value"),
        State("ensemble-size", "value"),
        State("epochs", "value"),
        State("learning-rate", "value"),
        State("num-proposals", "value"),
        State("plate-size", "value"),
        State("control-wells", "value"),
        State("plate-seed", "value"),
        State("evaluation-objective", "value"),
        State("evaluation-units", "value"),
        State("evaluation-uncertainty-type", "value"),
        State("evaluation-model-id", "value"),
        State("evaluation-data-id", "value"),
        running=[
            (Output("run-analysis-button", "children"), "Running analysis...", "Run analysis"),
        ],
        prevent_initial_call=True,
    )
    def run_analysis(
        n_clicks: int,
        assay_state: dict[str, Any] | None,
        candidate_state: dict[str, Any] | None,
        analysis_type: str,
        project: str,
        output_dir: str | None,
        selected_output: dict[str, Any] | None,
        sequence_col: str | None,
        target_col: str | None,
        prediction_col: str | None,
        uncertainty_col: str | None,
        id_col: str | None,
        group_cols: list[str] | None,
        metadata_cols: list[str] | None,
        task_type: str,
        positive_label: str | None,
        top_k: int,
        low_n: int,
        val_fraction: float,
        test_fraction: float,
        homology_threshold: float,
        homology_k: int,
        beta: float,
        diversity_penalty: float,
        seed: int,
        ensemble_size: int,
        epochs: int,
        learning_rate: float,
        num_proposals: int,
        plate_size: int,
        control_wells: int,
        plate_seed: int,
        evaluation_objective: str,
        evaluation_units: str,
        evaluation_uncertainty_type: str,
        evaluation_model_id: str,
        evaluation_data_id: str,
    ) -> tuple[Any, dict[str, Any]]:
        if not n_clicks:
            raise PreventUpdate
        if not assay_state:
            message = "Upload measured assay data before running analysis."
            return _status(message, kind="warning"), {"status": "error", "message": message}
        available_columns = set(str(column) for column in assay_state.get("columns", []))
        if sequence_col not in available_columns or target_col not in available_columns:
            message = "Select sequence and measured target columns from the loaded assay table."
            return _status(message, kind="warning"), {"status": "error", "message": message}
        if analysis_type == "prediction" and prediction_col not in available_columns:
            message = "Prediction evaluation requires a prediction column from the loaded assay table."
            return _status(message, kind="warning"), {"status": "error", "message": message}
        if task_type == "classification":
            classification_values = {
                option["value"]
                for option in _classification_label_options(assay_state, target_col)
            }
            if len(classification_values) != 2 or str(positive_label or "") not in classification_values:
                message = "Binary classification requires exactly two observed target labels and an explicit positive class."
                return _status(message, kind="warning"), {"status": "error", "message": message}
        invalid_optional = [
            column
            for column in [uncertainty_col, id_col, *(group_cols or []), *(metadata_cols or [])]
            if column and column not in available_columns
        ]
        if invalid_optional:
            message = f"These selected columns are not in the loaded assay table: {', '.join(invalid_optional)}"
            return _status(message, kind="warning"), {"status": "error", "message": message}
        try:
            assay_files = [Path(path) for path in (assay_state.get("paths") or [assay_state["path"]])]
            candidate_files = [Path(path) for path in (candidate_state.get("paths") or [candidate_state["path"]])] if candidate_state and candidate_state.get("path") else []
            output_path = _analysis_output_dir(
                output_dir,
                project,
                selected_output=selected_output,
            )
            runtime_config = ((assay_state.get("ui_config") or {}).get("runtime") or {})
            configured_manifest = assay_state.get("evaluation_manifest")
            evaluation_manifest = dict(configured_manifest) if isinstance(configured_manifest, dict) else inferred_manifest_for_args(task_type=task_type or "regression").to_dict()
            evaluation_manifest["task_type"] = str(task_type or "regression")
            evaluation_manifest["positive_label"] = str(positive_label) if task_type == "classification" else None
            evaluation_manifest["objective_direction"] = str(evaluation_objective or "maximize")
            evaluation_manifest["units"] = str(evaluation_units or "unspecified")
            evaluation_manifest["uncertainty_type"] = str(evaluation_uncertainty_type or "none")
            evaluation_manifest.setdefault("provenance", {})["model_identifier"] = str(evaluation_model_id or "ui_unspecified")
            evaluation_manifest["provenance"]["data_identifier"] = str(evaluation_data_id or assay_state.get("filename") or "ui_unspecified")
            evaluation_manifest["provenance"]["split_identifier"] = str(evaluation_manifest["provenance"].get("split_identifier") or "ui_leakage_aware_split")
            base_params = {
                "project": project or ("prediction_audit" if analysis_type == "prediction" else "assayready_design_rank"),
                "assay_files": assay_files,
                "candidate_files": candidate_files,
                "output_dir": output_path,
                "sequence_col": sequence_col,
                "target_col": target_col,
                "id_col": id_col or None,
                "metadata_cols": metadata_cols or [],
                "group_cols": group_cols or [],
                "task_type": task_type or "regression",
                "positive_label": str(positive_label) if task_type == "classification" else None,
                "column_aliases": dict(runtime_config.get("column_aliases") or {}),
                "low_n_threshold": int(_value_or_default(low_n, 200)),
                "val_fraction": float(_value_or_default(val_fraction, 0.15)),
                "test_fraction": float(_value_or_default(test_fraction, 0.15)),
                "homology_threshold": float(_value_or_default(homology_threshold, 0.90)),
                "homology_k": int(_value_or_default(homology_k, 8)),
                "similarity_policy": str(runtime_config.get("similarity_policy") or "canonical_kmer_jaccard"),
                "similarity_sensitivity_policies": list(runtime_config.get("similarity_sensitivity_policies") or []),
                "similarity_sensitivity_thresholds": list(runtime_config.get("similarity_sensitivity_thresholds") or []),
                "top_k": int(_value_or_default(top_k, 96)),
                "beta": float(_value_or_default(beta, 1.0)),
                "diversity_method": "kmer_cosine",
                "diversity_penalty": float(_value_or_default(diversity_penalty, 0.2)),
                "seed": int(_value_or_default(seed, 13)),
                "evaluation_manifest": evaluation_manifest,
                "manifest_source": "ui_reviewed_config" if configured_manifest else "ui_declared",
            }
            if analysis_type == "prediction":
                summary = run_prediction_audit(
                    **base_params,
                    prediction_col=prediction_col,
                    uncertainty_col=uncertainty_col or None,
                )
                report_name = "prediction_audit_report.md"
                workflow = "prediction"
            else:
                summary = run_assayready(
                    **base_params,
                    acquisition_method=str(runtime_config.get("acquisition_method") or "upper_confidence_bound"),
                    generate_candidates=bool(runtime_config.get("generate_candidates", not bool(candidate_files))) and not bool(candidate_files),
                    num_proposals=int(_value_or_default(num_proposals, 1000)),
                    sequence_length="infer_from_training",
                    gc_range=None,
                    ensemble_size=int(_value_or_default(ensemble_size, 5)),
                    epochs=int(_value_or_default(epochs, 80)),
                    learning_rate=float(_value_or_default(learning_rate, 0.001)),
                )
                summary = _add_client_plate_plan_artifacts(
                    summary,
                    plate_size=int(_value_or_default(plate_size, 96)),
                    control_wells=int(_value_or_default(control_wells, 8)),
                    plate_seed=int(_value_or_default(plate_seed, 13)),
                )
                report_name = "readiness_report.md"
                workflow = "internal"
            _store_result(summary, report_name, workflow)
            return _result_summary(summary, report_name), _analysis_run_rail_state(summary)
        except Exception as exc:
            return _failure_panel("Analysis failed", exc), {"status": "error", "message": f"Analysis failed: {exc}"}

    @app.callback(Output("analysis-download", "data"), Input("analysis-download-button", "n_clicks"), State("analysis-artifact-select", "value"), prevent_initial_call=True)
    def download_analysis_artifact(n_clicks: int, path: str | None) -> Any:
        if not n_clicks or not path:
            raise PreventUpdate
        safe_path = _safe_artifact_path(path)
        return dcc.send_file(str(safe_path))

    @app.callback(
        Output("runs-content", "children"),
        Input("index-runs-button", "n_clicks"),
        Input("refresh-runs-button", "n_clicks"),
        Input("location", "pathname"),
        prevent_initial_call=True,
    )
    def refresh_runs(index_clicks: int, refresh_clicks: int, pathname: str | None) -> Any:
        triggered = callback_context.triggered_id
        if triggered == "location" and pathname != "/runs":
            raise PreventUpdate
        status = None
        if triggered == "index-runs-button":
            count = index_existing_runs()
            status = _status(f"Indexed {count} artifact folder(s).", kind="success")
        return _runs_content(status)

    @app.callback(
        Output("runs-table-container", "children"),
        Input("runs-search", "value", allow_optional=True),
        Input("runs-workflow-filter", "value", allow_optional=True),
        Input("runs-assurance-filter", "value", allow_optional=True),
        Input("runs-evidence-filter", "value", allow_optional=True),
        State("runs-index-store", "data", allow_optional=True),
        State("runs-run-select", "value", allow_optional=True),
    )
    def filter_runs(
        search: str | None,
        workflow: str | None,
        evaluation: str | None,
        recommended_use: str | None,
        stored_runs: list[dict[str, Any]] | None,
        selected_run_id: str | None,
    ) -> Any:
        selected = list(stored_runs or [])
        query = str(search or "").strip().lower()
        if query:
            selected = [
                run for run in selected
                if query in " ".join(
                    [
                        _friendly_project_name(run.get("project")),
                        str(run.get("project") or ""),
                        _run_display_metadata(run)["workflow"],
                    ]
                ).lower()
            ]
        if workflow and workflow != "all":
            selected = [run for run in selected if _run_display_metadata(run)["workflow"] == workflow]
        if evaluation and evaluation != "all":
            selected = [run for run in selected if _run_display_metadata(run)["evaluation"] == evaluation]
        if recommended_use and recommended_use != "all":
            selected = [run for run in selected if _run_display_metadata(run)["recommended_use"] == recommended_use]
        if not selected:
            return _status("No runs match the selected filters.")
        display = _runs_display_frame(selected)
        selected_rows = display.loc[display["Run ID"].astype(str) == str(selected_run_id)].to_dict("records")
        return _data_table(
            display,
            page_size=12,
            height=480,
            table_id="runs-index-table",
            hidden_columns={"Run ID"},
            selectable=True,
            selected_rows=selected_rows,
        )

    @app.callback(
        Output("runs-run-select", "value"),
        Input("runs-index-table", "cellClicked", allow_optional=True),
        prevent_initial_call=True,
    )
    def select_run_from_grid(cell_clicked: dict[str, Any] | None) -> Any:
        row = (cell_clicked or {}).get("data") or {}
        run_id = row.get("Run ID")
        if not run_id:
            raise PreventUpdate
        return str(run_id)

    @app.callback(Output("runs-run-detail", "children"), Input("runs-run-select", "value"))
    def show_run_detail(run_id: str | None) -> Any:
        if not run_id:
            raise PreventUpdate
        summary = get_run_summary(run_id)
        if not summary:
            return _status("Could not load selected run summary.", kind="warning")
        run = next((item for item in list_runs(limit=500) if str(item.get("run_id")) == str(run_id)), None)
        return html.Div(
            [
                _run_summary_header(summary, run=run),
                _result_summary(
                    summary,
                    "prediction_audit_report.md" if "prediction_audit" in summary else "simulation_report.md" if "simulation_report" in summary else "readiness_report.md",
                    surface="runs",
                ),
            ],
            className="selected-run-detail",
        )

    @app.callback(Output("report-detail", "children"), Input("reports-run-select", "value"), Input("reports-run-select", "options"))
    def show_report(run_id: str | None, _options: list[dict[str, Any]] | None) -> Any:
        if not run_id:
            return _status("No reports are indexed yet.")
        summary = get_run_summary(run_id)
        run = next((item for item in list_runs(limit=500) if str(item.get("run_id")) == str(run_id)), None)
        if not summary or not run:
            return _status("Could not load the selected report.", kind="warning")
        if _is_non_audited_sandbox_run(summary, run):
            return _sandbox_report_empty_state(summary, run)
        artifact_dir = Path(str(run["artifact_dir"]))
        report_name = str(run.get("report_name") or "readiness_report.md")
        report_path = artifact_dir / report_name
        artifacts = pd.DataFrame(list_artifacts(str(run_id)))
        artifact_opts = _artifact_options(summary, report_name)
        report_body = report_path.read_text(encoding="utf-8") if report_path.exists() else "Report markdown was not found."
        return html.Div(
            [
                _run_summary_header(summary, run=run),
                _card(
                    [
                        html.H3("Artifacts", className="section-heading"),
                        html.P("Download the complete evidence package for review or choose an individual artifact.", className="chart-description"),
                        html.Button("Download evidence package (.zip)", id="report-package-button", n_clicks=0, style=BUTTON_STYLE),
                        dcc.Dropdown(id="report-artifact-select", options=artifact_opts, value=artifact_opts[0]["value"] if artifact_opts else None, clearable=False),
                        html.Button("Download selected artifact", id="report-download-button", n_clicks=0, style={**SECONDARY_BUTTON_STYLE, "marginTop": "10px"}),
                        html.Details([html.Summary("Local artifact location", className="details-summary"), html.Code(str(artifact_dir), className="path-value")], className="maintenance-details"),
                    ]
                ),
                _benchmark_section(summary, surface="reports"),
                html.Details(
                    [html.Summary("Rendered technical report", className="result-details-summary"), _card(dcc.Markdown(report_body), style={"marginTop": "12px"})],
                    className="result-details",
                ),
                html.Details(
                    [html.Summary("Artifact index", className="result-details-summary"), _data_table(artifacts[[column for column in ["name"] if column in artifacts.columns]], page_size=12)],
                    className="result-details",
                ),
            ]
        )

    @app.callback(Output("report-download", "data"), Input("report-download-button", "n_clicks"), State("report-artifact-select", "value"), prevent_initial_call=True)
    def download_report_artifact(n_clicks: int, path: str | None) -> Any:
        if not n_clicks or not path:
            raise PreventUpdate
        return dcc.send_file(str(_safe_artifact_path(path)))

    @app.callback(
        Output("report-package-download", "data"),
        Input("report-package-button", "n_clicks", allow_optional=True),
        State("reports-run-select", "value"),
        prevent_initial_call=True,
    )
    def download_evidence_package(n_clicks: int | None, run_id: str | None) -> Any:
        if not n_clicks or not run_id:
            raise PreventUpdate
        payload, filename = _evidence_package_bytes(str(run_id))
        return dcc.send_bytes(payload, filename)

    @app.callback(
        Output("candidate-table-container", "children"),
        Input("candidate-project-filter", "value"),
        Input("candidate-source-filter", "value"),
        Input("candidate-risk-filter", "value"),
        Input("candidate-index-store", "data"),
    )
    def filter_candidates(run_id: str, source: str, risk: str, rows: list[dict[str, Any]] | None) -> Any:
        selected = list(rows or [])
        if run_id and run_id != "all":
            selected = [row for row in selected if str(row.get("run_id")) == str(run_id)]
        if source == "model":
            selected = [row for row in selected if str(row.get("workflow")) != "simulation"]
        elif source == "simulation":
            selected = [row for row in selected if str(row.get("workflow")) == "simulation"]
        if risk == "none":
            selected = [row for row in selected if not _load_json_cell(row.get("risk_flags_json"), [])]
        elif risk == "flagged":
            selected = [row for row in selected if bool(_load_json_cell(row.get("risk_flags_json"), []))]
        if not selected:
            return _status("No candidates match the selected filters.")
        include_context = run_id == "all"
        frame = _candidate_display_frame(selected, include_context=include_context)
        context = _candidate_run_status(run_id)
        return html.Div(
            [
                context,
                _data_table(frame, page_size=15, height=_candidate_grid_height(len(frame)), table_id="candidate-index-table", hidden_columns={"Run ID"}, selectable=True),
            ],
            className="candidate-grid-stack",
        )

    @app.callback(
        Output("candidate-detail", "children"),
        Input("candidate-index-table", "cellClicked", allow_optional=True),
        Input("candidate-index-store", "data"),
        prevent_initial_call=True,
    )
    def inspect_candidate(cell_clicked: dict[str, Any] | None, all_rows: list[dict[str, Any]] | None) -> Any:
        if callback_context.triggered_id == "candidate-index-store":
            return _status("Click any candidate row to inspect its complete sequence, risks, neighborhood, and ranking context.")
        if not cell_clicked or not isinstance(cell_clicked.get("data"), dict):
            raise PreventUpdate
        visible = cell_clicked["data"]
        candidate = next(
            (
                row for row in (all_rows or [])
                if str(row.get("run_id")) == str(visible.get("Run ID"))
                and str(row.get("display_id")) == str(visible.get("Candidate"))
            ),
            {},
        )
        risks = [_human_label(item) for item in _load_json_cell(candidate.get("risk_flags_json"), [])]
        candidate_summary = get_run_summary(str(candidate.get("run_id"))) if candidate.get("run_id") else None
        candidate_report = _report_payload(candidate_summary or {})
        candidate_manifest = candidate_report.get("evaluation_manifest") or (candidate_summary or {}).get("evaluation_manifest") or {}
        units = _human_label(candidate_manifest.get("units"), fallback="units not declared")
        return html.Div(
            [
                html.Div(
                    [
                        html.Div("Candidate details", className="step-eyebrow"),
                        html.H3(str(candidate.get("display_id") or "Candidate"), className="section-heading"),
                        html.P(f"{_friendly_project_name(candidate.get('project'))} - prediction units: {units}", className="chart-description"),
                    ]
                ),
                html.Div(
                    [
                        _metric("Rank", candidate.get("rank")),
                        _metric("Prediction", candidate.get("prediction")),
                        _metric("Uncertainty", candidate.get("uncertainty")),
                        _metric("Acquisition", candidate.get("acquisition_score")),
                    ],
                    className="metric-grid",
                ),
                html.Div([_status_badge(_human_label(candidate.get("training_distribution_status"))), *[_status_badge(item, tone="warning") for item in risks]], className="product-badge-row"),
                html.Code(str(candidate.get("sequence") or "Sequence not indexed"), className="candidate-sequence"),
                html.P(_readable_reason(candidate.get("ranking_reason")) or "No ranking explanation was recorded.", className="chart-description"),
            ]
        )

    @app.callback(
        Output("simulation-sequence-summary", "children"),
        Output("simulation-action-summary", "children"),
        Input("sim-parent", "value"),
        Input("sim-mode", "value"),
        Input("sim-num-candidates", "value"),
        Input("sim-model", "value"),
    )
    def update_simulation_input_summary(
        parent: str | None,
        mode: str | None,
        num_candidates: int | None,
        model_name: str | None,
    ) -> tuple[Any, list[Any]]:
        count = max(1, int(_value_or_default(num_candidates, 96)))
        scorer_label = "Heuristic scoring selected" if model_name == DEMO_SCORER else "Model-backed scoring selected · verification required"
        return (
            _sequence_input_summary(parent, mode),
            [html.Strong(f"Up to {count:,} candidates"), html.Span(scorer_label)],
        )

    @app.callback(
        Output("sim-visuals", "options"),
        Output("sim-visuals", "value"),
        Output("simulation-visual-guidance", "children"),
        Output("simulation-mode-guidance", "children"),
        Input("sim-mode", "value"),
        Input("sim-visual-preset", "value"),
    )
    def update_simulation_visual_choices(mode: str | None, preset: str | None) -> tuple[Any, ...]:
        mode_text = {
            "Mask-fill": "Fills N or [MASK] positions while keeping declared bases fixed.",
            "Random mutagenesis": "Randomly changes declared bases at the requested rate; this is not a systematic saturation scan.",
            "Diversity sampling": "Generates a broader pool and retains variants spread across sequence space; observed distances are reported explicitly.",
        }.get(str(mode), "Choose a generation mode.")
        visuals = _simulation_visual_defaults(mode, preset)
        selected_labels = [dict((item["value"], item["label"]) for item in _simulation_visual_options(mode))[key] for key in visuals]
        return (
            _simulation_visual_options(mode),
            visuals,
            html.Span("Included: " + ", ".join(selected_labels) + ". You may change the checklist after choosing a preset."),
            html.Span(mode_text),
        )

    @app.callback(
        Output("sim-forbidden-motifs", "value"),
        Input("sim-constraint-preset", "value"),
    )
    def update_simulation_constraint_preset(preset: str | None) -> str:
        return "GGTCTC, CGTCTC" if preset == "golden_gate" else ""

    @app.callback(
        Output("custom-model-path", "value"),
        Output("sim-model", "value"),
        Output("custom-model-picker-status", "children"),
        Input("browse-custom-model-button", "n_clicks"),
        State("custom-model-path", "value"),
        prevent_initial_call=True,
    )
    def browse_custom_model(browse_clicks: int, current_value: str | None) -> tuple[Any, Any, Any]:
        if not browse_clicks:
            raise PreventUpdate
        initial = Path(str(current_value)).resolve() if str(current_value or "").strip() else _model_search_roots()[0]
        try:
            selected = _choose_directory(title="Select AssayReady model weights folder", initial_dir=initial)
            if not selected:
                return no_update, no_update, _status("Model-folder selection canceled.")
            return str(Path(selected).resolve()), CUSTOM_SCORER, _status("Model folder selected. Choose its assay head, then verify.", kind="success")
        except Exception as exc:
            return no_update, no_update, _status(f"Could not open the model-folder picker: {exc}", kind="warning")

    @app.callback(
        Output("recommended-model-install-status", "children"),
        Input("install-model-runtime-button", "n_clicks"),
        running=[
            (Output("install-model-runtime-button", "disabled"), True, False),
            (Output("install-model-runtime-button", "children"), "Installing modeling support...", "Install modeling support"),
        ],
        prevent_initial_call=True,
    )
    def install_model_runtime(runtime_clicks: int) -> Any:
        if not runtime_clicks:
            raise PreventUpdate
        requirements = [
            "einops>=0.8,<1", "huggingface-hub>=0.23,<2", "peft>=0.15,<1",
            "safetensors>=0.4,<1", "tokenizers>=0.22,<0.24", "torch>=2.12,<2.13",
            "transformers>=4.50,<6",
        ]
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *requirements],
                check=False,
                capture_output=True,
                text=True,
                timeout=1800,
            )
            importlib.invalidate_caches()
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "pip exited without details").strip().splitlines()[-1]
                return _status(f"Modeling-support installation failed: {detail}", kind="danger")
            return _status("Modeling support is installed. Continue with Download recommended model.", kind="success")
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _status(f"Modeling-support installation stopped: {exc}", kind="danger")

    @app.callback(
        Output("custom-model-path", "value", allow_duplicate=True),
        Output("sim-model", "value", allow_duplicate=True),
        Output("recommended-model-install-status", "children", allow_duplicate=True),
        Input("install-recommended-model-button", "n_clicks"),
        running=[
            (Output("install-recommended-model-button", "disabled"), True, False),
            (Output("install-recommended-model-button", "children"), "Downloading and verifying...", "Download recommended model"),
        ],
        prevent_initial_call=True,
    )
    def install_recommended_model(install_clicks: int) -> tuple[Any, Any, Any]:
        if not install_clicks:
            raise PreventUpdate
        bundle = _recommended_model_bundle()
        try:
            missing = [
                name
                for name in ["torch", "transformers", "tokenizers", "einops", "huggingface_hub", "safetensors"]
                if importlib.util.find_spec(name) is None
            ]
            if missing:
                return no_update, no_update, _status(
                    "Modeling support is missing: " + ", ".join(missing) + ". Click Install modeling support first.",
                    kind="warning",
                )
            ensure_dnabert2_scaffold(bundle)
            manifest = download_dnabert2(bundle / "weights")
            return (
                str((bundle / "weights").resolve()),
                CUSTOM_SCORER,
                _status(
                    f"Downloaded and authenticated {manifest.get('model_name', 'DNABERT-2')} at revision {str(manifest.get('revision') or '')[:12]}.... Next, upload training data and create the assay head.",
                    kind="success",
                ),
            )
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            return no_update, no_update, _status(f"Model installation stopped safely: {exc}", kind="danger")

    @app.callback(
        Output("scorer-training-upload-state", "data"),
        Output("scorer-training-preview", "children"),
        Output("scorer-training-sequence-col", "options"),
        Output("scorer-training-sequence-col", "value"),
        Output("scorer-training-target-col", "options"),
        Output("scorer-training-target-col", "value"),
        Input("scorer-training-upload", "contents"),
        State("scorer-training-upload", "filename"),
        prevent_initial_call=True,
    )
    def load_scorer_training_table(contents: str | None, filename: str | None) -> tuple[Any, ...]:
        if not contents:
            raise PreventUpdate
        try:
            path, blob = _save_upload(contents, str(filename or "scorer_training.csv"), kind="scorer_training")
            frame = _read_table_bytes(blob, str(filename or ""))
            if frame.empty:
                raise ValueError("The training table has no rows.")
            if len(frame) < 20:
                raise ValueError("At least 20 training rows are required.")
            columns = [str(column) for column in frame.columns]
            sequence_col = _first_valid_col(columns, None, ["sequence", "dna", "promoter_sequence", "insert_sequence"])
            target_col = _first_valid_col(columns, None, ["measured_value", "measured_expression", "expression", "activity", "fluorescence", "target"])
            if not sequence_col or not target_col or sequence_col == target_col:
                target_col = next((column for column in columns if column != sequence_col), None)
            normalized_path = path.with_suffix(".training.csv")
            frame.to_csv(normalized_path, index=False)
            label_values = {
                column: sorted({str(value) for value in frame[column].dropna().tolist()}, key=str.casefold)
                for column in columns
                if int(frame[column].nunique(dropna=True)) <= 50
            }
            state = {
                "path": str(normalized_path.resolve()),
                "filename": str(filename or path.name),
                "rows": int(len(frame)),
                "columns": columns,
                "label_values": label_values,
            }
            options = _dropdown_options(columns)
            preview = html.Div(
                [
                    _status(f"Loaded {len(frame):,} training rows and {len(columns):,} columns.", kind="success"),
                    html.Details(
                        [html.Summary("Preview first 5 rows", className="details-summary"), _data_table(frame.head(5), page_size=5, height=190, compact=True)],
                        className="data-preview-details",
                    ),
                ]
            )
            return state, preview, options, sequence_col, options, target_col
        except (OSError, ValueError, UnicodeError, pd.errors.ParserError) as exc:
            return None, _status(f"Could not use that training table: {exc}", kind="warning"), [], None, [], None

    @app.callback(
        Output("scorer-training-positive-label", "options"),
        Output("scorer-training-positive-label", "value"),
        Input("scorer-training-target-col", "value"),
        Input("scorer-training-task-type", "value"),
        State("scorer-training-upload-state", "data"),
    )
    def update_scorer_positive_labels(
        target_col: str | None,
        task_type: str | None,
        upload_state: dict[str, Any] | None,
    ) -> tuple[list[dict[str, str]], str | None]:
        if task_type != "classification" or not target_col or not upload_state:
            return [], None
        labels = [str(value) for value in (upload_state.get("label_values") or {}).get(str(target_col), [])]
        options = [{"label": value, "value": value} for value in labels]
        return options, labels[-1] if len(labels) == 2 else None

    @app.callback(
        Output("custom-model-path", "value", allow_duplicate=True),
        Output("custom-head-path", "value", allow_duplicate=True),
        Output("sim-model", "value", allow_duplicate=True),
        Output("scorer-training-status", "children"),
        Output("scorer-verification-state", "data"),
        Input("train-scorer-button", "n_clicks"),
        State("custom-model-path", "value"),
        State("scorer-training-upload-state", "data"),
        State("scorer-training-sequence-col", "value"),
        State("scorer-training-target-col", "value"),
        State("scorer-training-task-type", "value"),
        State("scorer-training-objective", "value"),
        State("scorer-training-positive-label", "value"),
        State("scorer-training-confirmation", "value"),
        running=[
            (Output("train-scorer-button", "disabled"), True, False),
            (Output("train-scorer-button", "children"), "Creating and checking scorer...", "Create and verify scorer"),
        ],
        prevent_initial_call=True,
    )
    def train_scorer_from_table(
        n_clicks: int,
        model_path: str | None,
        upload_state: dict[str, Any] | None,
        sequence_col: str | None,
        target_col: str | None,
        task_type: str | None,
        objective_direction: str | None,
        positive_label: str | None,
        confirmation: list[str] | None,
    ) -> tuple[Any, Any, Any, Any, Any]:
        if not n_clicks:
            raise PreventUpdate
        try:
            if "confirmed" not in (confirmation or []):
                raise ValueError("Confirm that the upload contains training rows only.")
            if not upload_state or not upload_state.get("path"):
                raise ValueError("Choose a training table first.")
            if not sequence_col or not target_col or sequence_col == target_col:
                raise ValueError("Choose different sequence and measured-outcome columns.")
            selected_model = Path(str(model_path or "")).expanduser().resolve()
            if not str(model_path or "").strip():
                raise ValueError("Download the recommended model or choose a compatible weights folder first.")
            if (selected_model / "assayready_model_manifest.json").is_file():
                weights_dir = selected_model
                bundle_root = selected_model.parent
            elif (selected_model / "weights" / "assayready_model_manifest.json").is_file():
                bundle_root = selected_model
                weights_dir = selected_model / "weights"
            else:
                raise ValueError("The selected folder does not contain an AssayReady model manifest.")
            manifest = json.loads((weights_dir / "assayready_model_manifest.json").read_text(encoding="utf-8"))
            if manifest.get("repository") != DNABERT2_REPO or manifest.get("revision") != DNABERT2_REVISION:
                raise ValueError("Direct in-app training supports only the authenticated recommended DNABERT-2 revision.")
            verified_files = verify_manifest_file_records(weights_dir, manifest)
            verify_pinned_dnabert2_snapshot(weights_dir, verified_files=verified_files)
            label_values = (upload_state.get("label_values") or {}).get(str(target_col), [])
            if task_type == "classification" and len(label_values) != 2:
                raise ValueError("Classification requires exactly two distinct outcome categories.")
            if task_type == "classification" and not positive_label:
                raise ValueError("Choose which category is positive.")
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            head_dir = bundle_root / "trained_heads" / f"{_slug(str(target_col))}_{stamp}"
            from .model_bundles.dnabert2_117m.train_head import train_frozen_head

            trained = train_frozen_head(
                model_dir=weights_dir,
                training=Path(str(upload_state["path"])),
                output_dir=head_dir,
                sequence_col=str(sequence_col),
                target_col=str(target_col),
                task_type=str(task_type or "regression"),
                positive_label=str(positive_label) if positive_label is not None else None,
                objective_direction=str(objective_direction or "maximize"),
            )
            scorer = load_verified_foundation_scorer(weights_dir, head_dir, cache_dir=_output_root() / "_model_cache")
            health = scorer.health_check()
            return (
                str(weights_dir.resolve()),
                str(head_dir.resolve()),
                CUSTOM_SCORER,
                _status(
                    f"Scorer ready. Trained {trained['ensemble_size']} frozen-backbone heads on {trained['training_rows']:,} rows; health check prediction={health['prediction']:.4g}, uncertainty={health['uncertainty']:.4g}. It is selected above and ready to use.",
                    kind="success",
                ),
                {
                    "model_path": str(weights_dir.resolve()),
                    "head_path": str(head_dir.resolve()),
                    "caption": scorer.provenance.caption,
                    "device": scorer.provenance.device,
                    "model_sha256": scorer.provenance.model_weights_sha256,
                    "head_sha256": scorer.provenance.head_sha256,
                },
            )
        except (ModelScoringError, OSError, ValueError, RuntimeError, ImportError, json.JSONDecodeError) as exc:
            return no_update, no_update, no_update, _status(f"Scorer creation stopped safely: {exc}", kind="danger"), no_update

    @app.callback(
        Output("custom-head-path", "value"),
        Output("custom-head-picker-status", "children"),
        Input("browse-custom-head-button", "n_clicks"),
        State("custom-head-path", "value"),
        State("custom-model-path", "value"),
        prevent_initial_call=True,
    )
    def browse_custom_head(n_clicks: int, current_value: str | None, model_path: str | None) -> tuple[Any, Any]:
        if not n_clicks:
            raise PreventUpdate
        if str(current_value or "").strip():
            initial = Path(str(current_value)).resolve()
        elif str(model_path or "").strip():
            initial = Path(str(model_path)).resolve().parent / "task_head"
        else:
            initial = _model_search_roots()[0]
        try:
            selected = _choose_directory(title="Select AssayReady assay-head folder", initial_dir=initial)
            if not selected:
                return no_update, _status("Head-folder selection canceled.")
            return str(Path(selected).resolve()), _status("Assay-head folder selected. Verify the scorer before use.", kind="success")
        except Exception as exc:
            return no_update, _status(f"Could not open the head-folder picker: {exc}", kind="warning")

    @app.callback(
        Output("model-installation-panel", "children"),
        Input("sim-model", "value"),
        Input("custom-model-path", "value"),
        Input("custom-head-path", "value"),
        Input("verify-scorer-button", "n_clicks"),
        Input("scorer-verification-state", "data"),
        prevent_initial_call=False,
    )
    def update_model_installation(
        model_name: str | None,
        custom_model_path: str | None,
        custom_head_path: str | None,
        verify_clicks: int,
        verification_state: dict[str, Any] | None,
    ) -> Any:
        if not model_name or model_name == DEMO_SCORER:
            return _status(
                "Heuristic scoring is active. Choose a trained local scorer for model-backed ranking.",
                kind="info",
            )
        try:
            selected_path = _resolve_scorer_path(model_name, custom_model_path)
            if not selected_path:
                raise ModelScoringError("Choose a trained local scorer or use heuristic scoring.")
            resolved_head_path = custom_head_path if model_name == CUSTOM_SCORER and str(custom_head_path or "").strip() else None
            if (
                verification_state
                and Path(str(verification_state.get("model_path") or "")).resolve() == Path(selected_path).resolve()
                and resolved_head_path
                and Path(str(verification_state.get("head_path") or "")).resolve() == Path(resolved_head_path).resolve()
            ):
                return _status(
                    f"Ready: {verification_state.get('caption')} on {verification_state.get('device')}. Integrity and forward-pass checks completed during setup.",
                    kind="success",
                )
            head_dir = Path(str(resolved_head_path)).resolve() if resolved_head_path else Path(selected_path).parent / "task_head"
            structural_files = [
                Path(selected_path) / "assayready_model_manifest.json",
                Path(selected_path) / "model.safetensors",
                head_dir / "head_manifest.json",
                head_dir / "head.npz",
            ]
            missing = [path.name for path in structural_files if not path.is_file()]
            if missing:
                return _status("Scorer setup is incomplete. Missing: " + ", ".join(missing) + ". Open the setup guide below.", kind="warning")
            if callback_context.triggered_id != "verify-scorer-button" or not verify_clicks:
                return _status(f"A structurally complete bundle was found at {selected_path}. Click Verify scorer for integrity checks and a forward pass.")
            scorer = load_verified_foundation_scorer(
                selected_path,
                resolved_head_path,
                cache_dir=_output_root() / "_model_cache",
            )
            provenance = scorer.provenance
            health = scorer.health_check()
            positive = f", positive label {provenance.positive_label!r}" if provenance.positive_label is not None else ""
            return _status(
                f"Verified and executed {provenance.model_name} on {provenance.device} with the {provenance.target_col} "
                f"{provenance.task_type} head ({provenance.objective_direction}{positive}; model {provenance.model_weights_sha256[:12]}..., "
                f"head {provenance.head_sha256[:12]}...). Health check: prediction={health['prediction']:.4g}, uncertainty={health['uncertainty']:.4g}.",
                kind="success",
            )
        except (ModelScoringError, OSError, ValueError, ImportError, RuntimeError) as exc:
            return _status(f"Scorer is not ready: {exc}", kind="warning")


    @app.callback(
        Output("simulation-result", "children"),
        Input("run-simulation-button", "n_clicks"),
        State("sim-mode", "value"),
        State("sim-model", "value"),
        State("sim-num-candidates", "value"),
        State("sim-context", "value"),
        State("sim-parent", "value"),
        State("sim-mutation-rate", "value"),
        State("sim-gc-low", "value"),
        State("sim-gc-high", "value"),
        State("sim-max-homopolymer", "value"),
        State("sim-forbidden-motifs", "value"),
        State("sim-check-reverse-complements", "value"),
        State("sim-seed", "value"),
        State("sim-project", "value"),
        State("sim-plate-size", "value"),
        State("sim-control-wells", "value"),
        State("sim-plate-seed", "value"),
        State("custom-model-path", "value"),
        State("custom-head-path", "value"),
        State("sim-visuals", "value"),
        running=[
            (Output("run-simulation-button", "disabled"), True, False),
            (Output("run-simulation-button", "children"), "Generating and scoring...", "Generate draft designs"),
            (Output("simulation-running-status", "hidden"), False, True),
        ],
        prevent_initial_call=True,
    )
    def run_simulation(
        n_clicks: int,
        mode: str,
        model_name: str,
        num_candidates: int,
        context_multiplier: float,
        parent: str,
        mutation_rate: float,
        gc_low: float,
        gc_high: float,
        max_homopolymer: int,
        forbidden_motifs_text: str | None,
        reverse_complement_options: list[str] | None,
        seed: int,
        project: str,
        plate_size: int,
        control_wells: int,
        plate_seed: int,
        custom_model_path: str | None,
        custom_head_path: str | None,
        selected_visuals: list[str] | None,
    ) -> Any:
        if not n_clicks:
            raise PreventUpdate
        if float(gc_low) > float(gc_high):
            return _status("Min GC must be less than or equal to Max GC.", kind="warning")
        try:
            forbidden_motifs = _parse_sequence_motifs(forbidden_motifs_text)
        except ValueError as exc:
            return _status(str(exc), kind="warning")
        check_reverse_complements = "both_strands" in (reverse_complement_options or [])

        requested_candidates = int(_value_or_default(num_candidates, 96))
        parent_compact = "".join(str(parent or "").upper().replace("[MASK]", "N").split())
        invalid_symbols = sorted(set(parent_compact) - set("ACGTN"))
        if not parent_compact or invalid_symbols:
            return _status(
                f"Enter a DNA sequence containing only A, C, G, T, or N. Invalid symbols: {invalid_symbols}",
                kind="warning",
            )
        if mode == "Mask-fill" and "N" not in parent_compact:
            return _status(
                "Mask-fill needs at least one N (or [MASK]) position. Add masked bases, or switch to Random mutagenesis or Diversity sampling.",
                kind="warning",
            )
        generation_warnings: list[str] = []
        if mode == "Mask-fill":
            theoretical_variants = 4 ** parent_compact.count("N")
            if theoretical_variants < requested_candidates:
                generation_warnings.append(
                    f"The mask contains only {parent_compact.count('N')} variable position(s), so at most {theoretical_variants:,} unique variants exist for the requested {requested_candidates:,}."
                )

        model_is_demo = model_name == DEMO_SCORER
        scorer = None
        scorer_provenance: dict[str, Any] | None = None
        if not model_is_demo:
            try:
                selected_path = _resolve_scorer_path(model_name, custom_model_path)
                if not selected_path:
                    raise ModelScoringError("Choose a trained local scorer or heuristic scoring.")
                resolved_head_path = custom_head_path if model_name == CUSTOM_SCORER and str(custom_head_path or "").strip() else None
                scorer = load_verified_foundation_scorer(
                    selected_path,
                    resolved_head_path,
                    cache_dir=_output_root() / "_model_cache",
                )
                scorer_provenance = scorer.provenance.to_dict()
            except (ModelScoringError, OSError, ValueError, ImportError, RuntimeError) as exc:
                return _status(
                    f"Model-backed scoring stopped before artifact creation: {exc}. "
                    "Select heuristic scoring if heuristic output is intended, or open the scorer setup guide.",
                    kind="danger",
                )
        rng = random.Random(int(_value_or_default(seed, 13)))
        generated_pool: list[str] = []
        seen: set[str] = set()
        attempts = 0
        pool_target = requested_candidates * 6 if mode == "Diversity sampling" else requested_candidates
        max_attempts = pool_target * 80
        while len(generated_pool) < pool_target and attempts < max_attempts:
            attempts += 1
            if mode == "Diversity sampling":
                effective_rate = min(0.5, float(_value_or_default(mutation_rate, 0.08)) * 2.5 + 0.04)
            elif mode == "Random mutagenesis":
                effective_rate = float(_value_or_default(mutation_rate, 0.08))
            else:
                effective_rate = float(_value_or_default(mutation_rate, 0.08)) * 0.35
            seq = _fill_masked_sequence(parent_compact, rng, str(mode), effective_rate)
            if not seq or seq in seen:
                continue
            seen.add(seq)
            generated_pool.append(seq)
        selected_sequences = (
            _select_diverse_sequences(generated_pool, parent=parent_compact.replace("N", "A"), count=requested_candidates)
            if mode == "Diversity sampling"
            else generated_pool[:requested_candidates]
        )
        model_identity = scorer.provenance.caption if scorer is not None else "Heuristic scoring"
        candidates = [
            {"design_id": f"design_sim_{index + 1:04d}", "model": model_identity, "mode": mode, "sequence": sequence}
            for index, sequence in enumerate(selected_sequences)
        ]
        rows = []
        target_gc = (float(gc_low), float(gc_high))
        for candidate in candidates:
            seq = str(candidate["sequence"])
            scores = _heuristic_sequence_scores(
                seq,
                target_gc,
                int(_value_or_default(max_homopolymer, 6)),
                rng,
                forbidden_motifs=forbidden_motifs,
                check_reverse_complements=check_reverse_complements,
            )
            rows.append({**candidate, **scores})
        frame = pd.DataFrame(rows)
        if frame.empty:
            return _status("No candidates were generated. Check the sequence prompt and filters.", kind="warning")
        if len(frame) < requested_candidates:
            generation_warnings.append(
                f"Generated {len(frame):,} of {requested_candidates:,} requested unique candidates before the duplicate-attempt limit was reached."
            )
        if scorer is not None:
            try:
                token_lengths = scorer.sequence_token_lengths(frame["sequence"].astype(str).tolist())
                max_tokens = max(token_lengths, default=0)
                if max_tokens > scorer.provenance.max_length:
                    return _status(
                        f"Model-backed scoring stopped to prevent silent truncation: at least one candidate uses {max_tokens:,} tokens, "
                        f"but this head was configured for at most {scorer.provenance.max_length:,}. Shorten the parent sequence.",
                        kind="warning",
                    )
                predictions, uncertainties = scorer.score_sequences(frame["sequence"].astype(str).tolist())
            except (ModelScoringError, OSError, ValueError, RuntimeError) as exc:
                return _status(
                    f"Model-backed scoring failed and no heuristic fallback was used: {exc}",
                    kind="danger",
                )
            frame["predicted_activity"] = predictions
            frame["model_uncertainty"] = uncertainties
            frame["assay_prediction"] = predictions
            frame["assay_uncertainty"] = uncertainties
            frame["objective_direction"] = scorer.provenance.objective_direction
            frame["scoring_backend"] = scorer.provenance.caption
            frame["input_tokens"] = token_lengths
        objective_direction = scorer.provenance.objective_direction if scorer is not None else "maximize"
        frame["acquisition_score"] = _acquisition_utility(
            frame["predicted_activity"],
            frame["model_uncertainty"],
            exploration_weight=float(_value_or_default(context_multiplier, 2.0)),
            objective_direction=objective_direction,
        )
        frame["objective_direction"] = objective_direction
        frame["input_nt"] = frame["sequence"].astype(str).str.len()
        frame = frame.sort_values("acquisition_score", ascending=False).reset_index(drop=True)
        frame.insert(0, "rank", range(1, len(frame) + 1))

        allowed_visuals = set(SIMULATION_VISUAL_LABELS)
        requested_visuals = selected_visuals if selected_visuals is not None else _simulation_visual_defaults(mode, "recommended")
        visual_keys = [key for key in requested_visuals if key in allowed_visuals]
        p_seed = int(_value_or_default(plate_seed, _value_or_default(seed, 13)))
        plate_plan = (
            _build_plate_plan(
                frame,
                plate_size=int(_value_or_default(plate_size, 96)),
                control_wells=int(_value_or_default(control_wells, 4)),
                seed=p_seed,
            )
            if "plate" in visual_keys
            else pd.DataFrame()
        )
        eligible_count = int(frame["passes_basic_filters"].astype(bool).sum())
        placed_count = int((~plate_plan["role"].astype(str).str.endswith("control")).sum()) if not plate_plan.empty else 0
        if not plate_plan.empty and placed_count < eligible_count:
            generation_warnings.append(
                f"The plate has room for {placed_count:,} eligible candidates after reserved controls; {eligible_count - placed_count:,} eligible candidate(s) remain ranked but unplated."
            )

        artifact_params = {
            "model_name": model_identity,
            "generation_mode": mode,
            "num_candidates": requested_candidates,
            "generated_candidates": int(len(frame)),
            "context_multiplier": float(_value_or_default(context_multiplier, 2.0)),
            "mutation_rate": float(_value_or_default(mutation_rate, 0.08)),
            "gc_low": float(gc_low),
            "gc_high": float(gc_high),
            "max_homopolymer": int(_value_or_default(max_homopolymer, 6)),
            "forbidden_motifs": forbidden_motifs,
            "check_reverse_complements": check_reverse_complements,
            "seed": int(_value_or_default(seed, 13)),
            "plate_size": int(_value_or_default(plate_size, 96)),
            "control_wells": int(_value_or_default(control_wells, 4)),
            "plate_seed": p_seed,
            "model_backed": scorer_provenance is not None,
            "objective_direction": scorer.provenance.objective_direction if scorer is not None else "maximize",
            "selected_visuals": visual_keys,
        }
        figure_registry: dict[str, Any] = {}
        if "landscape" in visual_keys:
            figure_registry["landscape"] = simulation_landscape_figure(
                frame,
                model_backed=scorer is not None,
                target_label=scorer.provenance.target_col if scorer is not None else None,
            )
        if "generation" in visual_keys:
            figure_registry["generation"] = simulation_generation_figure(frame, parent_sequence=parent_compact, mode=mode)
        if "constraints" in visual_keys:
            figure_registry["constraints"] = simulation_constraints_figure(
                frame,
                gc_low=float(gc_low),
                gc_high=float(gc_high),
                max_homopolymer=int(_value_or_default(max_homopolymer, 6)),
            )
        if not plate_plan.empty:
            figure_registry["plate"] = plate_layout_figure(plate_plan, plate_size=int(_value_or_default(plate_size, 96)))
        selected_figures = {key: figure_registry[key] for key in visual_keys if key in figure_registry}
        artifact_dir, summary = _save_simulation_artifacts(
            project=str(_value_or_default(project, "dash_sequence_simulation")),
            frame=frame,
            plate_plan=plate_plan,
            params=artifact_params,
            scoring_caption=scorer.provenance.caption if scorer is not None else "demo_heuristic_scorer",
            figures=selected_figures,
            scorer_provenance=scorer_provenance,
            generation_warnings=generation_warnings,
        )
        artifact_opts = _artifact_options(summary, "simulation_report.md")
        display_cols = [
            column
            for column in ["rank", "design_id", "predicted_activity", "model_uncertainty", "acquisition_score", "gc_fraction", "max_homopolymer"]
            if column in frame.columns
        ]
        candidate_payload = _json_clean(
            {
                "rows": frame.to_dict("records"),
                "plate_rows": plate_plan.to_dict("records") if not plate_plan.empty else [],
                "display_cols": display_cols,
                "parent_sequence": parent_compact,
                "gc_low": float(gc_low),
                "gc_high": float(gc_high),
                "max_homopolymer": int(_value_or_default(max_homopolymer, 6)),
                "model_backed": scorer is not None,
                "target_label": scorer.provenance.target_col if scorer is not None else None,
                "model_name": model_identity,
                "objective_direction": objective_direction,
                "scorer_provenance": scorer_provenance,
            }
        )
        landscape_description = (
            "Predictions and ensemble disagreement come from the verified local model/head pair; marker shape shows sequence-constraint eligibility."
            if scorer is not None
            else "Activity and uncertainty are heuristic scoring proxies; marker shape shows sequence-constraint eligibility."
        )
        preview_status = _status(f"{len(frame):,} shown \u00b7 {eligible_count:,} eligible under view settings \u00b7 saved run unchanged", kind="info")
        return html.Div(
            [
                dcc.Store(id="simulation-candidate-store", data=candidate_payload),
                dcc.Store(id="simulation-selection-store", data=None),
                dcc.Store(id="simulation-inspector-a11y"),
                _planning_status_strip(eligible=eligible_count, generated=len(frame), model_backed=scorer is not None),
                _generation_funnel(requested=requested_candidates, generated=len(frame), eligible=eligible_count),
                *[_status(message, kind="warning") for message in generation_warnings],
                html.Div(
                    [
                        _metric("Eligibility rate", f"{float(frame['passes_basic_filters'].mean()):.0%}", note="Under saved sequence constraints"),
                        _metric("Top ranking score", f"{frame['acquisition_score'].max():.3f}", note="Model-backed utility" if scorer is not None else "Heuristic scoring"),
                        _metric("Placed in draft layout", placed_count if not plate_plan.empty else "Not requested", note="Reserved controls reduce capacity" if not plate_plan.empty else None),
                    ],
                    className="metric-grid metric-grid-three",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Interactive candidate explorer", className="row-explorer-eyebrow"),
                                html.H3("Inspect candidates", className="row-explorer-title"),
                                html.P("Click a point or table row to open its evidence panel. View settings only change this screen.", className="chart-description"),
                            ]
                        ),
                        html.Div(
                            [
                                html.Div(
                                    _field(
                                        "Show candidates",
                                        dcc.Dropdown(
                                            id="simulation-pass-filter",
                                            options=[
                                                {"label": "All candidates", "value": "all"},
                                                {"label": "Eligible", "value": "pass"},
                                                {"label": "Excluded", "value": "fail"},
                                            ],
                                            value="all",
                                            clearable=False,
                                        ),
                                    ),
                                    className="simulation-show-filter",
                                ),
                                html.Details(
                                    [
                                        html.Summary("Try different sequence constraints", className="details-summary"),
                                        html.P("These view settings do not change the saved run. Rerun to save different sequence constraints.", className="chart-description"),
                                        html.Div(
                                            [
                                                _field("Min GC", dcc.Input(id="simulation-preview-gc-low", type="number", min=0, max=1, step=0.01, value=float(gc_low), style=INPUT_STYLE)),
                                                _field("Max GC", dcc.Input(id="simulation-preview-gc-high", type="number", min=0, max=1, step=0.01, value=float(gc_high), style=INPUT_STYLE)),
                                                _field("Max homopolymer", dcc.Input(id="simulation-preview-max-homopolymer", type="number", min=1, max=20, step=1, value=int(_value_or_default(max_homopolymer, 6)), style=INPUT_STYLE)),
                                            ],
                                            className="simulation-preview-filter-grid",
                                        ),
                                    ],
                                    className="simulation-preview-details",
                                ),
                            ],
                            className="simulation-filter-toolbar",
                        ),
                        html.Div(preview_status, id="simulation-preview-status"),
                        html.Div(
                            _graph_card(
                                selected_figures.get("landscape", {}),
                                landscape_description,
                                graph_id="simulation-landscape-graph",
                                interactive=True,
                            ),
                            id="simulation-landscape-container",
                            style={} if "landscape" in visual_keys else {"display": "none"},
                        ),
                        html.Details(
                            [
                                html.Summary(f"Browse candidate table ({len(frame):,})", className="result-details-summary"),
                                html.Div(
                                    [
                                        html.P("Select a row to highlight its point and open the evidence panel.", className="chart-description"),
                                        _data_table(frame[display_cols], page_size=12, height=390, table_id="simulation-candidate-table", selectable=True),
                                    ],
                                    className="simulation-table-panel",
                                ),
                            ],
                            className="result-details simulation-candidate-table-details",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div("Candidate evidence", className="inspector-eyebrow"),
                                        html.Button("Close", id="simulation-inspector-close", n_clicks=0, className="inspector-close-button", **{"aria-label": "Close candidate inspector"}),
                                    ],
                                    className="inspector-header",
                                ),
                                html.Div(
                                    _simulation_candidate_inspector(candidate_payload, None),
                                    id="simulation-inspector-body",
                                    className="observation-inspector-body",
                                ),
                            ],
                            id="simulation-inspector",
                            className="observation-inspector",
                            role="dialog",
                            **{"aria-modal": "true", "aria-hidden": "true", "aria-label": "Candidate evidence inspector"},
                        ),
                    ],
                    className="simulation-explorer results-enter",
                ),
                html.Div(
                    [
                        _graph_card(
                            selected_figures[key],
                            {
                                "landscape": landscape_description,
                                "generation": "This view describes changes made by the selected generator and does not infer biological effects.",
                                "constraints": "Dashed lines show requested GC and homopolymer limits and reveal restrictive generation settings.",
                                "plate": "This full plate is an unapproved planning preview. Empty wells are explicit and scientist review remains required.",
                            }[key],
                            class_name="constraint-distribution-card" if key == "constraints" else None,
                        )
                        for key in visual_keys
                        if key in selected_figures and key != "landscape"
                    ],
                    className="visualization-grid simulation-visualizations results-enter",
                ) if any(key != "landscape" for key in selected_figures) else None,
                html.Details(
                    [
                        html.Summary("Draft well-layout table", className="result-details-summary"),
                        _status("Not approved for execution. Control identities, replicates, quotas, balancing, and position effects require scientist review.", kind="warning"),
                        _data_table(plate_plan, page_size=12, height=360),
                    ],
                    className="result-details",
                ) if not plate_plan.empty else None,
                html.Details(
                    [
                        html.Summary("Downloads and draft files", className="result-details-summary"),
                        html.Div(
                            [
                                dcc.Dropdown(id="simulation-artifact-select", options=artifact_opts, value=artifact_opts[0]["value"] if artifact_opts else None, clearable=False),
                                html.Button("Download selected artifact", id="simulation-download-button", n_clicks=0, style={**BUTTON_STYLE, "marginTop": "10px"}),
                                dcc.Download(id="simulation-download"),
                                html.Details([html.Summary("Local artifact location", className="details-summary"), html.Code(str(artifact_dir), className="path-value")], className="maintenance-details"),
                            ],
                            className="artifact-download-panel",
                        ),
                    ],
                    className="result-details",
                ),
                _status(str(summary.get("run_store_warning")), kind="warning") if summary.get("run_store_warning") else None,
            ]
        )

    @app.callback(
        Output("simulation-selection-store", "data"),
        Input("simulation-landscape-graph", "clickData", allow_optional=True),
        Input("simulation-candidate-table", "cellClicked", allow_optional=True),
        Input("simulation-inspector-close", "n_clicks", allow_optional=True),
        prevent_initial_call=True,
    )
    def select_simulation_candidate(
        graph_click: dict[str, Any] | None,
        table_click: dict[str, Any] | None,
        _close_clicks: int | None,
    ) -> str | None:
        triggered = callback_context.triggered_id
        if triggered == "simulation-inspector-close":
            return None
        event = graph_click if triggered == "simulation-landscape-graph" else table_click
        candidate_id = _simulation_candidate_id(event)
        if not candidate_id:
            raise PreventUpdate
        return candidate_id

    @app.callback(
        Output("simulation-landscape-graph", "figure"),
        Output("simulation-candidate-table", "rowData"),
        Output("simulation-candidate-table", "selectedRows"),
        Output("simulation-preview-status", "children"),
        Output("simulation-inspector-body", "children"),
        Output("simulation-inspector", "className"),
        Output("simulation-inspector", "aria-hidden"),
        Input("simulation-selection-store", "data", allow_optional=True),
        Input("simulation-pass-filter", "value", allow_optional=True),
        Input("simulation-preview-gc-low", "value", allow_optional=True),
        Input("simulation-preview-gc-high", "value", allow_optional=True),
        Input("simulation-preview-max-homopolymer", "value", allow_optional=True),
        State("simulation-candidate-store", "data", allow_optional=True),
        prevent_initial_call=True,
    )
    def update_simulation_explorer(
        candidate_id: str | None,
        pass_filter: str | None,
        gc_low_preview: float | None,
        gc_high_preview: float | None,
        homopolymer_preview: int | None,
        payload: dict[str, Any] | None,
    ) -> tuple[Any, ...]:
        if not payload or not payload.get("rows"):
            raise PreventUpdate
        gc_min = float(gc_low_preview if gc_low_preview is not None else payload.get("gc_low", 0.0))
        gc_max = float(gc_high_preview if gc_high_preview is not None else payload.get("gc_high", 1.0))
        max_run = int(homopolymer_preview if homopolymer_preview is not None else payload.get("max_homopolymer", 20))
        frame = pd.DataFrame(payload["rows"])
        preview = _simulation_preview_frame(frame, gc_low=gc_min, gc_high=gc_max, max_homopolymer=max_run)
        constraint_error = gc_min > gc_max
        if pass_filter == "pass":
            visible = preview[preview["passes_preview_filters"].astype(bool)].copy()
        elif pass_filter == "fail":
            visible = preview[~preview["passes_preview_filters"].astype(bool)].copy()
        else:
            visible = preview.copy()
        display_cols = [str(column) for column in payload.get("display_cols") or [] if str(column) in visible.columns]
        display = visible[display_cols].copy()
        if "filter_result" in display and "preview_filter_result" in visible:
            display["filter_result"] = visible["preview_filter_result"].values
        visible_ids = set(visible.get("design_id", pd.Series(dtype=str)).astype(str))
        selected_rows = display.loc[visible.get("design_id", pd.Series(index=visible.index, dtype=str)).astype(str) == str(candidate_id)].to_dict("records") if candidate_id else []
        accepted = int(preview["passes_preview_filters"].astype(bool).sum())
        hidden_selection = bool(candidate_id and str(candidate_id) not in visible_ids)
        if constraint_error:
            status = _status("View-setting minimum GC must be less than or equal to maximum GC. All candidates are shown as excluded until the limits are corrected.", kind="warning")
        else:
            suffix = f" Selected candidate {candidate_id} is hidden by the current show filter." if hidden_selection else ""
            status = _status(
                f"{len(visible):,} of {len(preview):,} shown \u00b7 {accepted:,} eligible under view settings \u00b7 saved run unchanged.{suffix}",
                kind="info",
            )
        figure = simulation_landscape_figure(
            visible,
            model_backed=bool(payload.get("model_backed")),
            target_label=payload.get("target_label"),
            selected_ids={str(candidate_id)} if candidate_id and not hidden_selection else None,
        )
        inspector = _simulation_candidate_inspector(
            payload,
            candidate_id,
            gc_low=gc_min,
            gc_high=gc_max,
            max_homopolymer=max_run,
        )
        return (
            figure,
            _json_clean(display.to_dict("records")),
            _json_clean(selected_rows),
            status,
            inspector,
            "observation-inspector is-open" if candidate_id else "observation-inspector",
            "false" if candidate_id else "true",
        )

    app.clientside_callback(
        r"""
        function(className) {
            const isOpen = String(className || '').split(/\s+/).includes('is-open');
            const state = window.__assayreadySimulationInspectorA11y || {trigger: null, handler: null};
            window.__assayreadySimulationInspectorA11y = state;
            if (isOpen) {
                if (!state.trigger || !document.body.contains(state.trigger)) state.trigger = document.activeElement;
                if (!state.handler) {
                    state.handler = function(event) {
                        if (event.key === 'Escape') {
                            const closeButton = document.getElementById('simulation-inspector-close');
                            if (closeButton) {
                                event.preventDefault();
                                closeButton.click();
                            }
                        }
                    };
                    document.addEventListener('keydown', state.handler);
                }
                window.setTimeout(function() {
                    const closeButton = document.getElementById('simulation-inspector-close');
                    if (closeButton) closeButton.focus();
                }, 0);
            } else {
                if (state.handler) {
                    document.removeEventListener('keydown', state.handler);
                    state.handler = null;
                }
                const trigger = state.trigger;
                state.trigger = null;
                window.setTimeout(function() {
                    if (trigger && document.body.contains(trigger) && typeof trigger.focus === 'function') trigger.focus();
                }, 0);
            }
            return {open: isOpen, changed: Date.now()};
        }
        """,
        Output("simulation-inspector-a11y", "data"),
        Input("simulation-inspector", "className"),
        prevent_initial_call=True,
    )

    @app.callback(Output("simulation-download", "data"), Input("simulation-download-button", "n_clicks"), State("simulation-artifact-select", "value"), prevent_initial_call=True)
    def download_simulation_artifact(n_clicks: int, path: str | None) -> Any:
        if not n_clicks or not path:
            raise PreventUpdate
        safe_path = _safe_artifact_path(path)
        return dcc.send_file(str(safe_path))

    return app


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _is_local_host(host: str) -> bool:
    return str(host or "").strip().lower() in LOCAL_HOSTS


def _ui_request_is_allowed(
    host_header: str,
    method: str,
    origin: str | None,
    fetch_site: str | None,
    scheme: str = "http",
) -> bool:
    try:
        parsed_host = urlsplit(f"//{host_header}")
        host = parsed_host.hostname
    except ValueError:
        return False
    if not host or parsed_host.username is not None or parsed_host.password is not None or not _is_local_host(host):
        return False
    if method.upper() not in {"POST", "PUT", "PATCH", "DELETE"}:
        return True
    if str(fetch_site or "").strip().lower() == "cross-site":
        return False
    if not str(origin or "").strip():
        return True
    try:
        parsed_origin = urlsplit(str(origin).strip())
        origin_host = parsed_origin.hostname
        origin_port = parsed_origin.port
        request_port = parsed_host.port
    except ValueError:
        return False
    if (
        parsed_origin.scheme.lower() not in {"http", "https"}
        or parsed_origin.scheme.lower() != scheme.lower()
        or not origin_host
        or parsed_origin.username is not None
        or parsed_origin.password is not None
        or parsed_origin.path not in {"", "/"}
        or parsed_origin.query
        or parsed_origin.fragment
    ):
        return False
    origin_default = 443 if parsed_origin.scheme.lower() == "https" else 80
    return _is_local_host(origin_host) and (origin_port or origin_default) == (request_port or origin_default)


def _install_ui_request_guards(app: Dash) -> None:
    from flask import abort, request

    @app.server.before_request
    def reject_untrusted_browser_request() -> None:
        if not _ui_request_is_allowed(
            request.host,
            request.method,
            request.headers.get("Origin"),
            request.headers.get("Sec-Fetch-Site"),
            request.scheme,
        ):
            abort(403)

    @app.server.after_request
    def add_ui_security_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        return response


def run_server(host: str = "127.0.0.1", port: int = 8050, debug: bool = False, allow_remote: bool = False) -> None:
    if allow_remote:
        raise ValueError("Remote host binding is unsupported. Remove --allow-remote to launch on localhost.")
    if not _is_local_host(host):
        raise ValueError(
            f"Refusing to bind Dash to {host!r}. Remote host binding is currently unsupported."
        )
    if debug and (allow_remote or not _is_local_host(host)):
        raise ValueError("Running in debug mode with remote access enabled is a security risk and is blocked.")
    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server()
