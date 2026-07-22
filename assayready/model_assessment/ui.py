from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dash import MATCH, Dash, Input, Output, State, callback_context, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate

try:
    from .cli import _load_config, _prediction_params_from_config, run_assayready, run_prediction_audit
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
        regression_fit_figure,
        regression_slice_summary,
        residual_figure,
        simulation_constraints_figure,
        simulation_landscape_figure,
        split_composition_figure,
        uncertainty_figure,
    )
except ImportError:
    from model_assessment.cli import _load_config, _prediction_params_from_config, run_assayready, run_prediction_audit
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
        regression_fit_figure,
        regression_slice_summary,
        residual_figure,
        simulation_constraints_figure,
        simulation_landscape_figure,
        split_composition_figure,
        uncertainty_figure,
    )


NAV_ITEMS = [
    ("Analyze", "/analyze"),
    ("Benchmarks", "/benchmarks"),
    ("Runs", "/runs"),
    ("Candidates", "/candidates"),
    ("Reports", "/reports"),
    ("Design Simulation", "/simulations"),
    ("Settings", "/settings"),
]

DNA_MLM_MODELS = {
    "Demo heuristic scorer": {
        "status": "deterministic local demo",
        "context": "short promoter-scale",
        "notes": "Local heuristic scorer for UI demos. Not a trained biological model.",
    },
}


COLORS = {
    "ink": "#192124",
    "muted": "#617174",
    "line": "#d9e2df",
    "soft": "#f5f8f7",
    "panel": "#ffffff",
    "accent": "#0f766e",
    "accent_dark": "#115e59",
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
    "borderRadius": "8px",
    "boxShadow": "0 1px 2px rgba(16, 36, 33, 0.06)",
    "padding": "18px",
}

INPUT_STYLE = {
    "width": "100%",
    "boxSizing": "border-box",
    "border": f"1px solid {COLORS['line']}",
    "borderRadius": "6px",
    "padding": "9px 10px",
    "fontSize": "14px",
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
    "background": "#e7efed",
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
    project_root = _package_dir().parent
    if (project_root / "pyproject.toml").is_file():
        return (project_root / "outputs" / "assayready").resolve()
    return (Path.home() / ".assayready" / "outputs").resolve()


def _safe_output_dir(raw: str) -> Path:
    allowed_root = _output_root()
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        target = candidate.resolve()
    else:
        parts = candidate.parts
        if len(parts) >= 2 and tuple(part.lower() for part in parts[:2]) == ("outputs", "assayready"):
            candidate = Path(*parts[2:])
        target = (allowed_root / candidate).resolve()
    if not (target == allowed_root or target.is_relative_to(allowed_root)):
        raise ValueError("Paths must reside under outputs/assayready.")
    return target


def _safe_artifact_path(raw: str) -> Path:
    target = Path(raw).resolve()
    all_runs = list_runs(limit=1000)
    for run in all_runs:
        run_id = run.get("run_id")
        run_dir_str = run.get("artifact_dir")
        if run_id and run_dir_str:
            run_dir = Path(run_dir_str).resolve()
            if target == run_dir or target.is_relative_to(run_dir):
                for art in list_artifacts(run_id):
                    art_path = Path(art["path"]).resolve()
                    if target == art_path:
                        return target
    raise ValueError(f"Requested file is not an indexed AssayReady artifact: {raw}")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _data_table(frame: pd.DataFrame, *, page_size: int = 10, height: int | None = None, compact: bool = False) -> Any:
    if frame.empty:
        return html.Div("No rows to display.", style={"color": COLORS["muted"], "padding": "12px 0"})
    style_table = {"overflowX": "auto"}
    if height:
        style_table["height"] = f"{height}px"
        style_table["overflowY"] = "auto"
    try:
        render_limit = max(1, int(os.environ.get(MAX_TABLE_RENDER_ROWS_ENV, "1000")))
    except ValueError:
        render_limit = 1000
    clean = _safe_frame(frame, limit=render_limit)
    table = dash_table.DataTable(
        data=clean.to_dict("records"),
        columns=[{"name": str(column), "id": str(column)} for column in clean.columns],
        page_size=page_size,
        sort_action="native",
        filter_action="none" if compact else "native",
        page_action="none" if compact else "native",
        style_table=style_table,
        style_cell={
            "fontFamily": PAGE_STYLE["fontFamily"],
            "fontSize": "12px" if compact else "13px",
            "padding": "6px 8px" if compact else "8px",
            "textAlign": "left",
            "minWidth": "90px",
            "maxWidth": "300px" if compact else "340px",
            "whiteSpace": "nowrap",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        style_header={
            "backgroundColor": "#eaf2f0",
            "fontWeight": "700",
            "border": f"1px solid {COLORS['line']}",
        },
        style_data={
            "border": f"1px solid {COLORS['line']}",
        },
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
            html.H1(title, style={"fontSize": "28px", "margin": "0 0 6px"}),
            html.Div(subtitle, style={"color": COLORS["muted"], "fontSize": "15px", "maxWidth": "880px"}),
        ],
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


def _metric(label: str, value: Any) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "0", "color": COLORS["muted"]}),
            html.Div(_metric_text(value), style={"fontSize": "24px", "fontWeight": 800, "marginTop": "3px"}),
        ],
        style={**CARD_STYLE, "padding": "14px"},
    )


def _status(message: str, *, kind: str = "info") -> html.Div:
    palette = {
        "info": ("#e8f3f1", COLORS["accent_dark"]),
        "success": ("#e8f5ee", "#166534"),
        "warning": ("#fff7ed", COLORS["warn"]),
        "danger": ("#fef2f2", COLORS["danger"]),
    }
    bg, fg = palette.get(kind, palette["info"])
    return html.Div(message, style={"background": bg, "color": fg, "padding": "10px 12px", "borderRadius": "6px", "fontWeight": 650})


def _readiness_item(label: str, ready: bool) -> html.Div:
    return html.Div(
        [
            html.Span("OK" if ready else "WAIT", className=f"ready-pill {'ready' if ready else 'waiting'}"),
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
        record_run(summary, workflow=workflow, report_name=report_name)
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
    ranked_path = artifact_dir / "ranked_candidates.csv"
    if not ranked_path.exists():
        return pd.DataFrame()
    ranked = _prepare_ranked_frame(pd.read_csv(ranked_path))
    if ranked.empty:
        return ranked
    sequence_col = _find_sequence_column(ranked, summary)
    prediction_col = _prediction_column(ranked)
    uncertainty_col = _uncertainty_column(ranked)
    explanations = _explanation_map(artifact_dir)
    rows: list[dict[str, Any]] = []
    for _, row in ranked.head(top_n).iterrows():
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
                "acquisition_score": row.get("acquisition_score", ""),
                "diversity_cluster": row.get("diversity_cluster", ""),
                "why_this_rank": reason or "Ranked by configured acquisition score.",
            }
        )
    return pd.DataFrame(rows)


def _summary_metrics(summary: dict[str, Any]) -> list[tuple[str, Any]]:
    report = _report_payload(summary)
    best = report.get("best_simple_baseline") or {}
    if "prediction_audit" in summary:
        uncertainty = report.get("uncertainty_audit") or {}
        return [
            ("Rows", summary.get("valid_prediction_rows") or summary.get("audit", {}).get("accepted_rows")),
            ("Test Metric", (report.get("test_metrics") or {}).get("primary_metric")),
            ("Best Baseline", best.get("primary_metric")),
            ("Unc/Error", uncertainty.get("uncertainty_abs_error_spearman")),
        ]
    return [
        ("Rows", summary.get("audit", {}).get("accepted_rows") or summary.get("valid_prediction_rows")),
        ("Model Metric", report.get("task_head_primary_metric") or report.get("primary_metric")),
        ("Best Baseline", best.get("primary_metric")),
        ("Ranked", summary.get("ranked_candidates")),
    ]


GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "displayModeBar": "hover",
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
        className=f"visualization-card{' interactive-chart-card' if interactive else ''}",
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


def _claim_gate_panel(report: dict[str, Any]) -> html.Div:
    gate = report.get("claim_gate") or {}
    labels = [
        ("leakage_controlled", "Leakage / independence"),
        ("lift_claim", "Model lift"),
        ("uncertainty_usable", "Uncertainty usefulness"),
        ("candidate_constraints", "Candidate constraints"),
        ("recommended", "Recommendation"),
    ]
    items: list[Any] = []
    for key, label in labels:
        detail = gate.get(key) or {}
        ok = bool(detail.get("ok"))
        reasons = [str(reason) for reason in detail.get("reasons") or []]
        items.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("PASS" if ok else "NOT MET", className=f"gate-badge {'pass' if ok else 'fail'}"),
                            html.Strong(label),
                        ],
                        className="gate-heading",
                    ),
                    html.Ul([html.Li(reason) for reason in reasons]) if reasons else html.Div("No blocking reason recorded.", className="gate-reason"),
                ],
                className=f"gate-item {'pass' if ok else 'fail'}",
            )
        )
    return _card([html.H3("Evidence gates", className="section-heading"), html.Div(items, className="claim-gate-grid")], className="claim-gate-card")


def _provenance_panel(report: dict[str, Any], summary: dict[str, Any]) -> html.Div:
    manifest = report.get("evaluation_manifest") or summary.get("evaluation_manifest") or {}
    provenance = manifest.get("provenance") or {}
    values = [
        ("Objective", manifest.get("objective_direction", "unspecified")),
        ("Units", manifest.get("units", "unspecified")),
        ("Uncertainty", manifest.get("uncertainty_type", "none / undeclared")),
        ("Training independence", manifest.get("training_independence", "unverified")),
        ("Data ID", provenance.get("data_identifier", "not provided")),
        ("Model ID", provenance.get("model_identifier", "not provided")),
    ]
    return _card(
        [
            html.H3("Evaluation context", className="section-heading"),
            html.Dl([html.Div([html.Dt(label), html.Dd(str(value))], className="provenance-item") for label, value in values], className="provenance-grid"),
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
        if column in metadata and frame[column].nunique(dropna=True) <= 50
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


def _benchmark_section(summary: dict[str, Any], *, standalone: bool = False, surface: str = "benchmark") -> Any:
    report = _report_payload(summary)
    if not report.get("baselines"):
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
    model_ci_text = "—"
    if isinstance(model_ci, (list, tuple)) and len(model_ci) == 2:
        model_ci_text = f"{float(model_ci[0]):.3g}–{float(model_ci[1]):.3g}"
    scope = _benchmark_scope(summary, surface)
    run_charts: list[Any] = [
        _graph_card(
            benchmark_metric_figure(report),
            "Actual baselines were fitted on the persisted train split and evaluated on the held-out split. The interval applies to the model only.",
        ),
        _graph_card(
            split_composition_figure(split),
            "Group and sequence-similarity boundaries determine these splits; few clusters make performance estimates unstable.",
        ),
    ]
    required = {target_col, prediction_col}
    row_explorer: Any = None
    if not frame.empty and required.issubset(frame.columns):
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

    banner_kind = "success" if leakage_ok else "warning"
    title = "Retrospective benchmark evidence" if leakage_ok else "Retrospective benchmark — audit only"
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
            _status(str(report.get("verdict") or "Retrospective benchmark completed."), kind=banner_kind),
            html.Div(
                [
                    _metric("Held-out rows", test_metrics.get("num_rows") or (split.get("split_sizes") or {}).get("test")),
                    _metric("Leakage clusters", split.get("num_clusters")),
                    _metric("Model 95% interval", model_ci_text),
                    _metric("Lift delta", lift.get("delta")),
                ],
                className="metric-grid benchmark-metrics",
            ),
            _status(frame_note, kind="info") if frame_note else None,
            html.Div(run_charts, className="visualization-grid run-evidence-grid"),
            row_explorer,
            _claim_gate_panel(report),
            _provenance_panel(report, summary),
            _card(
                [
                    html.H3("Statistical cautions", className="section-heading"),
                    html.Ul(
                        [
                            html.Li("The displayed model interval is a row bootstrap, not a leakage-cluster bootstrap."),
                            html.Li("Baseline values are point estimates; the lift interval is not a paired model-vs-baseline interval."),
                            html.Li("External predictions require independently verified training and locked-holdout provenance for claim-grade conclusions."),
                            html.Li("K-mer similarity grouping is a leakage-control proxy, not proof of biological independence."),
                        ]
                    ),
                ],
                className="warning-card",
                style={"marginTop": "14px"},
            ),
        ],
        className="benchmark-section",
    )


def _result_summary(summary: dict[str, Any], report_name: str, *, surface: str = "analysis") -> html.Div:
    report = _report_payload(summary)
    top = _top_candidates(summary)
    artifact_opts = _artifact_options(summary, report_name)
    claim_gate = report.get("claim_gate") or {}
    if "simulation_report" in summary:
        verdict_kind = "warning"
    elif claim_gate:
        verdict_kind = "success" if bool((claim_gate.get("recommended") or {}).get("ok")) else "warning"
    else:
        verdict_kind = "success"
    return html.Div(
        [
            _status(str(report.get("verdict") or "Run completed."), kind=verdict_kind),
            html.Div([_metric(label, value) for label, value in _summary_metrics(summary)], className="metric-grid"),
            _benchmark_section(summary, surface=surface),
            html.H3("Top Candidates", style={"margin": "18px 0 10px"}),
            _data_table(top, page_size=10) if not top.empty else _status("No ranked candidates were produced for this run."),
            html.Div(
                [
                    html.H3("Artifacts", style={"margin": "0 0 10px"}),
                    html.Div(f"Saved to: {summary.get('artifact_dir')}", style={"color": COLORS["muted"], "marginBottom": "10px"}),
                    dcc.Dropdown(id="analysis-artifact-select", options=artifact_opts, value=artifact_opts[0]["value"] if artifact_opts else None, clearable=False),
                    html.Button("Download selected artifact", id="analysis-download-button", n_clicks=0, style={**BUTTON_STYLE, "marginTop": "10px"}),
                    dcc.Download(id="analysis-download"),
                ],
                style={"marginTop": "18px"},
            ),
            _status(str(summary.get("run_store_warning")), kind="warning") if summary.get("run_store_warning") else None,
        ],
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


def _plate_role_selection(frame: pd.DataFrame, slots: int) -> list[tuple[str, dict[str, Any]]]:
    if slots <= 0 or frame.empty:
        return []
    eligible = frame[frame.get("passes_basic_filters", True).astype(bool)].copy() if "passes_basic_filters" in frame.columns else frame.copy()
    if eligible.empty:
        eligible = frame.copy()
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
    if len(selections) < slots and "rank" in frame.columns:
        add_rows("backup_ranked_candidate", frame.sort_values("rank"), slots - len(selections))
    return selections[:slots]


def _build_plate_plan(frame: pd.DataFrame, *, plate_size: int, control_wells: int, seed: int) -> pd.DataFrame:
    wells = _well_names(int(plate_size))
    rng = random.Random(int(seed))
    rng.shuffle(wells)
    control_count = max(0, min(int(control_wells), len(wells)))
    candidate_wells = wells[control_count:]
    control_well_names = wells[:control_count]
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
        f"# Design & Rank Report: {summary.get('project')}",
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
        f"- Ranked candidates: {summary.get('ranked_candidates')}",
        f"- Plate wells: {len(plate_plan)}",
        "",
        "## Next Action",
        "",
        "Review warnings, inspect top ranked candidates, and treat the plate plan as an in-silico validation plan until wet-lab results are returned.",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(f"- {artifact}" for artifact in summary.get("artifacts", []))
    lines.append("")
    return "\n".join(lines)


def _add_client_plate_plan_artifacts(summary: dict[str, Any], *, plate_size: int, control_wells: int, plate_seed: int) -> dict[str, Any]:
    artifact_dir = Path(str(summary["artifact_dir"]))
    ranked_path = artifact_dir / "ranked_candidates.csv"
    if not ranked_path.exists():
        return summary
    ranked = pd.read_csv(ranked_path)
    plate_source = _prepare_plate_source(ranked, summary)
    plate_plan = _build_plate_plan(plate_source, plate_size=plate_size, control_wells=control_wells, seed=plate_seed)
    plate_plan.to_csv(artifact_dir / "plate_plan.csv", index=False)
    artifacts = list(summary.get("artifacts") or [])
    for artifact in ["simulation_report.md", "plate_plan.csv"]:
        if artifact not in artifacts:
            artifacts.append(artifact)
    summary["artifacts"] = artifacts
    summary.setdefault("params", {})
    summary["params"]["plate_size"] = int(plate_size)
    summary["params"]["control_wells"] = int(control_wells)
    summary["params"]["plate_seed"] = int(plate_seed)
    (artifact_dir / "simulation_report.md").write_text(_client_design_report_markdown(summary, plate_plan), encoding="utf-8")
    _write_json_file(artifact_dir / "run_summary.json", summary)
    return summary


def _normalize_dna(sequence: str) -> str:
    return "".join(base for base in sequence.upper() if base in {"A", "C", "G", "T"})


def _gc_fraction(sequence: str) -> float:
    seq = _normalize_dna(sequence)
    if not seq:
        return 0.0
    return float((seq.count("G") + seq.count("C")) / len(seq))


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


def _heuristic_sequence_scores(sequence: str, target_gc: tuple[float, float], max_homopolymer: int, rng: random.Random) -> dict[str, Any]:
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
    return {
        "gc_fraction": gc,
        "max_homopolymer": homopolymer,
        "mlm_plausibility": plausibility,
        "model_uncertainty": uncertainty,
        "predicted_activity": predicted_activity,
        "acquisition_score": acquisition,
        "novelty_proxy": novelty,
        "passes_basic_filters": target_gc[0] <= gc <= target_gc[1] and homopolymer <= max_homopolymer,
        "scoring_backend": "demo_heuristic_scorer",
    }


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
        f"- Ranked candidates: {summary.get('ranked_candidates')}",
        f"- Plate wells: {len(plate_plan)}",
        "",
        "## Top Candidates",
        "",
        "```text",
        top_frame.head(10).to_string(index=False),
        "```",
        "",
        "## Next Plate",
        "",
        "Use `plate_plan.csv` as a planning artifact. Reserved control wells must be filled with lab-specific controls before execution.",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(f"- {artifact}" for artifact in summary.get("artifacts", []))
    lines.append("")
    return "\n".join(lines)


def _save_simulation_artifacts(*, project: str, frame: pd.DataFrame, plate_plan: pd.DataFrame, params: dict[str, Any], scoring_caption: str) -> tuple[Path, dict[str, Any]]:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    artifact_dir = _output_root() / "simulations" / f"{_slug(project)}_{timestamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    public_frame = _public_simulation_frame(frame)
    explanations = _candidate_explanations_from_simulation(public_frame)
    public_frame.to_csv(artifact_dir / "ranked_candidates.csv", index=False)
    public_frame.to_csv(artifact_dir / "simulated_candidates.csv", index=False)
    plate_plan.to_csv(artifact_dir / "plate_plan.csv", index=False)
    pd.DataFrame(explanations).to_csv(artifact_dir / "candidate_explanations.csv", index=False)
    _write_json_file(artifact_dir / "candidate_explanations.json", {"candidates": explanations})
    summary = {
        "project": project,
        "artifact_dir": str(artifact_dir.resolve()),
        "ranked_candidates": int(len(public_frame)),
        "artifacts": [
            "simulation_report.md",
            "simulation_summary.json",
            "ranked_candidates.csv",
            "simulated_candidates.csv",
            "plate_plan.csv",
            "candidate_explanations.csv",
            "candidate_explanations.json",
        ],
        "params": params,
        "simulation_report": {
            "verdict": "Heuristic simulation completed. Treat these as draft planning candidates until model and wet-lab validation.",
            "primary_metric_name": "heuristic_acquisition_score",
            "primary_metric": float(public_frame["acquisition_score"].max()) if "acquisition_score" in public_frame and not public_frame.empty else None,
            "best_simple_baseline": {},
            "scoring_backend": scoring_caption,
            "evidence_level": "synthetic_planning_only",
            "warnings": ["Activity and uncertainty are deterministic heuristic proxies, not measured or trained-model outputs."],
        },
    }
    (artifact_dir / "simulation_report.md").write_text(_simulation_report_markdown(summary, public_frame, plate_plan), encoding="utf-8")
    _write_json_file(artifact_dir / "simulation_summary.json", summary)
    try:
        record_run(summary, workflow="simulation", report_name="simulation_report.md")
    except Exception as exc:
        summary["run_store_warning"] = f"Simulation artifacts were saved, but local run indexing failed: {exc}"
    return artifact_dir, summary


def _runs_display_frame(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        rows.append(
            {
                "updated_at": run.get("updated_at"),
                "project": run.get("project"),
                "workflow": run.get("workflow"),
                "verdict": run.get("verdict"),
                "primary_metric": run.get("primary_metric"),
                "best_baseline": run.get("best_baseline_metric"),
                "ranked_candidates": run.get("ranked_candidates"),
                "artifact_dir": run.get("artifact_dir"),
                "run_id": run.get("run_id"),
            }
        )
    return pd.DataFrame(rows)


def _run_label(run: dict[str, Any]) -> str:
    updated = str(run.get("updated_at") or "")[:19]
    return f"{updated} | {run.get('project')} | {run.get('workflow')} | {run.get('run_id')}"


def page_analyze() -> html.Div:
    return html.Div(
        [
            _page_header("Analyze", "Upload local assay data, map columns, run an audit or local assay model, then download the evidence package."),
            dcc.Store(id="assay-upload-state"),
            dcc.Store(id="candidate-upload-state"),
            _card(
                [
                    html.H2("1. Upload measured assay data", style={"fontSize": "18px", "margin": "0 0 10px"}),
                    dcc.Upload(
                        id="assay-upload",
                        children=html.Div(["Drag and drop or ", html.A("select assay CSV/TSV/XLSX or config JSON/YAML")]),
                        style={
                            "border": f"1px dashed {COLORS['accent']}",
                            "borderRadius": "8px",
                            "padding": "22px",
                            "textAlign": "center",
                            "background": "#f8fbfa",
                        },
                        multiple=False,
                    ),
                    html.Button("Load prediction example", id="load-prediction-example-button", n_clicks=0, style={**SECONDARY_BUTTON_STYLE, "marginTop": "12px"}),
                    html.Div(id="assay-preview", style={"marginTop": "14px"}),
                ]
            ),
            html.Div(
                [
                    _card(
                        [
                            html.H2("2. Analysis", style={"fontSize": "18px", "margin": "0 0 12px"}),
                            _field(
                                "Analysis type",
                                dcc.RadioItems(
                                    id="analysis-type",
                                    options=[
                                        {"label": "Audit existing predictions", "value": "prediction"},
                                        {"label": "Train local assay model", "value": "local_model"},
                                    ],
                                    value="prediction",
                                    labelStyle={"display": "block", "marginBottom": "8px"},
                                ),
                            ),
                            _field("Project name", dcc.Input(id="project-name", value="assayready_prediction_audit", style=INPUT_STYLE)),
                            _field("Output directory", dcc.Input(id="output-dir", value="", placeholder="Leave empty for timestamped outputs/assayready path", style=INPUT_STYLE)),
                            _field("Optional candidate file", dcc.Upload(id="candidate-upload", children=html.Div(["Select candidate CSV/TSV/XLSX"]), style={"border": f"1px dashed {COLORS['line']}", "borderRadius": "8px", "padding": "14px", "textAlign": "center"})),
                        ],
                        style={"height": "100%"},
                    ),
                    _card(
                        [
                            html.H2("3. Columns", style={"fontSize": "18px", "margin": "0 0 12px"}),
                            html.Div("Load an assay table to enable column mapping.", id="columns-empty-hint", className="empty-hint"),
                            _field("Sequence column", dcc.Dropdown(id="sequence-col", options=[])),
                            _field("Measured target column", dcc.Dropdown(id="target-col", options=[])),
                            _field("Prediction column", dcc.Dropdown(id="prediction-col", options=[])),
                            _field("Optional uncertainty column", dcc.Dropdown(id="uncertainty-col", options=[], clearable=True)),
                            _field("Optional ID column", dcc.Dropdown(id="id-col", options=[], clearable=True)),
                            _field(
                                "Independence boundary columns",
                                dcc.Dropdown(id="group-cols", options=[], multi=True),
                                "Rows sharing any value in any selected column are kept together. Low-cardinality fields such as organism can collapse the data into only a few testable groups; use metadata columns for descriptive slices instead.",
                            ),
                            _field("Metadata columns", dcc.Dropdown(id="metadata-cols", options=[], multi=True)),
                            _field(
                                "Task type",
                                dcc.Dropdown(
                                    id="task-type",
                                    options=[{"label": item.title(), "value": item} for item in ["regression", "classification", "ranking"]],
                                    value="regression",
                                    clearable=False,
                                ),
                            ),
                        ],
                        style={"alignSelf": "start"},
                        id="columns-card",
                        className="columns-card muted-card",
                    ),
                    _card(
                        [
                            html.H2("4. Run settings", style={"fontSize": "18px", "margin": "0 0 12px"}),
                            _field(
                                "Top candidates",
                                dcc.Input(id="top-k", type="number", min=1, max=384, step=1, value=96, style=INPUT_STYLE),
                                "Maximum candidates to include in the ranked output package.",
                            ),
                            html.Details(
                                [
                                    html.Summary("Split and ranking settings", className="details-summary"),
                                    html.Div(
                                        [
                                            _field("Low-N warning threshold", dcc.Input(id="low-n", type="number", min=0, step=25, value=200, style=INPUT_STYLE), "Flags assay datasets with fewer rows than this threshold."),
                                            _field("Validation fraction", dcc.Input(id="val-fraction", type="number", min=0, max=0.4, step=0.01, value=0.15, style=INPUT_STYLE), "Share of rows held out for validation after leakage-aware grouping."),
                                            _field("Test fraction", dcc.Input(id="test-fraction", type="number", min=0, max=0.4, step=0.01, value=0.15, style=INPUT_STYLE), "Share of rows held out for final model or prediction evaluation."),
                                            _field("Homology threshold", dcc.Input(id="homology-threshold", type="number", min=0, max=1, step=0.01, value=0.9, style=INPUT_STYLE), "Maximum sequence similarity allowed across split boundaries before rows are grouped."),
                                            _field("Homology k-mer size", dcc.Input(id="homology-k", type="number", min=2, max=12, step=1, value=8, style=INPUT_STYLE), "Length of sequence words used to estimate homology leakage."),
                                            _field("Uncertainty beta", dcc.Input(id="beta", type="number", min=0, max=10, step=0.1, value=1.0, style=INPUT_STYLE), "Weight applied to uncertainty when ranking candidates by upper confidence bound."),
                                            _field("Diversity penalty", dcc.Input(id="diversity-penalty", type="number", min=0, max=2, step=0.05, value=0.2, style=INPUT_STYLE), "Strength of the penalty that discourages near-duplicate selected sequences."),
                                            _field("Seed", dcc.Input(id="seed", type="number", min=0, step=1, value=13, style=INPUT_STYLE), "Random seed used for deterministic splits and ranking tie breaks."),
                                        ],
                                        className="settings-grid",
                                    ),
                                ],
                                className="settings-details",
                            ),
                            html.Details(
                                [
                                    html.Summary("Local model options", className="details-summary"),
                                    _field("Ensemble size", dcc.Input(id="ensemble-size", type="number", min=1, max=16, step=1, value=5, style=INPUT_STYLE), "Number of local models trained to estimate uncertainty."),
                                    _field("Epochs", dcc.Input(id="epochs", type="number", min=1, max=500, step=5, value=80, style=INPUT_STYLE), "Training passes for each local model."),
                                    _field("Learning rate", dcc.Input(id="learning-rate", type="number", min=0.00001, max=0.1, step=0.0001, value=0.001, style=INPUT_STYLE), "Optimizer step size for the local assay model."),
                                    _field("Generated candidate pool", dcc.Input(id="num-proposals", type="number", min=96, max=5000, step=96, value=1000, style=INPUT_STYLE), "Number of synthetic candidate sequences to propose when no candidate file is uploaded."),
                                    _field("Plate size", dcc.Dropdown(id="plate-size", options=[{"label": str(item), "value": item} for item in [24, 48, 96, 384]], value=96, clearable=False), "Well count used when building the optional plate plan."),
                                    _field("Reserved control wells", dcc.Input(id="control-wells", type="number", min=0, step=1, value=8, style=INPUT_STYLE), "Wells reserved for controls before assigning ranked candidates."),
                                    _field("Plate seed", dcc.Input(id="plate-seed", type="number", min=0, step=1, value=13, style=INPUT_STYLE), "Random seed used to distribute candidates across the plate."),
                                ]
                            ),
                            html.Details(
                                [
                                    html.Summary("Evaluation semantics and provenance", className="details-summary"),
                                    html.P("Declare what the supplied values mean. These declarations improve interpretation but do not, by themselves, verify an independent holdout.", className="chart-description"),
                                    html.Div(
                                        [
                                            _field("Objective direction", dcc.Dropdown(id="evaluation-objective", options=[{"label": "Maximize", "value": "maximize"}, {"label": "Minimize", "value": "minimize"}], value="maximize", clearable=False)),
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
                                className="settings-details",
                            ),
                        ],
                        style={"alignSelf": "start"},
                    ),
                ],
                className="analysis-grid",
            ),
            html.Div(id="candidate-preview", style={"marginTop": "14px"}),
            html.Div(
                [
                    html.Button("Run analysis", id="run-analysis-button", n_clicks=0, style={**SECONDARY_BUTTON_STYLE, "cursor": "not-allowed", "opacity": 0.62}, disabled=True),
                    html.Div(id="analysis-ready-state", className="run-ready-state"),
                ],
                className="run-bar",
            ),
            dcc.Loading(html.Div(id="analysis-result"), type="circle"),
        ]
    )


def _runs_content(status: Any | None = None) -> html.Div:
    runs = list_runs(limit=500)
    options = [{"label": _run_label(run), "value": str(run["run_id"])} for run in runs]
    return html.Div(
        [
            status,
            _data_table(_runs_display_frame(runs), page_size=12) if runs else _status("No runs are indexed yet. Run an analysis first."),
            _field("Open run", dcc.Dropdown(id="runs-run-select", options=options, value=options[0]["value"] if options else None, clearable=False)) if options else None,
            html.Div(id="runs-run-detail"),
        ]
    )


def page_runs() -> html.Div:
    return html.Div(
        [
            _page_header("Runs", "Local experiment history for audits, model checks, candidate rankings, and generated reports."),
            html.Div(f"Local run store: {default_db_path()}", style={"color": COLORS["muted"], "marginBottom": "12px"}),
            html.Button("Index existing artifacts", id="index-runs-button", n_clicks=0, style=SECONDARY_BUTTON_STYLE),
            html.Button("Refresh", id="refresh-runs-button", n_clicks=0, style={**SECONDARY_BUTTON_STYLE, "marginLeft": "8px"}),
            html.Div(id="runs-content", children=_runs_content(), style={"marginTop": "16px"}),
        ]
    )


def page_candidates() -> html.Div:
    candidates = list_candidates(limit=5000)
    if not candidates:
        body = _status("No candidates are indexed yet. Run an audit that produces candidate explanations.")
    else:
        frame = pd.DataFrame(candidates)
        frame["risk_flags"] = frame["risk_flags_json"].map(lambda value: ", ".join(_load_json_cell(value, [])))
        visible = [
            column
            for column in [
                "project",
                "workflow",
                "rank",
                "display_id",
                "prediction",
                "uncertainty",
                "acquisition_score",
                "diversity_cluster",
                "training_distribution_status",
                "nearest_train_id",
                "risk_flags",
                "updated_at",
                "run_id",
            ]
            if column in frame.columns
        ]
        body = _data_table(frame[visible], page_size=20, height=620)
    return html.Div([_page_header("Candidates", "Search and sort ranked sequences across local runs."), _card(body)])


def page_reports() -> html.Div:
    runs = list_runs(limit=500)
    options = [{"label": _run_label(run), "value": str(run["run_id"])} for run in runs]
    return html.Div(
        [
            _page_header("Reports", "Open a run report, review artifacts, and download evidence files."),
            _field("Report run", dcc.Dropdown(id="reports-run-select", options=options, value=options[0]["value"] if options else None, clearable=False)),
            html.Div(id="report-detail"),
            dcc.Download(id="report-download"),
        ]
    )


def page_benchmarks() -> html.Div:
    runs = [run for run in list_runs(limit=500) if str(run.get("workflow")) != "simulation"]
    options = [{"label": _run_label(run), "value": str(run["run_id"])} for run in runs]
    return html.Div(
        [
            _page_header(
                "Benchmarks",
                "Inspect real held-out measurements, fitted sequence baselines, leakage-aware splits, uncertainty behavior, and evidence gates.",
            ),
            _card(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2("Public DREAM promoter reference", style={"fontSize": "19px", "margin": "0 0 6px"}),
                                    html.P(
                                        "Run the bundled 400-row public promoter example with recorded source provenance. It is a real-data workflow demonstration, not a state-of-the-art model claim.",
                                        style={"margin": 0, "color": COLORS["muted"]},
                                    ),
                                ]
                            ),
                            html.Button("Run public benchmark", id="run-public-benchmark-button", n_clicks=0, style=BUTTON_STYLE),
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
            _card(
                [
                    _field("Indexed benchmark run", dcc.Dropdown(id="benchmark-run-select", options=options, value=options[0]["value"] if options else None, clearable=False)),
                    html.Div("Select a completed run to inspect its persisted evidence. Runs without fitted baselines will show an explanatory empty state.", style={"color": COLORS["muted"], "fontSize": "13px"}),
                ],
                style={"marginTop": "16px"},
            ),
            dcc.Loading(html.Div(id="benchmark-detail", style={"marginTop": "16px"}), type="circle"),
        ]
    )


def page_simulations() -> html.Div:
    return html.Div(
        [
            _page_header("Design simulation", "Explore DNA variants with a deterministic heuristic, inspect constraints graphically, and draft a plate plan."),
            _status("Planning sandbox only: simulated activity and uncertainty are heuristic proxies, not model validation or benchmark evidence.", kind="warning"),
            _card(
                [
                    html.Div(
                        [
                            _field("Generation mode", dcc.Dropdown(id="sim-mode", options=[{"label": item, "value": item} for item in ["Mask-fill", "Mutational scan", "Diversity proposal"]], value="Mask-fill", clearable=False)),
                            _field("Scorer", dcc.Dropdown(id="sim-model", options=[{"label": key, "value": key} for key in DNA_MLM_MODELS], value="Demo heuristic scorer", clearable=False)),
                            _field("Candidates", dcc.Input(id="sim-num-candidates", type="number", min=1, max=1000, step=1, value=96, style=INPUT_STYLE)),
                            _field("Context multiplier", dcc.Input(id="sim-context", type="number", min=1, max=8, step=0.5, value=2.0, style=INPUT_STYLE)),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(0, 1fr))", "gap": "14px"},
                        className="simulation-settings-grid",
                    ),
                    _field("Parent or masked sequence", dcc.Textarea(id="sim-parent", value="TATAATNNNNGGTTTT", style={**INPUT_STYLE, "minHeight": "92px"})),
                    html.Div(
                        [
                            _field("Mutation rate", dcc.Input(id="sim-mutation-rate", type="number", min=0, max=0.5, step=0.01, value=0.08, style=INPUT_STYLE)),
                            _field("Min GC", dcc.Input(id="sim-gc-low", type="number", min=0, max=1, step=0.01, value=0.30, style=INPUT_STYLE)),
                            _field("Max GC", dcc.Input(id="sim-gc-high", type="number", min=0, max=1, step=0.01, value=0.70, style=INPUT_STYLE)),
                            _field("Max homopolymer", dcc.Input(id="sim-max-homopolymer", type="number", min=1, max=20, step=1, value=6, style=INPUT_STYLE)),
                            _field("Seed", dcc.Input(id="sim-seed", type="number", min=0, step=1, value=13, style=INPUT_STYLE)),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "repeat(5, minmax(0, 1fr))", "gap": "14px"},
                        className="simulation-settings-grid",
                    ),
                    html.Div(
                        [
                            _field("Project", dcc.Input(id="sim-project", value="dash_sequence_simulation", style=INPUT_STYLE)),
                            _field("Plate size", dcc.Dropdown(id="sim-plate-size", options=[{"label": str(item), "value": item} for item in [24, 96, 384]], value=96, clearable=False)),
                            _field("Control wells", dcc.Input(id="sim-control-wells", type="number", min=0, step=1, value=4, style=INPUT_STYLE)),
                            _field("Plate seed", dcc.Input(id="sim-plate-seed", type="number", min=0, step=1, value=13, style=INPUT_STYLE)),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "2fr 1fr 1fr 1fr", "gap": "14px"},
                        className="simulation-settings-grid",
                    ),
                    html.Button("Run design simulation", id="run-simulation-button", n_clicks=0, style=BUTTON_STYLE),
                ]
            ),
            dcc.Loading(html.Div(id="simulation-result", style={"marginTop": "18px"}), type="circle"),
        ]
    )


def page_settings() -> html.Div:
    return html.Div(
        [
            _page_header("Settings", "Local storage and deployment notes for the Dash workbench."),
            _card(
                [
                    html.H2("Local Storage", style={"fontSize": "18px", "margin": "0 0 8px"}),
                    html.Div(f"Upload cache: {_upload_root()}"),
                    html.Div(f"Run index: {default_db_path()}"),
                ]
            ),
            _card(
                [
                    html.H2("Privacy", style={"fontSize": "18px", "margin": "0 0 8px"}),
                    html.P("The Dash UI does not send assay data to a hosted service. Uploaded files, outputs, and run metadata stay on this machine unless you move or expose them."),
                    html.P("For shared deployments, put Dash behind authenticated HTTPS, restrict filesystem access, and keep uploads size-limited at the reverse proxy."),
                ],
                style={"marginTop": "14px"},
            ),
        ]
    )


def _sidebar_links(pathname: str | None) -> list[Any]:
    path = pathname or "/analyze"
    links: list[Any] = []
    for label, href in NAV_ITEMS:
        active = path == href or (path == "/" and href == "/analyze")
        links.append(
            dcc.Link(
                label,
                href=href,
                className=f"nav-link {'active' if active else ''}",
            )
        )
    return links


def _sidebar() -> html.Div:
    return html.Div(
        [
            html.Div("ASSAYREADY", className="side-brand"),
            html.Div("Dash workbench", className="side-subtitle"),
            html.Nav(id="sidebar-links", children=_sidebar_links("/analyze")),
        ],
        style=SIDEBAR_STYLE,
        className="app-sidebar",
    )


def _app_shell() -> html.Div:
    return html.Div(
        [
            dcc.Location(id="location"),
            _sidebar(),
            html.Main(id="page-content", style=CONTENT_STYLE, className="app-content"),
        ],
        style=PAGE_STYLE,
        className="app-shell",
    )


def create_app() -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True, title="AssayReady", assets_folder=str(_package_dir() / "assets"))
    app.layout = _app_shell

    @app.callback(Output("sidebar-links", "children"), Input("location", "pathname"))
    def render_sidebar_links(pathname: str | None) -> list[Any]:
        return _sidebar_links(pathname)

    @app.callback(Output("page-content", "children"), Input("location", "pathname"))
    def render_page(pathname: str | None) -> Any:
        path = pathname or "/analyze"
        if path in {"/", "/analyze"}:
            return page_analyze()
        if path == "/benchmarks":
            return page_benchmarks()
        if path == "/runs":
            return page_runs()
        if path == "/candidates":
            return page_candidates()
        if path == "/reports":
            return page_reports()
        if path == "/simulations":
            return page_simulations()
        if path == "/settings":
            return page_settings()
        return html.Div([_page_header("Not Found", "Choose a page from the sidebar.")])

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

    @app.callback(Output("benchmark-detail", "children"), Input("benchmark-run-select", "value"))
    def show_benchmark(run_id: str | None) -> Any:
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
                "cardinalities": {str(column): int(frame[column].nunique(dropna=True)) for column in frame.columns},
                "evaluation_manifest": evaluation_manifest,
            }
            group_defaults = _valid_columns(columns, defaults.get("group_cols"))
            metadata_defaults = _valid_columns(columns, defaults.get("metadata_cols"))
            preview = html.Div(
                [
                    _status(f"Loaded assay table {filename}: {len(frame):,} rows, {len(columns):,} columns.", kind="success"),
                    _status(source_message, kind="info") if source_message else None,
                    html.Div(_data_table(frame.head(8), page_size=8, height=260, compact=True), style={"marginTop": "10px"}),
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
                group_defaults or [column for column in ["batch", "construct_family", "organism"] if column in columns],
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
                        html.Div(_data_table(frame.head(6), page_size=6, height=220, compact=True), style={"marginTop": "10px"}),
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
                    frame = _concat_tables(assay_paths)
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

    @app.callback(Output("project-name", "value"), Input("analysis-type", "value"), prevent_initial_call=True)
    def update_project_default(analysis_type: str) -> str:
        return "assayready_prediction_audit" if analysis_type == "prediction" else "assayready_design_rank"

    @app.callback(
        Output("run-analysis-button", "disabled"),
        Output("run-analysis-button", "style"),
        Output("analysis-ready-state", "children"),
        Output("columns-card", "className"),
        Output("columns-empty-hint", "style"),
        Input("assay-upload-state", "data"),
        Input("sequence-col", "value"),
        Input("target-col", "value"),
        Input("prediction-col", "value"),
        Input("group-cols", "value"),
        Input("analysis-type", "value"),
    )
    def update_analysis_readiness(
        assay_state: dict[str, Any] | None,
        sequence_col: str | None,
        target_col: str | None,
        prediction_col: str | None,
        group_cols: list[str] | None,
        analysis_type: str,
    ) -> tuple[bool, dict[str, Any], Any, str, dict[str, str]]:
        assay_ready = bool(assay_state and assay_state.get("path"))
        available_columns = set(str(column) for column in (assay_state or {}).get("columns", []))
        columns_ready = bool(sequence_col in available_columns and target_col in available_columns)
        prediction_ready = bool(prediction_col in available_columns) if analysis_type == "prediction" else True
        ready = assay_ready and columns_ready and prediction_ready
        style = dict(BUTTON_STYLE if ready else SECONDARY_BUTTON_STYLE)
        if not ready:
            style.update({"cursor": "not-allowed", "opacity": 0.62})
        card_class = "ui-card columns-card" if assay_ready else "ui-card columns-card muted-card"
        hint_style = {"display": "none"} if assay_ready else {"display": "block"}
        cardinalities = (assay_state or {}).get("cardinalities") or {}
        low_cardinality_groups = [
            f"{column} ({cardinalities.get(column)} levels)"
            for column in (group_cols or [])
            if isinstance(cardinalities.get(column), int) and cardinalities[column] < 3
        ]
        readiness = _readiness_panel(assay_ready=assay_ready, columns_ready=columns_ready, prediction_ready=prediction_ready, analysis_type=analysis_type)
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
        return (
            not ready,
            style,
            readiness,
            card_class,
            hint_style,
        )

    @app.callback(
        Output("analysis-result", "children"),
        Input("run-analysis-button", "n_clicks"),
        State("assay-upload-state", "data"),
        State("candidate-upload-state", "data"),
        State("analysis-type", "value"),
        State("project-name", "value"),
        State("output-dir", "value"),
        State("sequence-col", "value"),
        State("target-col", "value"),
        State("prediction-col", "value"),
        State("uncertainty-col", "value"),
        State("id-col", "value"),
        State("group-cols", "value"),
        State("metadata-cols", "value"),
        State("task-type", "value"),
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
        prevent_initial_call=True,
    )
    def run_analysis(
        n_clicks: int,
        assay_state: dict[str, Any] | None,
        candidate_state: dict[str, Any] | None,
        analysis_type: str,
        project: str,
        output_dir: str | None,
        sequence_col: str | None,
        target_col: str | None,
        prediction_col: str | None,
        uncertainty_col: str | None,
        id_col: str | None,
        group_cols: list[str] | None,
        metadata_cols: list[str] | None,
        task_type: str,
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
    ) -> Any:
        if not n_clicks:
            raise PreventUpdate
        if not assay_state:
            return _status("Upload measured assay data before running analysis.", kind="warning")
        available_columns = set(str(column) for column in assay_state.get("columns", []))
        if sequence_col not in available_columns or target_col not in available_columns:
            return _status("Select sequence and measured target columns from the loaded assay table.", kind="warning")
        if analysis_type == "prediction" and prediction_col not in available_columns:
            return _status("Prediction audits require a prediction column from the loaded assay table.", kind="warning")
        invalid_optional = [
            column
            for column in [uncertainty_col, id_col, *(group_cols or []), *(metadata_cols or [])]
            if column and column not in available_columns
        ]
        if invalid_optional:
            return _status(f"These selected columns are not in the loaded assay table: {', '.join(invalid_optional)}", kind="warning")
        try:
            assay_files = [Path(path) for path in (assay_state.get("paths") or [assay_state["path"]])]
            candidate_files = [Path(path) for path in (candidate_state.get("paths") or [candidate_state["path"]])] if candidate_state and candidate_state.get("path") else []
            output_path = _safe_output_dir(output_dir) if output_dir else None
            configured_manifest = assay_state.get("evaluation_manifest")
            evaluation_manifest = dict(configured_manifest) if isinstance(configured_manifest, dict) else inferred_manifest_for_args(task_type=task_type or "regression").to_dict()
            if not configured_manifest:
                evaluation_manifest["objective_direction"] = str(evaluation_objective or "maximize")
                evaluation_manifest["units"] = str(evaluation_units or "unspecified")
                evaluation_manifest["uncertainty_type"] = str(evaluation_uncertainty_type or "none")
                evaluation_manifest["provenance"]["model_identifier"] = str(evaluation_model_id or "ui_unspecified")
                evaluation_manifest["provenance"]["data_identifier"] = str(evaluation_data_id or assay_state.get("filename") or "ui_unspecified")
                evaluation_manifest["provenance"]["split_identifier"] = "ui_leakage_aware_split"
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
                "column_aliases": {},
                "low_n_threshold": int(_value_or_default(low_n, 200)),
                "val_fraction": float(_value_or_default(val_fraction, 0.15)),
                "test_fraction": float(_value_or_default(test_fraction, 0.15)),
                "homology_threshold": float(_value_or_default(homology_threshold, 0.90)),
                "homology_k": int(_value_or_default(homology_k, 8)),
                "top_k": int(_value_or_default(top_k, 96)),
                "beta": float(_value_or_default(beta, 1.0)),
                "diversity_method": "greedy_embedding_cosine",
                "diversity_penalty": float(_value_or_default(diversity_penalty, 0.2)),
                "seed": int(_value_or_default(seed, 13)),
                "evaluation_manifest": evaluation_manifest,
                "manifest_source": "config" if configured_manifest else "ui_declared",
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
                    acquisition_method="upper_confidence_bound",
                    generate_candidates=not bool(candidate_files),
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
            return _result_summary(summary, report_name)
        except Exception as exc:
            return _failure_panel("Analysis failed", exc)

    @app.callback(Output("analysis-download", "data"), Input("analysis-download-button", "n_clicks"), State("analysis-artifact-select", "value"), prevent_initial_call=True)
    def download_analysis_artifact(n_clicks: int, path: str | None) -> Any:
        if not n_clicks or not path:
            raise PreventUpdate
        safe_path = _safe_artifact_path(path)
        return dcc.send_file(str(safe_path))

    @app.callback(Output("runs-content", "children"), Input("index-runs-button", "n_clicks"), Input("refresh-runs-button", "n_clicks"), prevent_initial_call=True)
    def refresh_runs(index_clicks: int, refresh_clicks: int) -> Any:
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        status = None
        if triggered == "index-runs-button":
            count = index_existing_runs()
            status = _status(f"Indexed {count} artifact folder(s).", kind="success")
        return _runs_content(status)

    @app.callback(Output("runs-run-detail", "children"), Input("runs-run-select", "value"), prevent_initial_call=True)
    def show_run_detail(run_id: str | None) -> Any:
        if not run_id:
            raise PreventUpdate
        summary = get_run_summary(run_id)
        if not summary:
            return _status("Could not load selected run summary.", kind="warning")
        return _result_summary(
            summary,
            "prediction_audit_report.md" if "prediction_audit" in summary else "simulation_report.md" if "simulation_report" in summary else "readiness_report.md",
            surface="runs",
        )

    @app.callback(Output("report-detail", "children"), Input("reports-run-select", "value"))
    def show_report(run_id: str | None) -> Any:
        if not run_id:
            return _status("No reports are indexed yet.")
        summary = get_run_summary(run_id)
        run = next((item for item in list_runs(limit=500) if str(item.get("run_id")) == str(run_id)), None)
        if not summary or not run:
            return _status("Could not load the selected report.", kind="warning")
        artifact_dir = Path(str(run["artifact_dir"]))
        report_name = str(run.get("report_name") or "readiness_report.md")
        report_path = artifact_dir / report_name
        artifacts = pd.DataFrame(list_artifacts(str(run_id)))
        artifact_opts = _artifact_options(summary, report_name)
        report_body = report_path.read_text(encoding="utf-8") if report_path.exists() else "Report markdown was not found."
        return html.Div(
            [
                _card(
                    [
                        dcc.Dropdown(id="report-artifact-select", options=artifact_opts, value=artifact_opts[0]["value"] if artifact_opts else None, clearable=False),
                        html.Button("Download selected artifact", id="report-download-button", n_clicks=0, style={**BUTTON_STYLE, "marginTop": "10px"}),
                        html.Div(f"Artifacts: {artifact_dir}", style={"color": COLORS["muted"], "marginTop": "10px"}),
                    ]
                ),
                _benchmark_section(summary, surface="reports"),
                _card(dcc.Markdown(report_body), style={"marginTop": "14px"}),
                _card([html.H3("Artifact Index", style={"marginTop": 0}), _data_table(artifacts, page_size=12)], style={"marginTop": "14px"}),
            ]
        )

    @app.callback(Output("report-download", "data"), Input("report-download-button", "n_clicks"), State("report-artifact-select", "value"), prevent_initial_call=True)
    def download_report_artifact(n_clicks: int, path: str | None) -> Any:
        if not n_clicks or not path:
            raise PreventUpdate
        return dcc.send_file(path)

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
        State("sim-seed", "value"),
        State("sim-project", "value"),
        State("sim-plate-size", "value"),
        State("sim-control-wells", "value"),
        State("sim-plate-seed", "value"),
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
        seed: int,
        project: str,
        plate_size: int,
        control_wells: int,
        plate_seed: int,
    ) -> Any:
        if not n_clicks:
            raise PreventUpdate
        if float(gc_low) > float(gc_high):
            return _status("Min GC must be less than or equal to Max GC.", kind="warning")
        rng = random.Random(int(_value_or_default(seed, 13)))
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        attempts = 0
        max_attempts = int(_value_or_default(num_candidates, 96)) * 80
        while len(candidates) < int(_value_or_default(num_candidates, 96)) and attempts < max_attempts:
            attempts += 1
            if mode == "Diversity proposal":
                effective_rate = min(0.5, float(_value_or_default(mutation_rate, 0.08)) * 2.5 + 0.04)
            elif mode == "Mutational scan":
                effective_rate = float(_value_or_default(mutation_rate, 0.08))
            else:
                effective_rate = float(_value_or_default(mutation_rate, 0.08)) * 0.35
            seq = _fill_masked_sequence(parent or "", rng, str(mode), effective_rate)
            if not seq or seq in seen:
                continue
            seen.add(seq)
            candidates.append({"design_id": f"design_sim_{len(candidates) + 1:04d}", "model": model_name, "mode": mode, "sequence": seq})
        rows = []
        target_gc = (float(gc_low), float(gc_high))
        for candidate in candidates:
            seq = str(candidate["sequence"])
            scores = _heuristic_sequence_scores(seq, target_gc, int(_value_or_default(max_homopolymer, 6)), rng)
            rows.append({**candidate, **scores})
        frame = pd.DataFrame(rows)
        if frame.empty:
            return _status("No candidates were generated. Check the sequence prompt and filters.", kind="warning")
        frame = frame.sort_values("acquisition_score", ascending=False).reset_index(drop=True)
        frame.insert(0, "rank", range(1, len(frame) + 1))

        p_seed = int(_value_or_default(plate_seed, _value_or_default(seed, 13)))
        plate_plan = _build_plate_plan(
            frame,
            plate_size=int(_value_or_default(plate_size, 96)),
            control_wells=int(_value_or_default(control_wells, 4)),
            seed=p_seed,
        )

        artifact_params = {
            "model_name": model_name,
            "generation_mode": mode,
            "num_candidates": int(_value_or_default(num_candidates, 96)),
            "context_multiplier": float(_value_or_default(context_multiplier, 2.0)),
            "mutation_rate": float(_value_or_default(mutation_rate, 0.08)),
            "gc_low": float(gc_low),
            "gc_high": float(gc_high),
            "max_homopolymer": int(_value_or_default(max_homopolymer, 6)),
            "seed": int(_value_or_default(seed, 13)),
            "plate_size": int(_value_or_default(plate_size, 96)),
            "control_wells": int(_value_or_default(control_wells, 4)),
            "plate_seed": p_seed,
        }
        artifact_dir, summary = _save_simulation_artifacts(
            project=str(_value_or_default(project, "dash_sequence_simulation")),
            frame=frame,
            plate_plan=plate_plan,
            params=artifact_params,
            scoring_caption="demo_heuristic_scorer",
        )
        artifact_opts = _artifact_options(summary, "simulation_report.md")
        display_cols = [column for column in ["rank", "design_id", "sequence", "predicted_activity", "model_uncertainty", "acquisition_score", "mlm_plausibility", "gc_fraction", "max_homopolymer", "passes_basic_filters", "scoring_backend"] if column in frame.columns]
        return html.Div(
            [
                _status("Synthetic planning output only — these plots are not part of the real benchmark evidence.", kind="warning"),
                html.Div(
                    [
                        _metric("Generated", len(frame)),
                        _metric("Pass Rate", f"{float(frame['passes_basic_filters'].mean()):.0%}"),
                        _metric("Top Score", f"{frame['acquisition_score'].max():.3f}"),
                    ],
                    className="metric-grid metric-grid-three",
                ),
                html.Div(
                    [
                        _graph_card(
                            simulation_landscape_figure(frame),
                            "Activity and uncertainty are heuristic proxies. Marker size reflects sequence plausibility; colors indicate only the configured basic filters.",
                        ),
                        _graph_card(
                            simulation_constraints_figure(
                                frame,
                                gc_low=float(gc_low),
                                gc_high=float(gc_high),
                                max_homopolymer=int(_value_or_default(max_homopolymer, 6)),
                            ),
                            "Dashed lines show the requested GC and homopolymer limits; distributions help reveal an over-restrictive prompt or generator.",
                        ),
                        _graph_card(
                            plate_layout_figure(plate_plan),
                            "This randomized layout is a draft planning artifact. Add lab-specific controls, replicates, blocking, and edge-well policy before execution.",
                        ),
                    ],
                    className="visualization-grid simulation-visualizations",
                ),
                _card([html.H3("Ranked Candidates", style={"marginTop": 0}), _data_table(frame[display_cols], page_size=15, height=460)], style={"marginTop": "14px"}),
                _card([html.H3("Plate Plan", style={"marginTop": 0}), _data_table(plate_plan, page_size=12, height=360)], style={"marginTop": "14px"}),
                _card(
                    [
                        html.Div(f"Artifacts saved to: {artifact_dir}", style={"color": COLORS["muted"], "marginBottom": "10px"}),
                        dcc.Dropdown(id="simulation-artifact-select", options=artifact_opts, value=artifact_opts[0]["value"] if artifact_opts else None, clearable=False),
                        html.Button("Download selected artifact", id="simulation-download-button", n_clicks=0, style={**BUTTON_STYLE, "marginTop": "10px"}),
                        dcc.Download(id="simulation-download"),
                    ],
                    style={"marginTop": "14px"},
                ),
                _status(str(summary.get("run_store_warning")), kind="warning") if summary.get("run_store_warning") else None,
            ]
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
