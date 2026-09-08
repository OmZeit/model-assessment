from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_DB_ENV = "ASSAYREADY_RUN_DB"
OUTPUT_ROOT_ENV = "ASSAYREADY_OUTPUT_ROOT"
WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def default_output_root() -> Path:
    configured = os.environ.get(OUTPUT_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    repository_root = Path(__file__).resolve().parents[2]
    if (repository_root / "pyproject.toml").is_file():
        return (repository_root / "outputs" / "assayready").resolve()
    return (Path.home() / ".assayready" / "outputs").resolve()


def default_db_path() -> Path:
    configured = os.environ.get(RUN_DB_ENV)
    if configured:
        return Path(configured).expanduser()
    return default_output_root() / "assayready_runs.sqlite"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return normalized or "run"


def _canonical_path_key(raw_path: str | Path) -> str:
    text = str(raw_path)
    if text.startswith("/mnt/") and len(text) > 6:
        drive = text[5].lower()
        rest = text[7:].replace("\\", "/")
        return f"{drive}:/{rest}"
    match = WINDOWS_DRIVE_RE.match(text)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        return f"{drive}:/{rest}"
    return str(Path(raw_path).expanduser().resolve()).replace("\\", "/")


def _run_id(project: str, artifact_dir: Path) -> str:
    digest = hashlib.sha256(_canonical_path_key(artifact_dir).encode("utf-8")).hexdigest()[:12]
    return f"{_slug(project)}-{digest}"


def _host_path(raw_path: str | Path) -> Path:
    text = str(raw_path)
    if os.name == "nt" and text.startswith("/mnt/") and len(text) > 6:
        drive = text[5].upper()
        rest = text[7:].replace("/", "\\")
        return Path(f"{drive}:\\{rest}").resolve()
    if os.name != "nt":
        match = WINDOWS_DRIVE_RE.match(text)
        if match:
            drive = match.group(1).lower()
            rest = match.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{rest}").resolve()
    return Path(raw_path).expanduser().resolve()


def _localize_run(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ["artifact_dir", "summary_path"]:
        if out.get(key):
            out[key] = str(_host_path(out[key]))
    return out


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            artifact_key TEXT UNIQUE,
            project TEXT NOT NULL,
            workflow TEXT NOT NULL,
            artifact_dir TEXT NOT NULL UNIQUE,
            report_name TEXT NOT NULL,
            summary_path TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            verdict TEXT,
            primary_metric_name TEXT,
            primary_metric REAL,
            best_baseline_name TEXT,
            best_baseline_metric REAL,
            uncertainty_spearman REAL,
            ranked_candidates INTEGER,
            accepted_rows INTEGER,
            test_rows INTEGER,
            warnings_json TEXT NOT NULL,
            params_json TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            schema_version INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'completed',
            started_at TEXT,
            completed_at TEXT,
            duration_seconds REAL,
            manifest_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS run_artifacts (
            run_id TEXT NOT NULL,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            PRIMARY KEY (run_id, name),
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS candidates (
            run_id TEXT NOT NULL,
            rank INTEGER,
            display_id TEXT,
            sequence_hash TEXT,
            sequence TEXT,
            prediction REAL,
            uncertainty REAL,
            acquisition_score REAL,
            diversified_acquisition_score REAL,
            diversity_cluster TEXT,
            training_distribution_status TEXT,
            nearest_train_id TEXT,
            nearest_train_similarity REAL,
            nearest_train_edit_distance INTEGER,
            risk_flags_json TEXT NOT NULL,
            ranking_reason TEXT,
            PRIMARY KEY (run_id, rank, display_id),
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );
        """
    )
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    migrations = {
        "artifact_key": "TEXT",
        "schema_version": "INTEGER NOT NULL DEFAULT 1",
        "status": "TEXT NOT NULL DEFAULT 'completed'",
        "started_at": "TEXT",
        "completed_at": "TEXT",
        "duration_seconds": "REAL",
        "manifest_json": "TEXT NOT NULL DEFAULT '{}'",
    }
    for name, definition in migrations.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")
    for row in conn.execute("SELECT run_id, artifact_dir FROM runs WHERE artifact_key IS NULL OR artifact_key = ''").fetchall():
        conn.execute(
            "UPDATE runs SET artifact_key = ? WHERE run_id = ?",
            (_canonical_path_key(row["artifact_dir"]), row["run_id"]),
        )
    duplicate_keys = conn.execute(
        "SELECT artifact_key FROM runs WHERE artifact_key IS NOT NULL AND artifact_key != '' "
        "GROUP BY artifact_key HAVING COUNT(*) > 1 LIMIT 1"
    ).fetchone()
    if duplicate_keys is None:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_artifact_key "
            "ON runs(artifact_key) WHERE artifact_key IS NOT NULL AND artifact_key != ''"
        )
    conn.commit()


def _report_from_summary(summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "prediction_audit" in summary:
        return "prediction", summary["prediction_audit"]
    if "simulation_report" in summary:
        return "simulation", summary["simulation_report"]
    return "internal", summary.get("benchmark_report", {})


def _summary_path(artifact_dir: Path, workflow: str) -> Path:
    if workflow == "prediction":
        name = "prediction_audit_summary.json"
    elif workflow == "simulation":
        name = "simulation_summary.json"
    else:
        name = "run_summary.json"
    return artifact_dir / name


def _candidate_rows(artifact_dir: Path) -> list[dict[str, Any]]:
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
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return []


def record_run(
    summary: dict[str, Any],
    *,
    workflow: str | None = None,
    report_name: str | None = None,
    db_path: str | Path | None = None,
) -> str:
    inferred_workflow, report = _report_from_summary(summary)
    workflow = workflow or inferred_workflow
    report_name = report_name or ("prediction_audit_report.md" if workflow == "prediction" else "readiness_report.md")
    artifact_dir = _host_path(summary["artifact_dir"])
    artifact_key = _canonical_path_key(artifact_dir)
    run_id = str(summary.get("run_id") or _run_id(str(summary.get("project") or "assayready"), artifact_dir))
    split_sizes = (summary.get("split_diagnostics") or {}).get("split_sizes") or {}
    best = report.get("best_simple_baseline") or {}
    uncertainty = report.get("uncertainty_audit") or {}
    if workflow == "prediction":
        primary_metric = (report.get("test_metrics") or {}).get("primary_metric")
    elif workflow == "simulation":
        primary_metric = report.get("primary_metric")
    else:
        primary_metric = report.get("task_head_primary_metric")
    warnings = report.get("warnings") or []
    accepted_rows = summary.get("valid_prediction_rows")
    if accepted_rows is None:
        accepted_rows = summary.get("valid_prediction_rows")
    if accepted_rows is None:
        accepted_rows = (summary.get("audit") or {}).get("accepted_rows")
    execution = summary.get("execution_manifest") or {}
    schema_version = _safe_int(summary.get("schema_version")) or 1
    status = str(execution.get("status") or "completed")
    started_at = execution.get("started_at")
    completed_at = execution.get("completed_at")
    duration_seconds = _safe_float(execution.get("duration_seconds"))
    summary_path = _summary_path(artifact_dir, workflow)
    now = _now()

    with connect(db_path) as conn:
        existing = conn.execute(
            "SELECT run_id, created_at FROM runs WHERE run_id = ? OR artifact_key = ? OR artifact_dir = ?",
            (run_id, artifact_key, str(artifact_dir)),
        ).fetchone()
        if existing:
            run_id = existing["run_id"]
        created_at = existing["created_at"] if existing else now
        conn.execute(
            """
            INSERT INTO runs (
                run_id, artifact_key, project, workflow, artifact_dir, report_name, summary_path,
                created_at, updated_at, verdict, primary_metric_name, primary_metric,
                best_baseline_name, best_baseline_metric, uncertainty_spearman,
                ranked_candidates, accepted_rows, test_rows, warnings_json, params_json, summary_json,
                schema_version, status, started_at, completed_at, duration_seconds, manifest_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                artifact_key = excluded.artifact_key,
                project = excluded.project,
                workflow = excluded.workflow,
                artifact_dir = excluded.artifact_dir,
                report_name = excluded.report_name,
                summary_path = excluded.summary_path,
                updated_at = excluded.updated_at,
                verdict = excluded.verdict,
                primary_metric_name = excluded.primary_metric_name,
                primary_metric = excluded.primary_metric,
                best_baseline_name = excluded.best_baseline_name,
                best_baseline_metric = excluded.best_baseline_metric,
                uncertainty_spearman = excluded.uncertainty_spearman,
                ranked_candidates = excluded.ranked_candidates,
                accepted_rows = excluded.accepted_rows,
                test_rows = excluded.test_rows,
                warnings_json = excluded.warnings_json,
                params_json = excluded.params_json,
                summary_json = excluded.summary_json,
                schema_version = excluded.schema_version,
                status = excluded.status,
                started_at = excluded.started_at,
                completed_at = excluded.completed_at,
                duration_seconds = excluded.duration_seconds,
                manifest_json = excluded.manifest_json
            """,
            (
                run_id,
                artifact_key,
                str(summary.get("project") or ""),
                workflow,
                str(artifact_dir),
                report_name,
                str(summary_path),
                created_at,
                now,
                report.get("verdict"),
                report.get("primary_metric_name"),
                _safe_float(primary_metric),
                best.get("name"),
                _safe_float(best.get("primary_metric")),
                _safe_float(uncertainty.get("uncertainty_abs_error_spearman")),
                _safe_int(summary.get("ranked_candidates")),
                _safe_int(accepted_rows),
                _safe_int(split_sizes.get("test")),
                _json(warnings),
                _json(summary.get("params") or {}),
                _json(summary),
                schema_version,
                status,
                started_at,
                completed_at,
                duration_seconds,
                _json(execution),
            ),
        )

        conn.execute("DELETE FROM run_artifacts WHERE run_id = ?", (run_id,))
        for name in summary.get("artifacts") or []:
            conn.execute(
                "INSERT OR REPLACE INTO run_artifacts (run_id, name, path) VALUES (?, ?, ?)",
                (run_id, str(name), str(artifact_dir / str(name))),
            )

        conn.execute("DELETE FROM candidates WHERE run_id = ?", (run_id,))
        for row in _candidate_rows(artifact_dir):
            conn.execute(
                """
                INSERT OR REPLACE INTO candidates (
                    run_id, rank, display_id, sequence_hash, sequence, prediction, uncertainty,
                    acquisition_score, diversified_acquisition_score, diversity_cluster,
                    training_distribution_status, nearest_train_id, nearest_train_similarity,
                    nearest_train_edit_distance, risk_flags_json, ranking_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    _safe_int(row.get("rank")),
                    str(row.get("display_id") or ""),
                    str(row.get("sequence_hash") or ""),
                    str(row.get("sequence") or ""),
                    _safe_float(row.get("prediction")),
                    _safe_float(row.get("uncertainty")),
                    _safe_float(row.get("acquisition_score")),
                    _safe_float(row.get("diversified_acquisition_score")),
                    str(row.get("diversity_cluster") or ""),
                    str(row.get("training_distribution_status") or ""),
                    str(row.get("nearest_train_id") or ""),
                    _safe_float(row.get("nearest_train_similarity")),
                    _safe_int(row.get("nearest_train_edit_distance")),
                    _json(row.get("risk_flags") or []),
                    str(row.get("ranking_reason") or ""),
                ),
            )
        conn.commit()
    return run_id


def list_runs(*, limit: int = 200, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY updated_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [_localize_run(dict(row)) for row in rows]


def get_run(run_id: str, *, db_path: str | Path | None = None) -> dict[str, Any] | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return _localize_run(dict(row)) if row else None


def get_run_summary(run_id: str, *, db_path: str | Path | None = None) -> dict[str, Any] | None:
    run = get_run(run_id, db_path=db_path)
    if not run:
        return None
    summary_path = Path(run["summary_path"])
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = json.loads(run["summary_json"])
    summary["artifact_dir"] = run["artifact_dir"]
    summary["artifact_verification"] = verify_run_artifacts(summary)
    return summary


def verify_run_artifacts(summary: dict[str, Any]) -> dict[str, Any]:
    execution = summary.get("execution_manifest") or {}
    records = execution.get("artifacts") or []
    errors: list[str] = []
    checked = 0
    for record in records:
        if not isinstance(record, dict) or not record.get("path"):
            continue
        path = _host_path(record["path"])
        if not path.is_file():
            errors.append(f"missing artifact: {path}")
            continue
        checked += 1
        expected_size = _safe_int(record.get("size_bytes"))
        if expected_size is not None and path.stat().st_size != expected_size:
            errors.append(f"size mismatch: {path}")
            continue
        expected_hash = str(record.get("sha256") or "")
        if expected_hash:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                errors.append(f"checksum mismatch: {path}")
    return {"ok": not errors, "checked": checked, "errors": errors}


def list_candidates(*, run_id: str | None = None, limit: int = 1000, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        if run_id:
            rows = conn.execute(
                "SELECT * FROM candidates WHERE run_id = ? ORDER BY rank ASC LIMIT ?",
                (run_id, int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT c.*, r.project, r.workflow, r.updated_at, r.verdict
                FROM candidates c
                JOIN runs r ON r.run_id = c.run_id
                ORDER BY r.updated_at DESC, c.rank ASC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
    return [dict(row) for row in rows]


def list_artifacts(run_id: str, *, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM run_artifacts WHERE run_id = ? ORDER BY name",
            (run_id,),
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["path"] = str(_host_path(item["path"]))
        out.append(item)
    return out


def index_existing_runs(root: str | Path | None = None, *, db_path: str | Path | None = None) -> int:
    root_path = Path(root) if root is not None else default_output_root()
    if not root_path.exists():
        return 0
    count = 0
    for summary_name in ["prediction_audit_summary.json", "run_summary.json", "simulation_summary.json"]:
        for path in root_path.rglob(summary_name):
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
                if summary_name.startswith("prediction"):
                    workflow = "prediction"
                    report_name = "prediction_audit_report.md"
                elif summary_name.startswith("simulation"):
                    workflow = "simulation"
                    report_name = "simulation_report.md"
                else:
                    workflow = "internal"
                    report_name = "readiness_report.md"
                record_run(summary, workflow=workflow, report_name=report_name, db_path=db_path)
                count += 1
            except Exception:
                continue
    return count
