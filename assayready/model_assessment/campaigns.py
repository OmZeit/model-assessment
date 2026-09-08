from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from model_assessment.statistics import replicate_noise, spearman


CAMPAIGN_DB_ENV = "ASSAYREADY_CAMPAIGN_DB"
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_campaign_db_path() -> Path:
    configured = os.environ.get(CAMPAIGN_DB_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path("outputs") / "assayready" / "assayready_campaigns.sqlite"


def _identifier(value: str, *, field: str) -> str:
    text = str(value or "").strip()
    if not IDENTIFIER_RE.match(text):
        raise ValueError(f"{field} must contain 1-100 letters, numbers, periods, underscores, or hyphens.")
    return text


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CampaignStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else default_campaign_db_path()

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        self._init(connection)
        return connection

    @staticmethod
    def _init(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS campaigns (
                campaign_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                assay_type TEXT NOT NULL,
                objective_direction TEXT NOT NULL,
                owner TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS campaign_rounds (
                round_id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                round_number INTEGER NOT NULL,
                run_id TEXT,
                model_version TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                policy_name TEXT,
                policy_sha256 TEXT,
                selection_path TEXT NOT NULL,
                selection_sha256 TEXT NOT NULL,
                selected_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'selected',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                outcome_path TEXT,
                outcome_sha256 TEXT,
                measured_at TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(campaign_id, round_number),
                FOREIGN KEY(campaign_id) REFERENCES campaigns(campaign_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS round_candidates (
                round_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                rank INTEGER,
                sequence_hash TEXT,
                prediction REAL,
                uncertainty REAL,
                family TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY(round_id, candidate_id),
                FOREIGN KEY(round_id) REFERENCES campaign_rounds(round_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS outcomes (
                round_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                replicate TEXT NOT NULL,
                measured_value REAL NOT NULL,
                batch TEXT,
                measured_at TEXT NOT NULL,
                cost REAL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY(round_id, candidate_id, replicate),
                FOREIGN KEY(round_id, candidate_id) REFERENCES round_candidates(round_id, candidate_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                occurred_at TEXT NOT NULL,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                details_json TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE
            );
            """
        )
        round_columns = {row["name"] for row in connection.execute("PRAGMA table_info(campaign_rounds)")}
        for name in ("outcome_path", "outcome_sha256", "measured_at"):
            if name not in round_columns:
                connection.execute(f"ALTER TABLE campaign_rounds ADD COLUMN {name} TEXT")
        connection.commit()

    def _audit(
        self,
        connection: sqlite3.Connection,
        *,
        actor: str,
        action: str,
        resource_type: str,
        resource_id: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        previous = connection.execute(
            "SELECT event_hash FROM audit_events ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        previous_hash = previous["event_hash"] if previous else "0" * 64
        event_id = uuid.uuid4().hex
        occurred_at = _now()
        canonical = _json(
            {
                "event_id": event_id,
                "occurred_at": occurred_at,
                "actor": actor,
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "details": details or {},
                "previous_hash": previous_hash,
            }
        )
        event_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        connection.execute(
            "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                occurred_at,
                actor,
                action,
                resource_type,
                resource_id,
                _json(details or {}),
                previous_hash,
                event_hash,
            ),
        )

    def create_campaign(
        self,
        *,
        campaign_id: str,
        name: str,
        assay_type: str,
        objective_direction: str,
        owner: str,
        metadata: dict[str, Any] | None = None,
        actor: str = "local-cli",
    ) -> dict[str, Any]:
        campaign_id = _identifier(campaign_id, field="campaign_id")
        if objective_direction not in {"maximize", "minimize"}:
            raise ValueError("objective_direction must be maximize or minimize.")
        if not all(str(value).strip() for value in (name, assay_type, owner)):
            raise ValueError("name, assay_type, and owner are required.")
        now = _now()
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO campaigns VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)",
                (campaign_id, name.strip(), assay_type.strip(), objective_direction, owner.strip(), _json(metadata or {}), now, now),
            )
            self._audit(
                connection,
                actor=actor,
                action="campaign.created",
                resource_type="campaign",
                resource_id=campaign_id,
                details={"name": name, "assay_type": assay_type},
            )
        return self.get_campaign(campaign_id) or {}

    def list_campaigns(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM campaigns ORDER BY updated_at DESC").fetchall()
        return [dict(row) for row in rows]

    def get_campaign(self, campaign_id: str) -> dict[str, Any] | None:
        campaign_id = _identifier(campaign_id, field="campaign_id")
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM campaigns WHERE campaign_id = ?", (campaign_id,)).fetchone()
        return dict(row) if row else None

    def archive_campaign(self, campaign_id: str, *, actor: str = "local-cli") -> dict[str, Any]:
        campaign_id = _identifier(campaign_id, field="campaign_id")
        with self.connect() as connection:
            if connection.execute("SELECT 1 FROM campaigns WHERE campaign_id=?", (campaign_id,)).fetchone() is None:
                raise ValueError(f"unknown campaign: {campaign_id}")
            connection.execute(
                "UPDATE campaigns SET status='archived', updated_at=? WHERE campaign_id=?", (_now(), campaign_id)
            )
            self._audit(
                connection, actor=actor, action="campaign.archived",
                resource_type="campaign", resource_id=campaign_id,
            )
        return self.get_campaign(campaign_id) or {}

    def purge_campaign(
        self, campaign_id: str, *, confirmation: str, actor: str = "local-cli"
    ) -> dict[str, Any]:
        campaign_id = _identifier(campaign_id, field="campaign_id")
        if confirmation != campaign_id:
            raise ValueError("purge confirmation must exactly match campaign_id.")
        with self.connect() as connection:
            counts = {
                "rounds": connection.execute(
                    "SELECT COUNT(*) FROM campaign_rounds WHERE campaign_id=?", (campaign_id,)
                ).fetchone()[0],
                "outcomes": connection.execute(
                    """SELECT COUNT(*) FROM outcomes o JOIN campaign_rounds r ON r.round_id=o.round_id
                    WHERE r.campaign_id=?""", (campaign_id,)
                ).fetchone()[0],
            }
            if connection.execute("SELECT 1 FROM campaigns WHERE campaign_id=?", (campaign_id,)).fetchone() is None:
                raise ValueError(f"unknown campaign: {campaign_id}")
            self._audit(
                connection, actor=actor, action="campaign.purged",
                resource_type="campaign", resource_id=campaign_id, details=counts,
            )
            connection.execute("DELETE FROM campaigns WHERE campaign_id=?", (campaign_id,))
        return {"campaign_id": campaign_id, "purged": True, **counts}

    def register_round(
        self,
        *,
        campaign_id: str,
        round_number: int,
        model_version: str,
        dataset_version: str,
        selection_file: str | Path,
        id_column: str = "candidate_id",
        prediction_column: str = "prediction",
        uncertainty_column: str | None = "uncertainty",
        run_id: str | None = None,
        policy_name: str | None = None,
        policy_sha256: str | None = None,
        selected_at: str | None = None,
        family_column: str | None = None,
        metadata: dict[str, Any] | None = None,
        actor: str = "local-cli",
    ) -> dict[str, Any]:
        campaign_id = _identifier(campaign_id, field="campaign_id")
        if int(round_number) < 1:
            raise ValueError("round_number must be positive.")
        if not str(model_version).strip() or not str(dataset_version).strip():
            raise ValueError("model_version and dataset_version are required.")
        path = Path(selection_file).resolve()
        if not path.is_file():
            raise ValueError(f"selection file does not exist: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            candidates = list(csv.DictReader(handle))
        if not candidates or id_column not in candidates[0]:
            raise ValueError(f"selection file must contain rows and an {id_column!r} column.")
        candidate_ids = [str(row.get(id_column) or "").strip() for row in candidates]
        if any(not item for item in candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("selection candidate identifiers must be non-empty and unique.")
        round_id = f"{campaign_id}-r{int(round_number):03d}"
        timestamp = selected_at or _now()
        digest = _sha256(path)
        with self.connect() as connection:
            if connection.execute("SELECT 1 FROM campaigns WHERE campaign_id = ?", (campaign_id,)).fetchone() is None:
                raise ValueError(f"unknown campaign: {campaign_id}")
            connection.execute(
                """
                INSERT INTO campaign_rounds (
                    round_id, campaign_id, round_number, run_id, model_version, dataset_version,
                    policy_name, policy_sha256, selection_path, selection_sha256, selected_at,
                    status, metadata_json, outcome_path, outcome_sha256, measured_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'selected', ?, NULL, NULL, NULL, ?)
                """,
                (
                    round_id, campaign_id, int(round_number), run_id, model_version, dataset_version,
                    policy_name, policy_sha256, str(path), digest, timestamp, _json(metadata or {}), _now(),
                ),
            )
            for rank, (candidate_id, row) in enumerate(zip(candidate_ids, candidates), start=1):
                def number(column: str | None) -> float | None:
                    try:
                        return float(row.get(column)) if column and str(row.get(column) or "").strip() else None
                    except ValueError:
                        return None
                sequence = str(row.get("sequence") or "")
                sequence_hash = str(row.get("sequence_hash") or "") or (
                    hashlib.sha256(sequence.encode("utf-8")).hexdigest() if sequence else ""
                )
                connection.execute(
                    "INSERT INTO round_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        round_id, candidate_id, int(float(row.get("rank") or rank)), sequence_hash,
                        number(prediction_column), number(uncertainty_column),
                        str(row.get(family_column) or "") if family_column else "", _json(row),
                    ),
                )
            connection.execute("UPDATE campaigns SET updated_at = ? WHERE campaign_id = ?", (_now(), campaign_id))
            self._audit(
                connection,
                actor=actor,
                action="round.selection_locked",
                resource_type="round",
                resource_id=round_id,
                details={"selection_sha256": digest, "candidate_count": len(candidates)},
            )
        return self.get_round(round_id) or {}

    def get_round(self, round_id: str) -> dict[str, Any] | None:
        round_id = _identifier(round_id, field="round_id")
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM campaign_rounds WHERE round_id = ?", (round_id,)).fetchone()
            count = connection.execute("SELECT COUNT(*) AS n FROM round_candidates WHERE round_id = ?", (round_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["candidate_count"] = int(count["n"])
        return result

    def import_outcomes(
        self,
        *,
        round_id: str,
        outcome_file: str | Path,
        id_column: str = "candidate_id",
        value_column: str = "measured_value",
        replicate_column: str | None = "replicate",
        batch_column: str | None = "batch",
        cost_column: str | None = "cost",
        measured_at: str | None = None,
        actor: str = "local-cli",
    ) -> dict[str, Any]:
        round_id = _identifier(round_id, field="round_id")
        path = Path(outcome_file).resolve()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or id_column not in rows[0] or value_column not in rows[0]:
            raise ValueError("outcome file must contain candidate and measured-value columns.")
        timestamp = measured_at or _now()
        with self.connect() as connection:
            round_row = connection.execute("SELECT * FROM campaign_rounds WHERE round_id = ?", (round_id,)).fetchone()
            if not round_row:
                raise ValueError(f"unknown round: {round_id}")
            try:
                selected_time = datetime.fromisoformat(str(round_row["selected_at"]).replace("Z", "+00:00"))
                measured_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("selected_at and measured_at must be ISO-8601 timestamps.") from exc
            if selected_time.tzinfo is None or measured_time.tzinfo is None:
                raise ValueError("selected_at and measured_at must include an explicit UTC offset or Z.")
            if measured_time <= selected_time:
                raise ValueError("measured_at must be later than the locked selection timestamp.")
            known = {
                row["candidate_id"]
                for row in connection.execute("SELECT candidate_id FROM round_candidates WHERE round_id = ?", (round_id,))
            }
            unknown = sorted({str(row.get(id_column) or "").strip() for row in rows} - known)
            if unknown:
                raise ValueError(f"outcomes contain candidates outside the locked selection: {unknown[:10]}")
            for index, row in enumerate(rows, start=1):
                candidate_id = str(row[id_column]).strip()
                replicate = str(row.get(replicate_column) or index) if replicate_column else str(index)
                try:
                    measured_value = float(row[value_column])
                    cost = float(row[cost_column]) if cost_column and str(row.get(cost_column) or "").strip() else None
                except ValueError as exc:
                    raise ValueError(f"invalid numeric outcome on row {index}") from exc
                connection.execute(
                    """INSERT INTO outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(round_id, candidate_id, replicate) DO UPDATE SET
                        measured_value=excluded.measured_value, batch=excluded.batch,
                        measured_at=excluded.measured_at, cost=excluded.cost, metadata_json=excluded.metadata_json""",
                    (
                        round_id, candidate_id, replicate, measured_value,
                        str(row.get(batch_column) or "") if batch_column else "", timestamp, cost, _json(row),
                    ),
                )
            outcome_digest = _sha256(path)
            connection.execute(
                "UPDATE campaign_rounds SET status='measured', outcome_path=?, outcome_sha256=?, measured_at=? WHERE round_id=?",
                (str(path), outcome_digest, timestamp, round_id),
            )
            connection.execute("UPDATE campaigns SET updated_at = ? WHERE campaign_id = ?", (_now(), round_row["campaign_id"]))
            self._audit(
                connection,
                actor=actor,
                action="round.outcomes_imported",
                resource_type="round",
                resource_id=round_id,
                details={"outcome_sha256": outcome_digest, "row_count": len(rows)},
            )
        return self.summarize_round(round_id)

    def summarize_round(self, round_id: str) -> dict[str, Any]:
        round_info = self.get_round(round_id)
        if not round_info:
            raise ValueError(f"unknown round: {round_id}")
        with self.connect() as connection:
            rows = [dict(row) for row in connection.execute(
                """SELECT c.candidate_id, c.rank, c.prediction, c.uncertainty,
                o.replicate, o.measured_value, o.batch, o.cost
                FROM round_candidates c LEFT JOIN outcomes o
                ON o.round_id=c.round_id AND o.candidate_id=c.candidate_id
                WHERE c.round_id=? ORDER BY c.rank, c.candidate_id""", (round_id,)
            ).fetchall()]
        measured = [row for row in rows if row["measured_value"] is not None]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in measured:
            grouped.setdefault(row["candidate_id"], []).append(row)
        aggregates = [
            {
                "candidate_id": candidate_id,
                "rank": values[0]["rank"],
                "prediction": values[0]["prediction"],
                "measured_value": mean(row["measured_value"] for row in values),
                "replicates": len(values),
                "cost": sum(row["cost"] or 0 for row in values),
            }
            for candidate_id, values in grouped.items()
        ]
        predicted = [row for row in aggregates if row["prediction"] is not None]
        objective = self.get_campaign(round_info["campaign_id"])["objective_direction"]
        values = [row["measured_value"] for row in aggregates]
        if values:
            ordered = sorted(values, reverse=objective == "maximize")
            threshold = ordered[max(0, min(len(ordered) - 1, len(ordered) // 4))]
            hits = sum(
                value >= threshold if objective == "maximize" else value <= threshold for value in values
            )
        else:
            threshold, hits = None, 0
        noise = replicate_noise(measured, id_col="candidate_id", value_col="measured_value")
        return {
            "schema_version": 1,
            "round": round_info,
            "selected_candidates": round_info["candidate_count"],
            "measured_candidates": len(aggregates),
            "measurement_rows": len(measured),
            "coverage": len(aggregates) / round_info["candidate_count"] if round_info["candidate_count"] else 0.0,
            "prediction_outcome_spearman": spearman(
                [row["prediction"] for row in predicted], [row["measured_value"] for row in predicted]
            ),
            "observed_mean": mean(values) if values else None,
            "observed_best": (max(values) if objective == "maximize" else min(values)) if values else None,
            "descriptive_top_quartile_threshold": threshold,
            "descriptive_hit_rate": hits / len(values) if values else None,
            "total_reported_cost": sum(row["cost"] for row in aggregates),
            "replicate_noise": noise,
            "candidate_outcomes": aggregates,
        }

    def compare_rounds(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"unknown campaign: {campaign_id}")
        with self.connect() as connection:
            round_ids = [row["round_id"] for row in connection.execute(
                "SELECT round_id FROM campaign_rounds WHERE campaign_id=? ORDER BY round_number", (campaign_id,)
            )]
        summaries = [self.summarize_round(round_id) for round_id in round_ids]
        trend = []
        previous = None
        for summary in summaries:
            current = summary["observed_mean"]
            trend.append(
                {
                    "round_id": summary["round"]["round_id"],
                    "round_number": summary["round"]["round_number"],
                    "model_version": summary["round"]["model_version"],
                    "dataset_version": summary["round"]["dataset_version"],
                    "coverage": summary["coverage"],
                    "prediction_outcome_spearman": summary["prediction_outcome_spearman"],
                    "observed_mean": current,
                    "mean_change_from_previous": current - previous if current is not None and previous is not None else None,
                    "total_reported_cost": summary["total_reported_cost"],
                }
            )
            if current is not None:
                previous = current
        return {"schema_version": 1, "campaign": campaign, "rounds": trend}

    def prospective_protocol(self, round_id: str) -> dict[str, Any]:
        round_info = self.get_round(round_id)
        if not round_info:
            raise ValueError(f"unknown round: {round_id}")
        required = ["outcome_path", "outcome_sha256", "measured_at"]
        if any(not round_info.get(field) for field in required):
            raise ValueError("round does not yet have imported prospective outcomes.")
        return {
            "protocol_identifier": round_id,
            "candidate_selection_timestamp": round_info["selected_at"],
            "measurement_timestamp": round_info["measured_at"],
            "selection_manifest_path": round_info["selection_path"],
            "selection_manifest_sha256": round_info["selection_sha256"],
            "outcome_data_sha256": round_info["outcome_sha256"],
        }

    def export_campaign(self, campaign_id: str) -> dict[str, Any]:
        comparison = self.compare_rounds(campaign_id)
        with self.connect() as connection:
            events = [dict(row) for row in connection.execute(
                "SELECT * FROM audit_events WHERE resource_id=? OR resource_id LIKE ? ORDER BY rowid",
                (campaign_id, f"{campaign_id}-r%"),
            )]
        comparison["audit_events"] = events
        comparison["exported_at"] = _now()
        return comparison

    def verify_audit_chain(self) -> dict[str, Any]:
        with self.connect() as connection:
            rows = [dict(row) for row in connection.execute(
                "SELECT * FROM audit_events ORDER BY rowid"
            )]
        previous_hash = "0" * 64
        errors = []
        for row in rows:
            details = json.loads(row["details_json"])
            canonical = _json(
                {
                    "event_id": row["event_id"], "occurred_at": row["occurred_at"],
                    "actor": row["actor"], "action": row["action"],
                    "resource_type": row["resource_type"], "resource_id": row["resource_id"],
                    "details": details, "previous_hash": row["previous_hash"],
                }
            )
            expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if row["previous_hash"] != previous_hash or row["event_hash"] != expected:
                errors.append(row["event_id"])
            previous_hash = row["event_hash"]
        return {"ok": not errors, "events_checked": len(rows), "invalid_event_ids": errors}
