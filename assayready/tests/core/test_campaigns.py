import csv
import sqlite3
from pathlib import Path

import pytest

from model_assessment.campaigns import CampaignStore


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _store_with_round(tmp_path: Path) -> tuple[CampaignStore, str]:
    store = CampaignStore(tmp_path / "campaigns.sqlite")
    store.create_campaign(
        campaign_id="enzyme-a", name="Enzyme A", assay_type="activity",
        objective_direction="maximize", owner="team-a",
    )
    selection = tmp_path / "selection.csv"
    _write(selection, [
        {"candidate_id": "a", "rank": 1, "sequence": "AAAA", "prediction": 0.9, "uncertainty": 0.1},
        {"candidate_id": "b", "rank": 2, "sequence": "CCCC", "prediction": 0.5, "uncertainty": 0.2},
        {"candidate_id": "c", "rank": 3, "sequence": "GGGG", "prediction": 0.1, "uncertainty": 0.3},
    ])
    store.register_round(
        campaign_id="enzyme-a", round_number=1, model_version="model@abc",
        dataset_version="dataset@123", selection_file=selection,
        selected_at="2026-01-01T00:00:00+00:00",
    )
    return store, "enzyme-a-r001"


def test_campaign_round_outcome_loop_and_comparison(tmp_path: Path) -> None:
    store, round_id = _store_with_round(tmp_path)
    outcomes = tmp_path / "outcomes.csv"
    _write(outcomes, [
        {"candidate_id": "a", "replicate": 1, "measured_value": 9.0, "batch": "x", "cost": 3},
        {"candidate_id": "a", "replicate": 2, "measured_value": 8.0, "batch": "y", "cost": 3},
        {"candidate_id": "b", "replicate": 1, "measured_value": 5.0, "batch": "x", "cost": 3},
        {"candidate_id": "c", "replicate": 1, "measured_value": 1.0, "batch": "x", "cost": 3},
    ])
    summary = store.import_outcomes(
        round_id=round_id, outcome_file=outcomes, measured_at="2026-01-02T00:00:00+00:00"
    )
    assert summary["coverage"] == 1.0
    assert summary["prediction_outcome_spearman"] == pytest.approx(1.0)
    assert summary["replicate_noise"]["replicated_candidates"] == 1
    comparison = store.compare_rounds("enzyme-a")
    assert comparison["rounds"][0]["model_version"] == "model@abc"
    protocol = store.prospective_protocol(round_id)
    assert protocol["candidate_selection_timestamp"] < protocol["measurement_timestamp"]
    assert len(protocol["selection_manifest_sha256"]) == 64
    assert len(protocol["outcome_data_sha256"]) == 64
    assert store.verify_audit_chain()["ok"] is True


def test_outcomes_cannot_add_unselected_candidate_or_predate_selection(tmp_path: Path) -> None:
    store, round_id = _store_with_round(tmp_path)
    outcomes = tmp_path / "outcomes.csv"
    _write(outcomes, [{"candidate_id": "unknown", "replicate": 1, "measured_value": 9.0}])
    with pytest.raises(ValueError, match="outside the locked selection"):
        store.import_outcomes(round_id=round_id, outcome_file=outcomes, measured_at="2026-01-02T00:00:00+00:00")
    _write(outcomes, [{"candidate_id": "a", "replicate": 1, "measured_value": 9.0}])
    with pytest.raises(ValueError, match="later"):
        store.import_outcomes(round_id=round_id, outcome_file=outcomes, measured_at="2025-12-31T00:00:00+00:00")


def test_audit_chain_detects_tampering(tmp_path: Path) -> None:
    store, _ = _store_with_round(tmp_path)
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("UPDATE audit_events SET actor='tampered' WHERE action='campaign.created'")
        connection.commit()
    assert store.verify_audit_chain()["ok"] is False


def test_archive_is_recoverable_and_purge_requires_exact_confirmation(tmp_path: Path) -> None:
    store, _ = _store_with_round(tmp_path)
    assert store.archive_campaign("enzyme-a")["status"] == "archived"
    with pytest.raises(ValueError, match="exactly match"):
        store.purge_campaign("enzyme-a", confirmation="wrong")
    result = store.purge_campaign("enzyme-a", confirmation="enzyme-a", actor="admin")
    assert result["purged"] is True
    assert store.get_campaign("enzyme-a") is None
    assert store.verify_audit_chain()["ok"] is True
