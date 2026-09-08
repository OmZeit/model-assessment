from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import model_assessment.foundation_scoring as foundation_scoring
from model_assessment.foundation_scoring import (
    ModelScoringError,
    inspect_foundation_scorer,
    load_verified_foundation_scorer,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    weights = tmp_path / "weights"
    head = tmp_path / "task_head"
    weights.mkdir()
    head.mkdir()
    (weights / "model.safetensors").write_bytes(b"model-bytes")
    model_digest = _sha(weights / "model.safetensors")
    (weights / "assayready_model_manifest.json").write_text(
        json.dumps(
            {
                "model_name": "Test DNA model",
                "repository": "example/test-dna",
                "revision": "revision-1",
                "embedding_size": 3,
                "weights_file": "model.safetensors",
                "weights_sha256": model_digest,
            }
        ),
        encoding="utf-8",
    )
    np.savez(head / "head.npz", coefficients=np.ones((2, 3), dtype=np.float32), intercepts=np.zeros(2, dtype=np.float32))
    (head / "head_manifest.json").write_text(
        json.dumps(
            {
                "head_sha256": _sha(head / "head.npz"),
                "head_type": "frozen_embedding_linear_ensemble",
                "task_type": "regression",
                "target_col": "activity",
                "ensemble_size": 2,
                "model_revision": "revision-1",
                "model_weights_sha256": model_digest,
                "training_rows": 100,
                "training_sha256": "training-digest",
                "max_length": 128,
            }
        ),
        encoding="utf-8",
    )
    return weights, head


def test_inspection_auto_detects_and_verifies_adjacent_head(tmp_path: Path) -> None:
    weights, head = _bundle(tmp_path)

    model_dir, head_dir, provenance = inspect_foundation_scorer(weights)

    assert model_dir == weights.resolve()
    assert head_dir == head.resolve()
    assert provenance.model_name == "Test DNA model"
    assert provenance.target_col == "activity"
    assert provenance.ensemble_size == 2
    assert provenance.model_weights_sha256 == _sha(weights / "model.safetensors")
    assert provenance.head_sha256 == _sha(head / "head.npz")
    assert provenance.objective_direction == "maximize"
    assert provenance.positive_label is None


def test_inspection_rejects_head_for_different_backbone(tmp_path: Path) -> None:
    weights, head = _bundle(tmp_path)
    manifest_path = head / "head_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_weights_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelScoringError, match="different foundation-model weights"):
        inspect_foundation_scorer(weights)


def test_inspection_rejects_tampered_head(tmp_path: Path) -> None:
    weights, head = _bundle(tmp_path)
    (head / "head.npz").write_bytes((head / "head.npz").read_bytes() + b"tampered")

    with pytest.raises(ModelScoringError, match="does not match"):
        inspect_foundation_scorer(weights)


def test_inspection_verifies_every_model_manifest_file(tmp_path: Path) -> None:
    weights, _head = _bundle(tmp_path)
    config = weights / "config.json"
    config.write_bytes(b"trusted-config")
    manifest_path = weights / "assayready_model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        {
            "name": "model.safetensors",
            "size_bytes": (weights / "model.safetensors").stat().st_size,
            "sha256": _sha(weights / "model.safetensors"),
        },
        {"name": "config.json", "size_bytes": config.stat().st_size, "sha256": _sha(config)},
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    inspect_foundation_scorer(weights)
    config.write_bytes(b"changed-config")

    with pytest.raises(ModelScoringError, match="model manifest.*SHA-256 mismatch"):
        inspect_foundation_scorer(weights)


def test_inspection_rejects_manifest_file_traversal(tmp_path: Path) -> None:
    weights, _head = _bundle(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside")
    manifest_path = weights / "assayready_model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        {
            "name": "../outside.json",
            "size_bytes": outside.stat().st_size,
            "sha256": _sha(outside),
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelScoringError, match="model manifest.*unsafe path component"):
        inspect_foundation_scorer(weights)


def test_inspection_propagates_head_semantics(tmp_path: Path) -> None:
    weights, head = _bundle(tmp_path)
    manifest_path = head / "head_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"task_type": "classification", "positive_label": "active", "objective_direction": "minimize"})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _model_dir, _head_dir, provenance = inspect_foundation_scorer(weights)

    assert provenance.task_type == "classification"
    assert provenance.positive_label == "active"
    assert provenance.objective_direction == "minimize"
    assert provenance.to_dict()["positive_label"] == "active"


def test_inspection_rejects_invalid_objective_direction(tmp_path: Path) -> None:
    weights, head = _bundle(tmp_path)
    manifest_path = head / "head_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["objective_direction"] = "sideways"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelScoringError, match="objective_direction"):
        inspect_foundation_scorer(weights)


def test_loading_inspects_once_before_populating_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights, _head = _bundle(tmp_path)
    real_inspect = foundation_scoring.inspect_foundation_scorer
    inspections = 0

    def counted_inspect(*args: object, **kwargs: object):
        nonlocal inspections
        inspections += 1
        return real_inspect(*args, **kwargs)

    class StubScorer:
        def __init__(self, model_dir: Path, head_dir: Path, provenance: object, *, device: str) -> None:
            self.model_dir = model_dir
            self.head_dir = head_dir
            self.provenance = provenance
            self.device = device

    foundation_scoring._load_verified_scorer_cached.cache_clear()
    monkeypatch.setattr(foundation_scoring, "DNABERT2_REPO", "example/test-dna")
    monkeypatch.setattr(foundation_scoring, "DNABERT2_REVISION", "revision-1")
    monkeypatch.setattr(
        foundation_scoring,
        "verify_pinned_dnabert2_snapshot",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(foundation_scoring, "inspect_foundation_scorer", counted_inspect)
    monkeypatch.setattr(foundation_scoring, "VerifiedFoundationScorer", StubScorer)
    try:
        scorer = load_verified_foundation_scorer(weights)
        assert scorer.model_dir == weights.resolve()
        assert inspections == 1
    finally:
        foundation_scoring._load_verified_scorer_cached.cache_clear()


def test_direct_loading_rejects_non_pinned_model_before_runtime_import(tmp_path: Path) -> None:
    weights, _head = _bundle(tmp_path)

    with pytest.raises(ModelScoringError, match="Direct sandbox scoring supports only"):
        load_verified_foundation_scorer(weights)
