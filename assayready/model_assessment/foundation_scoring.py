from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .open_models import (
    DNABERT2_REPO,
    DNABERT2_REVISION,
    safe_manifest_file_path,
    verify_manifest_file_records,
    verify_pinned_dnabert2_snapshot,
)


class ModelScoringError(RuntimeError):
    """Raised when a local foundation-model scorer cannot be verified or run."""


@dataclass(frozen=True)
class ScorerProvenance:
    model_name: str
    model_repository: str
    model_revision: str
    model_weights_sha256: str
    head_type: str
    head_sha256: str
    task_type: str
    target_col: str
    ensemble_size: int
    training_rows: int
    training_sha256: str
    max_length: int
    device: str = "not_loaded"
    positive_label: str | None = None
    objective_direction: str = "maximize"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def caption(self) -> str:
        return f"{self.model_name} + verified {self.target_col} head"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelScoringError(f"Could not read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelScoringError(f"{label} must contain a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_scorer_paths(model_path: str | Path, head_path: str | Path | None = None) -> tuple[Path, Path]:
    selected = Path(model_path).expanduser().resolve()
    if (selected / "assayready_model_manifest.json").is_file():
        model_dir = selected
        bundle_root = selected.parent
    elif (selected / "weights" / "assayready_model_manifest.json").is_file():
        bundle_root = selected
        model_dir = selected / "weights"
    else:
        raise ModelScoringError(
            f"No AssayReady model manifest was found in {selected} or its weights subdirectory."
        )

    head_dir = Path(head_path).expanduser().resolve() if head_path else bundle_root / "task_head"
    if not (head_dir / "head_manifest.json").is_file() or not (head_dir / "head.npz").is_file():
        raise ModelScoringError(
            f"An assay-specific head is required. Expected head_manifest.json and head.npz in {head_dir}."
        )
    return model_dir, head_dir


def inspect_foundation_scorer(
    model_path: str | Path,
    head_path: str | Path | None = None,
    *,
    require_direct_runtime: bool = False,
) -> tuple[Path, Path, ScorerProvenance]:
    model_dir, head_dir = resolve_scorer_paths(model_path, head_path)
    model_manifest = _read_json(model_dir / "assayready_model_manifest.json", "model manifest")
    head_manifest = _read_json(head_dir / "head_manifest.json", "head manifest")

    try:
        verified_model_files = verify_manifest_file_records(model_dir, model_manifest)
        weights_name, weights_path = safe_manifest_file_path(
            model_dir,
            model_manifest.get("weights_file") or "model.safetensors",
            label="model weights",
        )
    except ValueError as exc:
        raise ModelScoringError(f"The model manifest is invalid: {exc}") from exc
    if require_direct_runtime:
        if (
            str(model_manifest.get("repository") or "") != DNABERT2_REPO
            or str(model_manifest.get("revision") or "") != DNABERT2_REVISION
        ):
            raise ModelScoringError(
                "Direct sandbox scoring supports only AssayReady's pinned DNABERT-2 runtime. "
                "Use a prediction-table audit or locked container for another model."
            )
        try:
            verify_pinned_dnabert2_snapshot(
                model_dir,
                verified_files=verified_model_files,
            )
        except ValueError as exc:
            raise ModelScoringError(f"Pinned DNABERT-2 runtime verification failed: {exc}") from exc
    expected_model_digest = str(model_manifest.get("weights_sha256") or "")
    verified_weights = verified_model_files.get(weights_name)
    try:
        actual_model_digest = (
            verified_weights[2] if verified_weights is not None else _sha256_file(weights_path)
        )
    except OSError as exc:
        raise ModelScoringError(f"Could not hash the model weights at {weights_path}: {exc}") from exc
    if not expected_model_digest or actual_model_digest != expected_model_digest:
        raise ModelScoringError("The model weights do not match the SHA-256 recorded in their manifest.")

    head_file = head_dir / "head.npz"
    expected_head_digest = str(head_manifest.get("head_sha256") or "")
    actual_head_digest = _sha256_file(head_file)
    if not expected_head_digest or actual_head_digest != expected_head_digest:
        raise ModelScoringError("The assay head does not match the SHA-256 recorded in its manifest.")
    if str(head_manifest.get("model_weights_sha256") or "") != actual_model_digest:
        raise ModelScoringError("The assay head was trained for different foundation-model weights.")
    if str(head_manifest.get("model_revision") or "") != str(model_manifest.get("revision") or ""):
        raise ModelScoringError("The assay head and model manifests name different model revisions.")
    if str(head_manifest.get("head_type") or "") != "frozen_embedding_linear_ensemble":
        raise ModelScoringError("Unsupported assay head type; expected frozen_embedding_linear_ensemble.")
    task_type = str(head_manifest.get("task_type") or "")
    if task_type not in {"regression", "classification"}:
        raise ModelScoringError(f"Unsupported assay-head task type: {task_type or 'missing'}.")
    objective_direction = (
        "maximize"
        if "objective_direction" not in head_manifest
        else str(head_manifest["objective_direction"]).strip().lower()
    )
    if objective_direction not in {"maximize", "minimize"}:
        raise ModelScoringError("Assay-head objective_direction must be 'maximize' or 'minimize'.")
    raw_positive_label = head_manifest.get("positive_label")
    positive_label = None if raw_positive_label is None else str(raw_positive_label)

    try:
        with np.load(head_file, allow_pickle=False) as payload:
            coefficients = np.asarray(payload["coefficients"])
            intercepts = np.asarray(payload["intercepts"])
    except (OSError, KeyError, ValueError) as exc:
        raise ModelScoringError(f"Could not load the assay head: {exc}") from exc
    expected_embedding = int(model_manifest.get("embedding_size") or 0)
    if coefficients.ndim != 2 or intercepts.shape != (coefficients.shape[0],):
        raise ModelScoringError("The assay head has invalid coefficient/intercept dimensions.")
    if coefficients.shape[0] < 2 or coefficients.shape[1] != expected_embedding:
        raise ModelScoringError(
            f"The assay head shape {coefficients.shape} is incompatible with embedding size {expected_embedding}."
        )
    if int(head_manifest.get("ensemble_size") or 0) != coefficients.shape[0]:
        raise ModelScoringError("The assay-head ensemble size does not match its manifest.")

    provenance = ScorerProvenance(
        model_name=str(model_manifest.get("model_name") or "Local foundation model"),
        model_repository=str(model_manifest.get("repository") or head_manifest.get("model_repository") or ""),
        model_revision=str(model_manifest.get("revision") or ""),
        model_weights_sha256=actual_model_digest,
        head_type=str(head_manifest["head_type"]),
        head_sha256=actual_head_digest,
        task_type=task_type,
        target_col=str(head_manifest.get("target_col") or "assay outcome"),
        ensemble_size=int(head_manifest["ensemble_size"]),
        training_rows=int(head_manifest.get("training_rows") or 0),
        training_sha256=str(head_manifest.get("training_sha256") or ""),
        max_length=int(head_manifest.get("max_length") or 512),
        positive_label=positive_label,
        objective_direction=objective_direction,
    )
    return model_dir, head_dir, provenance


class VerifiedFoundationScorer:
    def __init__(self, model_dir: Path, head_dir: Path, provenance: ScorerProvenance, *, device: str = "auto") -> None:
        from .model_bundles.dnabert2_117m.runtime import choose_device, load_frozen_backbone

        self.model_dir = model_dir
        self.head_dir = head_dir
        self.device = choose_device(device)
        self.tokenizer, self.model = load_frozen_backbone(model_dir, self.device)
        with np.load(head_dir / "head.npz", allow_pickle=False) as payload:
            self.coefficients = np.asarray(payload["coefficients"], dtype=np.float32)
            self.intercepts = np.asarray(payload["intercepts"], dtype=np.float32)
        self.provenance = ScorerProvenance(**{**provenance.to_dict(), "device": str(self.device)})

    def sequence_token_lengths(self, sequences: list[str]) -> list[int]:
        """Return untruncated token counts so callers can prevent silent truncation."""
        if not sequences:
            return []
        from .model_bundles.dnabert2_117m.runtime import normalize_sequences

        normalized = normalize_sequences(sequences)
        encoded = self.tokenizer(normalized, add_special_tokens=True, truncation=False)
        input_ids = encoded.get("input_ids") or []
        return [len(row) for row in input_ids]

    def score_sequences(self, sequences: list[str], *, batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
        if not sequences:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        from .model_bundles.dnabert2_117m.runtime import embed_sequences, normalize_sequences

        normalized = normalize_sequences(sequences)
        embeddings = embed_sequences(
            normalized,
            tokenizer=self.tokenizer,
            model=self.model,
            device=self.device,
            batch_size=max(1, int(batch_size)),
            max_length=self.provenance.max_length,
        )
        member_predictions = embeddings @ self.coefficients.T + self.intercepts[None, :]
        if self.provenance.task_type == "classification":
            member_predictions = 1.0 / (1.0 + np.exp(-np.clip(member_predictions, -40.0, 40.0)))
        predictions = member_predictions.mean(axis=1)
        uncertainty = member_predictions.std(axis=1, ddof=1)
        if not np.isfinite(predictions).all() or not np.isfinite(uncertainty).all():
            raise ModelScoringError("The model produced non-finite predictions or uncertainties.")
        return predictions.astype(float), uncertainty.astype(float)

    def health_check(self) -> dict[str, float]:
        prediction, uncertainty = self.score_sequences(["ACGTACGTACGTACGT"], batch_size=1)
        if not math.isfinite(float(prediction[0])) or not math.isfinite(float(uncertainty[0])):
            raise ModelScoringError("The scorer health check returned a non-finite value.")
        return {"prediction": float(prediction[0]), "uncertainty": float(uncertainty[0])}


@lru_cache(maxsize=4)
def _load_verified_scorer_cached(
    model_dir: str,
    head_dir: str,
    provenance: ScorerProvenance,
    model_manifest_digest: str,
    head_manifest_digest: str,
    device: str,
) -> VerifiedFoundationScorer:
    # Manifest digests are cache-key material: a legitimately replaced config or
    # semantics manifest must reload the backbone/head even when weights are unchanged.
    del model_manifest_digest, head_manifest_digest
    return VerifiedFoundationScorer(Path(model_dir), Path(head_dir), provenance, device=device)


def load_verified_foundation_scorer(
    model_path: str | Path,
    head_path: str | Path | None = None,
    *,
    device: str = "auto",
    cache_dir: str | Path | None = None,
) -> VerifiedFoundationScorer:
    model_dir, head_dir, provenance = inspect_foundation_scorer(
        model_path,
        head_path,
        require_direct_runtime=True,
    )
    if cache_dir:
        resolved_cache = str(Path(cache_dir).resolve())
        os.environ.setdefault("HF_HOME", resolved_cache)
    return _load_verified_scorer_cached(
        str(model_dir),
        str(head_dir),
        provenance,
        _sha256_file(model_dir / "assayready_model_manifest.json"),
        _sha256_file(head_dir / "head_manifest.json"),
        device,
    )
