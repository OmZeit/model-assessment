from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

def _runtime_functions():
    """Load optional foundation dependencies only when training is requested."""
    if __package__:
        from .runtime import (
            choose_device,
            embed_sequences,
            load_frozen_backbone,
            normalize_sequences,
            sha256_file,
            verify_model_manifest,
        )
    else:  # Keep direct script execution working from a scaffolded bundle.
        from runtime import (
            choose_device,
            embed_sequences,
            load_frozen_backbone,
            normalize_sequences,
            sha256_file,
            verify_model_manifest,
        )
    return (
        choose_device,
        embed_sequences,
        load_frozen_backbone,
        normalize_sequences,
        sha256_file,
        verify_model_manifest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a small assay head over frozen DNABERT-2 embeddings.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--task-type", choices=["regression", "classification"], default="regression")
    parser.add_argument("--positive-label")
    parser.add_argument(
        "--objective-direction",
        choices=["maximize", "minimize"],
        default="maximize",
        help="Whether larger or smaller assay outcomes are preferred (default: maximize).",
    )
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    return parser


def train_frozen_head(
    *,
    model_dir: str | Path,
    training: str | Path,
    output_dir: str | Path,
    sequence_col: str = "sequence",
    target_col: str,
    task_type: str = "regression",
    positive_label: str | None = None,
    objective_direction: str = "maximize",
    ensemble_size: int = 8,
    ridge_alpha: float = 10.0,
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "auto",
    seed: int = 13,
) -> dict[str, object]:
    """Fit and persist an assay head for the verified frozen backbone."""
    (
        choose_device,
        embed_sequences,
        load_frozen_backbone,
        normalize_sequences,
        sha256_file,
        verify_model_manifest,
    ) = _runtime_functions()
    if ensemble_size < 2:
        raise ValueError("ensemble-size must be at least two to estimate model disagreement.")
    if task_type not in {"regression", "classification"}:
        raise ValueError("task-type must be regression or classification.")
    if objective_direction not in {"maximize", "minimize"}:
        raise ValueError("objective-direction must be maximize or minimize.")
    training_path = Path(training).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(training_path)
    required = {sequence_col, target_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("training table is missing columns: " + ", ".join(sorted(missing)))
    if len(frame) < 20:
        raise ValueError("at least 20 training rows are required for this development head.")
    sequences = normalize_sequences(frame[sequence_col])
    if task_type == "regression":
        targets = pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=np.float64)
        resolved_positive_label = None
    else:
        labels = frame[target_col].astype(str)
        unique = sorted(labels.unique())
        if len(unique) != 2:
            raise ValueError("classification requires exactly two target labels.")
        resolved_positive_label = positive_label or unique[-1]
        if resolved_positive_label not in unique:
            raise ValueError(f"positive label {resolved_positive_label!r} is not present in training targets.")
        targets = (labels == resolved_positive_label).to_numpy(dtype=np.int64)

    model_manifest = verify_model_manifest(model_dir)
    selected_device = choose_device(device)
    tokenizer, model = load_frozen_backbone(model_dir, selected_device)
    embeddings = embed_sequences(
        sequences, tokenizer=tokenizer, model=model, device=selected_device,
        batch_size=batch_size, max_length=max_length,
    )
    rng = np.random.default_rng(seed)
    coefficients, intercepts = [], []
    for member in range(ensemble_size):
        if member == 0:
            indices = np.arange(len(targets))
        elif task_type == "classification":
            negative = np.flatnonzero(targets == 0)
            positive = np.flatnonzero(targets == 1)
            indices = np.concatenate(
                [rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)]
            )
            rng.shuffle(indices)
        else:
            indices = rng.choice(len(targets), len(targets), replace=True)
        if task_type == "regression":
            estimator = Ridge(alpha=ridge_alpha).fit(embeddings[indices], targets[indices])
        else:
            estimator = LogisticRegression(C=1.0, max_iter=2000, random_state=seed + member).fit(
                embeddings[indices], targets[indices]
            )
        coefficients.append(np.asarray(estimator.coef_).reshape(-1))
        intercepts.append(float(np.asarray(estimator.intercept_).reshape(-1)[0]))

    head_path = output_path / "head.npz"
    np.savez_compressed(
        head_path,
        coefficients=np.stack(coefficients).astype(np.float32),
        intercepts=np.asarray(intercepts, dtype=np.float32),
    )
    manifest = {
        "schema_version": 1,
        "head_type": "frozen_embedding_linear_ensemble",
        "task_type": task_type,
        "positive_label": resolved_positive_label,
        "objective_direction": objective_direction,
        "sequence_col": sequence_col,
        "target_col": target_col,
        "ensemble_size": ensemble_size,
        "ridge_alpha": ridge_alpha if task_type == "regression" else None,
        "seed": seed,
        "max_length": max_length,
        "training_rows": len(frame),
        "training_sha256": sha256_file(training_path),
        "model_repository": model_manifest["repository"],
        "model_revision": model_manifest["revision"],
        "model_weights_sha256": model_manifest["weights_sha256"],
        "head_sha256": sha256_file(head_path),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_independence_warning": (
            "Only train on AssayReady's training partition; never fit this head on validation, test, or candidate outcomes."
        ),
    }
    (output_path / "head_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = _parser().parse_args()
    manifest = train_frozen_head(
        model_dir=args.model_dir,
        training=args.training,
        output_dir=args.output_dir,
        sequence_col=args.sequence_col,
        target_col=args.target_col,
        task_type=args.task_type,
        positive_label=args.positive_label,
        objective_direction=args.objective_direction,
        ensemble_size=args.ensemble_size,
        ridge_alpha=args.ridge_alpha,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
