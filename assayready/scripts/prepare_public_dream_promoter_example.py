from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
for search_root in [str(REPO_ROOT), str(PACKAGE_ROOT)]:
    if search_root not in sys.path:
        sys.path.insert(0, search_root)

try:
    from model_assessment.ml_core.specialization import kmer_feature_matrix, stable_sequence_hash
except ModuleNotFoundError:  # Running from the repository root before editable install.
    from model_assessment.model_assessment.ml_core.specialization import kmer_feature_matrix, stable_sequence_hash


DATASET_ID = "HuggingFaceBio/random-promoter-dream-2022"
DATASET_CONFIG = "supervised"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
ZENODO_RECORD = "10633252"
ZENODO_FILES = {
    "train": "train.txt",
    "validation": "val.txt",
}


def _fetch_rows(split: str, *, limit: int, offset: int = 0, page_size: int = 100) -> list[dict[str, Any]]:
    try:
        rows = _fetch_huggingface_rows(split, limit=limit, offset=offset, page_size=page_size)
        if rows:
            return rows
    except Exception:
        pass
    return _fetch_zenodo_rows(split, limit=limit, offset=offset)


def _fetch_huggingface_rows(split: str, *, limit: int, offset: int = 0, page_size: int = 100) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    while len(rows) < limit:
        length = min(page_size, limit - len(rows))
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET_ID,
                "config": DATASET_CONFIG,
                "split": split,
                "offset": offset + len(rows),
                "length": length,
            }
        )
        url = f"{ROWS_ENDPOINT}?{query}"
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        for item in payload.get("rows", []):
            row = item.get("row", item)
            sequence = str(row.get("sequence", "") or "").upper().strip()
            activity = row.get("activity")
            if sequence and activity not in {None, ""}:
                rows.append(row)
        if len(payload.get("rows", [])) < length:
            break
    return rows[:limit]


def _fetch_zenodo_rows(split: str, *, limit: int, offset: int = 0) -> list[dict[str, Any]]:
    filename = ZENODO_FILES.get(split)
    if not filename:
        raise ValueError(f"No Zenodo file mapping is configured for split {split!r}.")
    url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{filename}?download=1"
    rows: list[dict[str, Any]] = []
    seen = 0
    with urllib.request.urlopen(url, timeout=120) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split(",")
            if len(parts) < 2:
                continue
            sequence, activity = parts[0].strip().upper(), parts[1].strip()
            if not sequence or not activity:
                continue
            if seen < offset:
                seen += 1
                continue
            rows.append(
                {
                    "sequence": sequence,
                    "activity": float(activity),
                    "sequence_length": len(sequence),
                    "source_file": filename,
                    "row_id": seen,
                }
            )
            seen += 1
            if len(rows) >= limit:
                break
    return rows


def _ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_target: np.ndarray, *, alpha: float = 1.0) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x_train.shape[0], 1), dtype=x_train.dtype), x_train], axis=1)
    target_aug = np.concatenate([np.ones((x_target.shape[0], 1), dtype=x_target.dtype), x_target], axis=1)
    reg = np.eye(x_aug.shape[1], dtype=x_train.dtype) * float(alpha)
    reg[0, 0] = 0.0
    coef = np.linalg.pinv(x_aug.T @ x_aug + reg) @ x_aug.T @ y_train
    return target_aug @ coef


def _fit_public_model(train_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x_train = kmer_feature_matrix([row["sequence"] for row in train_rows], k=3)
    y_train = np.asarray([float(row["activity"]) for row in train_rows], dtype=np.float64)
    x_target = kmer_feature_matrix([row["sequence"] for row in target_rows], k=3)

    try:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=96,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        per_tree = np.vstack([tree.predict(x_target) for tree in model.estimators_])
        return per_tree.mean(axis=0), per_tree.std(axis=0)
    except ImportError:
        pass

    rng = np.random.default_rng(seed)
    preds = []
    for idx in range(24):
        sample = rng.integers(0, len(train_rows), len(train_rows))
        preds.append(_ridge_predict(x_train[sample], y_train[sample], x_target, alpha=1.0 + idx * 0.1))
    per_tree = np.vstack(preds)
    return per_tree.mean(axis=0), per_tree.std(axis=0)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_rows(rows: list[dict[str, Any]], predictions: np.ndarray, uncertainty: np.ndarray, *, prefix: str, split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, (row, pred, std) in enumerate(zip(rows, predictions, uncertainty), start=1):
        sequence = str(row["sequence"]).upper()
        out.append(
            {
                "sequence_id": f"{prefix}_{idx:05d}",
                "sequence": sequence,
                "measured_activity": float(row["activity"]),
                "model_prediction": float(pred),
                "model_uncertainty": float(std),
                "source_split": split,
                "source_row_id": row.get("row_id", idx - 1),
                "source_file": row.get("source_file", ""),
                "sequence_hash": stable_sequence_hash(sequence),
            }
        )
    return out


def _write_config(path: Path, assay_path: Path, candidate_path: Path) -> None:
    config = {
        "project": "public_dream_promoter_audit",
        "assay_files": [assay_path.name],
        "candidate_files": [candidate_path.name],
        "output_dir": "outputs/assayready/public_dream_promoter_audit",
        "task": {
            "type": "regression",
            "sequence_col": "sequence",
            "target_col": "measured_activity",
            "prediction_col": "model_prediction",
            "uncertainty_col": "model_uncertainty",
            "id_col": "sequence_id",
        },
        "metadata": {"categorical_cols": ["source_split"]},
        "split": {
            "val_fraction": 0.15,
            "test_fraction": 0.15,
            "homology_threshold": 0.90,
            "homology_k": 8,
            "seed": 13,
        },
        "ingestion": {"low_n_threshold": 200},
        "acquisition": {"beta": 1.0, "top_k": 96, "diversity_penalty": 0.2},
        "evaluation": {
            "schema_version": 1,
            "semantics": {
                "objective_direction": "maximize",
                "units": "normalized activity",
                "uncertainty_type": "predictive_standard_deviation",
            },
            "constraints_verified": False,
            "biological_constraints": [
                "GC fraction should remain in the assay-supported range",
                "avoid long homopolymers that degrade synthesis reliability",
            ],
            "provenance": {
                "model_identifier": "public_dream_demo_baseline",
                "data_identifier": assay_path.name,
                "split_identifier": "similarity_group_split_v1",
                "training_independence": "unverified",
                "code_revision": "public_example_v1",
                "duration_seconds": 0,
                "artifact_verification": "local_checksum_not_provided",
                "dependency_versions": {
                    "numpy": "2.x",
                    "pandas": "3.x",
                    "scikit-learn": "1.x",
                },
            },
        },
    }
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def prepare(out_dir: Path, *, train_rows: int, assay_rows: int, candidate_rows: int, seed: int) -> dict[str, str]:
    train = _fetch_rows("train", limit=train_rows)
    validation = _fetch_rows("validation", limit=assay_rows + candidate_rows)
    if len(train) < 50 or len(validation) < assay_rows + candidate_rows:
        raise RuntimeError(
            f"Fetched too few public rows: train={len(train)}, validation={len(validation)}. "
            "Retry later or lower the requested row counts."
        )

    assay_source = validation[:assay_rows]
    candidate_source = validation[assay_rows : assay_rows + candidate_rows]
    assay_pred, assay_uncertainty = _fit_public_model(train, assay_source, seed=seed)
    candidate_pred, candidate_uncertainty = _fit_public_model(train, candidate_source, seed=seed)

    assay = _format_rows(assay_source, assay_pred, assay_uncertainty, prefix="dream_assay", split="validation")
    candidates = _format_rows(candidate_source, candidate_pred, candidate_uncertainty, prefix="dream_candidate", split="validation")

    out_dir.mkdir(parents=True, exist_ok=True)
    assay_path = out_dir / "public_dream_promoter_predictions.csv"
    candidate_path = out_dir / "public_dream_promoter_candidates.csv"
    config_path = out_dir / "public_dream_promoter_audit.json"
    source_path = out_dir / "public_dream_promoter_source.md"

    _write_csv(assay_path, assay)
    _write_csv(candidate_path, candidates)
    _write_config(config_path, assay_path, candidate_path)
    source_path.write_text(
        "\n".join(
            [
                "# Public DREAM Promoter Example",
                "",
                f"Source dataset: {DATASET_ID}, config `{DATASET_CONFIG}`.",
                "Original source: Random Promoter DREAM Challenge 2022, Zenodo DOI `10.5281/zenodo.10633252`.",
                f"Direct files used when available: Zenodo record `{ZENODO_RECORD}` `train.txt` and `val.txt`.",
                "License noted by the Hugging Face dataset card: CC BY 4.0.",
                "",
                "This script uses public measured promoter activity rows and trains a local random-forest baseline",
                "when scikit-learn is available, otherwise a bootstrapped ridge baseline,",
                "only to create model_prediction and model_uncertainty columns for exercising AssayReady.",
                "Do not market these predictions as a best-in-class model.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "assay_csv": str(assay_path),
        "candidate_csv": str(candidate_path),
        "config_json": str(config_path),
        "source_note": str(source_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a public Random Promoter DREAM 2022 AssayReady example.")
    parser.add_argument(
        "--out-dir",
        default=str(PACKAGE_ROOT / "model_assessment" / "examples"),
        help="Directory for generated CSV/config files (defaults to the package's bundled examples directory).",
    )
    parser.add_argument("--train-rows", type=int, default=800)
    parser.add_argument("--assay-rows", type=int, default=400)
    parser.add_argument("--candidate-rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    try:
        outputs = prepare(
            Path(args.out_dir),
            train_rows=args.train_rows,
            assay_rows=args.assay_rows,
            candidate_rows=args.candidate_rows,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"failed to prepare public DREAM promoter example: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
