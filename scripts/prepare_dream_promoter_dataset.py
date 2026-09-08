from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


DATASET_ID = "HuggingFaceBio/random-promoter-dream-2022"
REVISION = "095ecf8da1e69d70c21812af809df7b19c7d8a84"
LICENSE = "CC-BY-4.0"
ZENODO_DOI = "10.5281/zenodo.10633252"
FILES = {
    "train": [
        (
            "supervised/train-00000-of-00002.parquet",
            "00b6cdfe2511f8ed4769c56fb10734e64ad1d9ec6d127ff0af02d7375d6bbea9",
        ),
        (
            "supervised/train-00001-of-00002.parquet",
            "8c5e3fe6697b3ec282193e2a617eaf489e8f9a8a925388c457f4310f2fbecca8",
        ),
    ],
    "validation": [
        (
            "supervised/validation-00000-of-00001.parquet",
            "a4a7c321d9d392bc55aeedf959973d4136a336da90ff366ec7436af0e41d8505",
        )
    ],
    "test": [
        (
            "supervised/test-00000-of-00001.parquet",
            "9d89aac2e8b9dd64ff2ef09644efcc475ff68dec4f7952ccb23328bcf5566695",
        )
    ],
}
CANDIDATE_FILE = (
    "challenge_test_sequences/train-00000-of-00001.parquet",
    "aaf39b1050fb18e7b57cc7f82534db4860a52d2ff9cb7c2869da158c41b7a901",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_verify(root: Path) -> dict[str, list[Path]]:
    downloaded: dict[str, list[Path]] = {}
    for split, entries in FILES.items():
        split_paths = []
        for filename, expected_sha256 in entries:
            path = Path(
                hf_hub_download(
                    DATASET_ID,
                    filename,
                    repo_type="dataset",
                    revision=REVISION,
                    local_dir=root,
                )
            )
            actual_sha256 = sha256_file(path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"Checksum mismatch for {filename}: expected {expected_sha256}, got {actual_sha256}"
                )
            split_paths.append(path)
        downloaded[split] = split_paths
    candidate_name, candidate_sha256 = CANDIDATE_FILE
    candidate_path = Path(
        hf_hub_download(
            DATASET_ID,
            candidate_name,
            repo_type="dataset",
            revision=REVISION,
            local_dir=root,
        )
    )
    actual_candidate_sha256 = sha256_file(candidate_path)
    if actual_candidate_sha256 != candidate_sha256:
        raise RuntimeError(
            f"Checksum mismatch for {candidate_name}: expected {candidate_sha256}, got {actual_candidate_sha256}"
        )
    downloaded["candidates"] = [candidate_path]
    hf_hub_download(
        DATASET_ID,
        "README.md",
        repo_type="dataset",
        revision=REVISION,
        local_dir=root,
    )
    return downloaded


def reservoir_sample(paths: list[Path], *, split: str, limit: int, seed: int) -> tuple[list[dict[str, Any]], int]:
    rng = np.random.default_rng(seed)
    reservoir: list[dict[str, Any]] = []
    seen = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=65_536,
            columns=["sequence", "activity", "source_file", "row_id"],
        ):
            columns = batch.to_pydict()
            for sequence, activity, source_file, row_id in zip(
                columns["sequence"],
                columns["activity"],
                columns["source_file"],
                columns["row_id"],
            ):
                sequence = str(sequence or "").upper().strip()
                if not sequence or activity is None:
                    continue
                row = {
                    "sequence_id": f"dream_{split}_{int(row_id):09d}",
                    "sequence": sequence,
                    "measured_activity": float(activity),
                    "source_split": split,
                    "source_row_id": int(row_id),
                    "source_file": str(source_file or path.name),
                    "sequence_hash": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                }
                if len(reservoir) < limit:
                    reservoir.append(row)
                else:
                    replacement = int(rng.integers(0, seen + 1))
                    if replacement < limit:
                        reservoir[replacement] = row
                seen += 1
    reservoir.sort(key=lambda item: (item["source_row_id"], item["sequence_hash"]))
    return reservoir, seen


def reservoir_sample_candidates(paths: list[Path], *, limit: int, seed: int) -> tuple[list[dict[str, Any]], int]:
    rng = np.random.default_rng(seed)
    reservoir: list[dict[str, Any]] = []
    seen = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=65_536,
            columns=["sequence", "source_file", "row_id"],
        ):
            columns = batch.to_pydict()
            for sequence, source_file, row_id in zip(
                columns["sequence"], columns["source_file"], columns["row_id"]
            ):
                sequence = str(sequence or "").upper().strip()
                if not sequence:
                    continue
                row = {
                    "sequence_id": f"dream_candidate_{int(row_id):09d}",
                    "sequence": sequence,
                    "source_split": "challenge_test_sequences",
                    "source_row_id": int(row_id),
                    "source_file": str(source_file or path.name),
                    "sequence_hash": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                }
                if len(reservoir) < limit:
                    reservoir.append(row)
                else:
                    replacement = int(rng.integers(0, seen + 1))
                    if replacement < limit:
                        reservoir[replacement] = row
                seen += 1
    reservoir.sort(key=lambda item: (item["source_row_id"], item["sequence_hash"]))
    return reservoir, seen


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download, verify, and prepare the pinned Random Promoter DREAM regression dataset."
    )
    parser.add_argument("--root", type=Path, default=Path("datasets/random-promoter-dream-2022"))
    parser.add_argument("--train-rows", type=int, default=50_000)
    parser.add_argument("--validation-rows", type=int, default=10_000)
    parser.add_argument("--test-rows", type=int, default=10_000)
    parser.add_argument("--candidate-rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    limits = {
        "train": args.train_rows,
        "validation": args.validation_rows,
        "test": args.test_rows,
    }
    if any(limit <= 0 for limit in limits.values()):
        raise ValueError("Every prepared split must contain at least one row.")

    root = args.root.resolve()
    paths = download_and_verify(root)
    prepared_dir = root / "prepared"
    split_manifest: dict[str, Any] = {}
    for index, split in enumerate(("train", "validation", "test")):
        rows, available_rows = reservoir_sample(
            paths[split], split=split, limit=limits[split], seed=args.seed + index
        )
        if len(rows) < limits[split]:
            raise RuntimeError(
                f"Requested {limits[split]} {split} rows but only {len(rows)} valid rows were available."
            )
        output_path = prepared_dir / f"{split}.csv"
        write_csv(output_path, rows)
        selected_targets = np.asarray([row["measured_activity"] for row in rows], dtype=np.float64)
        split_manifest[split] = {
            "available_rows": available_rows,
            "selected_rows": len(rows),
            "prepared_file": str(output_path.relative_to(root)).replace("\\", "/"),
            "prepared_sha256": sha256_file(output_path),
            "selected_target_summary": {
                "minimum": float(selected_targets.min()),
                "mean": float(selected_targets.mean()),
                "maximum": float(selected_targets.max()),
            },
            "source_files": [
                {
                    "path": str(path.relative_to(root)).replace("\\", "/"),
                    "rows": pq.ParquetFile(path).metadata.num_rows,
                    "sha256": sha256_file(path),
                }
                for path in paths[split]
            ],
        }

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "dataset_revision": REVISION,
        "original_source_doi": ZENODO_DOI,
        "license": LICENSE,
        "task_type": "regression",
        "source_target_column": "activity",
        "canonical_target_column": "measured_activity",
        "target_semantics": "continuous measured synthetic yeast promoter activity; higher is better",
        "primary_evaluation_split": "validation",
        "test_scale_warning": (
            "The labeled designed-promoter test file uses MAUDE expression values on a different numeric scale "
            "from train/validation. Treat it as an external transfer set and do not combine its regression "
            "metrics with validation without an explicitly documented calibration."
        ),
        "sampling": {
            "method": "deterministic uniform reservoir sampling within each official split",
            "seed": args.seed,
            "training_independence": "Only the official train sample is intended for fitting. Validation and test remain holdouts.",
        },
        "splits": split_manifest,
    }
    candidate_rows, available_candidates = reservoir_sample_candidates(
        paths["candidates"], limit=args.candidate_rows, seed=args.seed + 3
    )
    if len(candidate_rows) < args.candidate_rows:
        raise RuntimeError(
            f"Requested {args.candidate_rows} candidate rows but only {len(candidate_rows)} were available."
        )
    candidate_path = prepared_dir / "candidates.csv"
    write_csv(candidate_path, candidate_rows)
    source_candidate_path = paths["candidates"][0]
    manifest["candidate_pool"] = {
        "available_rows": available_candidates,
        "selected_rows": len(candidate_rows),
        "prepared_file": str(candidate_path.relative_to(root)).replace("\\", "/"),
        "prepared_sha256": sha256_file(candidate_path),
        "labeled_outcomes_used": False,
        "source_file": {
            "path": str(source_candidate_path.relative_to(root)).replace("\\", "/"),
            "rows": pq.ParquetFile(source_candidate_path).metadata.num_rows,
            "sha256": sha256_file(source_candidate_path),
        },
    }
    manifest_path = root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
