from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GEO_ACCESSION = "GSE135464"
GEO_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135464"
PAPER_DOI = "10.1038/s41467-020-15977-4"
BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE135nnn/GSE135464/suppl"
SOURCE_FILES = {
    "GSE135464_filtered_read_table_miseq_GPD.txt.gz": "5fbc210b491980f0e9f53afea5cae7649db5f698ea2b9e7adb928ebfe887d085",
    "GSE135464_filtered_read_table_miseq_ZEV.txt.gz": "bb31ea19f33e91e2c4f316b38ce026b2cffd0f5342e5a8893ae8ae70896e9535",
    "GSE135464_final_means_ids_added.csv.gz": "7c1371c41cbfab0e2cfc262ce1ab7d2ba81edbacccd06b8c7a391808bbc6ecf8",
    "GSE135464_means_nextseq_GPD.csv.gz": "cd39d47096a17d058c54fc6dc74c7ffa791fce7835f507419e6ff2b60e067b3b",
    "GSE135464_means_nextseq_ZEV.csv.gz": "fa3d1470a12c24d8587834d998459c481c200b1dbeb859a4782620542ec5760b",
}
GPD_SEQUENCE_FILE = "GSE135464_filtered_read_table_miseq_GPD.txt.gz"
GPD_ACTIVITY_FILE = "GSE135464_means_nextseq_GPD.csv.gz"
MIN_PUBLISHED_ACTIVITY = -0.521
MAX_PUBLISHED_ACTIVITY = 0.560
MAX_REPLICATE_DIFFERENCE = 0.2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def obtain_sources(raw_dir: Path, *, download_missing: bool) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for filename, expected_sha256 in SOURCE_FILES.items():
        path = raw_dir / filename
        if not path.is_file():
            if not download_missing:
                raise FileNotFoundError(f"Missing {path}; rerun without --no-download.")
            partial = raw_dir / f"{filename}.partial"
            with urllib.request.urlopen(f"{BASE_URL}/{filename}", timeout=120) as response:
                with partial.open("wb") as handle:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        handle.write(chunk)
            partial.replace(path)
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {filename}: expected {expected_sha256}, got {actual_sha256}"
            )
        paths[filename] = path
    return paths


def split_for_hash(sequence_hash: str) -> str:
    bucket = int(sequence_hash[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(
    root: Path,
    *,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    seed: int,
    download_missing: bool,
) -> dict[str, Any]:
    root = root.resolve()
    paths = obtain_sources(root / "raw", download_missing=download_missing)
    sequence_path = paths[GPD_SEQUENCE_FILE]
    activity_path = paths[GPD_ACTIVITY_FILE]

    activities = pd.read_csv(activity_path).set_index("Seqs")
    if not activities.index.is_unique:
        raise ValueError("The GPD NextSeq activity prefixes must be unique.")
    prefix_counts: Counter[str] = Counter()
    for chunk in pd.read_csv(sequence_path, header=None, usecols=[0], chunksize=100_000):
        prefix_counts.update(chunk.iloc[:, 0].astype(str).str.upper().str[:35])
    ambiguous_prefixes = {prefix for prefix, count in prefix_counts.items() if count > 1}

    limits = {"train": train_rows, "validation": validation_rows, "test": test_rows}
    if any(limit <= 0 for limit in limits.values()):
        raise ValueError("Every prepared split must contain at least one row.")
    reservoirs: dict[str, list[dict[str, Any]]] = {split: [] for split in limits}
    available = {split: 0 for split in limits}
    rng = {split: np.random.default_rng(seed + index) for index, split in enumerate(limits)}
    counters = {
        "miseq_rows": 0,
        "joined_rows": 0,
        "ambiguous_prefix_rows": 0,
        "replicate_filter_failures": 0,
        "published_range_filter_failures": 0,
        "accepted_rows": 0,
    }
    all_targets: list[np.ndarray] = []

    for chunk in pd.read_csv(sequence_path, header=None, usecols=[0], chunksize=100_000):
        sequences = chunk.iloc[:, 0].astype(str).str.upper().str.strip()
        prefixes = sequences.str[:35]
        joined = activities.reindex(prefixes.to_numpy())
        means_a = joined["Means_A"].to_numpy(dtype=np.float64)
        means_b = joined["Means_B"].to_numpy(dtype=np.float64)
        present = np.isfinite(means_a) & np.isfinite(means_b)
        ambiguous = prefixes.isin(ambiguous_prefixes).to_numpy()
        replicate_ok = np.abs(means_a - means_b) <= MAX_REPLICATE_DIFFERENCE + 1e-12
        targets = (means_a + means_b) / 2.0
        range_ok = (
            (targets >= MIN_PUBLISHED_ACTIVITY - 1e-12)
            & (targets <= MAX_PUBLISHED_ACTIVITY + 1e-12)
        )
        accepted = present & ~ambiguous & replicate_ok & range_ok

        counters["miseq_rows"] += len(chunk)
        counters["joined_rows"] += int(present.sum())
        counters["ambiguous_prefix_rows"] += int((present & ambiguous).sum())
        counters["replicate_filter_failures"] += int((present & ~ambiguous & ~replicate_ok).sum())
        counters["published_range_filter_failures"] += int(
            (present & ~ambiguous & replicate_ok & ~range_ok).sum()
        )
        counters["accepted_rows"] += int(accepted.sum())
        all_targets.append(targets[accepted])

        accepted_indices = np.flatnonzero(accepted)
        for index in accepted_indices:
            sequence = sequences.iloc[index]
            sequence_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()
            split = split_for_hash(sequence_hash)
            row = {
                "sequence_id": f"gse135464_gpd_{sequence_hash[:20]}",
                "sequence": sequence,
                "measured_activity": float(targets[index]),
                "replicate_a_activity": float(means_a[index]),
                "replicate_b_activity": float(means_b[index]),
                "replicate_abs_difference": float(abs(means_a[index] - means_b[index])),
                "source_split": split,
                "source_assay": "GPD_constitutive_FACS_seq",
                "source_accession": GEO_ACCESSION,
                "sequence_hash": sequence_hash,
            }
            available[split] += 1
            reservoir = reservoirs[split]
            if len(reservoir) < limits[split]:
                reservoir.append(row)
            else:
                replacement = int(rng[split].integers(0, available[split]))
                if replacement < limits[split]:
                    reservoir[replacement] = row

    accepted_targets = np.concatenate(all_targets)
    prepared_dir = root / "prepared"
    split_manifest: dict[str, Any] = {}
    for split, rows in reservoirs.items():
        if len(rows) < limits[split]:
            raise RuntimeError(f"Requested {limits[split]} {split} rows, but only {len(rows)} were available.")
        rows.sort(key=lambda row: row["sequence_hash"])
        output_path = prepared_dir / f"gpd_{split}.csv"
        write_csv(output_path, rows)
        selected_targets = np.asarray([row["measured_activity"] for row in rows])
        split_manifest[split] = {
            "available_rows": available[split],
            "selected_rows": len(rows),
            "prepared_file": str(output_path.relative_to(root)).replace("\\", "/"),
            "prepared_sha256": sha256_file(output_path),
            "target_summary": {
                "minimum": float(selected_targets.min()),
                "mean": float(selected_targets.mean()),
                "maximum": float(selected_targets.max()),
            },
        }

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "geo_accession": GEO_ACCESSION,
        "geo_url": GEO_URL,
        "paper_doi": PAPER_DOI,
        "reuse_note": "Public GEO data; cite the study and review repository terms for the intended use.",
        "task_type": "regression",
        "library": "GPD constitutive promoter library",
        "organism": "Saccharomyces cerevisiae",
        "target_column": "measured_activity",
        "target_definition": "mean of replicate A and B log10(GFP:mCherry) promoter activity",
        "filter_policy": {
            "maximum_replicate_absolute_difference": MAX_REPLICATE_DIFFERENCE,
            "published_activity_range": [MIN_PUBLISHED_ACTIVITY, MAX_PUBLISHED_ACTIVITY],
            "ambiguous_35bp_prefixes_excluded": True,
        },
        "split_policy": "SHA-256 bucket: 80% train, 10% validation, 10% test before deterministic reservoir sampling",
        "seed": seed,
        "source_files": {
            filename: {"sha256": expected, "size_bytes": paths[filename].stat().st_size}
            for filename, expected in SOURCE_FILES.items()
        },
        "row_audit": counters,
        "accepted_target_summary": {
            "minimum": float(accepted_targets.min()),
            "mean": float(accepted_targets.mean()),
            "maximum": float(accepted_targets.max()),
        },
        "splits": split_manifest,
        "excluded_for_now": {
            "ZEV": "Separate uninduced and induced endpoints require a distinct multi-target or condition-specific workflow.",
            "designed_validation": "Experiment/design identifiers require objective mapping before they can be used as an external test set.",
        },
    }
    manifest_path = root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the GSE135464 constitutive GPD promoter regression task.")
    parser.add_argument("--root", type=Path, default=Path("datasets/GSE135464"))
    parser.add_argument("--train-rows", type=int, default=50_000)
    parser.add_argument("--validation-rows", type=int, default=10_000)
    parser.add_argument("--test-rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    manifest = prepare(
        args.root,
        train_rows=args.train_rows,
        validation_rows=args.validation_rows,
        test_rows=args.test_rows,
        seed=args.seed,
        download_missing=not args.no_download,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
