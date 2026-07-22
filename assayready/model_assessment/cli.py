from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .ml_core.specialization import (
    acquisition_scores,
    best_simple_baseline,
    edit_distance,
    fit_task_head_ensemble,
    generate_de_novo_candidate_pool,
    gc_fraction,
    greedy_diverse_rank,
    ingest_assay_tables,
    jaccard_similarity,
    leakage_safe_split,
    predict_with_task_head_ensemble,
    run_baselines,
    sequence_kmers,
    stable_sequence_hash,
    write_table,
    normal_cdf,
    normal_pdf,
)
from .run_store import get_run_summary, list_runs, record_run
from .schemas import inferred_manifest_for_args, validate_evaluation_manifest


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML configs require PyYAML. Use JSON or install PyYAML.") from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping/object.")
    return data


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _doctor_item(name: str, ok: bool, detail: str, *, required: bool = True) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "required": bool(required),
        "detail": detail,
    }


def _module_available(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def _doctor_checks() -> list[dict[str, Any]]:
    root = _project_root()
    checks: list[dict[str, Any]] = []

    python_ok = (3, 12) <= sys.version_info[:2] < (3, 13)
    checks.append(
        _doctor_item(
            "python",
            python_ok,
            f"{platform.python_implementation()} {platform.python_version()} at {sys.executable}",
        )
    )

    pixi_path = shutil.which("pixi")
    if not pixi_path:
        home = Path.home()
        for candidate in [home / ".pixi" / "bin" / "pixi.exe", home / ".pixi" / "bin" / "pixi"]:
            if candidate.exists():
                pixi_path = str(candidate)
                break
    checks.append(
        _doctor_item(
            "pixi",
            pixi_path is not None,
            pixi_path or "Pixi is not on PATH; pip fallback can still run CPU-only workflows.",
            required=False,
        )
    )

    required_modules = {
        "dash": "dash",
        "numpy": "numpy",
        "openpyxl": "openpyxl",
        "pandas": "pandas",
        "PyYAML": "yaml",
        "rapidfuzz": "rapidfuzz",
        "scikit-learn": "sklearn",
        "tokenizers": "tokenizers",
        "torch": "torch",
    }
    missing_modules = [name for name, import_name in required_modules.items() if not _module_available(import_name)]
    checks.append(
        _doctor_item(
            "runtime imports",
            not missing_modules,
            "all required runtime modules are importable" if not missing_modules else "missing: " + ", ".join(missing_modules),
        )
    )

    public_demo_files = [
        root / "examples" / "public_dream_promoter_audit.json",
        root / "examples" / "public_dream_promoter_predictions.csv",
        root / "examples" / "public_dream_promoter_candidates.csv",
    ]
    missing_demo = [str(path.relative_to(root)) for path in public_demo_files if not path.exists()]
    checks.append(
        _doctor_item(
            "public demo",
            not missing_demo,
            "bundled public DREAM promoter audit files are present" if not missing_demo else "missing: " + ", ".join(missing_demo),
        )
    )

    output_root = Path("outputs") / "assayready"
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / ".doctor-write-test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink(missing_ok=True)
        output_ok = True
        output_detail = f"writable: {output_root.resolve()}"
    except Exception as exc:
        output_ok = False
        output_detail = str(exc)
    checks.append(_doctor_item("output directory", output_ok, output_detail))

    return checks


def run_doctor(*, as_json: bool = False) -> int:
    checks = _doctor_checks()
    ok = all(item["ok"] or not item["required"] for item in checks)
    payload = {"ok": ok, "checks": checks}
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("AssayReady doctor")
        for item in checks:
            status = "OK" if item["ok"] else ("WARN" if not item["required"] else "FAIL")
            required = "required" if item["required"] else "optional"
            print(f"[{status}] {item['name']} ({required}) - {item['detail']}")
    return 0 if ok else 1


def run_registry_command(args: argparse.Namespace) -> int:
    if args.runs_command == "list":
        rows = list_runs(limit=max(1, int(args.limit)))
        if args.json:
            print(json.dumps(rows, indent=2, sort_keys=True, default=_json_default))
        else:
            for row in rows:
                print(
                    "\t".join(
                        str(row.get(key) or "")
                        for key in ["run_id", "updated_at", "project", "workflow", "status", "verdict"]
                    )
                )
        return 0
    summary = get_run_summary(str(args.run_id))
    if summary is None:
        print(f"assayready: run not found: {args.run_id}", file=sys.stderr)
        return 2
    if args.runs_command == "verify":
        verification = summary.get("artifact_verification") or {}
        print(json.dumps(verification, indent=2, sort_keys=True))
        return 0 if verification.get("ok") else 1
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return 0


def _resolve_paths(paths: list[str | Path], *, base_dir: Path) -> list[Path]:
    resolved: list[Path] = []
    for raw in paths:
        candidate = Path(raw)
        if candidate.is_absolute():
            resolved.append(candidate)
            continue
        for option in [base_dir / candidate, Path.cwd() / candidate]:
            if option.exists():
                resolved.append(option.resolve())
                break
        else:
            resolved.append((base_dir / candidate).resolve())
    return resolved


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not np.isfinite(out):
        return None
    return out


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    return sums[inverse] / counts[inverse]


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    return _pearson(_rankdata(left), _rankdata(right))


def _parse_aliases(values: list[str] | None) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Alias must use canonical=alias1,alias2 format, got {item!r}.")
        canonical, raw_aliases = item.split("=", 1)
        aliases[canonical.strip()] = [alias.strip() for alias in raw_aliases.split(",") if alias.strip()]
    return aliases


def _params_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if not args.assay:
        raise ValueError("--assay is required when --config is not provided.")
    if not args.target_col:
        raise ValueError("--target-col is required when --config is not provided.")
    manifest = inferred_manifest_for_args(task_type=args.task_type, positive_label=args.positive_label)
    return {
        "project": args.project,
        "assay_files": _resolve_paths(args.assay, base_dir=Path.cwd()),
        "candidate_files": _resolve_paths(args.candidates or [], base_dir=Path.cwd()),
        "output_dir": Path(args.output_dir) if args.output_dir else None,
        "task_type": args.task_type,
        "sequence_col": args.sequence_col,
        "target_col": args.target_col,
        "positive_label": args.positive_label,
        "id_col": args.id_col,
        "metadata_cols": args.metadata_cols or [],
        "group_cols": args.group_cols or [],
        "column_aliases": _parse_aliases(args.alias),
        "low_n_threshold": args.low_n_threshold,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "homology_threshold": args.homology_threshold,
        "homology_k": args.homology_k,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "ensemble_size": args.ensemble_size,
        "seed": args.seed,
        "acquisition_method": args.acquisition_method,
        "beta": args.beta,
        "diversity_method": args.diversity_method,
        "diversity_penalty": args.diversity_penalty,
        "top_k": args.top_k,
        "generate_candidates": args.generate_candidates,
        "num_proposals": args.num_proposals,
        "sequence_length": args.sequence_length,
        "gc_range": args.gc_range,
        "evaluation_manifest": manifest.to_dict(),
        "manifest_source": "inferred_from_args",
    }


def _params_from_config(config: dict[str, Any], *, config_dir: Path) -> dict[str, Any]:
    task = config.get("task") or {}
    split = config.get("split") or {}
    training = config.get("training") or {}
    acquisition = config.get("acquisition") or {}
    generation = config.get("generation") or {}
    ingestion = config.get("ingestion") or {}
    candidates = config.get("candidates") or {}
    metadata = config.get("metadata") or {}

    task_type = str(task.get("type", config.get("task_type", "regression"))).lower()
    manifest = validate_evaluation_manifest(config, task_type=task_type)

    assay_files = _as_list(config.get("assay") or ingestion.get("raw_files"))
    candidate_files = _as_list(config.get("candidate_files") or candidates.get("raw_files"))
    if not assay_files:
        raise ValueError("Config must provide assay or ingestion.raw_files.")

    output_dir = config.get("output_dir")
    return {
        "project": str(config.get("project") or "assayready_project"),
        "assay_files": _resolve_paths(assay_files, base_dir=config_dir),
        "candidate_files": _resolve_paths(candidate_files, base_dir=config_dir),
        "output_dir": Path(output_dir) if output_dir else None,
        "task_type": task_type,
        "sequence_col": str(task.get("sequence_col", config.get("sequence_col", "sequence"))),
        "target_col": str(task.get("target_col", config.get("target_col", "target"))),
        "positive_label": task.get("positive_label", config.get("positive_label")),
        "id_col": task.get("id_col", config.get("id_col")),
        "metadata_cols": list(metadata.get("categorical_cols") or config.get("metadata_cols") or []),
        "group_cols": list(split.get("group_cols") or config.get("group_cols") or []),
        "column_aliases": dict(ingestion.get("column_aliases") or config.get("column_aliases") or {}),
        "low_n_threshold": int(ingestion.get("low_n_threshold", config.get("low_n_threshold", 200))),
        "val_fraction": float(split.get("val_fraction", config.get("val_fraction", 0.15))),
        "test_fraction": float(split.get("test_fraction", config.get("test_fraction", 0.15))),
        "homology_threshold": float(split.get("homology_threshold", config.get("homology_threshold", 0.90))),
        "homology_k": int(split.get("homology_k", config.get("homology_k", 8))),
        "epochs": int(training.get("epochs", config.get("epochs", 80))),
        "learning_rate": float(training.get("learning_rate", config.get("learning_rate", 1e-3))),
        "ensemble_size": int((config.get("uq") or {}).get("ensemble_size", config.get("ensemble_size", 5))),
        "seed": int(training.get("seed", split.get("seed", config.get("seed", 13)))),
        "acquisition_method": str(acquisition.get("method", config.get("acquisition_method", "upper_confidence_bound"))),
        "beta": float(acquisition.get("beta", config.get("beta", 1.0))),
        "diversity_method": str(acquisition.get("diversity_method", config.get("diversity_method", "greedy_embedding_cosine"))),
        "diversity_penalty": float(acquisition.get("diversity_penalty", config.get("diversity_penalty", 0.2))),
        "top_k": int(acquisition.get("top_k", config.get("top_k", generation.get("plate_size", 96)))),
        "generate_candidates": bool(generation.get("enabled", config.get("generate_candidates", False))),
        "num_proposals": int(generation.get("num_proposals", config.get("num_proposals", 1000))),
        "sequence_length": generation.get("sequence_length", config.get("sequence_length", "infer_from_training")),
        "gc_range": generation.get("gc_range", config.get("gc_range")),
        "evaluation_manifest": manifest.to_dict(),
        "manifest_source": "config",
    }


def _prediction_params_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if not args.assay:
        raise ValueError("--assay is required when --config is not provided.")
    if not args.target_col:
        raise ValueError("--target-col is required when --config is not provided.")
    if not args.prediction_col:
        raise ValueError("--prediction-col is required when --config is not provided.")
    manifest = inferred_manifest_for_args(task_type=args.task_type, positive_label=args.positive_label)
    return {
        "project": args.project,
        "assay_files": _resolve_paths(args.assay, base_dir=Path.cwd()),
        "candidate_files": _resolve_paths(args.candidates or [], base_dir=Path.cwd()),
        "output_dir": Path(args.output_dir) if args.output_dir else None,
        "task_type": args.task_type,
        "sequence_col": args.sequence_col,
        "target_col": args.target_col,
        "prediction_col": args.prediction_col,
        "uncertainty_col": args.uncertainty_col,
        "positive_label": args.positive_label,
        "id_col": args.id_col,
        "metadata_cols": args.metadata_cols or [],
        "group_cols": args.group_cols or [],
        "column_aliases": _parse_aliases(args.alias),
        "low_n_threshold": args.low_n_threshold,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "homology_threshold": args.homology_threshold,
        "homology_k": args.homology_k,
        "seed": args.seed,
        "beta": args.beta,
        "diversity_method": args.diversity_method,
        "diversity_penalty": args.diversity_penalty,
        "top_k": args.top_k,
        "evaluation_manifest": manifest.to_dict(),
        "manifest_source": "inferred_from_args",
    }


def _prediction_params_from_config(config: dict[str, Any], *, config_dir: Path) -> dict[str, Any]:
    task = config.get("task") or {}
    split = config.get("split") or {}
    ingestion = config.get("ingestion") or {}
    candidates = config.get("candidates") or {}
    metadata = config.get("metadata") or {}
    acquisition = config.get("acquisition") or {}

    task_type = str(task.get("type", config.get("task_type", "regression"))).lower()
    manifest = validate_evaluation_manifest(config, task_type=task_type)

    assay_files = _as_list(config.get("assay") or config.get("assay_files") or ingestion.get("raw_files"))
    candidate_files = _as_list(config.get("candidate_files") or candidates.get("raw_files"))
    if not assay_files:
        raise ValueError("Config must provide assay_files, assay, or ingestion.raw_files.")

    output_dir = config.get("output_dir")
    prediction_col = task.get("prediction_col", config.get("prediction_col"))
    if not prediction_col:
        raise ValueError("Prediction audit config must provide prediction_col or task.prediction_col.")
    return {
        "project": str(config.get("project") or "prediction_audit"),
        "assay_files": _resolve_paths(assay_files, base_dir=config_dir),
        "candidate_files": _resolve_paths(candidate_files, base_dir=config_dir),
        "output_dir": Path(output_dir) if output_dir else None,
        "task_type": task_type,
        "sequence_col": str(task.get("sequence_col", config.get("sequence_col", "sequence"))),
        "target_col": str(task.get("target_col", config.get("target_col", "target"))),
        "prediction_col": str(prediction_col),
        "uncertainty_col": task.get("uncertainty_col", config.get("uncertainty_col")),
        "positive_label": task.get("positive_label", config.get("positive_label")),
        "id_col": task.get("id_col", config.get("id_col")),
        "metadata_cols": list(metadata.get("categorical_cols") or config.get("metadata_cols") or []),
        "group_cols": list(split.get("group_cols") or config.get("group_cols") or []),
        "column_aliases": dict(ingestion.get("column_aliases") or config.get("column_aliases") or {}),
        "low_n_threshold": int(ingestion.get("low_n_threshold", config.get("low_n_threshold", 200))),
        "val_fraction": float(split.get("val_fraction", config.get("val_fraction", 0.15))),
        "test_fraction": float(split.get("test_fraction", config.get("test_fraction", 0.15))),
        "homology_threshold": float(split.get("homology_threshold", config.get("homology_threshold", 0.90))),
        "homology_k": int(split.get("homology_k", config.get("homology_k", 8))),
        "seed": int(split.get("seed", config.get("seed", 13))),
        "beta": float(acquisition.get("beta", config.get("beta", 1.0))),
        "diversity_method": str(acquisition.get("diversity_method", config.get("diversity_method", "greedy_embedding_cosine"))),
        "diversity_penalty": float(acquisition.get("diversity_penalty", config.get("diversity_penalty", 0.2))),
        "top_k": int(acquisition.get("top_k", config.get("top_k", 96))),
        "evaluation_manifest": manifest.to_dict(),
        "manifest_source": "config",
    }


def _artifact_dir(project: str, output_dir: Path | None) -> Path:
    safe_project = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(project).strip()).strip("-._") or "assayready"
    run_token = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:8]}"
    if output_dir is not None:
        base = Path(output_dir).expanduser()
        if base.exists() and not base.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {base}")
        base.mkdir(parents=True, exist_ok=True)
        out = base / run_token
    else:
        out = Path("outputs") / "assayready" / safe_project / run_token
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_records(paths: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        records.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    return records


def _git_provenance() -> dict[str, Any]:
    candidate = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(candidate), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"revision": "unavailable", "dirty": None}


def _dependency_versions() -> dict[str, str]:
    names = ["model-assessment", "dash", "numpy", "pandas", "scikit-learn", "torch"]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _execution_manifest(
    *,
    workflow: str,
    params: dict[str, Any],
    artifact_dir: Path,
    artifact_names: list[str],
    started_at: str,
    duration_seconds: float,
) -> dict[str, Any]:
    input_paths = list(params.get("assay_files") or []) + list(params.get("candidate_files") or [])
    if params.get("config_path"):
        input_paths.append(params["config_path"])
    artifact_paths = [artifact_dir / name for name in artifact_names]
    return {
        "schema_version": 1,
        "execution_id": uuid.uuid4().hex,
        "workflow": workflow,
        "status": "completed",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_seconds": round(float(duration_seconds), 6),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": _dependency_versions(),
        "git": _git_provenance(),
        "inputs": _file_records(input_paths),
        "artifacts": _file_records(artifact_paths),
    }


def _record_run_safely(summary: dict[str, Any], *, workflow: str, report_name: str) -> str | None:
    try:
        return record_run(summary, workflow=workflow, report_name=report_name)
    except Exception as exc:
        summary.setdefault("run_store_warning", f"Run was completed but local run indexing failed: {exc}")
        return None


def _primary_metric_name(task_type: str) -> str:
    if task_type == "classification":
        return "auroc_or_balanced_accuracy"
    if task_type == "ranking":
        return "spearman_or_pairwise_accuracy"
    return "r2"


def _validate_runtime_params(params: dict[str, Any], *, includes_training: bool) -> None:
    val_fraction = float(params.get("val_fraction", 0.15))
    test_fraction = float(params.get("test_fraction", 0.15))
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("Validation and test fractions must each be in [0, 1).")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("Validation and test fractions must sum to less than 1 so training rows remain.")
    homology_threshold = float(params.get("homology_threshold", 0.9))
    if not 0.0 <= homology_threshold <= 1.0:
        raise ValueError("Homology/Jaccard threshold must be between 0 and 1.")
    if int(params.get("homology_k", 8)) < 1:
        raise ValueError("Homology k-mer size must be at least 1.")
    if int(params.get("top_k", 96)) < 1:
        raise ValueError("top_k must be at least 1.")
    if float(params.get("beta", 1.0)) < 0.0:
        raise ValueError("Uncertainty beta must be non-negative.")
    if float(params.get("diversity_penalty", 0.2)) < 0.0:
        raise ValueError("Diversity penalty must be non-negative.")
    if int(params.get("low_n_threshold", 200)) < 0:
        raise ValueError("Low-N threshold must be non-negative.")
    if includes_training:
        if int(params.get("ensemble_size", 5)) < 1 or int(params.get("epochs", 80)) < 1:
            raise ValueError("Ensemble size and epochs must be at least 1.")
        if float(params.get("learning_rate", 1e-3)) <= 0.0:
            raise ValueError("Learning rate must be positive.")


def _as_int_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def verify_cross_split_violations(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    sequence_col: str,
    homology_threshold: float,
    homology_k: int,
    group_cols: list[str],
) -> list[str]:
    violations = []
    if group_cols:
        for g_col in group_cols:
            train_groups = {str(row[g_col]).strip() for row in train_rows if row.get(g_col) is not None}
            test_groups = {str(row[g_col]).strip() for row in test_rows if row.get(g_col) is not None}
            overlap = train_groups.intersection(test_groups)
            if overlap:
                violations.append(f"Group overlap detected in '{g_col}': {overlap}")
    
    # Homology similarity check
    if homology_threshold > 0 and train_rows and test_rows:
        train_kmer_sets = [sequence_kmers(row[sequence_col], homology_k) for row in train_rows if row.get(sequence_col)]
        for test_row in test_rows:
            test_seq = test_row.get(sequence_col)
            if not test_seq:
                continue
            test_kmers = sequence_kmers(test_seq, homology_k)
            if not test_kmers:
                continue
            for train_kmers in train_kmer_sets:
                union_size = len(test_kmers.union(train_kmers))
                if union_size == 0:
                    continue
                jaccard = len(test_kmers.intersection(train_kmers)) / union_size
                if jaccard >= homology_threshold:
                    violations.append(
                        f"Homology similarity violation: test sequence has Jaccard similarity {jaccard:.2f} >= threshold {homology_threshold} with train sequence."
                    )
                    if len(violations) >= 5:
                        violations.append("Additional homology violations omitted...")
                        return violations
    return violations


def _bootstrap_metric_ci(
    rows: list[dict[str, Any]],
    *,
    task_type: str,
    target_col: str,
    prediction_col: str,
    positive_label: str | None,
    n_resamples: int = 200,
) -> tuple[float, float] | None:
    if not rows:
        return None
    metrics = []
    rng = np.random.default_rng(42)
    indices = np.arange(len(rows))
    for _ in range(n_resamples):
        resample_idx = rng.choice(indices, size=len(rows), replace=True)
        resampled_rows = [rows[idx] for idx in resample_idx]
        try:
            m = _prediction_metrics(
                resampled_rows,
                task_type=task_type,
                target_col=target_col,
                prediction_col=prediction_col,
                positive_label=positive_label,
            )
            val = m.get("primary_metric")
            if val is not None:
                metrics.append(val)
        except Exception:
            continue
    if not metrics:
        return None
    return float(np.percentile(metrics, 2.5)), float(np.percentile(metrics, 97.5))


def _lift_ci_from_metric_ci(
    model_metric_ci: tuple[float, float] | None,
    baseline_metric: float | None,
) -> tuple[float, float] | None:
    """Compare a bootstrapped model interval with a fixed baseline point estimate.

    This is intentionally not described as a paired lift interval: generating a
    paired interval requires per-row baseline predictions, which the current
    baseline API does not expose.
    """
    if model_metric_ci is None or baseline_metric is None:
        return None
    return (
        float(model_metric_ci[0]) - float(baseline_metric),
        float(model_metric_ci[1]) - float(baseline_metric),
    )


def _claim_gate(
    *,
    task_type: str,
    split_diagnostics: dict[str, Any],
    warnings: list[str],
    best_baseline_metric: float | None,
    model_metric: float | None,
    uncertainty_audit: dict[str, Any] | None,
    ranked_candidates: int,
    evaluation_manifest: dict[str, Any],
    cross_split_violations: list[str] | None = None,
    model_metric_ci: tuple[float, float] | None = None,
    lift_delta_ci: tuple[float, float] | None = None,
    external_predictions: bool = False,
) -> dict[str, Any]:
    thresholds = dict(evaluation_manifest.get("claim_thresholds") or {})
    min_test_rows = _as_int_default(thresholds.get("min_test_rows"), 20)
    min_num_clusters = _as_int_default(thresholds.get("min_num_clusters"), 3)
    min_lift_delta = _as_float_default(thresholds.get("min_lift_delta"), 0.0)
    min_uncertainty_spearman = _as_float_default(thresholds.get("min_uncertainty_spearman"), 0.20)
    max_calibration_gap_ratio = _as_float_default(thresholds.get("max_calibration_gap_ratio"), 1.0)

    split_sizes = split_diagnostics.get("split_sizes") or {}
    test_rows = _as_int_default(split_sizes.get("test"), 0)
    num_clusters = _as_int_default(split_diagnostics.get("num_clusters"), 0)
    severe_warning = any("DO NOT TRUST" in str(item) for item in warnings)
    training_independence = str(evaluation_manifest.get("training_independence") or "unverified").lower()

    leakage_reasons: list[str] = []
    if test_rows < min_test_rows:
        leakage_reasons.append(f"test rows {test_rows} < required {min_test_rows}")
    if num_clusters < min_num_clusters:
        leakage_reasons.append(f"clusters {num_clusters} < required {min_num_clusters}")
    if severe_warning:
        leakage_reasons.append("split quality warning flagged as DO NOT TRUST")
    if training_independence == "unverified":
        leakage_reasons.append("model-training independence is unverified")
    if external_predictions and training_independence != "verified_holdout":
        leakage_reasons.append("external predictions were not declared as generated on a verified locked holdout")
    if cross_split_violations:
        for violation in cross_split_violations:
            leakage_reasons.append(f"cross-split violation: {violation}")
    leakage_ok = not leakage_reasons

    lift_reasons: list[str] = []
    if model_metric is None:
        lift_reasons.append("model metric is missing")
    if best_baseline_metric is None:
        lift_reasons.append("baseline metric is missing")
    if leakage_reasons:
        lift_reasons.append("leakage gate failed")

    lift_delta: float | None = None
    if model_metric is not None and best_baseline_metric is not None:
        # Primary model metrics in this release are all higher-is-better. Target
        # objective direction affects candidate acquisition, not metric polarity.
        lift_delta = float(model_metric) - float(best_baseline_metric)
            
        if lift_delta < min_lift_delta:
            lift_reasons.append(f"lift delta {lift_delta:.6f} < required {min_lift_delta:.6f}")
            
        if lift_delta_ci is not None:
            lower_bound, _ = lift_delta_ci
            if lower_bound < min_lift_delta:
                lift_reasons.append(f"lift delta 95% CI lower bound {lower_bound:.6f} < required {min_lift_delta:.6f}")

    lift_ok = not lift_reasons

    uncertainty_type = str(evaluation_manifest.get("uncertainty_type") or "none").lower()
    calibration_reasons: list[str] = []
    if uncertainty_type == "none":
        calibration_reasons.append("uncertainty_type is 'none'")
    else:
        uncertainty = uncertainty_audit or {}
        if uncertainty.get("status") != "ok":
            calibration_reasons.append("uncertainty audit is not available")
        else:
            spearman = _safe_float(uncertainty.get("uncertainty_abs_error_spearman"))
            mean_error = _safe_float(uncertainty.get("mean_abs_error"))
            gap = _safe_float(uncertainty.get("mean_absolute_calibration_gap"))
            if spearman is None or spearman < min_uncertainty_spearman:
                calibration_reasons.append(
                    f"uncertainty/error spearman {spearman} < required {min_uncertainty_spearman}"
                )
            if uncertainty_type == "predicted_absolute_error":
                if mean_error is None or gap is None:
                    calibration_reasons.append("absolute-error scale metrics are missing")
                elif gap > max(1e-12, mean_error) * max_calibration_gap_ratio:
                    calibration_reasons.append(
                        f"absolute-error scale gap ratio {(gap / max(1e-12, mean_error)):.6f} exceeds "
                        f"{max_calibration_gap_ratio:.6f}"
                    )
            elif uncertainty_type not in {"predictive_standard_deviation"}:
                calibration_reasons.append(
                    f"uncertainty_type {uncertainty_type!r} is not dimensionally evaluable by this audit"
                )
    calibration_ok = not calibration_reasons

    constraints_verified = bool(evaluation_manifest.get("constraints_verified", False))
    constraint_reasons = [] if constraints_verified else ["candidate biological/synthesis constraints were not verified"]

    recommendation_reasons: list[str] = []
    if ranked_candidates <= 0:
        recommendation_reasons.append("no ranked candidates were produced")
    if severe_warning:
        recommendation_reasons.append("severe warning present")
    if not leakage_ok:
        recommendation_reasons.append("leakage gate failed")
    if not lift_ok:
        recommendation_reasons.append("lift gate failed")
    if uncertainty_type != "none" and not calibration_ok:
        recommendation_reasons.append("uncertainty usability gate failed")
    if constraint_reasons:
        recommendation_reasons.append("candidate constraint gate failed")
    recommendation_ok = not recommendation_reasons

    return {
        "thresholds": {
            "min_test_rows": min_test_rows,
            "min_num_clusters": min_num_clusters,
            "min_lift_delta": min_lift_delta,
            "min_uncertainty_spearman": min_uncertainty_spearman,
            "max_calibration_gap_ratio": max_calibration_gap_ratio,
        },
        "leakage_controlled": {"ok": leakage_ok, "reasons": leakage_reasons},
        "lift_claim": {
            "ok": lift_ok,
            "delta": lift_delta,
            "model_metric_ci": model_metric_ci,
            "lift_delta_ci": lift_delta_ci,
            "reasons": lift_reasons,
        },
        "uncertainty_usable": {"ok": calibration_ok, "reasons": calibration_reasons},
        "calibrated_uncertainty": {"ok": calibration_ok, "reasons": calibration_reasons},
        "candidate_constraints": {"ok": constraints_verified, "reasons": constraint_reasons},
        "recommended": {"ok": recommendation_ok, "reasons": recommendation_reasons},
    }


def _metric_lines(metrics: dict[str, Any]) -> list[str]:
    if not metrics:
        return ["- No metrics available."]
    lines: list[str] = []
    for key, value in metrics.items():
        if key in {"models", "raw"}:
            continue
        lines.append(f"- {key}: {value}")
    return lines or ["- No metrics available."]


def _benchmark_markdown(report: dict[str, Any]) -> str:
    claim_gate = report.get("claim_gate") or {}
    leakage_ok = (claim_gate.get("leakage_controlled") or {}).get("ok")
    lift_ok = (claim_gate.get("lift_claim") or {}).get("ok")
    calibration_ok = (claim_gate.get("uncertainty_usable") or {}).get("ok")
    recommended_ok = (claim_gate.get("recommended") or {}).get("ok")
    manifest = report.get("evaluation_manifest") or {}
    lines = [
        f"# AssayReady Benchmark: {report['project']}",
        "",
        "## Summary",
        "",
        f"- Task type: {report['task_type']}",
        f"- Primary metric: {report['primary_metric_name']}",
        f"- Best simple baseline: {report['best_simple_baseline']}",
        f"- Task-head primary metric: {report.get('task_head_primary_metric')}",
        f"- Verdict: {report['verdict']}",
        "",
        "## Evaluation Manifest",
        "",
        f"- Schema version: {manifest.get('schema_version')}",
        f"- Objective direction: {manifest.get('objective_direction')}",
        f"- Units: {manifest.get('units')}",
        f"- Uncertainty type: {manifest.get('uncertainty_type')}",
        f"- Training independence: {manifest.get('training_independence')}",
        f"- Candidate constraints verified: {manifest.get('constraints_verified')}",
        f"- Positive label: {manifest.get('positive_label')}",
        f"- Biological constraints: {manifest.get('biological_constraints')}",
        "",
        "## Claim Gate",
        "",
        f"- leakage_controlled: {leakage_ok}",
        f"- lift_claim: {lift_ok}",
        f"- uncertainty_usable: {calibration_ok}",
        f"- recommended: {recommended_ok}",
        "",
    ]
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    lines.extend(["## Baselines", ""])
    for name, metrics in report.get("baselines", {}).items():
        lines.append(f"### {name}")
        lines.extend(_metric_lines(metrics))
        lines.append("")
    lines.extend(["## Task Head", ""])
    lines.extend(_metric_lines(report.get("task_head", {})))
    lines.append("")
    return "\n".join(lines)


def _readiness_markdown(summary: dict[str, Any]) -> str:
    report = summary["benchmark_report"]
    audit = summary["audit"]
    split = summary["split_diagnostics"]
    lines = [
        f"# AssayReady Report: {summary['project']}",
        "",
        "## Verdict",
        "",
        report["verdict"],
        "",
        "## Data Readiness",
        "",
        f"- Accepted labeled rows: {audit.get('accepted_rows')}",
        f"- Duplicate sequences: {audit.get('duplicate_sequences')}",
        f"- Conflicting duplicate labels: {audit.get('conflicting_duplicate_labels')}",
        f"- Invalid DNA rows: {audit.get('invalid_dna_rows')}",
        f"- Split sizes: {split.get('split_sizes')}",
        f"- Canonical k-mer/group clusters: {split.get('num_clusters')}",
        "",
        "## Model Lift",
        "",
        f"- Primary metric: {report.get('primary_metric_name')}",
        f"- Best simple baseline: {report.get('best_simple_baseline')}",
        f"- Task-head metric: {report.get('task_head_primary_metric')}",
        "",
        "## Next Experiment",
        "",
        f"- Ranked candidate rows: {summary.get('ranked_candidates', 0)}",
        f"- Generated candidate rows: {summary.get('generated_candidates', 0)}",
        "- Candidate explanations include nearest training examples, training-distribution status, and risk flags.",
        "- Treat all ranked designs as in-silico hypotheses until wet-lab validated.",
        "",
        "## Artifacts",
        "",
    ]
    for artifact in summary["artifacts"]:
        lines.append(f"- {artifact}")
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.append("")
    return "\n".join(lines)


def _model_card_markdown(summary: dict[str, Any]) -> str:
    params = summary["params"]
    lines = [
        f"# Model Card: {summary['project']}",
        "",
        "## Intended Use",
        "",
        "- Local assay audit and candidate ranking for computational biology teams.",
        "- Exploratory assay-data and retrospective model audit with candidate prioritization.",
        "- Not intended for clinical, diagnostic, therapeutic, or environmental release decisions.",
        "- Not intended for patient-specific clinical decision support or regulated medical-device claims.",
        "- Generated or ranked sequences require wet-lab validation before any biological claim.",
        "",
        "## Input Contract",
        "",
        f"- Task type: {params['task_type']}",
        f"- Sequence column: {params['sequence_col']}",
        f"- Target column: {params['target_col']}",
        f"- ID column: {params.get('id_col')}",
        f"- Metadata columns: {params.get('metadata_cols')}",
        f"- Group columns: {params.get('group_cols')}",
        "",
        "## Validation Policy",
        "",
        "- Train/validation/test splits use declared groups and canonical k-mer Jaccard clustering.",
        "- Simple GC/length and k-mer baselines are reported before model lift is trusted.",
        "- Low-N, split-quality, and cross-split violations produce explicit warnings and block claims.",
        "",
        "## Result",
        "",
        f"- Verdict: {summary['benchmark_report']['verdict']}",
        f"- Artifact directory: {summary['artifact_dir']}",
        "",
    ]
    return "\n".join(lines)


def _normalize_binary_label(value: Any) -> str:
    try:
        val_float = float(value)
        if val_float.is_integer():
            return str(int(val_float))
        return str(val_float)
    except (TypeError, ValueError):
        return str(value).strip().lower()


def _resolve_binary_labels(raw_targets: list[str], *, positive_label: str | None) -> tuple[str, str]:
    labels = sorted(set(raw_targets))
    if not labels or len(labels) > 2:
        raise ValueError("Prediction audit classification requires one or two labels in each evaluated subset.")

    if positive_label is not None:
        positive = _normalize_binary_label(positive_label)
        if len(labels) == 2 and positive not in labels:
            raise ValueError(f"Configured positive_label={positive_label!r} is not present in measured labels: {labels}.")
    elif len(labels) == 2:
        preferred = ["1", "true", "yes", "positive", "pos", "active", "hit"]
        positive = next((label for label in preferred if label in labels), labels[-1])
    else:
        raise ValueError("positive_label is required when an evaluated classification subset contains one class.")
    negative = next((label for label in labels if label != positive), "__unobserved_negative__")
    return negative, positive


def _classification_arrays(
    rows: list[dict[str, Any]],
    *,
    target_col: str,
    prediction_col: str,
    positive_label: str | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    raw_targets = [_normalize_binary_label(row[target_col]) for row in rows]
    negative_label, positive_label_resolved = _resolve_binary_labels(raw_targets, positive_label=positive_label)
    mapping = {negative_label: 0, positive_label_resolved: 1}
    y_true = np.asarray([mapping[item] for item in raw_targets], dtype=np.float64)
    y_score = np.asarray([float(row[prediction_col]) for row in rows], dtype=np.float64)
    y_score = np.clip(y_score, 1e-7, 1.0 - 1e-7)
    return y_true, y_score, {"negative_label": negative_label, "positive_label": positive_label_resolved}


def _prediction_metrics(
    rows: list[dict[str, Any]],
    *,
    task_type: str,
    target_col: str,
    prediction_col: str,
    positive_label: str | None,
) -> dict[str, Any]:
    if not rows:
        return {"status": "skipped", "reason": "no rows with valid predictions", "primary_metric": None}
    if task_type == "classification":
        y_true, y_score, label_info = _classification_arrays(
            rows,
            target_col=target_col,
            prediction_col=prediction_col,
            positive_label=positive_label,
        )
        y_pred = (y_score >= 0.5).astype(np.float64)
        accuracy = float(np.mean(y_pred == y_true))
        brier = float(np.mean((y_score - y_true) ** 2))
        log_loss = float(-np.mean(y_true * np.log(y_score) + (1.0 - y_true) * np.log(1.0 - y_score)))
        discrimination_evaluable = len(set(y_true.tolist())) == 2
        auroc = None
        average_precision = None
        balanced_accuracy = None
        if discrimination_evaluable:
            positive_scores = y_score[y_true == 1]
            negative_scores = y_score[y_true == 0]
            comparisons = positive_scores[:, None] - negative_scores[None, :]
            auroc = float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))

            order = np.argsort(-y_score, kind="mergesort")
            sorted_true = y_true[order]
            cumulative_positive = np.cumsum(sorted_true)
            positive_positions = np.flatnonzero(sorted_true == 1)
            average_precision = float(
                np.mean(cumulative_positive[positive_positions] / (positive_positions + 1))
            )

            true_positive = float(np.sum((y_pred == 1) & (y_true == 1)))
            true_negative = float(np.sum((y_pred == 0) & (y_true == 0)))
            positive_count = float(np.sum(y_true == 1))
            negative_count = float(np.sum(y_true == 0))
            balanced_accuracy = 0.5 * (
                true_positive / positive_count + true_negative / negative_count
            )
        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "auroc": auroc,
            "average_precision": average_precision,
            "brier": brier,
            "log_loss": log_loss,
            "pearson": _pearson(y_true, y_score),
            "spearman": _spearman(y_true, y_score),
            "num_rows": len(rows),
            "primary_metric": auroc,
            "primary_metric_name": "auroc" if discrimination_evaluable else None,
            "status": "ok" if discrimination_evaluable else "not_evaluable_single_class",
            **label_info,
        }

    y_true = np.asarray([float(row[target_col]) for row in rows], dtype=np.float64)
    y_pred = np.asarray([float(row[prediction_col]) for row in rows], dtype=np.float64)
    residual = y_pred - y_true
    mse = float(np.mean(residual**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = None if denom <= 1e-12 else float(1.0 - np.sum(residual**2) / denom)
    spearman = _spearman(y_true, y_pred)
    primary_metric = spearman if task_type == "ranking" else r2
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": _pearson(y_true, y_pred),
        "spearman": spearman,
        "num_rows": len(rows),
        "primary_metric": primary_metric,
        "primary_metric_name": "spearman" if task_type == "ranking" else "r2",
    }


def _uncertainty_audit(
    rows: list[dict[str, Any]],
    *,
    task_type: str,
    target_col: str,
    prediction_col: str,
    uncertainty_col: str | None,
    positive_label: str | None,
) -> dict[str, Any]:
    if not uncertainty_col:
        return {"status": "skipped", "reason": "no uncertainty column provided"}
    usable_rows: list[dict[str, Any]] = []
    usable = usable_rows  # Backward-compatible alias for older editable installs during active development.
    usable_uncertainty: list[float] = []
    for row in rows:
        uncertainty = _safe_float(row.get(uncertainty_col))
        if uncertainty is None:
            continue
        usable_rows.append(row)
        usable_uncertainty.append(float(max(0.0, uncertainty)))
    if len(usable_rows) < 3:
        return {"status": "skipped", "reason": "fewer than 3 rows have valid uncertainty values", "num_rows": len(usable_rows)}

    uncertainty = np.asarray(usable_uncertainty, dtype=np.float64)
    if task_type == "classification":
        y_true, y_score, _ = _classification_arrays(
            usable_rows,
            target_col=target_col,
            prediction_col=prediction_col,
            positive_label=positive_label,
        )
        abs_error = np.abs(y_score - y_true)
    else:
        abs_error = np.asarray(
            [abs(float(row[prediction_col]) - float(row[target_col])) for row in usable_rows],
            dtype=np.float64,
        )
    order = np.argsort(uncertainty)
    quantile_bins = np.array_split(order, min(5, len(order)))
    bins: list[dict[str, Any]] = []
    for idx, bin_indices in enumerate(quantile_bins, start=1):
        if len(bin_indices) == 0:
            continue
        bins.append(
            {
                "bin": idx,
                "count": int(len(bin_indices)),
                "mean_uncertainty": float(np.mean(uncertainty[bin_indices])),
                "mean_abs_error": float(np.mean(abs_error[bin_indices])),
            }
        )
    return {
        "status": "ok",
        "num_rows": int(len(usable)),
        "uncertainty_abs_error_pearson": _pearson(uncertainty, abs_error),
        "uncertainty_abs_error_spearman": _spearman(uncertainty, abs_error),
        "mean_uncertainty": float(np.mean(uncertainty)),
        "mean_abs_error": float(np.mean(abs_error)),
        "mean_absolute_calibration_gap": float(np.mean(np.abs(uncertainty - abs_error))),
        "bins": bins,
    }


def _ranking_audit(
    rows: list[dict[str, Any]],
    *,
    task_type: str,
    target_col: str,
    prediction_col: str,
    top_k: int,
    positive_label: str | None,
) -> dict[str, Any]:
    if not rows:
        return {"status": "skipped", "reason": "no rows with valid predictions"}
    if task_type == "classification":
        y_true, y_pred, _ = _classification_arrays(
            rows,
            target_col=target_col,
            prediction_col=prediction_col,
            positive_label=positive_label,
        )
    else:
        y_true = np.asarray([float(row[target_col]) for row in rows], dtype=np.float64)
        y_pred = np.asarray([float(row[prediction_col]) for row in rows], dtype=np.float64)
    top_k = max(1, min(int(top_k), len(rows)))
    order = np.argsort(-y_pred)
    top_indices = order[:top_k]
    best_true_index = int(np.argmax(y_true))
    true_best_prediction_rank = int(np.where(order == best_true_index)[0][0] + 1)
    overall_mean = float(np.mean(y_true))
    top_mean = float(np.mean(y_true[top_indices]))
    return {
        "status": "ok",
        "num_rows": len(rows),
        "top_k": top_k,
        "overall_target_mean": overall_mean,
        "top_k_target_mean": top_mean,
        "top_k_enrichment": None if abs(overall_mean) < 1e-12 else float(top_mean / overall_mean),
        "best_true_item_rank_by_prediction": true_best_prediction_rank,
        "spearman": _spearman(y_true, y_pred),
    }


def _prediction_audit_markdown(summary: dict[str, Any]) -> str:
    report = summary["prediction_audit"]
    positive_label = report.get("positive_label")
    positive_label_line = [f"- Positive class label: {positive_label}"] if positive_label else []
    claim_gate = report.get("claim_gate") or {}
    leakage_ok = (claim_gate.get("leakage_controlled") or {}).get("ok")
    lift_ok = (claim_gate.get("lift_claim") or {}).get("ok")
    calibration_ok = (claim_gate.get("uncertainty_usable") or {}).get("ok")
    recommended_ok = (claim_gate.get("recommended") or {}).get("ok")
    manifest = report.get("evaluation_manifest") or {}
    lines = [
        f"# Model Prediction Audit: {summary['project']}",
        "",
        "## Verdict",
        "",
        report["verdict"],
        "",
        "## Data Readiness",
        "",
        f"- Accepted labeled rows: {summary['audit'].get('accepted_rows')}",
        f"- Rows with valid predictions: {summary.get('valid_prediction_rows')}",
        f"- Invalid prediction rows: {summary.get('invalid_prediction_rows')}",
        f"- Split sizes: {summary['split_diagnostics'].get('split_sizes')}",
        "",
        "## External Model Performance",
        "",
        f"- Primary metric: {report.get('primary_metric_name')}",
        *positive_label_line,
        f"- All rows metric: {report.get('all_rows_metrics', {}).get('primary_metric')}",
        f"- Retrospective clustered-split metric: {report.get('test_metrics', {}).get('primary_metric')}",
        f"- Best simple baseline: {report.get('best_simple_baseline')}",
        "",
        "## Evaluation Manifest",
        "",
        f"- Schema version: {manifest.get('schema_version')}",
        f"- Objective direction: {manifest.get('objective_direction')}",
        f"- Units: {manifest.get('units')}",
        f"- Uncertainty type: {manifest.get('uncertainty_type')}",
        f"- Training independence: {manifest.get('training_independence')}",
        f"- Candidate constraints verified: {manifest.get('constraints_verified')}",
        f"- Biological constraints: {manifest.get('biological_constraints')}",
        "",
        "## Claim Gate",
        "",
        f"- leakage_controlled: {leakage_ok}",
        f"- lift_claim: {lift_ok}",
        f"- uncertainty_usable: {calibration_ok}",
        f"- recommended: {recommended_ok}",
        "",
        "## Uncertainty",
        "",
    ]
    for key, value in report.get("uncertainty_audit", {}).items():
        if key == "bins":
            continue
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Ranking", ""])
    for key, value in report.get("ranking_audit", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Candidate Explainability",
            "",
            "- `candidate_explanations.csv` and `candidate_explanations.json` explain each ranked candidate.",
            "- Explanations include prediction, uncertainty, acquisition score, diversity cluster, nearest training examples, and training-distribution status.",
            "- Risk flags are propagated when data readiness, model lift, or uncertainty calibration is weak.",
        ]
    )
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.append("")
    return "\n".join(lines)


def _prediction_verdict(
    *,
    claim_gate: dict[str, Any],
    test_metric: float | None,
) -> str:
    if test_metric is None:
        return "Audit-only: external predictions did not produce a trusted retrospective evaluation metric."
    if not bool((claim_gate.get("leakage_controlled") or {}).get("ok")):
        return "Audit-only: leakage-control criteria did not pass for claim-grade conclusions."
    if not bool((claim_gate.get("lift_claim") or {}).get("ok")):
        return "Audit-only: model-vs-baseline lift criteria did not pass documented thresholds."
    if not bool((claim_gate.get("recommended") or {}).get("ok")):
        return "Promising only as a pilot: supplied predictions beat baselines, but data volume or split quality is weak."
    return "Recommended for next-round prioritization: claim gates passed for leakage control, lift, and recommendation readiness."


def _verdict(
    *,
    claim_gate: dict[str, Any],
    task_head_metric: float | None,
) -> str:
    if task_head_metric is None:
        return "Audit-only: task-head training or evaluation did not produce a trusted primary metric."
    if not bool((claim_gate.get("leakage_controlled") or {}).get("ok")):
        return "Audit-only: leakage-control criteria did not pass for claim-grade conclusions."
    if not bool((claim_gate.get("lift_claim") or {}).get("ok")):
        return "Audit-only: model-vs-baseline lift criteria did not pass documented thresholds."
    if not bool((claim_gate.get("recommended") or {}).get("ok")):
        return "Promising only as a pilot: model lift may exist, but data volume or split quality is not yet strong."
    return "Recommended for next-round prioritization: claim gates passed for leakage control, lift, and recommendation readiness."


def _observed_incumbent(train_rows: list[dict[str, Any]], *, target_col: str, task_type: str) -> float | None:
    if task_type == "classification":
        return None
    values: list[float] = []
    for row in train_rows:
        try:
            values.append(float(row[target_col]))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None


def _row_identifier(row: dict[str, Any], *, id_col: str | None, fallback_prefix: str, fallback_index: int) -> str:
    for column in [id_col, "sequence_id", "construct_id", "variant_id", "design_id", "id"]:
        if not column:
            continue
        value = str(row.get(column, "") or "").strip()
        if value:
            return value
    seq_hash = str(row.get("sequence_hash", "") or "").strip()
    if seq_hash:
        return f"hash:{seq_hash[:12]}"
    return f"{fallback_prefix}_{fallback_index:04d}"


def _nearest_training_examples(
    candidate: dict[str, Any],
    train_rows: list[dict[str, Any]],
    *,
    sequence_col: str,
    target_col: str | None,
    prediction_col: str | None,
    id_col: str | None,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    sequence = str(candidate.get(sequence_col, "") or "")
    if not sequence or not train_rows:
        return []
    candidate_kmers = sequence_kmers(sequence)
    scored: list[tuple[float, int, int, dict[str, Any]]] = []
    for idx, row in enumerate(train_rows):
        train_sequence = str(row.get(sequence_col, "") or "")
        similarity = jaccard_similarity(candidate_kmers, sequence_kmers(train_sequence))
        scored.append((similarity, 0, idx, row))
    scored.sort(key=lambda item: item[0], reverse=True)

    nearest: list[tuple[float, int, int, dict[str, Any]]] = []
    for similarity, _distance, idx, row in scored[: max(top_n * 3, top_n)]:
        distance = edit_distance(sequence, str(row.get(sequence_col, "") or ""))
        nearest.append((similarity, distance, idx, row))
    nearest.sort(key=lambda item: (-item[0], item[1]))

    out: list[dict[str, Any]] = []
    for similarity, distance, idx, row in nearest[:top_n]:
        item = {
            "id": _row_identifier(row, id_col=id_col, fallback_prefix="train", fallback_index=idx + 1),
            "sequence_hash": row.get("sequence_hash", ""),
            "similarity": float(similarity),
            "edit_distance": int(distance),
            "gc_fraction": float(gc_fraction(str(row.get(sequence_col, "") or ""))),
        }
        if target_col and target_col in row:
            item["target"] = row.get(target_col)
        if prediction_col and prediction_col in row:
            item["prediction"] = row.get(prediction_col)
        out.append(item)
    return out


def _training_distribution_status(candidate_sequence: str, nearest: list[dict[str, Any]]) -> str:
    if not nearest:
        return "unknown_no_training_neighbors"
    length = max(1, len(str(candidate_sequence or "")))
    best = nearest[0]
    similarity = float(best.get("similarity") or 0.0)
    normalized_distance = float(best.get("edit_distance") or length) / float(length)
    if similarity < 0.08 and normalized_distance > 0.35:
        return "outside_training_distribution"
    if similarity < 0.20 or normalized_distance > 0.25:
        return "sparse_training_neighborhood"
    return "near_training_distribution"


def _model_warning_tags(report: dict[str, Any], warnings: list[str]) -> list[str]:
    tags: list[str] = []
    joined = " ".join(warnings).lower()
    if "do not trust" in joined:
        tags.append("weak_data_readiness")
    if "baseline" in joined and ("beat" in joined or "matched" in joined):
        tags.append("model_lift_not_proven")
    uncertainty = report.get("uncertainty_audit") or {}
    if uncertainty.get("status") == "ok":
        spearman = uncertainty.get("uncertainty_abs_error_spearman")
        gap = uncertainty.get("mean_absolute_calibration_gap")
        mean_error = uncertainty.get("mean_abs_error")
        if spearman is None or float(spearman) < 0.20:
            tags.append("uncertainty_not_reliable")
        if gap is not None and mean_error is not None and float(gap) > float(mean_error):
            tags.append("uncertainty_calibration_warning")
    return tags


def _candidate_ranking_reason(row: dict[str, Any], status: str, flags: list[str]) -> str:
    prediction = _safe_float(row.get("prediction"))
    uncertainty = _safe_float(row.get("uncertainty"))
    acquisition = _safe_float(row.get("acquisition_score"))
    cluster = str(row.get("diversity_cluster", "") or "n/a")
    parts = [
        f"prediction={prediction:.4f}" if prediction is not None else "prediction=n/a",
        f"uncertainty={uncertainty:.4f}" if uncertainty is not None else "uncertainty=n/a",
        f"acquisition={acquisition:.4f}" if acquisition is not None else "acquisition=n/a",
        f"cluster={cluster}",
        f"training_status={status}",
    ]
    if flags:
        parts.append(f"warnings={','.join(flags)}")
    return "; ".join(parts)


def _candidate_explanations(
    ranked: list[dict[str, Any]],
    *,
    train_rows: list[dict[str, Any]],
    sequence_col: str,
    target_col: str | None,
    prediction_col: str | None,
    id_col: str | None,
    report: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    if not ranked:
        return []
    uncertainties = np.asarray([float(_safe_float(row.get("uncertainty")) or 0.0) for row in ranked], dtype=np.float64)
    high_uncertainty_cutoff = float(np.quantile(uncertainties, 0.75)) if uncertainties.size else 0.0
    global_tags = _model_warning_tags(report, warnings)

    explanations: list[dict[str, Any]] = []
    for idx, row in enumerate(ranked):
        nearest = _nearest_training_examples(
            row,
            train_rows,
            sequence_col=sequence_col,
            target_col=target_col,
            prediction_col=prediction_col,
            id_col=id_col,
        )
        sequence = str(row.get(sequence_col, "") or "")
        status = _training_distribution_status(sequence, nearest)
        flags = list(global_tags)
        uncertainty = float(_safe_float(row.get("uncertainty")) or 0.0)
        if uncertainty > 0 and uncertainty >= high_uncertainty_cutoff:
            flags.append("high_uncertainty")
        if status in {"outside_training_distribution", "sparse_training_neighborhood"}:
            flags.append(status)

        top_neighbor = nearest[0] if nearest else {}
        explanation = {
            "rank": row.get("rank", idx + 1),
            "display_id": _row_identifier(row, id_col=id_col, fallback_prefix="candidate", fallback_index=idx + 1),
            "sequence_hash": row.get("sequence_hash", ""),
            "sequence": sequence,
            "prediction": row.get("prediction"),
            "uncertainty": row.get("uncertainty"),
            "acquisition_score": row.get("acquisition_score"),
            "diversified_acquisition_score": row.get("diversified_acquisition_score"),
            "diversity_cluster": row.get("diversity_cluster", ""),
            "training_distribution_status": status,
            "nearest_training_examples": nearest,
            "nearest_train_id": top_neighbor.get("id", ""),
            "nearest_train_similarity": top_neighbor.get("similarity"),
            "nearest_train_edit_distance": top_neighbor.get("edit_distance"),
            "nearest_train_target": top_neighbor.get("target"),
            "nearest_train_prediction": top_neighbor.get("prediction"),
            "risk_flags": sorted(set(flags)),
        }
        explanation["ranking_reason"] = _candidate_ranking_reason(row, status, explanation["risk_flags"])
        explanations.append(explanation)
    return explanations


def _write_candidate_explanations(
    artifact_dir: Path,
    ranked: list[dict[str, Any]],
    *,
    train_rows: list[dict[str, Any]],
    sequence_col: str,
    target_col: str | None,
    prediction_col: str | None,
    id_col: str | None,
    report: dict[str, Any],
    warnings: list[str],
) -> None:
    explanations = _candidate_explanations(
        ranked,
        train_rows=train_rows,
        sequence_col=sequence_col,
        target_col=target_col,
        prediction_col=prediction_col,
        id_col=id_col,
        report=report,
        warnings=warnings,
    )
    _write_json(artifact_dir / "candidate_explanations.json", {"candidates": explanations})
    flat_rows = []
    for item in explanations:
        flat = dict(item)
        flat["risk_flags"] = ";".join(item.get("risk_flags") or [])
        flat.pop("nearest_training_examples", None)
        flat_rows.append(flat)
    write_table(
        artifact_dir / "candidate_explanations.csv",
        flat_rows,
        fieldnames=[
            "rank",
            "display_id",
            "sequence_hash",
            "sequence",
            "prediction",
            "uncertainty",
            "acquisition_score",
            "diversified_acquisition_score",
            "diversity_cluster",
            "training_distribution_status",
            "nearest_train_id",
            "nearest_train_similarity",
            "nearest_train_edit_distance",
            "nearest_train_target",
            "nearest_train_prediction",
            "risk_flags",
            "ranking_reason",
        ],
    )


def _rank_candidates(
    candidate_rows: list[dict[str, Any]],
    *,
    ensemble: dict[str, Any],
    train_rows: list[dict[str, Any]],
    sequence_col: str,
    target_col: str,
    task_type: str,
    acquisition_method: str,
    beta: float,
    diversity_method: str,
    diversity_penalty: float,
    top_k: int,
    objective_direction: str = "maximize",
) -> list[dict[str, Any]]:
    if not candidate_rows or ensemble.get("status") != "ok":
        return []
    pred = predict_with_task_head_ensemble(ensemble, candidate_rows)
    
    # We pass standard maximizing acquisition first
    acquisition = acquisition_scores(
        pred["mean"],
        pred["std"],
        method=acquisition_method,
        beta=beta,
        incumbent=_observed_incumbent(train_rows, target_col=target_col, task_type=task_type),
    )
    
    if objective_direction.lower() == "minimize":
        mean_arr = np.asarray(pred["mean"], dtype=np.float64)
        std_arr = np.asarray(pred["std"], dtype=np.float64)
        if acquisition_method.lower() in {"upper_confidence_bound", "ucb"}:
            acquisition_display = mean_arr - beta * std_arr
            acquisition_ranking = -acquisition_display
        elif acquisition_method.lower() in {"expected_improvement", "ei"}:
            best = float(np.min(mean_arr) if _observed_incumbent(train_rows, target_col=target_col, task_type=task_type) is None else _observed_incumbent(train_rows, target_col=target_col, task_type=task_type))
            improvement = best - mean_arr
            std_clipped = np.maximum(std_arr, 1e-9)
            z = improvement / std_clipped
            acquisition_display = improvement * normal_cdf(z) + std_clipped * normal_pdf(z)
            acquisition_ranking = acquisition_display
        else:
            acquisition_display = mean_arr
            acquisition_ranking = -mean_arr
    else:
        acquisition_display = acquisition
        acquisition_ranking = acquisition

    enriched: list[dict[str, Any]] = []
    for idx, row in enumerate(candidate_rows):
        out = dict(row)
        out["prediction"] = float(pred["mean"][idx])
        out["uncertainty"] = float(pred["std"][idx])
        out["acquisition_score"] = float(acquisition_display[idx])
        enriched.append(out)
        
    ranked = greedy_diverse_rank(
        enriched,
        acquisition_ranking,
        sequence_col=sequence_col,
        method=diversity_method,
        diversity_penalty=diversity_penalty,
        top_k=top_k,
    )
    
    if objective_direction.lower() == "minimize" and acquisition_method.lower() in {"upper_confidence_bound", "ucb", "prediction", "mean"}:
        for row in ranked:
            row["diversified_acquisition_score"] = -row["diversified_acquisition_score"]
            
    return ranked


def run_assayready(**params: Any) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started_clock = time.perf_counter()
    _validate_runtime_params(params, includes_training=True)
    project = params["project"]
    task_type = params["task_type"]
    sequence_col = params["sequence_col"]
    target_col = params["target_col"]
    id_col = params.get("id_col")
    metadata_cols = list(params.get("metadata_cols") or [])
    group_cols = list(params.get("group_cols") or [])
    evaluation_manifest = dict(params.get("evaluation_manifest") or inferred_manifest_for_args(task_type=task_type).to_dict())
    evaluation_manifest["training_independence"] = "internally_controlled"
    manifest_source = str(params.get("manifest_source") or "inferred_from_args")
    positive_label = params.get("positive_label") or evaluation_manifest.get("positive_label")
    artifact_dir = _artifact_dir(project, params.get("output_dir"))

    rows, audit, failures = ingest_assay_tables(
        params["assay_files"],
        project=project,
        task_type=task_type,
        sequence_col=sequence_col,
        target_col=target_col,
        id_col=id_col,
        metadata_cols=sorted(set(metadata_cols + group_cols)),
        group_cols=group_cols,
        column_aliases=params.get("column_aliases") or {},
        require_target=True,
        low_n_threshold=int(params.get("low_n_threshold", 200)),
    )
    _write_json(artifact_dir / "data_audit_report.json", audit.to_dict())
    _write_text(artifact_dir / "data_audit_report.md", audit.to_markdown())
    write_table(artifact_dir / "failure_cases.csv", failures, fieldnames=["row_index", "id", "source_file", "sequence", "reason"])

    splits, split_diagnostics = leakage_safe_split(
        rows,
        sequence_col=sequence_col,
        val_fraction=float(params["val_fraction"]),
        test_fraction=float(params["test_fraction"]),
        group_cols=group_cols,
        homology_threshold=float(params["homology_threshold"]),
        homology_k=int(params["homology_k"]),
        seed=int(params["seed"]),
    )
    processed_dir = artifact_dir / "processed"
    for split_name in ["train", "val", "test"]:
        write_table(processed_dir / f"{split_name}.csv", splits[split_name])
    _write_json(artifact_dir / "split_diagnostics.json", split_diagnostics)

    baselines = run_baselines(
        splits["train"],
        splits["test"],
        task_type=task_type,
        sequence_col=sequence_col,
        target_col=target_col,
        seed=int(params["seed"]),
        positive_label=positive_label,
    )
    try:
        ensemble = fit_task_head_ensemble(
            splits["train"],
            splits["val"],
            splits["test"],
            task_type=task_type,
            sequence_col=sequence_col,
            target_col=target_col,
            metadata_cols=metadata_cols,
            ensemble_size=int(params["ensemble_size"]),
            epochs=int(params["epochs"]),
            learning_rate=float(params["learning_rate"]),
            seed=int(params["seed"]),
            positive_label=positive_label,
        )
    except Exception as exc:
        ensemble = {"status": "failed", "reason": str(exc), "metrics": {"status": "failed", "reason": str(exc)}}

    prediction_rows: list[dict[str, Any]] = []
    if splits["test"] and ensemble.get("status") == "ok":
        pred = predict_with_task_head_ensemble(ensemble, splits["test"])
        for idx, row in enumerate(splits["test"]):
            prediction_rows.append(
                {
                    "id": row.get(id_col) if id_col else idx,
                    "sequence_hash": row.get("sequence_hash"),
                    "split": "test",
                    "target": row.get(target_col),
                    "prediction": float(pred["mean"][idx]),
                    "uncertainty": float(pred["std"][idx]),
                    sequence_col: row.get(sequence_col),
                }
            )
    write_table(
        artifact_dir / "predictions.csv",
        prediction_rows,
        fieldnames=["id", "sequence_hash", "split", "target", "prediction", "uncertainty", sequence_col],
    )

    best_name, best_value = best_simple_baseline(baselines, task_type)
    task_head_metrics = ensemble.get("metrics") or {}
    task_head_metric = task_head_metrics.get("primary_metric")
    warnings = list(audit.warnings) + list(split_diagnostics.get("warnings") or [])
    if manifest_source not in {"config", "ui_declared"}:
        warnings.append("Evaluation manifest was inferred from CLI arguments; provide config.evaluation for auditable provenance.")
    if ensemble.get("status") != "ok":
        warnings.append(f"Task-head training failed or was skipped: {ensemble.get('reason', 'unknown reason')}")
    if best_value is not None and task_head_metric is not None and best_value >= task_head_metric:
        warnings.append(f"Simple baseline {best_name} matched or beat the task head; do not market model lift yet.")
    best_baseline = {"name": best_name, "primary_metric": best_value}

    cross_split_violations = verify_cross_split_violations(
        splits["train"],
        splits["test"],
        sequence_col=sequence_col,
        homology_threshold=float(params["homology_threshold"]),
        homology_k=int(params["homology_k"]),
        group_cols=group_cols,
    )
    model_metric_ci = _bootstrap_metric_ci(
        prediction_rows,
        task_type=task_type,
        target_col="target",
        prediction_col="prediction",
        positive_label=positive_label,
    )
    lift_delta_ci = _lift_ci_from_metric_ci(model_metric_ci, best_value)

    benchmark_claim_gate = _claim_gate(
        task_type=task_type,
        split_diagnostics=split_diagnostics,
        warnings=warnings,
        best_baseline_metric=best_value,
        model_metric=task_head_metric,
        uncertainty_audit=None,
        ranked_candidates=0,
        evaluation_manifest=evaluation_manifest,
        cross_split_violations=cross_split_violations,
        model_metric_ci=model_metric_ci,
        lift_delta_ci=lift_delta_ci,
    )
    benchmark_report = {
        "schema_version": 1,
        "project": project,
        "task_type": task_type,
        "primary_metric_name": task_head_metrics.get("primary_metric_name") or _primary_metric_name(task_type),
        "baselines": baselines,
        "best_simple_baseline": best_baseline,
        "task_head": task_head_metrics,
        "task_head_primary_metric": task_head_metric,
        "evaluation_manifest": evaluation_manifest,
        "claim_gate": benchmark_claim_gate,
        "warnings": warnings,
        "verdict": _verdict(claim_gate=benchmark_claim_gate, task_head_metric=task_head_metric),
    }
    _write_json(artifact_dir / "evaluation_manifest.json", evaluation_manifest)

    candidate_rows: list[dict[str, Any]] = []
    generated_count = 0
    if params.get("candidate_files"):
        candidate_rows, _candidate_audit, candidate_failures = ingest_assay_tables(
            params["candidate_files"],
            project=f"{project}_candidates",
            task_type=task_type,
            sequence_col=sequence_col,
            target_col=None,
            id_col=id_col,
            metadata_cols=sorted(set(metadata_cols + group_cols)),
            group_cols=group_cols,
            column_aliases=params.get("column_aliases") or {},
            require_target=False,
            low_n_threshold=0,
        )
        write_table(
            artifact_dir / "candidate_failure_cases.csv",
            candidate_failures,
            fieldnames=["row_index", "id", "source_file", "sequence", "reason"],
        )
    elif params.get("generate_candidates") and ensemble.get("status") == "ok":
        generated, generation_report = generate_de_novo_candidate_pool(
            splits["train"],
            sequence_col=sequence_col,
            num_proposals=int(params["num_proposals"]),
            sequence_length=params.get("sequence_length") or "infer_from_training",
            gc_range=params.get("gc_range") or "infer_from_training",
            seed=int(params["seed"]),
        )
        candidate_rows = generated
        generated_count = len(generated)
        _write_json(artifact_dir / "generation_report.json", generation_report)
        write_table(artifact_dir / "generated_candidates.csv", generated)
    else:
        warnings.append("No candidate file or generated candidate pool was provided; ranked_candidates.csv is empty.")

    for row in candidate_rows:
        row.setdefault("sequence_hash", stable_sequence_hash(row.get(sequence_col, "")))
    ranked = _rank_candidates(
        candidate_rows,
        ensemble=ensemble,
        train_rows=splits["train"],
        sequence_col=sequence_col,
        target_col=target_col,
        task_type=task_type,
        acquisition_method=params["acquisition_method"],
        beta=float(params["beta"]),
        diversity_method=params["diversity_method"],
        diversity_penalty=float(params["diversity_penalty"]),
        top_k=int(params["top_k"]),
        objective_direction=evaluation_manifest.get("objective_direction", "maximize"),
    )
    ranked_fields = [
        "rank",
        "recommendation",
        "diversity_cluster",
        "sequence_hash",
        sequence_col,
        "prediction",
        "uncertainty",
        "acquisition_score",
        "diversified_acquisition_score",
    ]
    if id_col:
        ranked_fields.insert(3, id_col)

    benchmark_claim_gate = _claim_gate(
        task_type=task_type,
        split_diagnostics=split_diagnostics,
        warnings=warnings,
        best_baseline_metric=best_value,
        model_metric=task_head_metric,
        uncertainty_audit=None,
        ranked_candidates=len(ranked),
        evaluation_manifest=evaluation_manifest,
        cross_split_violations=cross_split_violations,
        model_metric_ci=model_metric_ci,
        lift_delta_ci=lift_delta_ci,
    )
    recommendation_ok = benchmark_claim_gate["recommended"]["ok"]
    for row in ranked:
        row["recommendation"] = recommendation_ok

    write_table(artifact_dir / "ranked_candidates.csv", ranked, fieldnames=ranked_fields)
    benchmark_report["claim_gate"] = benchmark_claim_gate
    benchmark_report["verdict"] = _verdict(claim_gate=benchmark_claim_gate, task_head_metric=task_head_metric)
    _write_json(artifact_dir / "benchmark_report.json", benchmark_report)
    _write_text(artifact_dir / "benchmark_report.md", _benchmark_markdown(benchmark_report))

    _write_candidate_explanations(
        artifact_dir,
        ranked,
        train_rows=splits["train"],
        sequence_col=sequence_col,
        target_col=target_col,
        prediction_col=None,
        id_col=id_col,
        report=benchmark_report,
        warnings=warnings,
    )

    artifacts = [
        "readiness_report.md",
        "data_audit_report.md",
        "data_audit_report.json",
        "benchmark_report.md",
        "benchmark_report.json",
        "evaluation_manifest.json",
        "ranked_candidates.csv",
        "candidate_explanations.csv",
        "candidate_explanations.json",
        "predictions.csv",
        "failure_cases.csv",
        "model_card.md",
        "processed/train.csv",
        "processed/val.csv",
        "processed/test.csv",
        "split_diagnostics.json",
    ]
    if generated_count:
        artifacts.extend(["generated_candidates.csv", "generation_report.json"])
    if params.get("candidate_files"):
        artifacts.append("candidate_failure_cases.csv")
    summary = {
        "schema_version": 1,
        "project": project,
        "artifact_dir": str(artifact_dir),
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in params.items()},
        "evaluation_manifest": evaluation_manifest,
        "audit": audit.to_dict(),
        "split_diagnostics": split_diagnostics,
        "benchmark_report": benchmark_report,
        "ranked_candidates": len(ranked),
        "generated_candidates": generated_count,
        "artifacts": artifacts,
    }
    _write_text(artifact_dir / "readiness_report.md", _readiness_markdown(summary))
    _write_text(artifact_dir / "model_card.md", _model_card_markdown(summary))
    artifacts.append("execution_manifest.json")
    execution = _execution_manifest(
        workflow="internal",
        params=params,
        artifact_dir=artifact_dir,
        artifact_names=[name for name in artifacts if name != "execution_manifest.json"],
        started_at=started_at,
        duration_seconds=time.perf_counter() - started_clock,
    )
    summary["run_id"] = execution["execution_id"]
    summary["execution_manifest"] = execution
    _write_json(artifact_dir / "execution_manifest.json", execution)
    _write_json(artifact_dir / "run_summary.json", summary)
    _record_run_safely(summary, workflow="internal", report_name="readiness_report.md")
    if summary.get("run_store_warning"):
        _write_json(artifact_dir / "run_summary.json", summary)
    return summary


def _filter_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    prediction_col: str,
    uncertainty_col: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        prediction = _safe_float(row.get(prediction_col))
        if prediction is None:
            failures.append(
                {
                    "row_index": idx,
                    "sequence_hash": row.get("sequence_hash", ""),
                    "reason": "missing_or_invalid_prediction",
                }
            )
            continue
        clean = dict(row)
        clean[prediction_col] = prediction
        if uncertainty_col:
            uncertainty = _safe_float(clean.get(uncertainty_col))
            if uncertainty is not None:
                clean[uncertainty_col] = max(0.0, uncertainty)
        valid.append(clean)
    return valid, failures


def _rank_external_predictions(
    rows: list[dict[str, Any]],
    *,
    sequence_col: str,
    prediction_col: str,
    uncertainty_col: str | None,
    beta: float,
    diversity_method: str,
    diversity_penalty: float,
    top_k: int,
    objective_direction: str = "maximize",
) -> list[dict[str, Any]]:
    if not rows:
        return []
    prediction = np.asarray([float(row[prediction_col]) for row in rows], dtype=np.float64)
    if uncertainty_col:
        uncertainty = np.asarray([float(_safe_float(row.get(uncertainty_col)) or 0.0) for row in rows], dtype=np.float64)
    else:
        uncertainty = np.zeros(len(rows), dtype=np.float64)
        
    if objective_direction.lower() == "minimize":
        acquisition_display = prediction - float(beta) * uncertainty
        acquisition_ranking = -acquisition_display
    else:
        acquisition_display = prediction + float(beta) * uncertainty
        acquisition_ranking = acquisition_display

    enriched: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        out = dict(row)
        out["prediction"] = float(prediction[idx])
        out["uncertainty"] = float(uncertainty[idx])
        out["acquisition_score"] = float(acquisition_display[idx])
        enriched.append(out)
        
    ranked = greedy_diverse_rank(
        enriched,
        acquisition_ranking,
        sequence_col=sequence_col,
        method=diversity_method,
        diversity_penalty=diversity_penalty,
        top_k=top_k,
    )
    
    if objective_direction.lower() == "minimize":
        for row in ranked:
            row["diversified_acquisition_score"] = -row["diversified_acquisition_score"]
            
    return ranked


def run_prediction_audit(**params: Any) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started_clock = time.perf_counter()
    _validate_runtime_params(params, includes_training=False)
    project = params["project"]
    task_type = params["task_type"]
    sequence_col = params["sequence_col"]
    target_col = params["target_col"]
    prediction_col = params["prediction_col"]
    uncertainty_col = params.get("uncertainty_col")
    positive_label = params.get("positive_label")
    id_col = params.get("id_col")
    metadata_cols = list(params.get("metadata_cols") or [])
    group_cols = list(params.get("group_cols") or [])
    evaluation_manifest = dict(
        params.get("evaluation_manifest")
        or inferred_manifest_for_args(task_type=task_type, positive_label=positive_label).to_dict()
    )
    manifest_source = str(params.get("manifest_source") or "inferred_from_args")
    artifact_dir = _artifact_dir(project, params.get("output_dir"))
    audit_metadata = sorted(set(metadata_cols + group_cols + [prediction_col] + ([uncertainty_col] if uncertainty_col else [])))

    rows, audit, failures = ingest_assay_tables(
        params["assay_files"],
        project=project,
        task_type=task_type,
        sequence_col=sequence_col,
        target_col=target_col,
        id_col=id_col,
        metadata_cols=audit_metadata,
        group_cols=group_cols,
        column_aliases=params.get("column_aliases") or {},
        require_target=True,
        low_n_threshold=int(params.get("low_n_threshold", 200)),
    )
    prediction_rows, prediction_failures = _filter_prediction_rows(
        rows,
        prediction_col=prediction_col,
        uncertainty_col=uncertainty_col,
    )
    failures.extend(prediction_failures)
    if not prediction_rows:
        raise ValueError("No rows contain valid external model predictions.")
    if task_type == "classification":
        all_labels = [_normalize_binary_label(row[target_col]) for row in prediction_rows]
        _negative_label, positive_label = _resolve_binary_labels(all_labels, positive_label=positive_label)
        evaluation_manifest["positive_label"] = positive_label

    _write_json(artifact_dir / "data_audit_report.json", audit.to_dict())
    _write_text(artifact_dir / "data_audit_report.md", audit.to_markdown())
    write_table(
        artifact_dir / "failure_cases.csv",
        failures,
        fieldnames=["row_index", "id", "source_file", "sequence", "sequence_hash", "reason"],
    )

    splits, split_diagnostics = leakage_safe_split(
        prediction_rows,
        sequence_col=sequence_col,
        val_fraction=float(params["val_fraction"]),
        test_fraction=float(params["test_fraction"]),
        group_cols=group_cols,
        homology_threshold=float(params["homology_threshold"]),
        homology_k=int(params["homology_k"]),
        seed=int(params["seed"]),
    )
    processed_dir = artifact_dir / "processed"
    for split_name in ["train", "val", "test"]:
        write_table(processed_dir / f"{split_name}.csv", splits[split_name])
    _write_json(artifact_dir / "split_diagnostics.json", split_diagnostics)

    baselines = run_baselines(
        splits["train"],
        splits["test"],
        task_type=task_type,
        sequence_col=sequence_col,
        target_col=target_col,
        seed=int(params["seed"]),
        positive_label=positive_label,
    )
    all_rows_metrics = _prediction_metrics(
        prediction_rows,
        task_type=task_type,
        target_col=target_col,
        prediction_col=prediction_col,
        positive_label=positive_label,
    )
    test_metrics = _prediction_metrics(
        splits["test"],
        task_type=task_type,
        target_col=target_col,
        prediction_col=prediction_col,
        positive_label=positive_label,
    )
    uncertainty = _uncertainty_audit(
        splits["test"] or prediction_rows,
        task_type=task_type,
        target_col=target_col,
        prediction_col=prediction_col,
        uncertainty_col=uncertainty_col,
        positive_label=positive_label,
    )
    uncertainty["declared_type"] = evaluation_manifest.get("uncertainty_type", "none")
    uncertainty["interpretation"] = (
        "The scale-gap statistic is calibration-like only when uncertainty is declared as predicted absolute error; "
        "for predictive standard deviation, rank correlation is treated as an uncertainty-usefulness diagnostic."
    )
    ranking = _ranking_audit(
        splits["test"] or prediction_rows,
        task_type=task_type,
        target_col=target_col,
        prediction_col=prediction_col,
        top_k=int(params["top_k"]),
        positive_label=positive_label,
    )
    best_name, best_value = best_simple_baseline(baselines, task_type)
    best_baseline = {"name": best_name, "primary_metric": best_value}
    warnings = list(audit.warnings) + list(split_diagnostics.get("warnings") or [])
    warnings.append("Warning: Splitting already-generated external predictions does not guarantee leakage-safety because model-training provenance is unknown. Retrospective split is for baseline comparison only.")
    if manifest_source not in {"config", "ui_declared"}:
        warnings.append("Evaluation manifest was inferred from CLI arguments; provide config.evaluation for auditable provenance.")
    if uncertainty_col and str(evaluation_manifest.get("uncertainty_type") or "none") == "none":
        warnings.append("An uncertainty column was supplied, but its semantics were not declared; uncertainty-based claims are disabled.")
    if prediction_failures:
        warnings.append(f"{len(prediction_failures)} rows were excluded because model predictions were missing or invalid.")
    test_primary = test_metrics.get("primary_metric")
    if best_value is not None and test_primary is not None and best_value >= test_primary:
        warnings.append(f"Simple baseline {best_name} matched or beat supplied model predictions; do not market model lift yet.")

    cross_split_violations = verify_cross_split_violations(
        splits["train"],
        splits["test"],
        sequence_col=sequence_col,
        homology_threshold=float(params["homology_threshold"]),
        homology_k=int(params["homology_k"]),
        group_cols=group_cols,
    )
    model_metric_ci = _bootstrap_metric_ci(
        splits["test"],
        task_type=task_type,
        target_col=target_col,
        prediction_col=prediction_col,
        positive_label=positive_label,
    )
    lift_delta_ci = _lift_ci_from_metric_ci(model_metric_ci, best_value)

    prediction_claim_gate = _claim_gate(
        task_type=task_type,
        split_diagnostics=split_diagnostics,
        warnings=warnings,
        best_baseline_metric=best_value,
        model_metric=test_primary,
        uncertainty_audit=uncertainty,
        ranked_candidates=0,
        evaluation_manifest=evaluation_manifest,
        cross_split_violations=cross_split_violations,
        model_metric_ci=model_metric_ci,
        lift_delta_ci=lift_delta_ci,
        external_predictions=True,
    )

    report = {
        "schema_version": 1,
        "project": project,
        "task_type": task_type,
        "primary_metric_name": test_metrics.get("primary_metric_name") or _primary_metric_name(task_type),
        "prediction_col": prediction_col,
        "uncertainty_col": uncertainty_col,
        "positive_label": all_rows_metrics.get("positive_label"),
        "evaluation_manifest": evaluation_manifest,
        "all_rows_metrics": all_rows_metrics,
        "test_metrics": test_metrics,
        "baselines": baselines,
        "best_simple_baseline": best_baseline,
        "uncertainty_audit": uncertainty,
        "ranking_audit": ranking,
        "claim_gate": prediction_claim_gate,
        "warnings": warnings,
        "verdict": _prediction_verdict(claim_gate=prediction_claim_gate, test_metric=test_primary),
    }
    _write_json(artifact_dir / "evaluation_manifest.json", evaluation_manifest)

    candidate_rows: list[dict[str, Any]]
    candidate_failures: list[dict[str, Any]] = []
    if params.get("candidate_files"):
        raw_candidates, _candidate_audit, candidate_ingest_failures = ingest_assay_tables(
            params["candidate_files"],
            project=f"{project}_candidates",
            task_type=task_type,
            sequence_col=sequence_col,
            target_col=None,
            id_col=id_col,
            metadata_cols=sorted(set(metadata_cols + group_cols + [prediction_col] + ([uncertainty_col] if uncertainty_col else []))),
            group_cols=group_cols,
            column_aliases=params.get("column_aliases") or {},
            require_target=False,
            low_n_threshold=0,
        )
        candidate_rows, candidate_prediction_failures = _filter_prediction_rows(
            raw_candidates,
            prediction_col=prediction_col,
            uncertainty_col=uncertainty_col,
        )
        candidate_failures = candidate_ingest_failures + candidate_prediction_failures
    else:
        candidate_rows = []
        warnings.append(
            "No prospective candidate pool was provided; measured evaluation rows were not relabeled as candidates."
        )

    ranked = _rank_external_predictions(
        candidate_rows,
        sequence_col=sequence_col,
        prediction_col=prediction_col,
        uncertainty_col=uncertainty_col,
        beta=float(params["beta"]),
        diversity_method=params["diversity_method"],
        diversity_penalty=float(params["diversity_penalty"]),
        top_k=int(params["top_k"]),
        objective_direction=evaluation_manifest.get("objective_direction", "maximize"),
    )
    ranked_fields = [
        "rank",
        "recommendation",
        "diversity_cluster",
        "sequence_hash",
        sequence_col,
        prediction_col,
        "prediction",
        "uncertainty",
        "acquisition_score",
        "diversified_acquisition_score",
    ]
    if uncertainty_col and uncertainty_col not in ranked_fields:
        ranked_fields.insert(7, uncertainty_col)
    if id_col:
        ranked_fields.insert(3, id_col)

    prediction_claim_gate = _claim_gate(
        task_type=task_type,
        split_diagnostics=split_diagnostics,
        warnings=warnings,
        best_baseline_metric=best_value,
        model_metric=test_primary,
        uncertainty_audit=uncertainty,
        ranked_candidates=len(ranked),
        evaluation_manifest=evaluation_manifest,
        cross_split_violations=cross_split_violations,
        model_metric_ci=model_metric_ci,
        lift_delta_ci=lift_delta_ci,
        external_predictions=True,
    )
    recommendation_ok = prediction_claim_gate["recommended"]["ok"]
    for row in ranked:
        row["recommendation"] = recommendation_ok

    write_table(artifact_dir / "ranked_candidates.csv", ranked, fieldnames=ranked_fields)
    report["claim_gate"] = prediction_claim_gate
    report["verdict"] = _prediction_verdict(claim_gate=prediction_claim_gate, test_metric=test_primary)
    _write_json(artifact_dir / "prediction_audit_report.json", report)
    _write_text(
        artifact_dir / "prediction_audit_report.md",
        _prediction_audit_markdown(
            {
                "project": project,
                "audit": audit.to_dict(),
                "valid_prediction_rows": len(prediction_rows),
                "invalid_prediction_rows": len(prediction_failures),
                "split_diagnostics": split_diagnostics,
                "prediction_audit": report,
            }
        ),
    )

    _write_candidate_explanations(
        artifact_dir,
        ranked,
        train_rows=splits["train"],
        sequence_col=sequence_col,
        target_col=target_col,
        prediction_col=prediction_col,
        id_col=id_col,
        report=report,
        warnings=warnings,
    )
    write_table(
        artifact_dir / "candidate_failure_cases.csv",
        candidate_failures,
        fieldnames=["row_index", "id", "source_file", "sequence", "sequence_hash", "reason"],
    )

    artifact_names = [
        "prediction_audit_report.md",
        "prediction_audit_report.json",
        "evaluation_manifest.json",
        "ranked_candidates.csv",
        "candidate_explanations.csv",
        "candidate_explanations.json",
        "data_audit_report.md",
        "data_audit_report.json",
        "failure_cases.csv",
        "candidate_failure_cases.csv",
        "processed/train.csv",
        "processed/val.csv",
        "processed/test.csv",
        "split_diagnostics.json",
        "execution_manifest.json",
    ]
    execution = _execution_manifest(
        workflow="prediction",
        params=params,
        artifact_dir=artifact_dir,
        artifact_names=[name for name in artifact_names if name != "execution_manifest.json"],
        started_at=started_at,
        duration_seconds=time.perf_counter() - started_clock,
    )
    _write_json(artifact_dir / "execution_manifest.json", execution)
    summary = {
        "schema_version": 1,
        "run_id": execution["execution_id"],
        "project": project,
        "artifact_dir": str(artifact_dir),
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in params.items()},
        "evaluation_manifest": evaluation_manifest,
        "execution_manifest": execution,
        "audit": audit.to_dict(),
        "valid_prediction_rows": len(prediction_rows),
        "invalid_prediction_rows": len(prediction_failures),
        "split_diagnostics": split_diagnostics,
        "prediction_audit": report,
        "ranked_candidates": len(ranked),
        "artifacts": artifact_names,
    }
    _write_json(artifact_dir / "prediction_audit_summary.json", summary)
    _record_run_safely(summary, workflow="prediction", report_name="prediction_audit_report.md")
    if summary.get("run_store_warning"):
        _write_json(artifact_dir / "prediction_audit_summary.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assayready",
        description="Local assay audit, clustered retrospective benchmark, and candidate ranking.",
    )
    parser.add_argument("--version", action="version", version=f"AssayReady {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    run = subparsers.add_parser("run", help="Run an exploratory assay-data and local-model audit.")
    run.add_argument("--config", help="JSON or YAML config file.")
    run.add_argument("--assay", nargs="+", help="Input assay CSV/TSV/XLSX files.")
    run.add_argument("--candidates", nargs="*", help="Optional candidate CSV/TSV/XLSX files to rank.")
    run.add_argument("--output-dir", help="Artifact output directory.")
    run.add_argument("--project", default="assayready_project")
    run.add_argument("--task-type", default="regression", choices=["regression", "classification", "ranking"])
    run.add_argument("--sequence-col", default="sequence")
    run.add_argument("--target-col")
    run.add_argument("--positive-label", help="Explicit positive class label for binary classification task heads.")
    run.add_argument("--id-col")
    run.add_argument("--metadata-cols", nargs="*", default=[])
    run.add_argument("--group-cols", nargs="*", default=[])
    run.add_argument("--alias", action="append", help="Column alias mapping: canonical=alias1,alias2")
    run.add_argument("--low-n-threshold", type=int, default=200)
    run.add_argument("--val-fraction", type=float, default=0.15)
    run.add_argument("--test-fraction", type=float, default=0.15)
    run.add_argument("--homology-threshold", type=float, default=0.90)
    run.add_argument("--homology-k", type=int, default=8)
    run.add_argument("--epochs", type=int, default=80)
    run.add_argument("--learning-rate", type=float, default=1e-3)
    run.add_argument("--ensemble-size", type=int, default=5)
    run.add_argument("--seed", type=int, default=13)
    run.add_argument("--acquisition-method", default="upper_confidence_bound")
    run.add_argument("--beta", type=float, default=1.0)
    run.add_argument("--diversity-method", default="greedy_embedding_cosine")
    run.add_argument("--diversity-penalty", type=float, default=0.2)
    run.add_argument("--top-k", type=int, default=96)
    run.add_argument("--generate-candidates", action="store_true")
    run.add_argument("--num-proposals", type=int, default=1000)
    run.add_argument("--sequence-length", default="infer_from_training")
    run.add_argument("--gc-range", nargs=2, type=float)
    run.add_argument("--print-json", action="store_true")

    audit = subparsers.add_parser("audit-predictions", help="Audit externally supplied model predictions.")
    audit.add_argument("--config", help="JSON or YAML config file.")
    audit.add_argument("--assay", nargs="+", help="Input assay CSV/TSV/XLSX files with measured values and predictions.")
    audit.add_argument("--candidates", nargs="*", help="Optional candidate files with model predictions to rank.")
    audit.add_argument("--output-dir", help="Artifact output directory.")
    audit.add_argument("--project", default="prediction_audit")
    audit.add_argument("--task-type", default="regression", choices=["regression", "classification", "ranking"])
    audit.add_argument("--sequence-col", default="sequence")
    audit.add_argument("--target-col")
    audit.add_argument("--prediction-col")
    audit.add_argument("--uncertainty-col")
    audit.add_argument("--positive-label", help="Explicit positive class label for binary classification prediction audits.")
    audit.add_argument("--id-col")
    audit.add_argument("--metadata-cols", nargs="*", default=[])
    audit.add_argument("--group-cols", nargs="*", default=[])
    audit.add_argument("--alias", action="append", help="Column alias mapping: canonical=alias1,alias2")
    audit.add_argument("--low-n-threshold", type=int, default=200)
    audit.add_argument("--val-fraction", type=float, default=0.15)
    audit.add_argument("--test-fraction", type=float, default=0.15)
    audit.add_argument("--homology-threshold", type=float, default=0.90)
    audit.add_argument("--homology-k", type=int, default=8)
    audit.add_argument("--seed", type=int, default=13)
    audit.add_argument("--beta", type=float, default=1.0)
    audit.add_argument("--diversity-method", default="greedy_embedding_cosine")
    audit.add_argument("--diversity-penalty", type=float, default=0.2)
    audit.add_argument("--top-k", type=int, default=96)
    audit.add_argument("--print-json", action="store_true")

    doctor = subparsers.add_parser("doctor", help="Check the local AssayReady environment and bundled public demo.")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable check results.")
    runs = subparsers.add_parser("runs", help="Inspect immutable local run records.")
    runs_subparsers = runs.add_subparsers(dest="runs_command", required=True)
    runs_list = runs_subparsers.add_parser("list", help="List recent runs.")
    runs_list.add_argument("--limit", type=int, default=50)
    runs_list.add_argument("--json", action="store_true")
    runs_show = runs_subparsers.add_parser("show", help="Print a stored run summary.")
    runs_show.add_argument("run_id")
    runs_verify = runs_subparsers.add_parser("verify", help="Verify stored artifact sizes and checksums.")
    runs_verify.add_argument("run_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"run", "audit-predictions", "doctor", "runs", "--version"}:
        argv = ["run", *argv]
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return run_doctor(as_json=bool(args.json))
    if args.command == "runs":
        return run_registry_command(args)
    try:
        if args.command == "audit-predictions":
            if args.config:
                config_path = Path(args.config).resolve()
                params = _prediction_params_from_config(_load_config(config_path), config_dir=config_path.parent)
                params["config_path"] = config_path
            else:
                params = _prediction_params_from_args(args)
            summary = run_prediction_audit(**params)
        else:
            if args.config:
                config_path = Path(args.config).resolve()
                params = _params_from_config(_load_config(config_path), config_dir=config_path.parent)
                params["config_path"] = config_path
            else:
                params = _params_from_args(args)
            summary = run_assayready(**params)
    except Exception as exc:
        print(f"assayready: error: {exc}", file=sys.stderr)
        return 2
    if args.print_json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    else:
        print(f"Wrote AssayReady artifacts to {summary['artifact_dir']}")
        if args.command == "audit-predictions":
            print(f"Verdict: {summary['prediction_audit']['verdict']}")
        else:
            print(f"Verdict: {summary['benchmark_report']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
