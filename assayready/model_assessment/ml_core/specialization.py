from __future__ import annotations

import csv
import hashlib
import math
import random
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from threading import RLock
from typing import Any, Callable

import numpy as np

try:  # Optional at import time so parser-only tests stay light.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - exercised only in stripped environments.
    torch = None
    nn = None
    F = None

VALID_DNA = set("ACGTN")
MISSING_VALUE = "__missing__"
UNKNOWN_VALUE = "__unknown__"
DNA_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def clean_column_name(name: str) -> str:
    """Normalize lab spreadsheet headers into stable snake_case keys."""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


def normalize_sequence(sequence: Any) -> str:
    if sequence is None or not isinstance(sequence, str):
        return ""
    text = str(sequence).strip()
    if not text or text.lower() in {"na", "nan", "none", "null", "n/a"}:
        return ""
    return re.sub(r"\s+", "", text).upper()


def stable_sequence_hash(sequence: str) -> str:
    return hashlib.sha256(normalize_sequence(sequence).encode("utf-8")).hexdigest()


def reverse_complement(sequence: str) -> str:
    return normalize_sequence(sequence).translate(DNA_COMPLEMENT)[::-1]


def canonical_sequence_hash(sequence: str) -> str:
    seq = normalize_sequence(sequence)
    rc = reverse_complement(seq)
    return hashlib.sha256(min(seq, rc).encode("utf-8")).hexdigest()


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.lower() in {"na", "nan", "none", "null", "n/a"}


def _normalize_class_label(value: Any) -> str | None:
    if _is_missing_value(value):
        return None
    try:
        val_float = float(value)
        if val_float.is_integer():
            return str(int(val_float))
        return str(val_float)
    except (TypeError, ValueError):
        return str(value).strip().lower()


def read_table(path: str | Path) -> list[dict[str, Any]]:
    """Read CSV, TSV, or XLSX assay tables without making network calls."""
    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        delimiter = "\t" if suffix in {".tsv", ".txt"} else ","
        with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on local env.
            raise RuntimeError("Reading XLSX assay files requires pandas and openpyxl.") from exc
        frame = pd.read_excel(table_path)
        return frame.fillna("").to_dict(orient="records")
    raise ValueError(f"Unsupported assay table format for {table_path}; expected CSV, TSV, or XLSX.")


def write_table(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_sanitize_csv_row(row) for row in rows)


_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@")
_CSV_CONTROL_PREFIXES = ("\t", "\r", "\n")


def _sanitize_csv_cell(value: Any) -> Any:
    if not isinstance(value, str) or value == "":
        return value
    stripped = value.lstrip()
    if value.startswith(_CSV_CONTROL_PREFIXES) or (stripped and stripped.startswith(_CSV_FORMULA_PREFIXES)):
        return "'" + value
    return value


def _sanitize_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _sanitize_csv_cell(value) for key, value in row.items()}


@dataclass
class AssayAudit:
    project: str
    total_rows: int = 0
    accepted_rows: int = 0
    invalid_dna_rows: int = 0
    missing_sequence_rows: int = 0
    missing_target_rows: int = 0
    duplicate_sequences: int = 0
    conflicting_duplicate_labels: int = 0
    target_summary: dict[str, Any] = field(default_factory=dict)
    metadata_coverage: dict[str, float] = field(default_factory=dict)
    batch_imbalance: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "total_rows": self.total_rows,
            "accepted_rows": self.accepted_rows,
            "invalid_dna_rows": self.invalid_dna_rows,
            "missing_sequence_rows": self.missing_sequence_rows,
            "missing_target_rows": self.missing_target_rows,
            "duplicate_sequences": self.duplicate_sequences,
            "conflicting_duplicate_labels": self.conflicting_duplicate_labels,
            "target_summary": self.target_summary,
            "metadata_coverage": self.metadata_coverage,
            "batch_imbalance": self.batch_imbalance,
            "warnings": self.warnings,
            "errors": self.errors,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Data Audit: {self.project}",
            "",
            "## Row Counts",
            "",
            f"- Total rows: {self.total_rows}",
            f"- Accepted labeled rows: {self.accepted_rows}",
            f"- Missing sequence rows: {self.missing_sequence_rows}",
            f"- Invalid DNA rows: {self.invalid_dna_rows}",
            f"- Missing target rows: {self.missing_target_rows}",
            f"- Duplicate sequences: {self.duplicate_sequences}",
            f"- Conflicting duplicate labels: {self.conflicting_duplicate_labels}",
            "",
            "## Target Summary",
            "",
        ]
        if self.target_summary:
            for key, value in self.target_summary.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("- No target summary available.")
        lines.extend(["", "## Metadata Coverage", ""])
        if self.metadata_coverage:
            for key, value in self.metadata_coverage.items():
                lines.append(f"- {key}: {value:.3f}")
        else:
            lines.append("- No metadata columns configured.")
        lines.extend(["", "## Batch And Group Balance", ""])
        if self.batch_imbalance:
            for key, stats in self.batch_imbalance.items():
                lines.append(
                    f"- {key}: {stats.get('num_groups', 0)} groups, "
                    f"largest_fraction={stats.get('largest_fraction', 0.0):.3f}"
                )
        else:
            lines.append("- No group columns configured.")
        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)
        if self.errors:
            lines.extend(["", "## Errors", ""])
            lines.extend(f"- {error}" for error in self.errors)
        lines.append("")
        return "\n".join(lines)


def _alias_lookup(column_aliases: dict[str, list[str]] | None, canonical_names: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    aliases = column_aliases or {}
    for canonical in canonical_names:
        names = [canonical, clean_column_name(canonical)]
        names.extend(aliases.get(canonical, []))
        names.extend(aliases.get(clean_column_name(canonical), []))
        for name in names:
            lookup[clean_column_name(name)] = canonical
    return lookup


def normalize_row_columns(row: dict[str, Any], alias_lookup: dict[str, str]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        cleaned = clean_column_name(key)
        canonical = alias_lookup.get(cleaned, cleaned)
        if canonical not in normalized or normalized[canonical] in {None, ""}:
            normalized[canonical] = value
    return normalized


def validate_dna(sequence: str) -> bool:
    return bool(sequence) and set(sequence) <= VALID_DNA


def _target_summary(values: list[Any], task_type: str) -> dict[str, Any]:
    if not values:
        return {}
    if task_type == "classification":
        counts = Counter(str(value) for value in values)
        return {"classes": dict(sorted(counts.items())), "num_classes": len(counts)}
    numeric = np.asarray([float(value) for value in values], dtype=np.float64)
    return {
        "count": int(numeric.size),
        "mean": float(np.mean(numeric)),
        "std": float(np.std(numeric)),
        "min": float(np.min(numeric)),
        "median": float(np.median(numeric)),
        "max": float(np.max(numeric)),
    }


def _coverage(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    if not rows:
        return {col: 0.0 for col in columns}
    for col in columns:
        present = sum(1 for row in rows if str(row.get(col, "")).strip())
        out[col] = present / len(rows)
    return out


def _group_balance(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, dict[str, Any]]:
    balance: dict[str, dict[str, Any]] = {}
    for col in columns:
        counts = Counter(str(row.get(col, "") or MISSING_VALUE) for row in rows)
        if not counts:
            continue
        largest = max(counts.values()) / max(1, len(rows))
        balance[col] = {"num_groups": len(counts), "largest_fraction": largest, "counts": dict(counts)}
    return balance


def ingest_assay_tables(
    raw_files: list[str | Path],
    *,
    project: str,
    task_type: str,
    sequence_col: str,
    target_col: str | None,
    id_col: str | None = None,
    metadata_cols: list[str] | None = None,
    group_cols: list[str] | None = None,
    column_aliases: dict[str, list[str]] | None = None,
    require_target: bool = True,
    low_n_threshold: int = 200,
) -> tuple[list[dict[str, Any]], AssayAudit, list[dict[str, Any]]]:
    """Load assay tables, normalize columns, and reject unsafe rows."""
    metadata_cols = list(metadata_cols or [])
    group_cols = list(group_cols or [])
    canonical_names = [sequence_col]
    if target_col:
        canonical_names.append(target_col)
    if id_col:
        canonical_names.append(id_col)
    canonical_names.extend(metadata_cols)
    canonical_names.extend(group_cols)
    alias_lookup = _alias_lookup(column_aliases, canonical_names)

    raw_rows: list[dict[str, Any]] = []
    for raw_file in raw_files:
        for row in read_table(raw_file):
            normalized = normalize_row_columns(row, alias_lookup)
            normalized["_source_file"] = str(raw_file)
            raw_rows.append(normalized)

    audit = AssayAudit(project=project, total_rows=len(raw_rows))
    accepted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen: dict[str, Any] = {}

    for idx, row in enumerate(raw_rows):
        seq = normalize_sequence(row.get(sequence_col))
        target = row.get(target_col) if target_col else None
        row_id = row.get(id_col) if id_col else ""
        failure_base = {
            "row_index": idx,
            "id": row_id,
            "source_file": row.get("_source_file", ""),
            "sequence": seq,
        }
        if not seq:
            audit.missing_sequence_rows += 1
            failures.append({**failure_base, "reason": "missing_sequence"})
            continue
        if not validate_dna(seq):
            audit.invalid_dna_rows += 1
            failures.append({**failure_base, "reason": "invalid_dna"})
            continue
        if require_target:
            if target_col is None or _is_missing_value(target):
                audit.missing_target_rows += 1
                failures.append({**failure_base, "reason": "missing_target"})
                continue
            if task_type in {"regression", "ranking"}:
                numeric_target = parse_float(target)
                if numeric_target is None:
                    audit.missing_target_rows += 1
                    failures.append({**failure_base, "reason": "non_numeric_target"})
                    continue
                target = numeric_target
            elif task_type == "classification":
                class_label = _normalize_class_label(target)
                if class_label is None:
                    audit.missing_target_rows += 1
                    failures.append({**failure_base, "reason": "missing_target"})
                    continue
                target = class_label

        seq_hash = stable_sequence_hash(seq)
        if seq_hash in seen:
            audit.duplicate_sequences += 1
            if require_target and str(seen[seq_hash]) != str(target):
                audit.conflicting_duplicate_labels += 1
        else:
            seen[seq_hash] = target

        clean_row = dict(row)
        clean_row[sequence_col] = seq
        clean_row["sequence_hash"] = seq_hash
        clean_row["rc_stable_hash"] = canonical_sequence_hash(seq)
        clean_row["length"] = len(seq)
        if target_col and require_target:
            clean_row[target_col] = target
        if id_col and not str(clean_row.get(id_col, "")).strip():
            clean_row[id_col] = f"{project}_{len(accepted):06d}"
        accepted.append(clean_row)

    audit.accepted_rows = len(accepted)
    if require_target and target_col:
        audit.target_summary = _target_summary([row[target_col] for row in accepted], task_type)
    audit.metadata_coverage = _coverage(accepted, metadata_cols)
    audit.batch_imbalance = _group_balance(accepted, group_cols)

    if audit.accepted_rows < low_n_threshold:
        audit.warnings.append(
            f"DO NOT TRUST: only {audit.accepted_rows} labeled rows passed validation; "
            f"use this as a data audit or pilot benchmark until validation performance is strong."
        )
    for col, stats in audit.batch_imbalance.items():
        if stats["num_groups"] < 2:
            audit.warnings.append(f"Group column {col!r} has fewer than 2 groups after validation.")
        elif stats["largest_fraction"] >= 0.80:
            audit.warnings.append(
                f"Group column {col!r} is imbalanced; largest group contains "
                f"{stats['largest_fraction']:.1%} of accepted rows."
            )
    return accepted, audit, failures


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def sequence_kmers(sequence: str, k: int = 8) -> set[str]:
    seq = normalize_sequence(sequence)
    if len(seq) < k:
        return {seq} if seq else set()
    kmers: set[str] = set()
    for i in range(0, len(seq) - k + 1):
        kmer = seq[i : i + k]
        if "N" in kmer:
            continue
        # The parent sequence is already normalized, so avoid normalizing every
        # overlapping window again while selecting its canonical orientation.
        kmers.add(min(kmer, kmer.translate(DNA_COMPLEMENT)[::-1]))
    return kmers


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


_SIMILARITY_POLICY_ALIASES = {
    "canonical_kmer_jaccard": "canonical_kmer_jaccard",
    "kmer_jaccard": "canonical_kmer_jaccard",
    "minhash_kmer": "canonical_kmer_jaccard",
    "exact_reverse_complement": "exact_reverse_complement",
    "edit_distance": "edit_distance",
    "position_aware_motif": "position_aware_motif",
    "customer_family_labels": "customer_family_labels",
}
_BUILTIN_SIMILARITY_POLICIES = frozenset(_SIMILARITY_POLICY_ALIASES.values())
_RESERVED_DIVERSITY_POLICY_NAMES = frozenset(
    {
        "greedy_embedding_cosine",
        "greedy_edit_distance",
        "kmer_cosine",
    }
)
_REGISTERED_SIMILARITY_POLICIES: dict[str, Callable[[str, str], float]] = {}
_SIMILARITY_POLICY_LOCK = RLock()
_SIMILARITY_POLICY_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _normalize_similarity_policy_name(name: str) -> str:
    normalized = re.sub(r"[-\s]+", "_", str(name or "").strip().lower())
    if not _SIMILARITY_POLICY_NAME.fullmatch(normalized):
        raise ValueError(
            "Similarity policy names must start with a letter and contain only "
            "lowercase letters, digits, and underscores."
        )
    return normalized


def register_similarity_policy(
    name: str,
    callback: Callable[[str, str], float],
    *,
    replace: bool = False,
) -> str:
    """Register an in-process assay-specific sequence-similarity callback.

    Plugins are Python callables, not import paths. They receive two normalized
    DNA sequences and must return a finite value in ``[0, 1]``. Built-in and
    compatibility-alias names cannot be replaced.
    """
    normalized = _normalize_similarity_policy_name(name)
    if normalized in _SIMILARITY_POLICY_ALIASES or normalized in _RESERVED_DIVERSITY_POLICY_NAMES:
        raise ValueError(f"Similarity policy name {normalized!r} is reserved by AssayReady.")
    if not callable(callback):
        raise TypeError("Similarity policy callback must be callable.")
    with _SIMILARITY_POLICY_LOCK:
        if normalized in _REGISTERED_SIMILARITY_POLICIES and not replace:
            raise ValueError(f"Similarity policy {normalized!r} is already registered.")
        _REGISTERED_SIMILARITY_POLICIES[normalized] = callback
    return normalized


def unregister_similarity_policy(name: str) -> bool:
    """Remove a previously registered plugin policy, returning whether it existed."""
    normalized = _normalize_similarity_policy_name(name)
    if normalized in _SIMILARITY_POLICY_ALIASES:
        raise ValueError(f"Built-in similarity policy {normalized!r} cannot be unregistered.")
    with _SIMILARITY_POLICY_LOCK:
        return _REGISTERED_SIMILARITY_POLICIES.pop(normalized, None) is not None


def registered_similarity_policies() -> tuple[str, ...]:
    """Return registered plugin names in deterministic order."""
    with _SIMILARITY_POLICY_LOCK:
        return tuple(sorted(_REGISTERED_SIMILARITY_POLICIES))


def _resolve_similarity_policy(name: str) -> tuple[str, str]:
    requested = _normalize_similarity_policy_name(name)
    resolved = _SIMILARITY_POLICY_ALIASES.get(requested, requested)
    if resolved in _BUILTIN_SIMILARITY_POLICIES:
        return requested, resolved
    with _SIMILARITY_POLICY_LOCK:
        if resolved in _REGISTERED_SIMILARITY_POLICIES:
            return requested, resolved
        registered = sorted(_REGISTERED_SIMILARITY_POLICIES)
    available = sorted(set(_SIMILARITY_POLICY_ALIASES) | set(registered))
    raise ValueError(
        f"Unknown similarity_policy={name!r}. Available policies: {', '.join(available)}. "
        "Register assay-specific or learned-embedding policies with register_similarity_policy()."
    )


def _position_aware_motif_tokens(sequence: str, *, k: int) -> set[tuple[int, str]]:
    seq = normalize_sequence(sequence)
    if not seq:
        return set()
    if len(seq) < k:
        return {(0, seq)}
    return {
        (position, seq[position : position + k])
        for position in range(0, len(seq) - k + 1)
        if "N" not in seq[position : position + k]
    }


def _validated_similarity(value: Any, *, policy: str) -> float:
    try:
        similarity = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Similarity policy {policy!r} returned a non-numeric value.") from exc
    if not math.isfinite(similarity) or not 0.0 <= similarity <= 1.0:
        raise ValueError(f"Similarity policy {policy!r} must return a finite value in [0, 1]; got {value!r}.")
    return similarity


def sequence_similarity(
    left: str,
    right: str,
    *,
    policy: str | None = None,
    similarity_policy: str | None = None,
    k: int = 8,
    left_row: dict[str, Any] | None = None,
    right_row: dict[str, Any] | None = None,
    group_cols: list[str] | tuple[str, ...] | None = None,
) -> float:
    """Compare two sequences using a named, auditable similarity policy."""
    if int(k) <= 0:
        raise ValueError(f"Similarity k must be positive, got {k}.")
    selected_policy = similarity_policy or policy or "canonical_kmer_jaccard"
    if policy is not None and similarity_policy is not None:
        _policy_requested, policy_resolved = _resolve_similarity_policy(policy)
        _similarity_requested, similarity_resolved = _resolve_similarity_policy(similarity_policy)
        if policy_resolved != similarity_resolved:
            raise ValueError(
                f"Conflicting policy={policy!r} and similarity_policy={similarity_policy!r} arguments."
            )
    _requested, resolved = _resolve_similarity_policy(selected_policy)
    left_norm = normalize_sequence(left)
    right_norm = normalize_sequence(right)

    if resolved == "canonical_kmer_jaccard":
        value = jaccard_similarity(sequence_kmers(left_norm, k=int(k)), sequence_kmers(right_norm, k=int(k)))
    elif resolved == "exact_reverse_complement":
        left_canonical = min(left_norm, reverse_complement(left_norm))
        right_canonical = min(right_norm, reverse_complement(right_norm))
        value = float(left_canonical == right_canonical)
    elif resolved == "edit_distance":
        max_length = max(len(left_norm), len(right_norm), 1)
        value = 1.0 - edit_distance(left_norm, right_norm) / max_length
    elif resolved == "position_aware_motif":
        value = jaccard_similarity(
            _position_aware_motif_tokens(left_norm, k=int(k)),
            _position_aware_motif_tokens(right_norm, k=int(k)),
        )
    elif resolved == "customer_family_labels":
        columns = list(group_cols or [])
        if not columns or left_row is None or right_row is None:
            raise ValueError(
                "customer_family_labels requires left_row, right_row, and at least one group_cols entry."
            )
        value = float(
            any(
                not _is_missing_value(left_row.get(column))
                and not _is_missing_value(right_row.get(column))
                and str(left_row.get(column)).strip() == str(right_row.get(column)).strip()
                for column in columns
            )
        )
    else:
        with _SIMILARITY_POLICY_LOCK:
            callback = _REGISTERED_SIMILARITY_POLICIES[resolved]
        value = callback(left_norm, right_norm)
    return _validated_similarity(value, policy=resolved)


def _union_homologous_pairs(
    rows: list[dict[str, Any]],
    *,
    sequence_col: str,
    policy: str,
    threshold: float,
    k: int,
    union_find: UnionFind,
) -> dict[str, Any]:
    """Deterministically union pairs selected by a generic similarity policy."""
    possible_pairs = len(rows) * (len(rows) - 1) // 2
    candidate_pairs = 0
    homology_edges = 0
    duplicate_rows = 0
    component_skipped_pairs = 0
    for right in range(1, len(rows)):
        for left in range(right):
            if union_find.find(left) == union_find.find(right):
                component_skipped_pairs += 1
                continue
            candidate_pairs += 1
            similarity = sequence_similarity(
                str(rows[left].get(sequence_col, "") or ""),
                str(rows[right].get(sequence_col, "") or ""),
                policy=policy,
                k=k,
            )
            if similarity >= threshold:
                if normalize_sequence(rows[left].get(sequence_col)) == normalize_sequence(rows[right].get(sequence_col)):
                    duplicate_rows += 1
                union_find.union(left, right)
                homology_edges += 1
    return {
        "homology_search_strategy": f"deterministic_pairwise_{policy}",
        "homology_possible_pairs": possible_pairs,
        "homology_prefix_candidates": candidate_pairs,
        "homology_candidate_pairs": candidate_pairs,
        "homology_candidate_fraction": candidate_pairs / possible_pairs if possible_pairs else 0.0,
        "homology_edges": homology_edges,
        "homology_duplicate_rows": duplicate_rows,
        "homology_position_pruned_pairs": 0,
        "homology_component_skipped_pairs": component_skipped_pairs,
    }


def _union_homologous_kmer_sets(
    kmers: list[set[str]],
    *,
    threshold: float,
    union_find: UnionFind,
) -> dict[str, Any]:
    """Union exact Jaccard matches using a no-false-negative PPJoin index."""
    possible_pairs = len(kmers) * (len(kmers) - 1) // 2
    token_frequency = Counter(token for tokens in kmers for token in tokens)
    ordered_vocabulary = sorted(token_frequency, key=lambda token: (token_frequency[token], token))
    token_ids = {token: index for index, token in enumerate(ordered_vocabulary)}
    encoded_kmers = [frozenset(token_ids[token] for token in tokens) for tokens in kmers]
    ordered_kmers = [tuple(sorted(tokens)) for tokens in encoded_kmers]

    # Posting tuples stay ordered because rows are processed by nondecreasing
    # set size and then original row index. This lets bisect discard candidates
    # that cannot meet the Jaccard length bound before materializing them.
    prefix_index: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    representative_by_tokens: dict[frozenset[int], int] = {}
    prefix_candidates = 0
    candidate_pairs = 0
    homology_edges = 0
    duplicate_rows = 0
    position_pruned_pairs = 0
    component_skipped_pairs = 0

    processing_order = sorted(range(len(encoded_kmers)), key=lambda index: (len(encoded_kmers[index]), index))
    for right in processing_order:
        right_tokens = encoded_kmers[right]
        token_signature = right_tokens
        if token_signature in representative_by_tokens:
            # Identical sets always have Jaccard similarity 1.0. One indexed
            # representative carries all of their edges to other components.
            union_find.union(representative_by_tokens[token_signature], right)
            homology_edges += 1
            duplicate_rows += 1
            continue
        representative_by_tokens[token_signature] = right

        if not right_tokens:
            # An empty set can only match another empty set, handled by the
            # identical-set representative above.
            continue

        # For Jaccard(A, B) >= threshold, each set must overlap the other in
        # at least ceil(threshold * |set|) tokens. With a shared global token
        # order, prefixes of this length are therefore guaranteed to intersect.
        right_size = len(right_tokens)
        prefix_length = right_size - math.ceil(threshold * right_size) + 1
        prefix = ordered_kmers[right][:prefix_length]
        minimum_left_size = math.ceil(threshold * right_size)
        candidate_overlap: dict[int, int] = {}
        pruned_candidates: set[int] = set()

        for right_position, token in enumerate(prefix):
            postings = prefix_index[token]
            posting_start = bisect_left(postings, (minimum_left_size, -1, -1))
            for left_size, left, left_position in postings[posting_start:]:
                if left in pruned_candidates:
                    continue
                if left not in candidate_overlap:
                    prefix_candidates += 1
                overlap = candidate_overlap.get(left, 0) + 1
                required_overlap = math.ceil(threshold * (left_size + right_size) / (1.0 + threshold))
                overlap_upper_bound = overlap + min(
                    left_size - left_position - 1,
                    right_size - right_position - 1,
                )
                if overlap_upper_bound < required_overlap:
                    candidate_overlap.pop(left, None)
                    pruned_candidates.add(left)
                    position_pruned_pairs += 1
                else:
                    candidate_overlap[left] = overlap

        for left in candidate_overlap:
            if union_find.find(left) == union_find.find(right):
                component_skipped_pairs += 1
                continue
            left_size = len(encoded_kmers[left])
            required_overlap = math.ceil(threshold * (left_size + right_size) / (1.0 + threshold))
            candidate_pairs += 1
            if len(encoded_kmers[left] & right_tokens) >= required_overlap:
                union_find.union(left, right)
                homology_edges += 1

        for position, token in enumerate(prefix):
            prefix_index[token].append((right_size, right, position))

    return {
        "homology_search_strategy": "exact_jaccard_ppjoin",
        "homology_possible_pairs": possible_pairs,
        "homology_prefix_candidates": prefix_candidates,
        "homology_candidate_pairs": candidate_pairs,
        "homology_candidate_fraction": candidate_pairs / possible_pairs if possible_pairs else 0.0,
        "homology_edges": homology_edges,
        "homology_duplicate_rows": duplicate_rows,
        "homology_position_pruned_pairs": position_pruned_pairs,
        "homology_component_skipped_pairs": component_skipped_pairs,
    }


def leakage_safe_split(
    rows: list[dict[str, Any]],
    *,
    sequence_col: str,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    group_cols: list[str] | None = None,
    homology_threshold: float = 0.90,
    homology_k: int = 8,
    similarity_policy: str = "canonical_kmer_jaccard",
    seed: int = 13,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Split by declared groups and an explicitly named similarity policy."""
    group_cols = list(group_cols or [])
    requested_similarity_policy, resolved_similarity_policy = _resolve_similarity_policy(similarity_policy)
    if not 0.0 <= float(val_fraction) < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if not 0.0 <= float(test_fraction) < 1.0:
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")
    if float(val_fraction) + float(test_fraction) >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1")
    if not 0.0 <= float(homology_threshold) <= 1.0:
        raise ValueError(f"homology_threshold must be in [0, 1], got {homology_threshold}")
    if int(homology_k) <= 0:
        raise ValueError(f"homology_k must be positive, got {homology_k}")
    if resolved_similarity_policy != "customer_family_labels" and any(sequence_col not in row for row in rows):
        raise ValueError(f"Every row must contain sequence column {sequence_col!r}.")
    absent_group_cols = [col for col in group_cols if not any(col in row for row in rows)]
    if absent_group_cols:
        raise ValueError(f"Configured group columns are absent from all rows: {absent_group_cols}")
    if resolved_similarity_policy == "customer_family_labels" and not group_cols:
        raise ValueError("customer_family_labels similarity policy requires at least one group_cols entry.")

    n_rows = len(rows)
    uf = UnionFind(n_rows)
    group_to_first: dict[tuple[str, str], int] = {}

    for idx, row in enumerate(rows):
        # Each configured column represents an independent leakage boundary.
        # Missing values do not prove that two rows belong to the same group.
        for col in group_cols:
            value = row.get(col)
            if _is_missing_value(value):
                continue
            group_key = (col, str(value).strip())
            if group_key in group_to_first:
                uf.union(group_to_first[group_key], idx)
            else:
                group_to_first[group_key] = idx

    homology_search = {
        "homology_search_strategy": "disabled",
        "homology_possible_pairs": n_rows * (n_rows - 1) // 2,
        "homology_prefix_candidates": 0,
        "homology_candidate_pairs": 0,
        "homology_candidate_fraction": 0.0,
        "homology_edges": 0,
        "homology_duplicate_rows": 0,
        "homology_position_pruned_pairs": 0,
        "homology_component_skipped_pairs": 0,
    }
    if resolved_similarity_policy == "customer_family_labels":
        homology_search["homology_search_strategy"] = "customer_family_labels_only"
    elif homology_threshold > 0 and n_rows > 1 and resolved_similarity_policy == "canonical_kmer_jaccard":
        kmers = [sequence_kmers(row.get(sequence_col, ""), k=homology_k) for row in rows]
        homology_search = _union_homologous_kmer_sets(
            kmers,
            threshold=homology_threshold,
            union_find=uf,
        )
    elif homology_threshold > 0 and n_rows > 1:
        homology_search = _union_homologous_pairs(
            rows,
            sequence_col=sequence_col,
            policy=resolved_similarity_policy,
            threshold=homology_threshold,
            k=homology_k,
            union_find=uf,
        )

    clusters_by_root: dict[int, list[int]] = defaultdict(list)
    for idx in range(n_rows):
        clusters_by_root[uf.find(idx)].append(idx)
    clusters = list(clusters_by_root.values())

    def cluster_fingerprint(cluster: list[int]) -> str:
        members = []
        for idx in cluster:
            row = rows[idx]
            group_values = [
                f"{col}={str(row.get(col)).strip()}"
                for col in group_cols
                if not _is_missing_value(row.get(col))
            ]
            members.append("|".join([stable_sequence_hash(row.get(sequence_col, "")), *group_values]))
        return hashlib.sha256("\n".join(sorted(members)).encode("utf-8")).hexdigest()

    cluster_ids = {id(cluster): cluster_fingerprint(cluster) for cluster in clusters}
    clusters.sort(
        key=lambda cluster: (
            -len(cluster),
            hashlib.sha256(f"{int(seed)}:{cluster_ids[id(cluster)]}".encode("utf-8")).hexdigest(),
        )
    )

    diagnostics: dict[str, Any] = {
        "num_rows": n_rows,
        "num_clusters": len(clusters),
        "group_cols": group_cols,
        "group_semantics": "shared non-missing value in any configured group column",
        "similarity_policy": resolved_similarity_policy,
        "similarity_policy_requested": requested_similarity_policy,
        "similarity_threshold": homology_threshold,
        "homology_threshold": homology_threshold,
        "homology_k": homology_k,
        "warnings": [],
        **homology_search,
    }
    assignments: dict[int, str] = {}
    if not clusters:
        return {"train": [], "val": [], "test": []}, diagnostics
    if len(clusters) < 3:
        diagnostics["warnings"].append(
            "DO NOT TRUST: fewer than 3 leakage-safe clusters are available; validation/test splits may be empty."
        )

    desired_test = int(round(n_rows * test_fraction))
    desired_val = int(round(n_rows * val_fraction))
    if len(clusters) >= 3 and test_fraction > 0:
        desired_test = max(1, desired_test)
    if len(clusters) >= 3 and val_fraction > 0:
        desired_val = max(1, desired_val)

    split_sizes = {"train": 0, "val": 0, "test": 0}
    unassigned = list(clusters)

    def assign_next(split: str, desired: int) -> None:
        nonlocal unassigned
        while desired > 0 and unassigned and len(unassigned) > 1 and split_sizes[split] < desired:
            cluster = unassigned.pop()
            for idx in cluster:
                assignments[idx] = split
            split_sizes[split] += len(cluster)

    assign_next("test", desired_test)
    assign_next("val", desired_val)
    for cluster in unassigned:
        for idx in cluster:
            assignments[idx] = "train"
        split_sizes["train"] += len(cluster)

    splits = {"train": [], "val": [], "test": []}
    for idx, row in enumerate(rows):
        split = assignments.get(idx, "train")
        with_split = dict(row)
        with_split["split"] = split
        cluster = clusters_by_root[uf.find(idx)]
        with_split["leakage_cluster"] = cluster_ids[id(cluster)][:16]
        splits[split].append(with_split)
    diagnostics["split_sizes"] = {key: len(value) for key, value in splits.items()}
    diagnostics["split_cluster_counts"] = {
        key: len(
            {
                str(row.get("leakage_cluster"))
                for row in split_rows
                if str(row.get("leakage_cluster") or "").strip()
            }
        )
        for key, split_rows in splits.items()
    }
    if not splits["val"] or not splits["test"]:
        required_empty = []
        if val_fraction > 0 and not splits["val"]:
            required_empty.append("validation")
        if test_fraction > 0 and not splits["test"]:
            required_empty.append("test")
        if required_empty:
            diagnostics["warnings"].append(
                "DO NOT TRUST: requested leakage-safe " + " and ".join(required_empty)
                + " split is empty; collect more independent groups."
            )
    return splits, diagnostics


def all_kmers(k: int) -> list[str]:
    return ["".join(chars) for chars in product("ACGT", repeat=k)]


def gc_fraction(sequence: str) -> float:
    seq = normalize_sequence(sequence)
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def gc_length_features(sequences: list[str]) -> np.ndarray:
    if not sequences:
        return np.empty((0, 2), dtype=np.float32)
    lengths = np.asarray([len(normalize_sequence(seq)) for seq in sequences], dtype=np.float32)
    log_lengths = np.log1p(lengths)
    gc = np.asarray([gc_fraction(seq) for seq in sequences], dtype=np.float32)
    return np.column_stack([log_lengths, gc]).astype(np.float32)


def kmer_feature_matrix(sequences: list[str], *, k: int = 3, include_gc_length: bool = True) -> np.ndarray:
    vocab = all_kmers(k)
    index = {kmer: idx for idx, kmer in enumerate(vocab)}
    features = np.zeros((len(sequences), len(vocab)), dtype=np.float32)
    for row_idx, sequence in enumerate(sequences):
        seq = normalize_sequence(sequence)
        total = 0
        for pos in range(0, max(0, len(seq) - k + 1)):
            kmer = seq[pos : pos + k]
            idx = index.get(kmer)
            if idx is None:
                continue
            features[row_idx, idx] += 1.0
            total += 1
        if total:
            features[row_idx] /= float(total)
    if include_gc_length:
        return np.concatenate([gc_length_features(sequences), features], axis=1)
    return features


def row_sequence_feature_matrix(
    rows: list[dict[str, Any]],
    *,
    sequence_col: str,
    sequence_feature_key: str | None = None,
) -> np.ndarray:
    """Return supervised sequence features from precomputed embeddings or k-mers."""
    if not sequence_feature_key:
        return kmer_feature_matrix([row[sequence_col] for row in rows], k=3)
    vectors: list[np.ndarray] = []
    missing = 0
    expected_dim: int | None = None
    for row in rows:
        value = row.get(sequence_feature_key)
        if value is None:
            missing += 1
            continue
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if expected_dim is None:
            expected_dim = int(vector.size)
        elif int(vector.size) != expected_dim:
            raise ValueError(
                f"Precomputed sequence feature {sequence_feature_key!r} has inconsistent dimensions: "
                f"expected {expected_dim}, got {vector.size}."
            )
        vectors.append(vector)
    if missing:
        raise ValueError(f"{missing} rows are missing precomputed sequence feature {sequence_feature_key!r}.")
    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(vectors).astype(np.float32)


def edit_distance(left: str, right: str) -> int:
    left_norm = normalize_sequence(left)
    right_norm = normalize_sequence(right)
    try:
        from rapidfuzz.distance import Levenshtein

        return int(Levenshtein.distance(left_norm, right_norm))
    except Exception:
        return sum(a != b for a, b in zip(left_norm, right_norm)) + abs(len(left_norm) - len(right_norm))


def min_edit_distance(sequence: str, references: list[str]) -> int:
    if not references:
        return len(normalize_sequence(sequence))
    return min(edit_distance(sequence, reference) for reference in references)


def max_homopolymer_run(sequence: str) -> int:
    seq = normalize_sequence(sequence)
    if not seq:
        return 0
    longest = 1
    current = 1
    for idx in range(1, len(seq)):
        if seq[idx] == seq[idx - 1]:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    return max(longest, current)


def infer_sequence_length(sequences: list[str], configured: Any = "infer_from_training") -> int:
    if configured is not None and configured != "" and configured != "infer_from_training":
        length = int(configured)
        if length <= 0:
            raise ValueError(f"generation.sequence_length must be positive, got {configured!r}.")
        return length
    lengths = [len(normalize_sequence(seq)) for seq in sequences if normalize_sequence(seq)]
    if not lengths:
        raise ValueError("Cannot infer generation.sequence_length from an empty training set.")
    return max(1, int(round(float(np.median(lengths)))))


def infer_gc_range(sequences: list[str], configured: Any = "infer_from_training") -> tuple[float, float]:
    if configured is not None and configured != "" and configured != "infer_from_training":
        if not isinstance(configured, (list, tuple)) or len(configured) != 2:
            raise ValueError("generation.gc_range must be infer_from_training or a [min, max] pair.")
        low = float(configured[0])
        high = float(configured[1])
    else:
        values = np.asarray([gc_fraction(seq) for seq in sequences if normalize_sequence(seq)], dtype=np.float64)
        if values.size == 0:
            low, high = 0.30, 0.70
        else:
            center = float(np.mean(values))
            spread = max(float(np.std(values)) * 2.0, 0.08)
            low = center - spread
            high = center + spread
    low = max(0.0, min(1.0, low))
    high = max(0.0, min(1.0, high))
    if low > high:
        low, high = high, low
    return low, high


def _random_dna_sequence(length: int, gc_target: float, rng: random.Random) -> str:
    gc_target = max(0.0, min(1.0, gc_target))
    seq: list[str] = []
    for _ in range(length):
        if rng.random() < gc_target:
            seq.append("G" if rng.random() < 0.5 else "C")
        else:
            seq.append("A" if rng.random() < 0.5 else "T")
    return "".join(seq)


def _mutate_sequence(sequence: str, rng: random.Random, *, mutation_rate: float = 0.08) -> str:
    bases = "ACGT"
    out: list[str] = []
    for base in normalize_sequence(sequence):
        if rng.random() < mutation_rate:
            choices = [candidate for candidate in bases if candidate != base]
            out.append(rng.choice(choices))
        else:
            out.append(base if base in bases else rng.choice(bases))
    return "".join(out)


def _recombine_sequences(left: str, right: str, rng: random.Random) -> str:
    left_norm = normalize_sequence(left)
    right_norm = normalize_sequence(right)
    length = min(len(left_norm), len(right_norm))
    if length <= 1:
        return left_norm or right_norm
    cut = rng.randint(1, length - 1)
    return left_norm[:cut] + right_norm[cut:length]


def synthesis_filter_reason(
    sequence: str,
    *,
    references: list[str],
    gc_range: tuple[float, float],
    max_homopolymer: int,
    forbidden_motifs: list[str] | None = None,
    novelty_min_edit_distance: int = 0,
) -> tuple[str | None, int, float]:
    seq = normalize_sequence(sequence)
    gc = gc_fraction(seq)
    novelty = min_edit_distance(seq, references)
    forbidden = [normalize_sequence(motif) for motif in (forbidden_motifs or []) if normalize_sequence(motif)]
    if not validate_dna(seq) or "N" in seq:
        return "invalid_dna", novelty, gc
    if gc < gc_range[0] or gc > gc_range[1]:
        return "gc_out_of_range", novelty, gc
    if max_homopolymer > 0 and max_homopolymer_run(seq) > max_homopolymer:
        return "homopolymer_too_long", novelty, gc
    if any(motif in seq for motif in forbidden):
        return "forbidden_motif", novelty, gc
    if novelty < novelty_min_edit_distance:
        return "too_close_to_training", novelty, gc
    return None, novelty, gc


def generate_de_novo_candidate_pool(
    train_rows: list[dict[str, Any]],
    *,
    sequence_col: str,
    num_proposals: int = 50000,
    sequence_length: Any = "infer_from_training",
    gc_range: Any = "infer_from_training",
    max_homopolymer: int = 6,
    forbidden_motifs: list[str] | None = None,
    novelty_min_edit_distance: int = 8,
    method: str = "evolutionary_mcmc",
    seed: int = 13,
    batch_size: int = 512,
    parent_pool_size: int = 256,
    mutation_rate: float = 0.08,
    score_callback: Callable[[list[dict[str, Any]]], np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate synthesis-filtered de novo DNA candidates with evolutionary search."""
    if method != "evolutionary_mcmc":
        raise ValueError("generation.method currently supports only evolutionary_mcmc.")
    if num_proposals <= 0:
        return [], {"enabled": True, "accepted": 0, "rejected": 0, "warnings": ["generation.num_proposals <= 0"]}

    references = [normalize_sequence(row[sequence_col]) for row in train_rows if normalize_sequence(row.get(sequence_col))]
    length = infer_sequence_length(references, sequence_length)
    inferred_gc_range = infer_gc_range(references, gc_range)
    rng = random.Random(seed)
    seen = set(references)
    accepted: list[dict[str, Any]] = []
    parent_pool: list[tuple[float, str]] = []
    rejected = Counter()
    attempts = 0
    max_attempts = max(num_proposals * 50, 2000)
    batch_size = max(1, int(batch_size))
    parent_pool_size = max(2, int(parent_pool_size))

    def make_candidate() -> str:
        if len(parent_pool) >= 2 and rng.random() < 0.70:
            left = rng.choice(parent_pool)[1]
            right = rng.choice(parent_pool)[1]
            return _mutate_sequence(_recombine_sequences(left, right, rng), rng, mutation_rate=mutation_rate)
        if parent_pool and rng.random() < 0.75:
            return _mutate_sequence(rng.choice(parent_pool)[1], rng, mutation_rate=mutation_rate)
        gc_target = rng.uniform(inferred_gc_range[0], inferred_gc_range[1])
        return _random_dna_sequence(length, gc_target, rng)

    while len(accepted) < num_proposals and attempts < max_attempts:
        batch: list[dict[str, Any]] = []
        while len(batch) < batch_size and len(accepted) + len(batch) < num_proposals and attempts < max_attempts:
            attempts += 1
            seq = make_candidate()
            if len(seq) != length:
                seq = (seq + _random_dna_sequence(length, sum(inferred_gc_range) / 2.0, rng))[:length]
            if seq in seen:
                rejected["duplicate"] += 1
                continue
            reason, novelty, gc = synthesis_filter_reason(
                seq,
                references=references,
                gc_range=inferred_gc_range,
                max_homopolymer=max_homopolymer,
                forbidden_motifs=forbidden_motifs,
                novelty_min_edit_distance=novelty_min_edit_distance,
            )
            if reason:
                rejected[reason] += 1
                continue
            seen.add(seq)
            batch.append(
                {
                    "design_id": f"de_novo_{len(accepted) + len(batch) + 1:06d}",
                    sequence_col: seq,
                    "sequence_hash": stable_sequence_hash(seq),
                    "generation_method": method,
                    "gc_fraction": float(gc),
                    "novelty_edit_distance": int(novelty),
                    "novelty_score": float(novelty / max(1, length)),
                    "synthesis_status": "passed_filters",
                    "prioritization_note": "in_silico_design_requires_scientist_review_and_wet_lab_validation",
                }
            )
        if not batch:
            continue
        if score_callback is None:
            scores = np.zeros(len(batch), dtype=np.float64)
        else:
            scores = np.asarray(score_callback(batch), dtype=np.float64)
        if scores.size != len(batch):
            raise ValueError(
                f"Generation score callback returned {scores.size} scores for {len(batch)} candidate rows."
            )
        for row, score in zip(batch, scores):
            row["_generation_score"] = float(score)
        parent_pool.extend((float(row["_generation_score"]), row[sequence_col]) for row in batch)
        parent_pool.sort(key=lambda item: item[0], reverse=True)
        del parent_pool[parent_pool_size:]
        accepted.extend(batch)

    report = {
        "enabled": True,
        "mode": "de_novo",
        "method": method,
        "sequence_length": length,
        "gc_range": list(inferred_gc_range),
        "num_proposals_requested": int(num_proposals),
        "accepted": len(accepted),
        "attempts": attempts,
        "rejected": int(sum(rejected.values())),
        "rejection_reasons": dict(sorted(rejected.items())),
        "novelty_min_edit_distance": int(novelty_min_edit_distance),
        "max_homopolymer": int(max_homopolymer),
        "forbidden_motifs": forbidden_motifs or [],
        "warnings": [],
    }
    if len(accepted) < num_proposals:
        report["warnings"].append(
            f"Generated {len(accepted)} candidates after {attempts} attempts; filters may be too strict."
        )
    return accepted, report


def encode_class_labels(
    values: list[Any],
    mapping: dict[str, int] | None = None,
    *,
    positive_label: Any | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    normalized = [_normalize_class_label(value) for value in values]
    if any(label is None for label in normalized):
        raise ValueError("Classification targets contain missing labels.")
    labels = [str(label) for label in normalized]

    if mapping is None:
        classes = sorted(set(labels))
        if positive_label is not None:
            positive = _normalize_class_label(positive_label)
            if positive not in classes:
                raise ValueError(
                    f"Configured positive_label={positive_label!r} is not present in classification labels {classes}."
                )
            if len(classes) != 2:
                raise ValueError("positive_label requires exactly two classification labels.")
            negative = next(label for label in classes if label != positive)
            mapping = {negative: 0, str(positive): 1}
        else:
            preferred = {
                "low": 0, "inactive": 0, "false": 0, "no": 0, "0": 0,
                "high": 1, "active": 1, "true": 1, "yes": 1, "1": 1,
            }
            if set(classes) <= set(preferred):
                mapping = {
                    label: preferred[label]
                    for label in sorted(classes, key=lambda item: (preferred[item], item))
                }
            else:
                mapping = {label: idx for idx, label in enumerate(classes)}
    else:
        mapping = {
            str(_normalize_class_label(label)): int(encoded)
            for label, encoded in mapping.items()
        }
        if positive_label is not None:
            positive = _normalize_class_label(positive_label)
            if positive not in mapping or mapping[str(positive)] != 1:
                raise ValueError("positive_label must identify the label encoded as class 1.")

    unknown = sorted(set(labels) - set(mapping))
    if unknown:
        raise ValueError(f"Classification labels are not present in the fitted mapping: {unknown}")
    encoded = np.asarray([mapping[label] for label in labels])
    return encoded.astype(np.int64), mapping


def regression_metrics(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.size == 0:
        return {"rmse": None, "mae": None, "r2": None, "pearson": None, "num_test": 0}
    residual = pred - true
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = None if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
    pearson = None
    if true.size > 1 and float(np.std(true)) > 0 and float(np.std(pred)) > 0:
        pearson = float(np.corrcoef(true, pred)[0, 1])
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": pearson, "num_test": int(true.size)}


def _binary_auroc(true: np.ndarray, score: np.ndarray) -> float | None:
    """Compute binary AUROC from average ranks, including tied scores."""
    positive = true == 1
    num_positive = int(np.sum(positive))
    num_negative = int(true.size - num_positive)
    if num_positive == 0 or num_negative == 0:
        return None

    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=np.float64)
    start = 0
    while start < score.size:
        end = start + 1
        while end < score.size and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end

    rank_sum = float(np.sum(ranks[positive]))
    u_statistic = rank_sum - num_positive * (num_positive + 1) / 2.0
    return u_statistic / (num_positive * num_negative)


def classification_metrics(y_true: list[int] | np.ndarray, y_score: list[float] | np.ndarray) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(y_score, dtype=np.float64)
    if true.size == 0:
        return {"accuracy": None, "f1": None, "auroc": None, "num_test": 0}
    pred = (score >= 0.5).astype(np.int64)
    accuracy = float(np.mean(pred == true))
    tp = float(np.sum((pred == 1) & (true == 1)))
    fp = float(np.sum((pred == 1) & (true == 0)))
    fn = float(np.sum((pred == 0) & (true == 1)))
    tn = float(np.sum((pred == 0) & (true == 0)))
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    class_recalls = []
    if tp + fn > 0:
        class_recalls.append(recall)
    if tn + fp > 0:
        class_recalls.append(tn / (tn + fp))
    balanced_accuracy = float(np.mean(class_recalls)) if class_recalls else None
    auroc = _binary_auroc(true, score)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": float(f1),
        "auroc": auroc,
        "num_test": int(true.size),
    }


def ranking_metrics(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> dict[str, Any]:
    out = regression_metrics(y_true, y_pred)
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    total = 0
    correct = 0
    for i in range(len(true)):
        for j in range(i + 1, len(true)):
            diff_true = true[i] - true[j]
            if diff_true == 0:
                continue
            total += 1
            diff_pred = pred[i] - pred[j]
            if diff_true * diff_pred > 0:
                correct += 1
    out["pairwise_accuracy"] = None if total == 0 else correct / total
    spearman = None
    if len(true) > 1:
        def average_ranks(values: np.ndarray) -> np.ndarray:
            order = np.argsort(values, kind="mergesort")
            ranks = np.empty(len(values), dtype=np.float64)
            start = 0
            while start < len(values):
                end = start + 1
                while end < len(values) and values[order[end]] == values[order[start]]:
                    end += 1
                ranks[order[start:end]] = (start + end - 1) / 2.0
                start = end
            return ranks

        true_ranks = average_ranks(true)
        pred_ranks = average_ranks(pred)
        if float(np.std(true_ranks)) > 0 and float(np.std(pred_ranks)) > 0:
            spearman = float(np.corrcoef(true_ranks, pred_ranks)[0, 1])
    out["spearman"] = spearman
    return out


def _fit_linear_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x_train.shape[0], 1), dtype=x_train.dtype), x_train], axis=1)
    coef = np.linalg.pinv(x_aug.T @ x_aug + np.eye(x_aug.shape[1], dtype=x_train.dtype) * 1e-3) @ x_aug.T @ y_train
    x_test_aug = np.concatenate([np.ones((x_test.shape[0], 1), dtype=x_test.dtype), x_test], axis=1)
    return x_test_aug @ coef


def _fit_logistic_fallback(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    if len(set(y_train.tolist())) < 2:
        return np.full(x_test.shape[0], float(y_train[0]) if y_train.size else 0.0, dtype=np.float64)
    pred = _fit_linear_regression(x_train, y_train.astype(np.float64), x_test)
    return 1.0 / (1.0 + np.exp(-pred))


def _primary_metric(task_type: str, metrics: dict[str, Any]) -> float | None:
    key = _primary_metric_name(task_type, metrics)
    if key is None:
        return None
    value = metrics.get(key)
    return None if value is None else float(value)


def _primary_metric_name(task_type: str, metrics: dict[str, Any]) -> str | None:
    if task_type == "classification":
        return "auroc" if metrics.get("auroc") is not None else "balanced_accuracy"
    if task_type == "ranking":
        return "spearman" if metrics.get("spearman") is not None else "pairwise_accuracy"
    if task_type == "regression":
        return "r2"
    return None


def run_baselines(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    task_type: str,
    sequence_col: str,
    target_col: str,
    seed: int = 13,
    positive_label: Any | None = None,
    include_predictions: bool = False,
) -> dict[str, dict[str, Any]]:
    """Fit simple comparison baselines on the caller-provided train/test split."""
    baselines: dict[str, dict[str, Any]] = {}
    if not train_rows or not test_rows:
        warning = {"status": "skipped", "reason": "empty train or test split"}
        for name in [
            "gc_length_baseline",
            "kmer_ridge_or_logistic",
            "kmer_random_forest",
            "kmer4_linear_probe",
        ]:
            baselines[name] = dict(warning)
        return baselines

    train_seq = [row[sequence_col] for row in train_rows]
    test_seq = [row[sequence_col] for row in test_rows]
    x_train_gc = gc_length_features(train_seq)
    x_test_gc = gc_length_features(test_seq)
    x_train_kmer = kmer_feature_matrix(train_seq, k=3)
    x_test_kmer = kmer_feature_matrix(test_seq, k=3)
    x_train_embed = kmer_feature_matrix(train_seq, k=4)
    x_test_embed = kmer_feature_matrix(test_seq, k=4)

    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        sklearn_ok = True
    except Exception:
        sklearn_ok = False

    def store_baseline(
        name: str,
        predictions: np.ndarray,
        metric_fn: Callable[[Any, Any], dict[str, Any]],
        y_test: np.ndarray,
        *,
        status: str | None = None,
    ) -> None:
        aligned_predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
        if len(aligned_predictions) != len(test_rows):
            raise ValueError(
                f"Baseline {name!r} produced {len(aligned_predictions)} predictions for {len(test_rows)} test rows."
            )
        metrics = metric_fn(y_test, aligned_predictions)
        if status is not None:
            metrics["status"] = status
        if include_predictions:
            metrics["test_predictions"] = [float(value) for value in aligned_predictions]
        baselines[name] = metrics

    if task_type == "classification":
        y_train, mapping = encode_class_labels(
            [row[target_col] for row in train_rows],
            positive_label=positive_label,
        )
        y_test, _ = encode_class_labels([row[target_col] for row in test_rows], mapping=mapping)
        if len(mapping) > 2:
            raise ValueError("Classification task heads currently support binary high/low labels.")
        if sklearn_ok and len(set(y_train.tolist())) >= 2:
            gc_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
            gc_model.fit(x_train_gc, y_train)
            store_baseline(
                "gc_length_baseline",
                gc_model.predict_proba(x_test_gc)[:, 1],
                classification_metrics,
                y_test,
            )

            kmer_model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, random_state=seed))
            kmer_model.fit(x_train_kmer, y_train)
            store_baseline(
                "kmer_ridge_or_logistic",
                kmer_model.predict_proba(x_test_kmer)[:, 1],
                classification_metrics,
                y_test,
            )

            rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=seed)
            rf.fit(x_train_kmer, y_train)
            store_baseline(
                "kmer_random_forest",
                rf.predict_proba(x_test_kmer)[:, 1],
                classification_metrics,
                y_test,
            )

            probe = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, random_state=seed))
            probe.fit(x_train_embed, y_train)
            store_baseline(
                "kmer4_linear_probe",
                probe.predict_proba(x_test_embed)[:, 1],
                classification_metrics,
                y_test,
            )
        else:
            for name, x_train, x_test in [
                ("gc_length_baseline", x_train_gc, x_test_gc),
                ("kmer_ridge_or_logistic", x_train_kmer, x_test_kmer),
                ("kmer_random_forest", x_train_kmer, x_test_kmer),
                ("kmer4_linear_probe", x_train_embed, x_test_embed),
            ]:
                score = _fit_logistic_fallback(x_train, y_train, x_test)
                store_baseline(name, score, classification_metrics, y_test, status="fallback")
    else:
        y_train = np.asarray([float(row[target_col]) for row in train_rows], dtype=np.float64)
        y_test = np.asarray([float(row[target_col]) for row in test_rows], dtype=np.float64)
        metric_fn = ranking_metrics if task_type == "ranking" else regression_metrics
        if sklearn_ok:
            gc_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed))
            gc_model.fit(x_train_gc, y_train)
            store_baseline("gc_length_baseline", gc_model.predict(x_test_gc), metric_fn, y_test)

            kmer_model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0, random_state=seed))
            kmer_model.fit(x_train_kmer, y_train)
            store_baseline("kmer_ridge_or_logistic", kmer_model.predict(x_test_kmer), metric_fn, y_test)

            rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=seed)
            rf.fit(x_train_kmer, y_train)
            store_baseline("kmer_random_forest", rf.predict(x_test_kmer), metric_fn, y_test)

            probe = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0, random_state=seed))
            probe.fit(x_train_embed, y_train)
            store_baseline("kmer4_linear_probe", probe.predict(x_test_embed), metric_fn, y_test)
        else:
            for name, x_train, x_test in [
                ("gc_length_baseline", x_train_gc, x_test_gc),
                ("kmer_ridge_or_logistic", x_train_kmer, x_test_kmer),
                ("kmer_random_forest", x_train_kmer, x_test_kmer),
                ("kmer4_linear_probe", x_train_embed, x_test_embed),
            ]:
                pred = _fit_linear_regression(x_train, y_train, x_test)
                store_baseline(name, pred, metric_fn, y_test, status="fallback")

    for metrics in baselines.values():
        metrics["primary_metric"] = _primary_metric(task_type, metrics)
        metrics["primary_metric_name"] = _primary_metric_name(task_type, metrics)
    return baselines


class MetadataEncoder:
    def __init__(self, categorical_cols: list[str] | None = None, max_categories: int = 100) -> None:
        self.categorical_cols = list(categorical_cols or [])
        self.max_categories = max_categories
        self.mappings: dict[str, dict[str, int]] = {}

    @property
    def output_dim(self) -> int:
        return sum(len(mapping) for mapping in self.mappings.values())

    def fit(self, rows: list[dict[str, Any]]) -> "MetadataEncoder":
        self.mappings = {}
        for col in self.categorical_cols:
            counts = Counter(str(row.get(col, "") or MISSING_VALUE) for row in rows)
            most_common = [value for value, _count in counts.most_common(self.max_categories - 1)]
            if UNKNOWN_VALUE not in most_common:
                most_common.append(UNKNOWN_VALUE)
            self.mappings[col] = {value: idx for idx, value in enumerate(most_common)}
        return self

    def transform(self, rows: list[dict[str, Any]]) -> np.ndarray:
        if not self.categorical_cols:
            return np.empty((len(rows), 0), dtype=np.float32)
        out = np.zeros((len(rows), self.output_dim), dtype=np.float32)
        offset = 0
        for col in self.categorical_cols:
            mapping = self.mappings.get(col, {UNKNOWN_VALUE: 0})
            unknown_idx = mapping.get(UNKNOWN_VALUE, 0)
            for row_idx, row in enumerate(rows):
                value = str(row.get(col, "") or MISSING_VALUE)
                out[row_idx, offset + mapping.get(value, unknown_idx)] = 1.0
            offset += len(mapping)
        return out

    def fit_transform(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return self.fit(rows).transform(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "categorical_cols": self.categorical_cols,
            "max_categories": self.max_categories,
            "mappings": self.mappings,
            "output_dim": self.output_dim,
        }


if nn is not None:

    class SupervisedTaskHead(nn.Module):
        """Task head over pooled DNA embeddings plus optional metadata features."""

        def __init__(
            self,
            sequence_dim: int,
            *,
            metadata_dim: int = 0,
            task_type: str = "regression",
            hidden_dim: int = 128,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            if task_type not in {"regression", "classification", "ranking"}:
                raise ValueError(f"Unsupported task_type={task_type!r}")
            self.task_type = task_type
            self.net = nn.Sequential(
                nn.Linear(sequence_dim + metadata_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, sequence_embedding: "torch.Tensor", metadata_features: "torch.Tensor | None" = None) -> "torch.Tensor":
            if metadata_features is not None and metadata_features.numel() > 0:
                x = torch.cat([sequence_embedding, metadata_features.to(sequence_embedding.device)], dim=-1)
            else:
                x = sequence_embedding
            return self.net(x).squeeze(-1)

else:
    SupervisedTaskHead = None  # type: ignore


def pool_hidden_state(
    hidden_state: "torch.Tensor",
    attention_mask: "torch.Tensor | None" = None,
    *,
    method: str = "mean",
    attention_pooler: Any | None = None,
) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("pool_hidden_state requires torch.")
    method = method.lower()
    if method == "cls":
        return hidden_state[:, 0]
    if method == "attention":
        if attention_pooler is None:
            scores = hidden_state.mean(dim=-1)
        else:
            scores = attention_pooler(hidden_state).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(hidden_state * weights.unsqueeze(-1), dim=1)
    if method != "mean":
        raise ValueError(f"Unsupported pooling method={method!r}; expected mean, cls, or attention.")
    if attention_mask is None:
        return hidden_state.mean(dim=1)
    mask = attention_mask.to(hidden_state.device, dtype=hidden_state.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden_state * mask).sum(dim=1) / denom


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (x - mean) / std, mean, std


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _target_arrays(
    rows: list[dict[str, Any]],
    *,
    target_col: str,
    task_type: str,
    class_mapping: dict[str, int] | None = None,
    positive_label: Any | None = None,
) -> tuple[np.ndarray, dict[str, int] | None]:
    values = [row[target_col] for row in rows]
    if task_type == "classification":
        y, mapping = encode_class_labels(
            values,
            mapping=class_mapping,
            positive_label=positive_label,
        )
        return y.astype(np.float32), mapping
    return np.asarray([float(value) for value in values], dtype=np.float32), class_mapping


def _pairwise_ranking_loss(scores: "torch.Tensor", target: "torch.Tensor", margin: float = 0.05) -> "torch.Tensor":
    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    mask = target_diff > 0
    if not bool(mask.any()):
        return F.mse_loss(scores, target)
    return torch.relu(margin - score_diff[mask]).mean()


def fit_task_head_ensemble(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    task_type: str,
    sequence_col: str,
    target_col: str,
    metadata_cols: list[str] | None = None,
    ensemble_size: int = 5,
    epochs: int = 80,
    learning_rate: float = 1e-3,
    seed: int = 13,
    sequence_feature_key: str | None = None,
    feature_source: str = "kmer_task_head",
    positive_label: Any | None = None,
) -> dict[str, Any]:
    if torch is None or SupervisedTaskHead is None:
        raise RuntimeError("Task-head training requires torch.")
    if not train_rows:
        return {"status": "skipped", "reason": "empty train split", "models": []}

    metadata_encoder = MetadataEncoder(metadata_cols).fit(train_rows)
    x_seq_train_raw = row_sequence_feature_matrix(
        train_rows,
        sequence_col=sequence_col,
        sequence_feature_key=sequence_feature_key,
    )
    x_seq_train, seq_mean, seq_std = _standardize_fit(x_seq_train_raw)
    x_meta_train = metadata_encoder.transform(train_rows)
    y_train, class_mapping = _target_arrays(
        train_rows,
        target_col=target_col,
        task_type=task_type,
        positive_label=positive_label,
    )

    if task_type == "classification" and len(set(y_train.tolist())) > 2:
        raise ValueError("Task-head classification currently supports binary high/low labels.")

    device = torch.device("cpu")
    models: list[Any] = []
    losses: list[float] = []
    rng = np.random.default_rng(seed)
    x_seq_tensor = torch.tensor(x_seq_train, dtype=torch.float32, device=device)
    x_meta_tensor = torch.tensor(x_meta_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

    for member in range(max(1, ensemble_size)):
        torch.manual_seed(seed + member)
        model = SupervisedTaskHead(
            x_seq_train.shape[1],
            metadata_dim=x_meta_train.shape[1],
            task_type=task_type,
            hidden_dim=min(128, max(16, x_seq_train.shape[1] // 2)),
            dropout=0.1,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        if len(train_rows) > 1:
            indices = rng.integers(0, len(train_rows), size=len(train_rows))
        else:
            indices = np.asarray([0])
        index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        final_loss = 0.0
        for _epoch in range(max(1, epochs)):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_seq_tensor[index_tensor], x_meta_tensor[index_tensor])
            target = y_tensor[index_tensor]
            if task_type == "classification":
                loss = F.binary_cross_entropy_with_logits(logits, target)
            elif task_type == "ranking":
                loss = _pairwise_ranking_loss(logits, target) + 0.1 * F.mse_loss(logits, target)
            else:
                loss = F.mse_loss(logits, target)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
        models.append(model.eval())
        losses.append(final_loss)

    ensemble = {
        "status": "ok",
        "models": models,
        "metadata_encoder": metadata_encoder,
        "sequence_mean": seq_mean,
        "sequence_std": seq_std,
        "task_type": task_type,
        "sequence_col": sequence_col,
        "sequence_feature_key": sequence_feature_key,
        "feature_source": feature_source,
        "target_col": target_col,
        "class_mapping": class_mapping,
        "positive_label": (
            next((label for label, encoded in (class_mapping or {}).items() if encoded == 1), None)
            if task_type == "classification"
            else None
        ),
        "training_losses": losses,
        "calibration_error": 0.0,
    }

    calibration_rows = val_rows if val_rows else train_rows
    if calibration_rows:
        cal_pred = predict_with_task_head_ensemble(ensemble, calibration_rows, include_calibration=False)
        if task_type == "classification":
            y_cal, _ = _target_arrays(
                calibration_rows,
                target_col=target_col,
                task_type=task_type,
                class_mapping=class_mapping,
            )
            calibration_error = float(np.sqrt(np.mean((cal_pred["mean"] - y_cal) ** 2)))
        else:
            y_cal = np.asarray([float(row[target_col]) for row in calibration_rows], dtype=np.float32)
            calibration_error = float(np.sqrt(np.mean((cal_pred["mean"] - y_cal) ** 2)))
        ensemble["calibration_error"] = calibration_error

    if test_rows:
        pred = predict_with_task_head_ensemble(ensemble, test_rows)
        if task_type == "classification":
            y_test, _ = _target_arrays(test_rows, target_col=target_col, task_type=task_type, class_mapping=class_mapping)
            metrics = classification_metrics(y_test.astype(np.int64), pred["mean"])
        elif task_type == "ranking":
            metrics = ranking_metrics([float(row[target_col]) for row in test_rows], pred["mean"])
        else:
            metrics = regression_metrics([float(row[target_col]) for row in test_rows], pred["mean"])
    else:
        metrics = {"status": "skipped", "reason": "empty test split"}
        pred = {"mean": np.asarray([]), "std": np.asarray([]), "raw": np.empty((0, 0))}
    metrics["primary_metric"] = _primary_metric(task_type, metrics)
    metrics["primary_metric_name"] = _primary_metric_name(task_type, metrics)
    ensemble["metrics"] = metrics
    ensemble["test_prediction_summary"] = {
        "mean": pred["mean"].tolist(),
        "std": pred["std"].tolist(),
    }
    return ensemble


def predict_with_task_head_ensemble(
    ensemble: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    include_calibration: bool = True,
) -> dict[str, np.ndarray]:
    if torch is None:
        raise RuntimeError("Task-head prediction requires torch.")
    models = ensemble.get("models") or []
    if not rows or not models:
        return {"mean": np.asarray([], dtype=np.float32), "std": np.asarray([], dtype=np.float32), "raw": np.empty((0, 0))}

    sequence_col = ensemble["sequence_col"]
    x_seq = row_sequence_feature_matrix(
        rows,
        sequence_col=sequence_col,
        sequence_feature_key=ensemble.get("sequence_feature_key"),
    )
    x_seq = _standardize_apply(x_seq, ensemble["sequence_mean"], ensemble["sequence_std"])
    x_meta = ensemble["metadata_encoder"].transform(rows)
    x_seq_tensor = torch.tensor(x_seq, dtype=torch.float32)
    x_meta_tensor = torch.tensor(x_meta, dtype=torch.float32)
    raw: list[np.ndarray] = []
    with torch.no_grad():
        for model in models:
            logits = model(x_seq_tensor, x_meta_tensor)
            if ensemble["task_type"] == "classification":
                values = torch.sigmoid(logits)
            else:
                values = logits
            raw.append(values.detach().cpu().numpy())
    raw_array = np.stack(raw, axis=0)
    mean = np.mean(raw_array, axis=0)
    std = np.std(raw_array, axis=0)
    if include_calibration:
        calibration_error = float(ensemble.get("calibration_error", 0.0) or 0.0)
        std = np.sqrt(std**2 + calibration_error**2)
    return {"mean": mean, "std": std, "raw": raw_array}


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def acquisition_scores(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    method: str = "upper_confidence_bound",
    beta: float = 1.0,
    incumbent: float | None = None,
) -> np.ndarray:
    method = method.lower()
    std = np.maximum(np.asarray(std, dtype=np.float64), 1e-9)
    mean = np.asarray(mean, dtype=np.float64)
    if method in {"upper_confidence_bound", "ucb"}:
        return mean + beta * std
    if method in {"expected_improvement", "ei"}:
        best = float(np.max(mean) if incumbent is None else incumbent)
        improvement = mean - best
        z = improvement / std
        return improvement * normal_cdf(z) + std * normal_pdf(z)
    if method in {"prediction", "mean"}:
        return mean
    raise ValueError(f"Unsupported acquisition method={method!r}.")


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    denom = np.linalg.norm(features, axis=1, keepdims=True)
    denom[denom < 1e-9] = 1.0
    normed = features / denom
    return normed @ normed.T


def _edit_similarity(left: str, right: str) -> float:
    return sequence_similarity(left, right, policy="edit_distance")


_DIVERSITY_POLICY_ALIASES = {
    "greedy_embedding_cosine": "kmer_cosine",
    "kmer_cosine": "kmer_cosine",
    "greedy_edit_distance": "edit_distance",
    "edit_distance": "edit_distance",
    "exact_reverse_complement": "exact_reverse_complement",
    "canonical_kmer_jaccard": "kmer_jaccard",
    "kmer_jaccard": "kmer_jaccard",
    "minhash_kmer": "kmer_jaccard",
    "position_aware_motif": "position_aware_motif",
}


def _resolve_diversity_policy(method: str) -> tuple[str, str, str | None]:
    requested = _normalize_similarity_policy_name(method)
    if requested in _DIVERSITY_POLICY_ALIASES:
        resolved = _DIVERSITY_POLICY_ALIASES[requested]
        sequence_policy = {
            "edit_distance": "edit_distance",
            "exact_reverse_complement": "exact_reverse_complement",
            "kmer_jaccard": "canonical_kmer_jaccard",
            "position_aware_motif": "position_aware_motif",
        }.get(resolved)
        return requested, resolved, sequence_policy
    try:
        _plugin_requested, plugin_policy = _resolve_similarity_policy(requested)
    except ValueError as exc:
        supported = sorted(_DIVERSITY_POLICY_ALIASES)
        raise ValueError(
            f"Unsupported diversity method={method!r}. Supported built-ins: {', '.join(supported)}; "
            "registered similarity plugins are also accepted."
        ) from exc
    if plugin_policy == "customer_family_labels":
        raise ValueError("customer_family_labels is group-based and cannot be used as a sequence diversity method.")
    return requested, plugin_policy, plugin_policy


def _pairwise_sequence_similarity_matrix(
    sequences: list[str],
    *,
    policy: str,
    k: int,
) -> np.ndarray:
    matrix = np.eye(len(sequences), dtype=np.float32)
    for right in range(1, len(sequences)):
        for left in range(right):
            similarity = sequence_similarity(sequences[left], sequences[right], policy=policy, k=k)
            matrix[left, right] = similarity
            matrix[right, left] = similarity
    return matrix


def greedy_diverse_rank(
    rows: list[dict[str, Any]],
    acquisition: np.ndarray,
    *,
    sequence_col: str,
    method: str = "greedy_embedding_cosine",
    diversity_penalty: float = 0.2,
    top_k: int = 96,
    similarity_k: int = 3,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    requested_policy, resolved_policy, sequence_policy = _resolve_diversity_policy(method)
    if int(similarity_k) <= 0:
        raise ValueError(f"similarity_k must be positive, got {similarity_k}.")
    acquisition_values = np.asarray(acquisition, dtype=np.float64).reshape(-1)
    if len(acquisition_values) != len(rows):
        raise ValueError(
            f"acquisition must contain one score per row; got {len(acquisition_values)} scores for {len(rows)} rows."
        )
    top_k = min(top_k, len(rows))
    sequences = [row[sequence_col] for row in rows]
    if resolved_policy == "kmer_cosine":
        # Preserve the historical default: 3-mer frequencies plus GC/length
        # features, now named explicitly instead of implying learned embeddings.
        features = kmer_feature_matrix(sequences, k=int(similarity_k))
        similarities = cosine_similarity_matrix(features)
    else:
        if sequence_policy is None:  # Defensive guard for future built-ins.
            raise ValueError(f"Diversity policy {resolved_policy!r} has no sequence-similarity implementation.")
        similarities = _pairwise_sequence_similarity_matrix(
            sequences,
            policy=sequence_policy,
            k=int(similarity_k),
        )
    selected: list[int] = []
    remaining = set(range(len(rows)))
    clusters = [-1 for _ in rows]
    selected_scores: dict[int, float] = {}
    selected_similarities: dict[int, float] = {}

    for rank in range(1, top_k + 1):
        best_idx = None
        best_score = -float("inf")
        best_similarity = 0.0
        for idx in sorted(remaining):
            if selected:
                similarity = max(float(similarities[idx, selected_idx]) for selected_idx in selected)
            else:
                similarity = 0.0
            score = float(acquisition_values[idx]) - diversity_penalty * similarity
            if score > best_score:
                best_score = score
                best_idx = idx
                best_similarity = similarity
        if best_idx is None:
            break
        remaining.remove(best_idx)
        selected.append(best_idx)
        selected_scores[best_idx] = best_score
        selected_similarities[best_idx] = best_similarity
        clusters[best_idx] = rank
        for idx in sorted(remaining):
            if clusters[idx] != -1:
                continue
            similarity = float(similarities[idx, best_idx])
            if similarity >= 0.95:
                clusters[idx] = rank

    ranked: list[dict[str, Any]] = []
    for rank, idx in enumerate(selected, start=1):
        row = dict(rows[idx])
        row["rank"] = rank
        row["diversity_cluster"] = clusters[idx] if clusters[idx] != -1 else rank
        row["diversified_acquisition_score"] = float(selected_scores.get(idx, acquisition_values[idx]))
        row["max_similarity_to_previous_selection"] = float(selected_similarities.get(idx, 0.0))
        row["diversity_penalty_applied"] = float(
            diversity_penalty * selected_similarities.get(idx, 0.0)
        )
        row["diversity_policy"] = resolved_policy
        row["diversity_policy_requested"] = requested_policy
        row["prioritization_status"] = "unreviewed_candidate_prioritization"
        ranked.append(row)
    return ranked


def best_simple_baseline(baselines: dict[str, dict[str, Any]], task_type: str) -> tuple[str | None, float | None]:
    best_name = None
    best_value = None
    for name, metrics in baselines.items():
        value = _primary_metric(task_type, metrics)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_name = name
            best_value = value
    return best_name, best_value
