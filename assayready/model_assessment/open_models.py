from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


DNABERT2_REPO = "zhihan1996/DNABERT-2-117M"
DNABERT2_REVISION = "a3d38b3f41cec05e370a4d3eeb8664fcb4bce227"
DNABERT2_WEIGHTS_SHA256 = "bc91ac0d972a698b7ff12ea6815e966b1feff9d7d1bef3a10535f2f4332609ac"
DNABERT2_WEIGHTS_SIZE = 468_313_032

# Expected bytes for every file downloaded from the executable model snapshot.
# A self-authored manifest can prove local consistency, but only these constants
# authenticate the pinned revision before Transformers imports its remote code.
DNABERT2_PINNED_FILES: dict[str, tuple[int, str]] = {
    "LICENSE": (11_357, "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
    "README.md": (1_316, "48b18abd051eb4952e0c0a50a0740387a1cea145be06517b67ab73a29431a3bc"),
    "bert_layers.py": (40_690, "317ad7e9667980ac724c07f0174ba265c8446ca5230f6b473068ff1b911edcf9"),
    "bert_padding.py": (6_099, "44d1c68afb1f585fdc66c150d4c60f1ed44a89c006abc57d50531d71940d7421"),
    "config.json": (904, "ba9bdafaff0cc3e30556927474d4a179519a9864012bed2628e9f1bc23c84bfd"),
    "configuration_bert.py": (1_011, "95fc868641b87bbcd7a32d2cd7b9f4769c27592e129daf167d14b5b8c74ec4c5"),
    "flash_attn_triton.py": (42_737, "568d1ac3beca0b5e1df528a1f136aa19b6489a616fcf3784f33336a50bb1de81"),
    "generation_config.json": (90, "c993e393c12525ea015019130f83c90502efaa6de8d019e94856555614d9fae3"),
    "model.safetensors": (DNABERT2_WEIGHTS_SIZE, DNABERT2_WEIGHTS_SHA256),
    "tokenizer.json": (167_908, "5d178e8ce2ba55df97fff197f4b30f40133b95d7096be398c2df6b526c5d8cd3"),
    "tokenizer_config.json": (158, "f9d18c81f4dd9dd7db02e9f27cc1203228147d890bfce9167c3af6465ff5b769"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_record_value(
    record: dict[str, Any],
    primary: str,
    alias: str,
    *,
    index: int,
) -> Any:
    """Read a canonical manifest field while accepting one historical alias."""
    has_primary = primary in record
    has_alias = alias in record
    if has_primary and has_alias and record[primary] != record[alias]:
        raise ValueError(
            f"model manifest file record {index} has conflicting {primary!r} and {alias!r} values."
        )
    if has_primary:
        return record[primary]
    if has_alias:
        return record[alias]
    raise ValueError(f"model manifest file record {index} is missing {primary!r}.")


def safe_manifest_file_path(root: str | Path, name: Any, *, label: str = "manifest file") -> tuple[str, Path]:
    """Resolve a manifest-owned relative filename without permitting an escape from *root*."""
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError(f"{label} path must be a non-empty string.")
    windows_path = PureWindowsPath(name)
    posix_path = PurePosixPath(name)
    if windows_path.is_absolute() or windows_path.drive or posix_path.is_absolute():
        raise ValueError(f"{label} path must be relative: {name!r}.")

    # Treat either slash style as a separator on every platform. This prevents a
    # Windows traversal from becoming an innocuous-looking filename on POSIX.
    parts = name.replace("\\", "/").split("/")
    if any(part in {"", ".", ".."} for part in parts) or any(":" in part for part in parts):
        raise ValueError(f"{label} path contains an unsafe path component: {name!r}.")

    root_path = Path(root).expanduser().resolve()
    candidate = root_path.joinpath(*parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} is missing or inaccessible: {name!r}.") from exc
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the model directory: {name!r}.") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {name!r}.")
    return "/".join(parts), resolved


def verify_manifest_file_records(
    root: str | Path,
    manifest: dict[str, Any],
) -> dict[str, tuple[Path, int, str]]:
    """Verify every declared model-manifest file and return its observed metadata.

    Manifests created before per-file records were introduced remain supported.
    Once a ``files`` member is present, however, malformed or unverifiable records
    are rejected instead of silently falling back to weights-only verification.
    """
    if "files" not in manifest:
        return {}
    records = manifest["files"]
    if not isinstance(records, list):
        raise ValueError("model manifest 'files' must be a list.")

    verified: dict[str, tuple[Path, int, str]] = {}
    resolved_names: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"model manifest file record {index} must be an object.")
        raw_name = _manifest_record_value(record, "name", "path", index=index)
        raw_size = _manifest_record_value(record, "size_bytes", "bytes", index=index)
        expected_digest = record.get("sha256")
        if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
            raise ValueError(
                f"model manifest file record {index} size_bytes must be a non-negative integer."
            )
        if (
            not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in expected_digest)
        ):
            raise ValueError(f"model manifest file record {index} has an invalid SHA-256 digest.")

        relative_name, path = safe_manifest_file_path(
            root, raw_name, label=f"model manifest file record {index}"
        )
        resolved_key = os.path.normcase(str(path))
        if resolved_key in resolved_names:
            raise ValueError(f"model manifest contains duplicate file record path: {raw_name!r}.")
        resolved_names.add(resolved_key)

        actual_size = path.stat().st_size
        if actual_size != raw_size:
            raise ValueError(
                f"model manifest size mismatch for {relative_name!r}: expected {raw_size}, got {actual_size}."
            )
        try:
            actual_digest = _sha256(path)
        except OSError as exc:
            raise ValueError(f"could not hash model manifest file {relative_name!r}: {exc}") from exc
        if actual_digest != expected_digest.lower():
            raise ValueError(f"model manifest SHA-256 mismatch for {relative_name!r}.")
        verified[relative_name] = (path, actual_size, actual_digest)
    return verified


def verify_pinned_dnabert2_snapshot(
    root: str | Path,
    *,
    verified_files: dict[str, tuple[Path, int, str]] | None = None,
) -> dict[str, tuple[Path, int, str]]:
    """Authenticate every file in the DNABERT-2 revision AssayReady executes."""
    verified_files = verified_files or {}
    observed: dict[str, tuple[Path, int, str]] = {}
    for name, (expected_size, expected_digest) in DNABERT2_PINNED_FILES.items():
        record = verified_files.get(name)
        if record is None:
            _relative_name, path = safe_manifest_file_path(
                root,
                name,
                label=f"pinned DNABERT-2 file {name!r}",
            )
            try:
                record = (path, path.stat().st_size, _sha256(path))
            except OSError as exc:
                raise ValueError(f"could not verify pinned DNABERT-2 file {name!r}: {exc}") from exc
        path, actual_size, actual_digest = record
        if actual_size != expected_size:
            raise ValueError(
                f"pinned DNABERT-2 size mismatch for {name!r}: expected {expected_size}, got {actual_size}."
            )
        if actual_digest != expected_digest:
            raise ValueError(f"pinned DNABERT-2 SHA-256 mismatch for {name!r}.")
        observed[name] = (path, actual_size, actual_digest)
    return observed


def dnabert2_lock() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_name": "DNABERT-2 117M",
        "repository": DNABERT2_REPO,
        "revision": DNABERT2_REVISION,
        "weights_file": "model.safetensors",
        "weights_sha256": DNABERT2_WEIGHTS_SHA256,
        "parameters": 117_000_000,
        "embedding_size": 768,
        "license": "Apache-2.0",
        "intended_device": "one NVIDIA GPU with 16 GB VRAM",
        "selection_rationale": (
            "Commercial-friendly, efficient multi-species DNA foundation model with ample "
            "16 GB inference headroom; the frozen backbone still requires an assay-specific head."
        ),
    }


def scaffold_dnabert2_bundle(output_dir: str | Path) -> Path:
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty model bundle: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source = files("model_assessment").joinpath("model_bundles", "dnabert2_117m")
    for item in source.iterdir():
        if item.is_file():
            destination = output / item.name
            with item.open("rb") as reader, destination.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
    (output / "model.lock.json").write_text(
        json.dumps(dnabert2_lock(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "weights").mkdir(exist_ok=True)
    (output / "task_head").mkdir(exist_ok=True)
    return output


def ensure_dnabert2_scaffold(output_dir: str | Path) -> Path:
    """Create or safely resume an AssayReady-owned DNABERT-2 scaffold.

    The low-level ``scaffold_dnabert2_bundle`` operation deliberately refuses
    every non-empty directory.  The user-facing ``prepare-dnabert2`` command,
    however, needs to be rerunnable after a completed setup or an interrupted
    download.  A valid AssayReady lock file is the ownership marker: existing
    files are never overwritten, while missing packaged scaffold files and
    directories are restored.
    """
    output = Path(output_dir).resolve()
    if not output.exists() or not any(output.iterdir()):
        return scaffold_dnabert2_bundle(output)

    lock_path = output / "model.lock.json"
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FileExistsError(
            f"refusing to resume an unrecognized non-empty model bundle: {output}"
        ) from exc
    expected_lock = dnabert2_lock()
    ownership_fields = ("schema_version", "repository", "revision", "weights_file")
    if not isinstance(lock, dict) or any(
        lock.get(field) != expected_lock[field] for field in ownership_fields
    ):
        raise FileExistsError(
            f"refusing to resume an unrecognized non-empty model bundle: {output}"
        )

    source = files("model_assessment").joinpath("model_bundles", "dnabert2_117m")
    for item in source.iterdir():
        if not item.is_file():
            continue
        destination = output / item.name
        if destination.exists():
            if not destination.is_file():
                raise FileExistsError(
                    f"refusing to replace non-file scaffold path: {destination}"
                )
            continue
        with item.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
    (output / "weights").mkdir(exist_ok=True)
    (output / "task_head").mkdir(exist_ok=True)
    return output


def download_dnabert2(output_dir: str | Path) -> dict[str, Any]:
    """Download and verify the exact Safetensors snapshot selected by AssayReady."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "assayready_model_manifest.json"
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                raise ValueError("model manifest must contain a JSON object.")
            if existing.get("revision") != DNABERT2_REVISION:
                raise ValueError("model revision is not the pinned DNABERT-2 revision.")
            verified_files = verify_manifest_file_records(output, existing)
            verify_pinned_dnabert2_snapshot(output, verified_files=verified_files)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"existing model directory does not match the pinned DNABERT-2 snapshot: {exc}"
            ) from exc
        return existing

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface-hub to download the pinned model snapshot.") from exc

    snapshot_download(
        repo_id=DNABERT2_REPO,
        revision=DNABERT2_REVISION,
        local_dir=output,
        allow_patterns=[
            "LICENSE", "README.md", "config.json", "configuration_bert.py",
            "bert_layers.py", "bert_padding.py", "flash_attn_triton.py",
            "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "model.safetensors",
        ],
    )
    try:
        pinned_files = verify_pinned_dnabert2_snapshot(output)
    except ValueError as exc:
        raise RuntimeError(f"Downloaded DNABERT-2 snapshot failed pinned-file verification: {exc}") from exc
    file_records = [
        {"name": name, "size_bytes": size, "sha256": digest}
        for name, (_path, size, digest) in sorted(pinned_files.items())
    ]
    manifest = {
        **dnabert2_lock(),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": file_records,
        "remote_code": "authenticated_pinned_snapshot_requires_operator_review",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
