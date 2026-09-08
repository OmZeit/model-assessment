from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


IMAGE_DIGEST_RE = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$", re.IGNORECASE)
MEMORY_LIMIT_RE = re.compile(r"^[1-9][0-9]*[bkmg]$", re.IGNORECASE)
CONTROLLED_OUTPUT_ROOT_ENV = "ASSAYREADY_CONTROLLED_OUTPUT_ROOT"


def controlled_output_root(base_dir: Path | None = None) -> Path:
    base = (base_dir or Path.cwd()).resolve()
    configured = os.environ.get(CONTROLLED_OUTPUT_ROOT_ENV, "").strip()
    if not configured:
        return (base / "outputs").resolve()
    root = Path(configured).expanduser()
    if not root.is_absolute():
        raise RuntimeError(f"{CONTROLLED_OUTPUT_ROOT_ENV} must be an absolute path.")
    return root.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ContainerExecutionSpec:
    image: str
    input_file: Path
    output_dir: Path
    output_filename: str = "predictions.csv"
    id_column: str = "sequence_id"
    prediction_column: str = "prediction"
    uncertainty_column: str | None = None
    runtime: str = "docker"
    timeout_seconds: int = 1800
    memory: str = "4g"
    cpus: float = 2.0
    gpu_device: str | None = None
    command: tuple[str, ...] = field(default_factory=tuple)
    allowed_output_root: Path | None = field(default=None, repr=False)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], *, base_dir: Path | None = None) -> "ContainerExecutionSpec":
        base = (base_dir or Path.cwd()).resolve()
        input_path = Path(str(payload.get("input_file") or ""))
        output_path = Path(str(payload.get("output_dir") or ""))
        if not input_path.is_absolute():
            input_path = base / input_path
        if not output_path.is_absolute():
            output_path = base / output_path
        return cls(
            image=str(payload.get("image") or ""),
            input_file=input_path.resolve(),
            output_dir=output_path.resolve(),
            allowed_output_root=controlled_output_root(base),
            output_filename=str(payload.get("output_filename") or "predictions.csv"),
            id_column=str(payload.get("id_column") or "sequence_id"),
            prediction_column=str(payload.get("prediction_column") or "prediction"),
            uncertainty_column=(str(payload["uncertainty_column"]) if payload.get("uncertainty_column") else None),
            runtime=str(payload.get("runtime") or "docker"),
            timeout_seconds=int(payload.get("timeout_seconds", 1800)),
            memory=str(payload.get("memory") or "4g"),
            cpus=float(payload.get("cpus", 2.0)),
            gpu_device=(str(payload["gpu_device"]) if payload.get("gpu_device") is not None else None),
            command=tuple(str(item) for item in payload.get("command") or ()),
        )

    def validate(self) -> None:
        if not IMAGE_DIGEST_RE.match(self.image):
            if "replace" in self.image.lower() or "placeholder" in self.image.lower():
                raise ValueError(
                    "image must be replaced with a built and pushed container digest such as "
                    "registry.example/assayready-dnabert2@sha256:<64 hex characters>."
                )
            raise ValueError("image must be immutable and pinned as repository@sha256:<64 hex characters>.")
        if self.runtime not in {"docker", "podman"}:
            raise ValueError("runtime must be docker or podman.")
        if not self.input_file.is_file():
            raise ValueError(f"input_file does not exist: {self.input_file}")
        allowed_output_root = (
            self.allowed_output_root.resolve()
            if self.allowed_output_root is not None
            else controlled_output_root()
        )
        resolved_output = self.output_dir.resolve()
        if resolved_output != allowed_output_root and not resolved_output.is_relative_to(allowed_output_root):
            raise ValueError(
                f"output_dir must reside within the controlled output root: {allowed_output_root}"
            )
        if Path(self.output_filename).name != self.output_filename or self.output_filename in {"", ".", ".."}:
            raise ValueError("output_filename must be a plain file name.")
        if self.timeout_seconds < 1 or self.timeout_seconds > 86400:
            raise ValueError("timeout_seconds must be between 1 and 86400.")
        if self.cpus <= 0 or self.cpus > 128:
            raise ValueError("cpus must be between zero and 128.")
        if not MEMORY_LIMIT_RE.fullmatch(self.memory):
            raise ValueError("memory must be a positive integer followed by b, k, m, or g.")
        if self.gpu_device is not None and not re.fullmatch(r"all|[0-9]+", self.gpu_device):
            raise ValueError("gpu_device must be 'all' or a non-negative integer device index.")
        if any("\x00" in item for item in self.command):
            raise ValueError("command arguments may not contain NUL bytes.")


def build_runtime_command(spec: ContainerExecutionSpec) -> list[str]:
    spec.validate()
    command = [
        spec.runtime,
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=256m",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=256",
        f"--memory={spec.memory}",
        f"--cpus={spec.cpus:g}",
        "--mount",
        f"type=bind,source={spec.input_file},target=/assayready/input/{spec.input_file.name},readonly",
        "--mount",
        f"type=bind,source={spec.output_dir},target=/assayready/output",
    ]
    if spec.gpu_device is not None:
        if spec.runtime == "docker":
            gpu_value = "all" if spec.gpu_device == "all" else f"device={spec.gpu_device}"
            command.append(f"--gpus={gpu_value}")
        else:
            command.append(f"--device=nvidia.com/gpu={spec.gpu_device}")
    return [
        *command,
        spec.image,
        *spec.command,
        f"/assayready/input/{spec.input_file.name}",
        f"/assayready/output/{spec.output_filename}",
    ]


def validate_prediction_output(spec: ContainerExecutionSpec, output_path: Path) -> dict[str, Any]:
    if output_path.is_symlink():
        raise ValueError("prediction output may not be a symbolic link.")
    if not output_path.is_file():
        raise ValueError(f"container did not create {output_path.name}.")
    with spec.input_file.open("r", encoding="utf-8-sig", newline="") as handle:
        input_rows = list(csv.DictReader(handle))
    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        output_rows = list(csv.DictReader(handle))
    if not input_rows:
        raise ValueError("locked input contains no data rows.")
    if not output_rows:
        raise ValueError("prediction output contains no data rows.")
    required = {spec.id_column, spec.prediction_column}
    if spec.uncertainty_column:
        required.add(spec.uncertainty_column)
    missing = required - set(output_rows[0])
    if missing:
        raise ValueError("prediction output is missing columns: " + ", ".join(sorted(missing)))
    input_ids = [str(row.get(spec.id_column) or "") for row in input_rows]
    output_ids = [str(row.get(spec.id_column) or "") for row in output_rows]
    if len(set(input_ids)) != len(input_ids) or any(not value for value in input_ids):
        raise ValueError("locked input identifiers must be non-empty and unique.")
    if len(set(output_ids)) != len(output_ids) or any(not value for value in output_ids):
        raise ValueError("prediction output identifiers must be non-empty and unique.")
    if set(input_ids) != set(output_ids):
        missing_ids = sorted(set(input_ids) - set(output_ids))[:10]
        extra_ids = sorted(set(output_ids) - set(input_ids))[:10]
        raise ValueError(f"prediction identifiers do not match locked input; missing={missing_ids}, extra={extra_ids}")
    invalid_predictions = []
    for row in output_rows:
        try:
            float(row[spec.prediction_column])
        except (TypeError, ValueError):
            invalid_predictions.append(str(row.get(spec.id_column)))
    if invalid_predictions:
        raise ValueError(f"non-numeric predictions for identifiers: {invalid_predictions[:10]}")
    return {
        "input_rows": len(input_rows),
        "prediction_rows": len(output_rows),
        "id_set_match": True,
        "output_sha256": sha256_file(output_path),
    }


def execute_locked_container(spec: ContainerExecutionSpec) -> dict[str, Any]:
    spec.validate()
    if shutil.which(spec.runtime) is None:
        raise RuntimeError(f"{spec.runtime} is not installed or not available on PATH.")
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = spec.output_dir / spec.output_filename
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing prediction output: {output_path}")
    command = build_runtime_command(spec)
    started = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    completed = subprocess.run(
        command,
        shell=False,
        capture_output=True,
        text=True,
        timeout=spec.timeout_seconds,
        check=False,
    )
    duration = time.perf_counter() - started_clock
    if completed.returncode != 0:
        raise RuntimeError(
            f"controlled model execution failed with exit code {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    validation = validate_prediction_output(spec, output_path)
    manifest = {
        "schema_version": 1,
        "execution_mode": "controlled_container",
        "training_independence": "internally_controlled",
        "image": spec.image,
        "runtime": spec.runtime,
        "network": "none",
        "root_filesystem": "read_only",
        "capabilities": "dropped",
        "gpu_device": spec.gpu_device,
        "input_file": str(spec.input_file),
        "input_sha256": sha256_file(spec.input_file),
        "output_file": str(output_path),
        "output_sha256": validation["output_sha256"],
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "validation": validation,
        "command_arguments": list(spec.command),
    }
    manifest_path = spec.output_dir / "controlled_execution_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_execution_spec(path: str | Path) -> ContainerExecutionSpec:
    spec_path = Path(path).resolve()
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("execution spec must be a JSON object.")
    return ContainerExecutionSpec.from_mapping(payload, base_dir=spec_path.parent)
