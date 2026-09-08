from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

import yaml


POLICY_SCHEMA_VERSION = 1


def policy_pack_names() -> list[str]:
    root = files("model_assessment").joinpath("policy_packs")
    return sorted(item.name.removesuffix(".yaml") for item in root.iterdir() if item.name.endswith(".yaml"))


def _validate(pack: Mapping[str, Any]) -> dict[str, Any]:
    if int(pack.get("schema_version", 0)) != POLICY_SCHEMA_VERSION:
        raise ValueError(f"policy schema_version must be {POLICY_SCHEMA_VERSION}.")
    for field in ("name", "version", "status", "assay_type", "rationale"):
        if not str(pack.get(field) or "").strip():
            raise ValueError(f"policy field is required: {field}")
    if pack["status"] not in {"development_template", "customer_approved", "expert_reviewed"}:
        raise ValueError("policy status is not recognized.")
    evaluation = pack.get("evaluation")
    batch_design = pack.get("batch_design")
    if not isinstance(evaluation, Mapping) or not isinstance(batch_design, Mapping):
        raise ValueError("policy evaluation and batch_design sections are required.")
    normalized = json.loads(json.dumps(dict(pack), sort_keys=True, default=str))
    normalized.pop("sha256", None)
    normalized.pop("requires_scientist_approval", None)
    digest_payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    normalized["sha256"] = hashlib.sha256(digest_payload).hexdigest()
    normalized["requires_scientist_approval"] = pack["status"] == "development_template"
    return normalized


def load_policy_pack(name_or_path: str | Path) -> dict[str, Any]:
    candidate = Path(name_or_path)
    if candidate.is_file():
        payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    else:
        name = str(name_or_path).removesuffix(".yaml")
        resource = files("model_assessment").joinpath("policy_packs", f"{name}.yaml")
        if not resource.is_file():
            raise ValueError(f"Unknown policy pack {name!r}; available: {', '.join(policy_pack_names())}")
        payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("policy pack must contain a YAML object.")
    return _validate(payload)


def record_policy_approval(
    pack: Mapping[str, Any], *, approver: str, rationale: str
) -> dict[str, Any]:
    if not approver.strip() or not rationale.strip():
        raise ValueError("approver and rationale are required.")
    approved = dict(pack)
    approved["status"] = "customer_approved"
    approved["approval"] = {"approver": approver.strip(), "rationale": rationale.strip()}
    return _validate(approved)
