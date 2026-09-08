from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


API_KEYS_FILE_ENV = "ASSAYREADY_API_KEYS_FILE"
TRUST_PROXY_ENV = "ASSAYREADY_TRUST_PROXY_IDENTITY"
TRUSTED_PROXIES_ENV = "ASSAYREADY_TRUSTED_PROXIES"
PROXY_SECRET_ENV = "ASSAYREADY_PROXY_SECRET"
PROXY_SECRET_HEADER_ENV = "ASSAYREADY_PROXY_SECRET_HEADER"
IDENTITY_HEADER_ENV = "ASSAYREADY_IDENTITY_HEADER"
ROLE_HEADER_ENV = "ASSAYREADY_ROLE_HEADER"
DEFAULT_PROXY_SECRET_HEADER = "X-AssayReady-Proxy-Secret"

ROLE_PERMISSIONS = {
    "viewer": frozenset({"campaign:read", "run:read", "artifact:read"}),
    "scientist": frozenset(
        {"campaign:read", "campaign:write", "round:write", "outcome:write", "run:read", "artifact:read"}
    ),
    "admin": frozenset({"*"}),
}


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str
    authentication_method: str

    def can(self, permission: str) -> bool:
        permissions = ROLE_PERMISSIONS.get(self.role, frozenset())
        return "*" in permissions or permission in permissions


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def load_api_key_records(path: str | Path | None = None) -> list[dict[str, str]]:
    raw_path = str(path) if path is not None else os.environ.get(API_KEYS_FILE_ENV, "")
    if not raw_path.strip():
        return []
    configured = Path(raw_path)
    if not configured.is_file():
        raise RuntimeError(f"API key record file does not exist: {configured}")
    payload = json.loads(configured.read_text(encoding="utf-8"))
    records = payload.get("keys") if isinstance(payload, Mapping) else None
    if not isinstance(records, list):
        raise ValueError("API key file must contain a keys list.")
    normalized = []
    for record in records:
        if not isinstance(record, Mapping) or record.get("role") not in ROLE_PERMISSIONS:
            raise ValueError("each API key record requires id, sha256, and a recognized role.")
        digest = str(record.get("sha256") or "").lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("API key sha256 must be a 64-character hexadecimal digest.")
        normalized.append({"id": str(record.get("id") or ""), "sha256": digest, "role": str(record["role"])})
    return normalized


def authenticate_api_key(api_key: str, records: list[dict[str, str]]) -> Principal | None:
    supplied = hash_api_key(api_key)
    matched: dict[str, str] | None = None
    for record in records:
        if hmac.compare_digest(supplied, record["sha256"]):
            matched = record
    return Principal(matched["id"], matched["role"], "api_key") if matched else None


def _trusted_proxy_networks() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    configured = os.environ.get(TRUSTED_PROXIES_ENV, "")
    entries = [entry.strip() for entry in configured.replace(";", ",").split(",") if entry.strip()]
    if not entries:
        raise RuntimeError(
            f"{TRUSTED_PROXIES_ENV} must contain explicit proxy IP addresses or CIDR networks."
        )
    try:
        networks = tuple(ipaddress.ip_network(entry, strict=False) for entry in entries)
    except ValueError as exc:
        raise RuntimeError(f"{TRUSTED_PROXIES_ENV} contains an invalid IP address or CIDR network.") from exc
    if any(network.prefixlen == 0 for network in networks):
        raise RuntimeError(f"{TRUSTED_PROXIES_ENV} may not trust every address.")
    return networks


def validate_trusted_proxy_configuration() -> None:
    if os.environ.get(TRUST_PROXY_ENV, "").lower() not in {"1", "true", "yes"}:
        return
    _trusted_proxy_networks()
    secret = os.environ.get(PROXY_SECRET_ENV, "")
    if len(secret) < 32:
        raise RuntimeError(f"{PROXY_SECRET_ENV} must be set to a secret of at least 32 characters.")


def trusted_proxy_principal(
    headers: Mapping[str, str],
    client_host: str | None = None,
) -> Principal | None:
    if os.environ.get(TRUST_PROXY_ENV, "").lower() not in {"1", "true", "yes"}:
        return None
    validate_trusted_proxy_configuration()
    try:
        client_address = ipaddress.ip_address(str(client_host or "").split("%")[0])
    except ValueError:
        return None
    if isinstance(client_address, ipaddress.IPv6Address) and client_address.ipv4_mapped:
        client_address = client_address.ipv4_mapped
    if not any(
        client_address.version == network.version and client_address in network
        for network in _trusted_proxy_networks()
    ):
        return None
    identity_header = os.environ.get(IDENTITY_HEADER_ENV, "X-AssayReady-User").lower()
    role_header = os.environ.get(ROLE_HEADER_ENV, "X-AssayReady-Role").lower()
    secret_header = os.environ.get(PROXY_SECRET_HEADER_ENV, DEFAULT_PROXY_SECRET_HEADER).lower()
    normalized = {str(key).lower(): str(value) for key, value in headers.items()}
    supplied_secret = normalized.get(secret_header, "")
    expected_secret = os.environ[PROXY_SECRET_ENV]
    if not supplied_secret or not hmac.compare_digest(
        supplied_secret.encode("utf-8"), expected_secret.encode("utf-8")
    ):
        return None
    subject = normalized.get(identity_header, "").strip()
    role = normalized.get(role_header, "").strip().lower()
    if not subject or len(subject) > 256 or any(ord(character) < 32 for character in subject):
        return None
    if role not in ROLE_PERMISSIONS:
        return None
    return Principal(subject, role, "trusted_reverse_proxy")


def require_permission(principal: Principal, permission: str) -> None:
    if not principal.can(permission):
        raise PermissionError(f"role {principal.role!r} lacks permission {permission!r}.")


def security_posture() -> dict[str, Any]:
    proxy_enabled = os.environ.get(TRUST_PROXY_ENV, "").lower() in {"1", "true", "yes"}
    return {
        "api_keys_configured": bool(os.environ.get(API_KEYS_FILE_ENV)),
        "trusted_proxy_identity_enabled": proxy_enabled,
        "trusted_proxies_configured": bool(os.environ.get(TRUSTED_PROXIES_ENV, "").strip()),
        "proxy_shared_secret_configured": bool(os.environ.get(PROXY_SECRET_ENV, "")),
        "identity_header": os.environ.get(IDENTITY_HEADER_ENV, "X-AssayReady-User"),
        "role_header": os.environ.get(ROLE_HEADER_ENV, "X-AssayReady-Role"),
        "proxy_secret_header": os.environ.get(PROXY_SECRET_HEADER_ENV, DEFAULT_PROXY_SECRET_HEADER),
        "roles": {name: sorted(values) for name, values in ROLE_PERMISSIONS.items()},
        "warning": (
            "Terminate TLS and validate identity at the reverse proxy; strip client-supplied "
            "X-AssayReady-* headers before adding trusted identity headers."
        ),
    }
