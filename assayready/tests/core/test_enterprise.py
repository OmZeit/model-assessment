import json
from pathlib import Path

import pytest

from model_assessment.enterprise import (
    Principal,
    authenticate_api_key,
    hash_api_key,
    load_api_key_records,
    require_permission,
    trusted_proxy_principal,
)


def test_api_key_records_store_only_hashes(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"keys": [{"id": "pilot", "sha256": hash_api_key("secret"), "role": "scientist"}]}))
    records = load_api_key_records(path)
    principal = authenticate_api_key("secret", records)
    assert principal == Principal("pilot", "scientist", "api_key")
    assert authenticate_api_key("wrong", records) is None
    require_permission(principal, "outcome:write")
    with pytest.raises(PermissionError):
        require_permission(Principal("reader", "viewer", "api_key"), "outcome:write")


def test_proxy_headers_are_ignored_until_explicitly_enabled(monkeypatch) -> None:
    secret = "correct-horse-battery-staple-proxy-secret"
    headers = {
        "X-AssayReady-User": "alice",
        "X-AssayReady-Role": "admin",
        "X-AssayReady-Proxy-Secret": secret,
    }
    monkeypatch.delenv("ASSAYREADY_TRUST_PROXY_IDENTITY", raising=False)
    assert trusted_proxy_principal(headers) is None
    monkeypatch.setenv("ASSAYREADY_TRUST_PROXY_IDENTITY", "true")
    monkeypatch.setenv("ASSAYREADY_TRUSTED_PROXIES", "127.0.0.1,10.0.0.0/8")
    monkeypatch.setenv("ASSAYREADY_PROXY_SECRET", secret)

    assert trusted_proxy_principal(headers) is None
    assert trusted_proxy_principal(headers, "192.0.2.20") is None
    assert trusted_proxy_principal({**headers, "X-AssayReady-Proxy-Secret": "wrong"}, "127.0.0.1") is None
    assert trusted_proxy_principal(headers, "127.0.0.1") == Principal(
        "alice", "admin", "trusted_reverse_proxy"
    )
    assert trusted_proxy_principal(headers, "10.2.3.4") == Principal(
        "alice", "admin", "trusted_reverse_proxy"
    )


def test_enabled_proxy_identity_requires_complete_secure_configuration(monkeypatch) -> None:
    monkeypatch.setenv("ASSAYREADY_TRUST_PROXY_IDENTITY", "true")
    monkeypatch.delenv("ASSAYREADY_TRUSTED_PROXIES", raising=False)
    monkeypatch.delenv("ASSAYREADY_PROXY_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="ASSAYREADY_TRUSTED_PROXIES"):
        trusted_proxy_principal({}, "127.0.0.1")

    monkeypatch.setenv("ASSAYREADY_TRUSTED_PROXIES", "127.0.0.1")
    with pytest.raises(RuntimeError, match="at least 32"):
        trusted_proxy_principal({}, "127.0.0.1")

    monkeypatch.setenv("ASSAYREADY_TRUSTED_PROXIES", "0.0.0.0/0")
    with pytest.raises(RuntimeError, match="every address"):
        trusted_proxy_principal({}, "127.0.0.1")
