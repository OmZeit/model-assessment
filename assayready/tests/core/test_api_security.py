from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_assessment import api
from model_assessment.enterprise import Principal, hash_api_key


def _write_keys(path: Path, api_key: str, subject: str) -> None:
    path.write_text(
        json.dumps(
            {
                "keys": [
                    {
                        "id": subject,
                        "sha256": hash_api_key(api_key),
                        "role": "viewer",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_bearer_authentication_reloads_rotated_key_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "keys.json"
    _write_keys(key_file, "first-secret", "first-client")
    monkeypatch.setenv("ASSAYREADY_API_KEYS_FILE", str(key_file))

    assert api._authenticate_bearer("Bearer first-secret") == Principal(
        "first-client", "viewer", "api_key"
    )

    _write_keys(key_file, "replacement-secret", "replacement-client")
    assert api._authenticate_bearer("Bearer first-secret") is None
    assert api._authenticate_bearer("Bearer replacement-secret") == Principal(
        "replacement-client", "viewer", "api_key"
    )


def test_sliding_window_rate_limiter_recovers_after_window() -> None:
    limiter = api.SlidingWindowRateLimiter(limit=2, window_seconds=60)

    assert limiter.check("client", now=100) == (True, 1, 0.0)
    assert limiter.check("client", now=101) == (True, 0, 0.0)
    accepted, remaining, retry_after = limiter.check("client", now=102)
    assert accepted is False
    assert remaining == 0
    assert retry_after == pytest.approx(58)
    assert limiter.check("client", now=161) == (True, 1, 0.0)


def test_browser_origin_policy_defaults_to_same_origin() -> None:
    assert api._origin_is_allowed("http://127.0.0.1:8060", "http://127.0.0.1:8060/v1/runs", frozenset())
    assert not api._origin_is_allowed("https://malicious.example", "http://127.0.0.1:8060/v1/runs", frozenset())
    assert api._origin_is_allowed(
        "https://eln.example",
        "http://127.0.0.1:8060/v1/runs",
        frozenset({"https://eln.example"}),
    )


def test_cors_origin_configuration_rejects_wildcards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(api.API_ALLOWED_ORIGINS_ENV, "*")
    with pytest.raises(RuntimeError, match="wildcard"):
        api._allowed_origins_from_environment()
