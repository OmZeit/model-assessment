import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from model_assessment.campaigns import CampaignStore
from model_assessment.enterprise import (
    Principal,
    authenticate_api_key,
    load_api_key_records,
    require_permission,
    security_posture,
    trusted_proxy_principal,
    validate_trusted_proxy_configuration,
)
from model_assessment.run_store import get_run_summary, list_runs


LOGGER = logging.getLogger(__name__)
API_RATE_LIMIT_ENV = "ASSAYREADY_API_RATE_LIMIT_PER_MINUTE"
API_ALLOWED_ORIGINS_ENV = "ASSAYREADY_API_ALLOWED_ORIGINS"


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        if limit < 1 or window_seconds <= 0:
            raise ValueError("rate-limit values must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, *, now: float | None = None) -> tuple[bool, int, float]:
        current = time.monotonic() if now is None else now
        cutoff = current - self.window_seconds
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                retry_after = max(events[0] + self.window_seconds - current, 0.001)
                return False, 0, retry_after
            events.append(current)
            return True, self.limit - len(events), 0.0


def _rate_limit_from_environment() -> int:
    raw = os.environ.get(API_RATE_LIMIT_ENV, "120").strip()
    try:
        limit = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{API_RATE_LIMIT_ENV} must be an integer.") from exc
    if limit < 1 or limit > 100_000:
        raise RuntimeError(f"{API_RATE_LIMIT_ENV} must be between 1 and 100000.")
    return limit


def _normalize_origin(value: str) -> str:
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("origin must contain only an http(s) scheme and authority")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("origin contains an invalid port") from exc
    hostname = parsed.hostname.lower()
    host = f"[{hostname}]" if ":" in hostname else hostname
    default_port = 80 if parsed.scheme.lower() == "http" else 443
    port_suffix = f":{port}" if port is not None and port != default_port else ""
    return f"{parsed.scheme.lower()}://{host}{port_suffix}"


def _allowed_origins_from_environment() -> frozenset[str]:
    values = [value.strip() for value in os.environ.get(API_ALLOWED_ORIGINS_ENV, "").split(",")]
    normalized: set[str] = set()
    for value in values:
        if not value:
            continue
        if value == "*":
            raise RuntimeError(f"{API_ALLOWED_ORIGINS_ENV} does not permit a wildcard origin.")
        try:
            normalized.add(_normalize_origin(value))
        except ValueError as exc:
            raise RuntimeError(f"{API_ALLOWED_ORIGINS_ENV} contains an invalid origin.") from exc
    return frozenset(normalized)


def _origin_is_allowed(origin: str, request_url: str, allowed_origins: frozenset[str]) -> bool:
    try:
        normalized_origin = _normalize_origin(origin)
        parsed = urlsplit(request_url)
        if not parsed.scheme or not parsed.netloc:
            return False
        request_origin = _normalize_origin(f"{parsed.scheme}://{parsed.netloc}")
    except ValueError:
        return False
    return normalized_origin == request_origin or normalized_origin in allowed_origins


def _authenticate_bearer(authorization: str | None) -> Principal | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    api_key = authorization.removeprefix("Bearer ").strip()
    if not api_key:
        return None
    # Deliberately reload on every attempt so deleting or rotating a key takes
    # effect without restarting the API process.
    return authenticate_api_key(api_key, load_api_key_records())


def create_app(*, campaign_db: str | Path | None = None):
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field
        from starlette.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - exercised by installed-server smoke tests
        raise RuntimeError('Install the server extra with: pip install "assayready[server]"') from exc

    app = FastAPI(title="AssayReady API", version="1.0.0")
    store = CampaignStore(campaign_db)
    load_api_key_records()
    validate_trusted_proxy_configuration()
    allowed_origins = _allowed_origins_from_environment()
    rate_limit = _rate_limit_from_environment()
    limiter = SlidingWindowRateLimiter(rate_limit)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["Retry-After", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )

    @app.middleware("http")
    async def browser_and_rate_limit_guards(request: Request, call_next):
        origin = request.headers.get("origin", "").strip()
        fetch_site = request.headers.get("sec-fetch-site", "").strip().lower()
        if (origin and not _origin_is_allowed(origin, str(request.url), allowed_origins)) or (
            fetch_site == "cross-site" and not origin
        ):
            return JSONResponse(status_code=403, content={"detail": "cross-origin request rejected"})

        remaining: int | None = None
        if request.url.path.startswith("/v1/") and request.method != "OPTIONS":
            client_host = request.client.host if request.client else "unknown"
            accepted, remaining, retry_after = limiter.check(client_host)
            if not accepted:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "rate limit exceeded"},
                    headers={
                        "Retry-After": str(max(1, int(retry_after + 0.999))),
                        "X-RateLimit-Limit": str(rate_limit),
                        "X-RateLimit-Remaining": "0",
                    },
                )

        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        if remaining is not None:
            response.headers["X-RateLimit-Limit"] = str(rate_limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    class CampaignRequest(BaseModel):
        campaign_id: str
        name: str
        assay_type: str
        objective_direction: str = "maximize"
        owner: str
        metadata: dict[str, Any] = Field(default_factory=dict)

    class RoundRequest(BaseModel):
        round_number: int
        model_version: str
        dataset_version: str
        selection_path: str
        id_column: str = "candidate_id"
        prediction_column: str = "prediction"
        uncertainty_column: str | None = "uncertainty"
        family_column: str | None = None
        run_id: str | None = None
        selected_at: str | None = None

    class OutcomeRequest(BaseModel):
        outcome_path: str
        id_column: str = "candidate_id"
        value_column: str = "measured_value"
        replicate_column: str | None = "replicate"
        batch_column: str | None = "batch"
        cost_column: str | None = "cost"
        measured_at: str | None = None

    def integration_path(raw: str) -> Path:
        configured = os.environ.get("ASSAYREADY_INTEGRATION_ROOT", "").strip()
        if not configured:
            raise HTTPException(status_code=503, detail="ASSAYREADY_INTEGRATION_ROOT is not configured")
        root = Path(configured).resolve()
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise HTTPException(status_code=400, detail="integration file must exist within the configured root")
        return candidate

    def principal_for_request(request: Request, authorization: str | None = Header(default=None)) -> Principal:
        client_host = request.client.host if request.client else None
        proxy = trusted_proxy_principal(request.headers, client_host)
        if proxy:
            return proxy
        try:
            principal = _authenticate_bearer(authorization)
        except (OSError, ValueError, RuntimeError):
            LOGGER.exception("API key configuration could not be loaded")
            raise HTTPException(status_code=503, detail="API key configuration unavailable")
        if principal:
            return principal
        raise HTTPException(status_code=401, detail="valid bearer API key or trusted proxy identity required")

    def authorized(permission: str):
        def dependency(principal: Principal = Depends(principal_for_request)) -> Principal:
            try:
                require_permission(principal, permission)
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail="permission denied") from exc
            return principal
        return dependency

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "authentication_required": True}

    @app.get("/v1/security/posture")
    def get_security_posture(_: Principal = Depends(authorized("campaign:read"))) -> dict[str, Any]:
        return security_posture()

    @app.get("/v1/runs")
    def runs(limit: int = 50, _: Principal = Depends(authorized("run:read"))) -> list[dict[str, Any]]:
        return list_runs(limit=min(max(limit, 1), 500))

    @app.get("/v1/runs/{run_id}")
    def run(run_id: str, _: Principal = Depends(authorized("run:read"))) -> dict[str, Any]:
        summary = get_run_summary(run_id)
        if not summary:
            raise HTTPException(status_code=404, detail="run not found")
        return summary

    @app.get("/v1/campaigns")
    def campaigns(_: Principal = Depends(authorized("campaign:read"))) -> list[dict[str, Any]]:
        return store.list_campaigns()

    @app.get("/v1/campaigns/{campaign_id}")
    def campaign(campaign_id: str, _: Principal = Depends(authorized("campaign:read"))) -> dict[str, Any]:
        try:
            return store.export_campaign(campaign_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="campaign not found") from exc

    @app.post("/v1/campaigns", status_code=201)
    def create_campaign(
        payload: CampaignRequest,
        principal: Principal = Depends(authorized("campaign:write")),
    ) -> dict[str, Any]:
        try:
            return store.create_campaign(**payload.model_dump(), actor=principal.subject)
        except Exception as exc:
            LOGGER.warning("campaign creation rejected", exc_info=True)
            raise HTTPException(status_code=400, detail="campaign could not be created") from exc

    @app.get("/v1/campaigns/{campaign_id}/comparison")
    def comparison(campaign_id: str, _: Principal = Depends(authorized("campaign:read"))) -> dict[str, Any]:
        try:
            return store.compare_rounds(campaign_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="campaign not found") from exc

    @app.post("/v1/campaigns/{campaign_id}/rounds", status_code=201)
    def create_round(
        campaign_id: str,
        payload: RoundRequest,
        principal: Principal = Depends(authorized("round:write")),
    ) -> dict[str, Any]:
        values = payload.model_dump()
        values["selection_file"] = integration_path(values.pop("selection_path"))
        try:
            return store.register_round(campaign_id=campaign_id, **values, actor=principal.subject)
        except Exception as exc:
            LOGGER.warning("round creation rejected", exc_info=True)
            raise HTTPException(status_code=400, detail="round could not be created") from exc

    @app.post("/v1/rounds/{round_id}/outcomes")
    def import_outcomes(
        round_id: str,
        payload: OutcomeRequest,
        principal: Principal = Depends(authorized("outcome:write")),
    ) -> dict[str, Any]:
        values = payload.model_dump()
        values["outcome_file"] = integration_path(values.pop("outcome_path"))
        try:
            return store.import_outcomes(round_id=round_id, **values, actor=principal.subject)
        except Exception as exc:
            LOGGER.warning("outcome import rejected", exc_info=True)
            raise HTTPException(status_code=400, detail="outcomes could not be imported") from exc

    @app.post("/v1/campaigns/{campaign_id}/archive")
    def archive_campaign(
        campaign_id: str, principal: Principal = Depends(authorized("campaign:write"))
    ) -> dict[str, Any]:
        try:
            return store.archive_campaign(campaign_id, actor=principal.subject)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="campaign not found") from exc

    @app.delete("/v1/campaigns/{campaign_id}")
    def purge_campaign(
        campaign_id: str,
        confirm: str,
        principal: Principal = Depends(authorized("campaign:purge")),
    ) -> dict[str, Any]:
        try:
            return store.purge_campaign(campaign_id, confirmation=confirm, actor=principal.subject)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="campaign purge request rejected") from exc

    return app


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit('Install the server extra with: pip install "assayready[server]"') from exc
    host = os.environ.get("ASSAYREADY_API_HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "localhost", "::1"} and not os.environ.get("ASSAYREADY_ALLOW_REMOTE_API"):
        raise SystemExit("Remote API binding requires ASSAYREADY_ALLOW_REMOTE_API=1 and a TLS/auth reverse proxy.")
    uvicorn.run(
        create_app(),
        host=host,
        port=int(os.environ.get("ASSAYREADY_API_PORT", "8060")),
        proxy_headers=False,
    )
