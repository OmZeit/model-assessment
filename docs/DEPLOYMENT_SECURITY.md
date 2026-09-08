# Deployment and enterprise security

AssayReady is local-first. The Dash development server and the optional API
bind to loopback by default. Production deployments should run behind an
organization-managed TLS reverse proxy or API gateway.

## Authentication and roles

Install the optional server dependencies with `pip install "assayready[server]"`.
The API requires either a bearer API key or explicitly enabled identity headers
from a trusted reverse proxy.

API key files store SHA-256 digests, never plaintext keys:

```json
{
  "keys": [
    {"id": "automation-client", "sha256": "<64-hex-digest>", "role": "scientist"}
  ]
}
```

Set `ASSAYREADY_API_KEYS_FILE` to the absolute path, restrict file permissions,
and rotate keys through the deployment secret manager. Roles are `viewer`,
`scientist`, and `admin`.

For SSO, let a verified reverse proxy authenticate the user and strip any
incoming identity headers before adding `X-AssayReady-User` and
`X-AssayReady-Role`. The proxy must also add
`X-AssayReady-Proxy-Secret`, using a random secret of at least 32 characters.
Configure all of the following before enabling trusted identity headers:

```text
ASSAYREADY_TRUST_PROXY_IDENTITY=true
ASSAYREADY_TRUSTED_PROXIES=127.0.0.1,10.20.0.0/16
ASSAYREADY_PROXY_SECRET=<random-secret-of-at-least-32-characters>
```

`ASSAYREADY_TRUSTED_PROXIES` accepts literal IP addresses and CIDR networks;
hostnames and all-address (`/0`) networks are intentionally unsupported. The
secret header name can be changed with `ASSAYREADY_PROXY_SECRET_HEADER`.
AssayReady disables Uvicorn forwarded-header rewriting so the address check
uses the direct TCP peer. Never expose a trusted-header deployment directly to
clients.

API key records are reloaded on every bearer authentication attempt, so a key
file update or revocation takes effect without restarting the server. API
routes use a per-client sliding-window limit of 120 requests per minute by
default; change it with `ASSAYREADY_API_RATE_LIMIT_PER_MINUTE`. Browser origins
are denied unless they are the API's own origin or are explicitly listed in
the comma-separated `ASSAYREADY_API_ALLOWED_ORIGINS`. Wildcards are rejected.

## Controlled model execution

`assayready model execute` accepts only Docker or Podman images pinned by
SHA-256 digest. The container receives a read-only input mount, a dedicated
output mount, no network, a read-only root filesystem, dropped capabilities,
`no-new-privileges`, and bounded CPU, memory, process count, and wall time.
Only the named input file is mounted, not its parent directory. By default,
outputs must be beneath an `outputs/` directory next to the execution spec.
Operators can set an absolute `ASSAYREADY_CONTROLLED_OUTPUT_ROOT` instead.

This reduces risk but is not a complete hostile-code sandbox. Production
operators should also use a dedicated worker identity, rootless containers,
host-level egress controls, signed-image admission policy, vulnerability
scanning, resource quotas, and an isolated worker host.

## Auditability, backup, and retention

Campaign state is stored in SQLite. Campaign mutations append a hash-chained
audit event, and `assayready campaign verify-audit` checks the chain. This
detects accidental or unsophisticated modification; it is not a substitute for
an externally anchored append-only audit service.

Back up the run database, campaign database, and artifact directories together.
Define organization-specific retention and deletion procedures. No automated
destructive retention job is enabled by default because data ownership and
legal-hold requirements must be explicit.

`campaign archive` is the recoverable default. `campaign purge` permanently
deletes campaign rounds and outcomes only when the caller repeats the exact
campaign ID; the minimal purge audit event remains. The API restricts purge to
the admin role. Backups and external artifacts are separate retention domains
and are not silently deleted by this operation.

## Current boundaries

- No native SAML/OIDC implementation; SSO is delegated to a trusted proxy.
- No tenant isolation claim; deploy separate instances or databases when hard
  tenant boundaries are required.
- No regulated-system certification is claimed.
- No secrets should be placed in configs, candidate tables, or audit details.
