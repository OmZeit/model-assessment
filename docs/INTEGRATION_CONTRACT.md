# Integration contract

AssayReady's integration boundary is deliberately vendor-neutral so the same
workflow can connect to an ELN, LIMS, object store, or model registry without
changing scientific semantics.

## Inputs

- Locked model input: CSV with a unique candidate identifier and sequence.
- Predictions: the same identifier plus a numeric prediction and optional
  declared uncertainty.
- Candidate selection: unique candidate identifier, rank, prediction,
  uncertainty, sequence hash, and any family, cost, or constraint columns.
- Outcomes: candidate identifier, measured value, replicate, batch, timestamp,
  cost, and assay-specific metadata.

Column names are configurable at CLI boundaries. Original input rows are
retained as JSON metadata in the campaign store so adapter-specific fields are
not discarded.

## Outputs

`assayready campaign export` produces a JSON package containing campaign
identity, model and dataset versions, per-round results, policy and selection
digests, and the audit trail. The optional authenticated REST API exposes:

- `GET /v1/runs`
- `GET /v1/runs/{run_id}`
- `GET /v1/campaigns`
- `GET /v1/campaigns/{campaign_id}`
- `GET /v1/campaigns/{campaign_id}/comparison`
- `POST /v1/campaigns`
- `POST /v1/campaigns/{campaign_id}/rounds`
- `POST /v1/rounds/{round_id}/outcomes`
- `POST /v1/campaigns/{campaign_id}/archive`
- `DELETE /v1/campaigns/{campaign_id}` (admin-only, exact confirmation required)

File-based write endpoints resolve paths only beneath
`ASSAYREADY_INTEGRATION_ROOT`. An ELN/LIMS connector should deposit an immutable
export there, call the API with the relative path, and retain the resulting
campaign or round identifier. This prevents the API from becoming an arbitrary
host-file reader.

Specific vendor adapters should translate records at this boundary and retain
the upstream record ID and revision in metadata. They should use idempotency
keys, reject silent unit conversion, and preserve the AssayReady hashes. Direct
Benchling/LIMS credentials are intentionally not embedded in the core package.
