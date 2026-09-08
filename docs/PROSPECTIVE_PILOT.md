# Prospective pilot protocol

This template is for testing whether AssayReady improves an experimental
decision. It is not evidence that a prospective study has already succeeded.

## Decision and hypotheses

Predeclare one decision: which candidates enter the next experimental round.
Choose a primary endpoint before measurements exist, such as hit rate above a
scientifically meaningful threshold, enrichment relative to the candidate
pool, or cost per confirmed improvement. Define the meaningful effect size and
assay noise from prior rounds; do not choose them after seeing the result.

## Recommended comparison

Use three non-overlapping, randomized candidate arms when capacity permits:

1. The team's normal selection process.
2. The highest raw model predictions.
3. AssayReady's uncertainty- and diversity-aware constrained selection.

Include shared positive, negative, blank, and process controls. Randomize wells
within batch, balance arms across plates, and blind analysts to arm labels until
the outcome table is locked. Replicate a predeclared subset to estimate assay
noise. Record exclusions without replacing failed candidates after unblinding.

## Lock before measurement

Before wet-lab work begins, create the campaign and round. The `add-round`
command hashes the exact selection file and records its timestamp.

```bash
assayready campaign create \
  --campaign-id enzyme-pilot --name "Enzyme pilot" \
  --assay-type activity --objective-direction maximize --owner protein-team

assayready campaign add-round \
  --campaign-id enzyme-pilot --round-number 1 \
  --model-version model-v7 --dataset-version dataset-v4 \
  --selection candidate_prioritization.csv \
  --policy-pack protein_activity_v1
```

Archive the model digest, dataset digest, policy digest, randomization seed,
arm assignment, sample-size calculation, inclusion/exclusion rules, and primary
analysis alongside the selection manifest. A Git commit alone is not a data or
model digest.

## Import and analyze

The outcome file may contain repeated candidate IDs when a `replicate` column
distinguishes measurements.

```bash
assayready campaign import-outcomes \
  --round-id enzyme-pilot-r001 --outcomes measured_outcomes.csv

assayready campaign compare --campaign-id enzyme-pilot
assayready campaign protocol --round-id enzyme-pilot-r001 \
  --output prospective_protocol.json
```

Report all predeclared endpoints, uncertainty intervals, missingness, assay
failures, batch effects, and protocol deviations. The built-in comparison is
descriptive; use an assay-appropriate statistical model for confirmatory
inference.

## Commercial pilot success criteria

In addition to scientific endpoints, record whether the customer:

- uses the report in an actual candidate-review meeting;
- asks to connect the next model or experimental round;
- requests an ELN/LIMS or model-registry integration;
- pays for the next round without a bespoke rescoping exercise.

Three paid retrospective pilots and at least one prospective comparison are a
reasonable evidence threshold before treating annual-license pricing as
validated.
