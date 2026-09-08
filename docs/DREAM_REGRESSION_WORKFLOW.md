# DREAM promoter regression workflow

## Decision

Use **regression**, with the source `activity` field mapped to
`measured_activity`. The product ranks biological sequences by a continuous
experimental outcome, so thresholding this signal into classes would discard
magnitude and ordering information. Classification remains appropriate only
for a future assay whose real endpoint is categorical (for example, pass/fail
toxicity or binder/non-binder).

## Pinned public data

The acquisition script uses
`HuggingFaceBio/random-promoter-dream-2022` at revision
`095ecf8da1e69d70c21812af809df7b19c7d8a84`. This is a CC BY 4.0 repackaging
of Random Promoter DREAM Challenge 2022 data from Zenodo DOI
`10.5281/zenodo.10633252`. Every Parquet shard is checked against its expected
SHA-256 before use.

Run:

```powershell
python scripts\prepare_dream_promoter_dataset.py
```

The local, git-ignored dataset lives under
`datasets/random-promoter-dream-2022`. The prepared pilot uses a deterministic
uniform sample of 50,000 official training rows, 10,000 validation rows, and
10,000 labeled designed-promoter test rows. The full pinned source shards are
retained locally. Only `prepared/train.csv` may be used to fit the head.

Use `prepared/validation.csv` for the primary same-scale evaluation. The
labeled designed-promoter test file contains MAUDE expression values on a
different scale from train/validation; treat it as an external transfer set,
not as a directly pooled regression benchmark, unless a calibration protocol
is defined and locked first.

## Docker/WSL status

Docker Desktop is installed per-user with the WSL 2 backend and Windows
containers disabled. `scripts/enable_wsl2.ps1` enables only the WSL and Virtual
Machine Platform optional features and writes its result to
`outputs/setup/wsl2-status.json`. Windows must be restarted after first-time
feature enablement. After restart, update WSL, start Docker Desktop, review and
accept Docker's terms in the UI, then verify GPU access before training.

```powershell
wsl --update
wsl --set-default-version 2
docker version
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

Then follow `docs/DNABERT2_16GB.md`, using
`datasets/random-promoter-dream-2022/prepared/train.csv` as the training input
and `prepared/validation.csv` as the locked evaluation input.

## Local pilot result

The RTX 5080 run uses PyTorch 2.7.1 with CUDA 12.8; the earlier PyTorch 2.2
base could enumerate the GPU but could not execute its Blackwell `sm_120`
kernels. A real CUDA tensor test and DNABERT-2 forward pass now pass.

The frozen DNABERT-2 backbone was fitted with an eight-member ridge head on
50,000 official training rows. The resulting local image is
`assayready-dnabert2@sha256:bf1a5b38f0bf0920bb60f5dd0a2dcda2d58dccbe273c5fd1fe26ee96828fc740`.
On 10,000 untouched official validation rows it produced R2 0.327, RMSE
1.934, Pearson 0.573, and Spearman 0.589. On the later 1,500-row
similarity-held audit input, R2 was 0.335.

This is useful signal, but it is not yet evidence that DNABERT-2 beats the
simple sequence baselines: the audit's model-minus-best-baseline R2 lift was
about 0.015 and its paired confidence interval crossed zero. Ensemble
disagreement also did not correlate with absolute error, so the candidate run
sets acquisition beta to zero and does not reward that uncertainty value.

The pinned unlabeled challenge pool was scored through the network-disabled,
read-only image. The latest audit writes 96 review candidates to
`outputs/assayready/dnabert2_dream_candidate_ranking/<run-id>/ranked_candidates.csv`.
These are computational nominations, not experimentally validated promoter
designs.
