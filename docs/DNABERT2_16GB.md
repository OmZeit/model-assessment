# DNABERT-2 on a 16 GB GPU

AssayReady's default open DNA bundle uses
[`zhihan1996/DNABERT-2-117M`](https://huggingface.co/zhihan1996/DNABERT-2-117M).
The official ICLR 2024 project reports strong results across its multi-species
Genome Understanding Evaluation while using 117 million parameters. The model
and official implementation are Apache-2.0. AssayReady pins the converted
Safetensors snapshot at commit
`a3d38b3f41cec05e370a4d3eeb8664fcb4bce227` and verifies the 468 MB weights as
SHA-256 `bc91ac0d972a698b7ff12ea6815e966b1feff9d7d1bef3a10535f2f4332609ac`.

This is a practical default for the short DNA/promoter workflow, not a claim
that one foundation model is best for every organism, assay, or sequence
length. The model has ample inference and frozen-head-training headroom on a
16 GB GPU. Measure it against AssayReady's simple baselines before adopting it.

## 1. Create and download the bundle

The command downloads about 470 MB and verifies the exact Safetensors,
configuration, tokenizer, and executable custom-model files. It is safe to
rerun after a completed setup or interrupted download; unknown non-empty
directories are still refused.

```bash
assayready model prepare-dnabert2 --output-dir model_bundles_local/dnabert2
cd model_bundles_local/dnabert2
```

Downloaded remote Python files are also hashed in
`weights/assayready_model_manifest.json`. They are pinned but still deserve
code review before production because DNABERT-2 requires custom model code.

## 2. Produce an independent training partition

Use a previously locked, similarity-separated AssayReady training partition.
Never train on validation, test, candidate, or future-outcome rows. One way to
create the partition is an ordinary AssayReady run with the same group and
similarity policy that will govern the audit:

```bash
assayready run --assay measured_assay.csv \
  --sequence-col sequence --target-col measured_activity \
  --group-cols construct_family batch --project dnabert2_split
```

Use only that run's `processed/train.csv` in the next step. Record the run ID,
split hash, and model/head manifests in the later evaluation configuration.

## 3. Build the training image

```bash
docker build -t assayready-dnabert2:trainer .
```

The base is PyTorch 2.7.1 with CUDA 12.8 and cuDNN 9, pinned to its
Linux/amd64 image digest. CUDA 12.8 is required for the RTX 50-series
Blackwell `sm_120` architecture. The runtime selects DNABERT-2's standard
PyTorch attention implementation instead of its older optional Triton kernel.
The frozen 117M-parameter backbone is baked into the image and Hugging Face is
forced offline at runtime.

## 4. Train the small assay head

From the bundle directory, mount the locked training CSV and task-head output:

```bash
docker run --rm --gpus device=0 \
  --entrypoint python \
  --mount type=bind,source=/absolute/path/train.csv,target=/data/train.csv,readonly \
  --mount type=bind,source=/absolute/path/task_head,target=/output \
  assayready-dnabert2:trainer \
  /app/train_head.py \
  --model-dir /model --training /data/train.csv --output-dir /output \
  --sequence-col sequence --target-col measured_activity \
  --task-type regression --objective-direction maximize \
  --ensemble-size 8 --device cuda:0
```

For binary classification, use `--task-type classification` and explicitly set
`--positive-label`. Set `--objective-direction minimize` when lower assay values
are preferred. The backbone remains frozen. The fitted artifact is a
bootstrapped linear ensemble saved as non-pickle NumPy arrays. Its standard
deviation is an ensemble-disagreement signal, not a calibrated prediction
interval.

## 5. Build and publish the final immutable image

After `task_head/head.npz` and `head_manifest.json` exist:

```bash
docker build -t registry.example.org/team/assayready-dnabert2:v1 .
docker push registry.example.org/team/assayready-dnabert2:v1
docker inspect --format='{{index .RepoDigests 0}}' \
  registry.example.org/team/assayready-dnabert2:v1
```

To retain multiple assay-specific heads in one local bundle, store them in
separate directories and select one at build time:

```bash
docker build --build-arg TASK_HEAD_DIR=task_heads/gpd \
  -t assayready-dnabert2:gpd-regression-v1 .
```

Copy the resulting `repository@sha256:...` value into
`controlled_execution.json`. Tags such as `latest` are rejected.

## 6. Execute and audit

The input must have unique `sequence_id` and `sequence` columns. It may retain a
measured target column; prediction preserves every input column and adds
`model_prediction` and `model_uncertainty`.

```bash
assayready model execute --spec controlled_execution.json

assayready audit-predictions \
  --assay controlled_output/predictions.csv \
  --sequence-col sequence --target-col measured_activity \
  --prediction-col model_prediction --uncertainty-col model_uncertainty \
  --id-col sequence_id
```

Run candidate-only tables through the same immutable image, then use their
predictions as the prospective candidate input to the audited campaign.

## Use another model

The Design Sandbox's direct in-process scorer is deliberately narrow: it loads
only the authenticated DNABERT-2 snapshot above plus a compatible AssayReady
head. It does not execute an arbitrary Hugging Face repository, pickle, or
user-supplied Python adapter.

For another model, use either of these supported paths:

1. Run the model in its own environment and export the original row identifier,
   sequence, `model_prediction`, and optional `model_uncertainty`. Join those
   predictions to measured outcomes and choose **Analyze > Audit existing
   predictions**.
2. Package the model behind `controlled_execution.json`, pin the container by
   digest, and run `assayready model execute --spec controlled_execution.json`.
   Audit the resulting prediction table in the same way.

In both cases, declare objective direction, positive class for classification,
uncertainty meaning, model/data identifiers, and the independence of the
evaluation set. These declarations preserve interpretation and provenance; they
do not by themselves prove that the model was trained independently.

## Operational limits

- On Windows, Docker GPU access requires Docker Desktop's WSL 2 backend, a
  current WSL kernel, and a current NVIDIA Windows driver.
- AssayReady selects one GPU but cannot enforce a hard per-process VRAM quota;
  monitor the worker and keep it isolated from other GPU workloads.
- Very long sequences increase attention memory quadratically. The bundle
  defaults to 512 tokens and a batch size of 32; lower the batch size before
  increasing context.
- Commercially permissive licensing does not prove biological suitability or
  freedom to operate. Retain model attribution and obtain legal review for the
  intended distribution model.
