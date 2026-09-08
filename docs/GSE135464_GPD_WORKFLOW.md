# GSE135464 public wet-lab promoter workflow

## Scope and endpoint

This workflow imports the public `GSE135464` yeast promoter measurements and
defines a constitutive GPD-promoter **regression** task. The input is the
313-base DNA sequence and the target is `measured_activity`, defined as the
mean of replicate A and B log10(GFP:mCherry) measurements.

The importer applies the study's replicate-consistency and published activity
range filters, rejects ambiguous 35-base sequence joins, and assigns rows to
train, validation, or test from a SHA-256 sequence hash before sampling. This
keeps the test selection independent of row order and model fitting.

ZEV data are deliberately excluded from this first task. Its A and B columns
represent uninduced and induced conditions rather than technical replicates,
so they require separate condition-specific targets or a multi-output model.

## Reproduce the prepared dataset

From the repository root:

```powershell
python scripts\prepare_gse135464_gpd.py
```

The command downloads the five official GEO supplementary files, verifies
their pinned SHA-256 checksums, and writes git-ignored artifacts under
`datasets/GSE135464`. The current cohort contains 675,380 accepted sequences;
the prepared benchmark samples 50,000 train, 10,000 validation, and 10,000
test rows. Exact source, filter, split, and output hashes are recorded in
`datasets/GSE135464/dataset_manifest.json`.

Only `prepared/gpd_train.csv` may be used to fit the assay head. Keep
`gpd_validation.csv` for development decisions and `gpd_test.csv` locked for
the final evaluation.

## Current 16 GB GPU model

The pilot freezes the pinned DNABERT-2 117M backbone and fits an eight-member
ridge ensemble on the 50,000-row training sample. Its task head is stored at
`model_bundles_local/dnabert2/task_heads/gpd`; the separate DREAM task head is
retained at `task_heads/dream`.

The immutable GPD inference image is:

```text
assayready-dnabert2@sha256:32ac00a296441f631e61e7ccfd20904dcad8b76f559e0e4104c9560184822fc2
```

It was built with:

```powershell
docker build --build-arg TASK_HEAD_DIR=task_heads/gpd `
  -t assayready-dnabert2:gpd-regression-v1 `
  model_bundles_local\dnabert2
```

Run the locked test specification through the controlled, network-disabled
executor:

```powershell
assayready model execute `
  --spec outputs\gse135464_gpd_test_spec.json
```

## Pilot results

| Evaluation | Rows | R2 | RMSE | MAE | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 10,000 | 0.136 | 0.182 | 0.147 | 0.371 | 0.367 |
| Locked test | 10,000 | 0.130 | 0.182 | 0.146 | 0.362 | 0.355 |

The replicate measurements themselves agree strongly on the test sample
(replicate R2 0.937), so the weak model result is not explained primarily by
assay noise. On the full 50,000-to-10,000 comparison, the best simple k-mer
ridge result was R2 0.057. DNABERT-2 therefore adds measurable sequence signal,
but the frozen linear head materially underfits this assay.

The provenance-verified audit sample produced model R2 0.115 versus R2 0.037
for its best simple baseline. The paired model-minus-baseline R2 interval was
[0.008, 0.151]. That clears the declared retrospective lift check, but the
conclusion changes under stricter minimum-lift policies. Ensemble disagreement
also had Spearman -0.076 with absolute error, below the 0.2 usability threshold.
Consequently, the audit is descriptive only: do not use its uncertainty value
to prioritize experiments and do not claim prospective biological utility.

## Recommended next model iteration

Use the same locked split and compare either a task-trained convolutional
promoter model or parameter-efficient DNABERT-2 fine-tuning against this frozen
head. Choose on validation, then evaluate the selected configuration once on
the locked test partition. Preserve the k-mer and GC/length baselines and add
family/similarity exclusion before making a biological generalization claim.

The study's reported task-trained convolutional models are an important
reference point, but their performance is not directly attributable to this
split or implementation. A local reproduction is required before making a
head-to-head claim.
