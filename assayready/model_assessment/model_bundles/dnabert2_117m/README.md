# AssayReady DNABERT-2 117M bundle

This bundle uses the Apache-2.0 DNABERT-2 117M backbone pinned to an immutable
Hugging Face Safetensors revision. The backbone is frozen. A small bootstrapped
linear head learns the assay-specific target and supplies ensemble disagreement
as an uncertainty signal.

1. Download weights with `assayready model download-dnabert2 --output-dir weights`.
2. Fit only on an AssayReady training partition:

   `python train_head.py --model-dir weights --training train.csv --output-dir task_head --target-col measured_activity`

3. Build and push the image, then replace the image digest in
   `controlled_execution.json`.
4. Run `assayready model execute --spec controlled_execution.json`.

The uncertainty is ensemble disagreement, not a calibrated prediction interval.
The model requires pinned remote Python code, which the downloader hashes; an
operator should still review that code before production deployment.
