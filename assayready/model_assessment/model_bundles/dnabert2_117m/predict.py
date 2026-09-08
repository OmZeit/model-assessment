from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from runtime import (
    choose_device,
    embed_sequences,
    load_frozen_backbone,
    normalize_sequences,
    sha256_file,
    verify_model_manifest,
)


MODEL_DIR = Path("/model")
HEAD_DIR = Path("/head")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: predict.py INPUT.csv OUTPUT.csv")
    input_path, output_path = Path(sys.argv[1]), Path(sys.argv[2])
    model_manifest = verify_model_manifest(MODEL_DIR)
    head_manifest = json.loads((HEAD_DIR / "head_manifest.json").read_text(encoding="utf-8"))
    head_path = HEAD_DIR / "head.npz"
    if sha256_file(head_path) != head_manifest.get("head_sha256"):
        raise ValueError("task-head checksum does not match head_manifest.json.")
    for field in ("model_revision", "model_weights_sha256"):
        model_field = "revision" if field == "model_revision" else "weights_sha256"
        if head_manifest.get(field) != model_manifest.get(model_field):
            raise ValueError(f"task head and frozen model disagree on {field}.")

    frame = pd.read_csv(input_path)
    sequence_col = str(head_manifest["sequence_col"])
    if sequence_col not in frame.columns:
        raise ValueError(f"input table is missing sequence column {sequence_col!r}.")
    sequences = normalize_sequences(frame[sequence_col])
    device = choose_device("auto")
    tokenizer, model = load_frozen_backbone(MODEL_DIR, device)
    embeddings = embed_sequences(
        sequences, tokenizer=tokenizer, model=model, device=device,
        batch_size=32, max_length=int(head_manifest["max_length"]),
    )
    with np.load(head_path, allow_pickle=False) as head:
        scores = embeddings @ head["coefficients"].T + head["intercepts"][None, :]
    if head_manifest["task_type"] == "classification":
        scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -40, 40)))
    frame["model_prediction"] = scores.mean(axis=1)
    frame["model_uncertainty"] = scores.std(axis=1, ddof=1)
    frame.to_csv(output_path, index=False)
    print(
        json.dumps(
            {"rows": len(frame), "device": str(device), "output": str(output_path),
             "model_revision": model_manifest["revision"]},
            sort_keys=True,
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
