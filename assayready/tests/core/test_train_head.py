from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from model_assessment.model_bundles.dnabert2_117m import train_head


def test_train_frozen_head_library_api_persists_a_compatible_manifest(tmp_path: Path, monkeypatch) -> None:
    training = tmp_path / "training.csv"
    pd.DataFrame(
        {
            "sequence": ["ACGT" + "A" * (index % 4) for index in range(24)],
            "activity": [float(index) for index in range(24)],
        }
    ).to_csv(training, index=False)
    model_dir = tmp_path / "weights"
    model_dir.mkdir()

    def fake_embeddings(sequences, **_kwargs):
        return np.asarray(
            [[float(index), float(index % 3), 1.0] for index, _sequence in enumerate(sequences)],
            dtype=np.float32,
        )

    monkeypatch.setattr(
        train_head,
        "_runtime_functions",
        lambda: (
            lambda _device: "cpu",
            fake_embeddings,
            lambda _model_dir, _device: (object(), object()),
            lambda values: [str(value) for value in values],
            lambda _path: "a" * 64,
            lambda _model_dir: {
                "repository": "example/repository",
                "revision": "fixed-revision",
                "weights_sha256": "b" * 64,
            },
        ),
    )

    output = tmp_path / "head"
    manifest = train_head.train_frozen_head(
        model_dir=model_dir,
        training=training,
        output_dir=output,
        target_col="activity",
        objective_direction="minimize",
        ensemble_size=3,
    )

    assert manifest["objective_direction"] == "minimize"
    assert manifest["training_rows"] == 24
    assert manifest["ensemble_size"] == 3
    assert (output / "head.npz").is_file()
    assert (output / "head_manifest.json").is_file()
