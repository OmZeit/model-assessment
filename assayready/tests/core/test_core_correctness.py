from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from model_assessment.data_core.config import DnaConfig
from model_assessment.ml_core.adapters import HuggingFaceAdapter, NativeAdapter
from model_assessment.ml_core.specialization import (
    _primary_metric,
    _primary_metric_name,
    classification_metrics,
    encode_class_labels,
    fit_task_head_ensemble,
    leakage_safe_split,
    ranking_metrics,
    normalize_sequence,
    run_baselines,
)


def test_scoped_modules_import_cleanly() -> None:
    modules = [
        "model_assessment.data_core.dataset",
        "model_assessment.data_core.loaders",
        "model_assessment.data_core.weighted_loader",
        "model_assessment.ml_core.masking",
        "model_assessment.ml_core.backbones",
        "model_assessment.ml_core.model",
        "model_assessment.ml_core.adapters",
    ]
    for module in modules:
        importlib.import_module(module)


def test_native_adapter_uses_current_model_api() -> None:
    config = DnaConfig(
        max_input_bases=16,
        hidden_size=16,
        num_attention_heads=4,
        high_level_layers=1,
        vocab_size=16,
        use_kmer_mix=False,
        dropout=0.0,
        backbone="transformer",
    )
    adapter = NativeAdapter(config)
    hidden = adapter(
        torch.tensor([[2, 5, 6, 3]], dtype=torch.long),
        torch.ones((1, 4), dtype=torch.long),
    )
    assert hidden.shape == (1, 4, 16)
    assert adapter.get_hidden_size() == 16


def test_huggingface_adapter_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match="optional 'transformers'"):
        HuggingFaceAdapter("unused")


def test_leakage_split_uses_any_group_column_and_ignores_missing_groups() -> None:
    rows = [
        {"id": "a", "sequence": "AAAAACCC", "donor": "d1", "batch": "b1"},
        {"id": "b", "sequence": "CCCCCGGG", "donor": "d1", "batch": "b2"},
        {"id": "c", "sequence": "GGGGGTTT", "donor": "d2", "batch": "b2"},
        {"id": "d", "sequence": "TTTTTAAA", "donor": "d3", "batch": "b3"},
        {"id": "e", "sequence": "ACACACAC", "donor": None, "batch": None},
        {"id": "f", "sequence": "GTGTGTGT", "donor": "", "batch": ""},
    ]
    splits, diagnostics = leakage_safe_split(
        rows,
        sequence_col="sequence",
        val_fraction=0.0,
        test_fraction=0.0,
        group_cols=["donor", "batch"],
        homology_threshold=0.0,
    )
    clusters = {row["id"]: row["leakage_cluster"] for row in splits["train"]}
    assert clusters["a"] == clusters["b"] == clusters["c"]
    assert clusters["e"] != clusters["f"]
    assert diagnostics["num_clusters"] == 4


def test_leakage_split_is_seeded_and_row_order_invariant() -> None:
    rows = [
        {"id": str(idx), "sequence": f"ACGT{idx:08b}".replace("0", "A").replace("1", "C")}
        for idx in range(12)
    ]
    first, _ = leakage_safe_split(
        rows,
        sequence_col="sequence",
        val_fraction=0.2,
        test_fraction=0.2,
        homology_threshold=0.0,
        seed=91,
    )
    second, _ = leakage_safe_split(
        list(reversed(rows)),
        sequence_col="sequence",
        val_fraction=0.2,
        test_fraction=0.2,
        homology_threshold=0.0,
        seed=91,
    )
    assignment = lambda split: {
        row["id"]: name for name, values in split.items() for row in values
    }
    assert assignment(first) == assignment(second)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"val_fraction": -0.1},
        {"test_fraction": 1.0},
        {"val_fraction": 0.6, "test_fraction": 0.4},
        {"homology_threshold": 1.1},
        {"homology_k": 0},
    ],
)
def test_leakage_split_validates_configuration(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        leakage_safe_split([{"sequence": "ACGT"}], sequence_col="sequence", **kwargs)


def test_explicit_positive_label_controls_class_one() -> None:
    encoded, mapping = encode_class_labels(
        ["active", "inactive", "active"],
        positive_label="inactive",
    )
    assert mapping == {"active": 0, "inactive": 1}
    assert encoded.tolist() == [0, 1, 0]

def test_primary_metrics_match_task_semantics() -> None:
    classification = classification_metrics([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.9])
    assert _primary_metric_name("classification", classification) == "auroc"
    assert _primary_metric("classification", classification) == classification["auroc"]

    one_class = classification_metrics([1, 1], [0.9, 0.1])
    assert _primary_metric_name("classification", one_class) == "balanced_accuracy"
    assert _primary_metric("classification", one_class) == one_class["balanced_accuracy"]

    ranking = ranking_metrics([3.0, 1.0, 2.0], [0.9, 0.1, 0.5])
    assert _primary_metric_name("ranking", ranking) == "spearman"
    assert _primary_metric("ranking", ranking) == ranking["spearman"]


def test_kmer4_baseline_is_not_labeled_as_a_frozen_embedding() -> None:
    train = [
        {"sequence": "ACGTACGT" + base, "target": float(index)}
        for index, base in enumerate("ACGTAC")
    ]
    test = [
        {"sequence": "TGCATGCA" + base, "target": float(index)}
        for index, base in enumerate("GT")
    ]
    baselines = run_baselines(
        train,
        test,
        task_type="regression",
        sequence_col="sequence",
        target_col="target",
    )
    assert "kmer4_linear_probe" in baselines
    assert "frozen_embedding_linear_probe" not in baselines


def test_task_head_persists_explicit_positive_label() -> None:
    rows = [
        {"sequence": "ACGTACGT" + base, "target": label}
        for base, label in zip("ACGTACGT", ["active", "inactive"] * 4)
    ]
    ensemble = fit_task_head_ensemble(
        rows[:6],
        rows[6:7],
        rows[7:],
        task_type="classification",
        sequence_col="sequence",
        target_col="target",
        ensemble_size=1,
        epochs=1,
        positive_label="inactive",
    )
    assert ensemble["class_mapping"] == {"active": 0, "inactive": 1}
    assert ensemble["positive_label"] == "inactive"


def test_specialization_normalize_sequence_rejects_nan() -> None:
    assert normalize_sequence(np.nan) == ""
    assert normalize_sequence(0) == ""
