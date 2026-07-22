from __future__ import annotations

import json
from importlib import resources


EXPECTED_RESOURCES = [
    "assets/assayready.css",
    "data_core/bpe_vocab/tokenizer.json",
    "examples/prediction_audit.csv",
    "examples/prediction_audit.json",
    "examples/prediction_candidates.csv",
    "examples/public_dream_promoter_audit.json",
    "examples/public_dream_promoter_candidates.csv",
    "examples/public_dream_promoter_predictions.csv",
    "examples/public_dream_promoter_source.md",
]


def test_runtime_resources_are_in_the_importable_package() -> None:
    package_root = resources.files("model_assessment")
    missing = [name for name in EXPECTED_RESOURCES if not package_root.joinpath(name).is_file()]
    assert not missing
    assert "app-shell" in package_root.joinpath("assets/assayready.css").read_text(encoding="utf-8")


def test_bundled_audit_configs_use_sibling_resource_paths() -> None:
    examples = resources.files("model_assessment").joinpath("examples")
    for config_name in ["prediction_audit.json", "public_dream_promoter_audit.json"]:
        config = json.loads(examples.joinpath(config_name).read_text(encoding="utf-8"))
        references = list(config.get("assay_files") or (config.get("ingestion") or {}).get("raw_files") or [])
        references += list(config.get("candidate_files") or (config.get("candidates") or {}).get("raw_files") or [])
        assert references
        for reference in references:
            assert not reference.startswith("model_assessment/examples/")
            assert examples.joinpath(reference).is_file(), f"{config_name} references missing {reference}"


def test_bundled_prediction_examples_disclose_unverified_independence() -> None:
    examples = resources.files("model_assessment").joinpath("examples")
    for config_name in ["prediction_audit.json", "public_dream_promoter_audit.json"]:
        config = json.loads(examples.joinpath(config_name).read_text(encoding="utf-8"))
        evaluation = config["evaluation"]
        assert evaluation["constraints_verified"] is False
        assert evaluation["provenance"]["training_independence"] == "unverified"
        assert evaluation["semantics"]["uncertainty_type"] == "predictive_standard_deviation"
