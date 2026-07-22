import json
from pathlib import Path
from model_assessment.cli import _load_config, _prediction_params_from_config


def test_example_configs_load_without_error() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent / "model_assessment"
    examples_dir = project_root / "examples"

    for conf_name in ["public_dream_promoter_audit.json", "prediction_audit.json"]:
        conf_path = examples_dir / conf_name
        config = _load_config(conf_path)
        params = _prediction_params_from_config(config, config_dir=examples_dir)

        assert "project" in params
        assert "output_dir" in params
        assert "task_type" in params
        assert "sequence_col" in params
        assert "target_col" in params
