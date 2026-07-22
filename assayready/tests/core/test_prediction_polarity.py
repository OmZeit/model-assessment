from __future__ import annotations

import pytest

from model_assessment.cli import _prediction_metrics


def test_prediction_metrics_respects_explicit_positive_label() -> None:
    rows = [
        {"target": "active", "prediction": 0.9},
        {"target": "inactive", "prediction": 0.1},
    ]

    active_positive = _prediction_metrics(
        rows,
        task_type="classification",
        target_col="target",
        prediction_col="prediction",
        positive_label="active",
    )
    inactive_positive = _prediction_metrics(
        rows,
        task_type="classification",
        target_col="target",
        prediction_col="prediction",
        positive_label="inactive",
    )

    assert active_positive["accuracy"] == 1.0
    assert inactive_positive["accuracy"] == 0.0
    assert active_positive["positive_label"] == "active"
    assert inactive_positive["positive_label"] == "inactive"


def test_prediction_metrics_rejects_unknown_positive_label() -> None:
    rows = [
        {"target": "yes", "prediction": 0.8},
        {"target": "no", "prediction": 0.2},
    ]
    with pytest.raises(ValueError, match="positive_label"):
        _prediction_metrics(
            rows,
            task_type="classification",
            target_col="target",
            prediction_col="prediction",
            positive_label="maybe",
        )


def test_single_class_subset_is_reported_as_not_evaluable() -> None:
    metrics = _prediction_metrics(
        [{"target": "inactive", "prediction": 0.2}],
        task_type="classification",
        target_col="target",
        prediction_col="prediction",
        positive_label="active",
    )
    assert metrics["status"] == "not_evaluable_single_class"
    assert metrics["primary_metric"] is None
