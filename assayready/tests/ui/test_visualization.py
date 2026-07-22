from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from dash import dcc

from model_assessment import ui
from model_assessment.visualization import (
    benchmark_metric_figure,
    classification_discrimination_figure,
    plate_layout_figure,
    regression_fit_figure,
    regression_slice_summary,
    residual_figure,
    simulation_constraints_figure,
    simulation_landscape_figure,
    uncertainty_figure,
)


def _components(component: Any, target_type: type) -> list[Any]:
    found = [component] if isinstance(component, target_type) else []
    children = getattr(component, "children", None)
    if children is None:
        return found
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        found.extend(_components(child, target_type))
    return found


def _pattern_id_types(component: Any) -> set[str]:
    found: set[str] = set()
    component_id = getattr(component, "id", None)
    if isinstance(component_id, dict) and isinstance(component_id.get("type"), str):
        found.add(component_id["type"])
    children = getattr(component, "children", None)
    if children is None:
        return found
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        found.update(_pattern_id_types(child))
    return found


def _component_text(component: Any) -> str:
    if component is None:
        return ""
    if isinstance(component, (str, int, float)):
        return str(component)
    children = getattr(component, "children", None)
    if children is None:
        return ""
    if not isinstance(children, (list, tuple)):
        children = [children]
    return " ".join(_component_text(child) for child in children)


def _benchmark_report() -> dict[str, Any]:
    return {
        "task_type": "regression",
        "primary_metric_name": "r2",
        "test_metrics": {"primary_metric": 0.72, "num_rows": 6},
        "baselines": {
            "gc_length_baseline": {"primary_metric": 0.20},
            "kmer_ridge": {"primary_metric": 0.51},
        },
        "best_simple_baseline": {"name": "kmer_ridge", "primary_metric": 0.51},
        "claim_gate": {
            "lift_claim": {"ok": True, "delta": 0.21, "model_metric_ci": [0.62, 0.79], "reasons": []},
            "leakage_controlled": {"ok": True, "reasons": []},
            "uncertainty_usable": {"ok": False, "reasons": ["semantics undeclared"]},
            "candidate_constraints": {"ok": False, "reasons": ["not verified"]},
            "recommended": {"ok": False, "reasons": ["no candidates"]},
        },
        "uncertainty_audit": {
            "declared_type": "none",
            "bins": [{"bin": 1, "count": 3, "mean_uncertainty": 0.1, "mean_abs_error": 0.15}],
        },
        "evaluation_manifest": {
            "objective_direction": "maximize",
            "units": "fluorescence",
            "uncertainty_type": "none",
            "training_independence": "unverified",
            "provenance": {"data_identifier": "data-v1", "model_identifier": "model-v2"},
        },
        "verdict": "Retrospective audit only.",
    }


def test_metric_figure_uses_real_report_values_and_model_interval() -> None:
    figure = benchmark_metric_figure(_benchmark_report())
    assert list(figure.data[0].x) == [0.72, 0.20, 0.51]
    assert list(figure.data[0].error_x.array) == pytest.approx([0.07, 0.0, 0.0])
    assert list(figure.data[0].error_x.arrayminus) == pytest.approx([0.10, 0.0, 0.0])


def test_metric_figure_keeps_exact_values_in_hover_without_bar_labels() -> None:
    figure = benchmark_metric_figure(_benchmark_report())
    bar = figure.data[0]
    assert bar.text is None or len(bar.text) == 0
    assert bar.texttemplate in (None, "")
    assert "%{x" in str(bar.hovertemplate)


def test_regression_and_residual_figures_use_held_out_rows() -> None:
    frame = pd.DataFrame({"id": ["a", "b", "c"], "target": [1.0, 2.0, 3.0], "prediction": [1.1, 1.8, 3.2]})
    fit = regression_fit_figure(frame, target_col="target", prediction_col="prediction", id_col="id")
    residual = residual_figure(frame, target_col="target", prediction_col="prediction", id_col="id")
    assert list(fit.data[0].x) == [1.0, 2.0, 3.0]
    assert list(fit.data[0].y) == [1.1, 1.8, 3.2]
    assert list(residual.data[0].y) == pytest.approx([0.1, -0.2, 0.2])


def test_linked_observation_figures_expose_row_keys_metadata_and_selection() -> None:
    frame = pd.DataFrame(
        {
            "row_key": ["row-a", "row-b", "row-c"],
            "Sequence ID": ["p001", "p005", "p009"],
            "Organism": ["E. coli", "S. cerevisiae", "E. coli"],
            "Batch": ["batch-1", "batch-3", "batch-3"],
            "target": [1.0, 1.34, 0.8],
            "prediction": [1.1, 1.21, 0.7],
            "uncertainty": [0.1, 0.3, 0.2],
        }
    )
    common = {
        "target_col": "target",
        "prediction_col": "prediction",
        "row_key_col": "row_key",
        "hover_cols": ["Sequence ID", "Organism", "Batch"],
        "selected_row_keys": ["row-b"],
    }
    figures = [
        regression_fit_figure(frame, **common),
        residual_figure(frame, **common),
        uncertainty_figure(
            frame,
            uncertainty_col="uncertainty",
            audit={"declared_type": "none"},
            **common,
        ),
    ]

    for figure in figures:
        observations = next(
            trace
            for trace in figure.data
            if trace.type == "scatter" and "markers" in str(trace.mode) and len(trace.x) == len(frame)
        )
        assert [row[0] for row in observations.customdata] == ["row-a", "row-b", "row-c"]
        assert list(observations.selectedpoints) == [1]
        for label in ["Sequence ID", "Organism", "Batch"]:
            assert label in str(observations.hovertemplate)


def test_uncertainty_language_depends_on_declared_semantics() -> None:
    frame = pd.DataFrame({"target": [1.0, 2.0], "prediction": [1.2, 1.8], "uncertainty": [0.1, 0.3]})
    undeclared = uncertainty_figure(frame, target_col="target", prediction_col="prediction", uncertainty_col="uncertainty", audit={"declared_type": "none"})
    absolute_error = uncertainty_figure(frame, target_col="target", prediction_col="prediction", uncertainty_col="uncertainty", audit={"declared_type": "predicted_absolute_error"})
    assert "semantics undeclared" in undeclared.layout.title.text
    assert "calibration" in absolute_error.layout.title.text.lower()
    assert len(absolute_error.data) == len(undeclared.data) + 1


def test_classification_plot_uses_score_ordering_without_probability_calibration() -> None:
    frame = pd.DataFrame({"label": [0, 1, 0, 1], "score": [0.1, 0.9, 0.4, 0.7]})
    figure = classification_discrimination_figure(frame, target_col="label", prediction_col="score", positive_label=1)
    assert len(figure.data) == 3
    assert "discrimination" in figure.layout.title.text.lower()


def test_slice_summary_flags_small_groups() -> None:
    frame = pd.DataFrame({"target": [1.0, 2.0, 3.0], "prediction": [1.1, 1.9, 2.7], "batch": ["a", "a", "b"]})
    summary = regression_slice_summary(frame, target_col="target", prediction_col="prediction", slice_cols=["batch"], minimum_n=3)
    assert set(summary["level"]) == {"a", "b"}
    assert set(summary["stability"]) == {"low n — descriptive only"}


def test_simulation_figures_render_candidate_constraints_and_plate() -> None:
    candidates = pd.DataFrame(
        {
            "design_id": ["d1", "d2"],
            "predicted_activity": [1.0, 1.5],
            "model_uncertainty": [0.2, 0.3],
            "acquisition_score": [1.2, 1.8],
            "mlm_plausibility": [0.8, 0.6],
            "gc_fraction": [0.4, 0.8],
            "max_homopolymer": [3, 7],
            "passes_basic_filters": [True, False],
        }
    )
    plate = pd.DataFrame(
        {
            "well": ["A1", "A2"],
            "plate_row": ["A", "A"],
            "plate_column": [1, 2],
            "role": ["positive_control", "exploit_top_prediction"],
            "design_id": ["", "d1"],
        }
    )
    assert len(simulation_landscape_figure(candidates).data) == 2
    assert len(simulation_constraints_figure(candidates, gc_low=0.3, gc_high=0.7, max_homopolymer=6).data) == 2
    assert len(plate_layout_figure(plate).data) == 1


def test_benchmark_section_reads_persisted_test_rows(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "target": [1.0, 2.0, 3.0],
            "prediction": [1.1, 1.8, 3.2],
            "uncertainty": [0.1, 0.2, 0.3],
            "batch": ["x", "x", "y"],
            "leakage_cluster": ["c1", "c1", "c2"],
        }
    ).to_csv(processed / "test.csv", index=False)
    summary = {
        "artifact_dir": str(tmp_path),
        "params": {
            "target_col": "target",
            "prediction_col": "prediction",
            "uncertainty_col": "uncertainty",
            "id_col": "id",
            "metadata_cols": ["batch"],
        },
        "prediction_audit": _benchmark_report(),
        "split_diagnostics": {"num_clusters": 2, "split_sizes": {"train": 6, "val": 2, "test": 3}},
    }
    section = ui._benchmark_section(summary, standalone=True)
    graphs = _components(section, dcc.Graph)
    assert len(graphs) == 5


def test_benchmark_section_exposes_linked_interaction_targets(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame(
        {
            "id": ["p001", "p005", "p009"],
            "sequence": ["ACGT", "GATTACA", "TTAA"],
            "target": [1.0, 1.34, 0.8],
            "prediction": [1.1, 1.21, 0.7],
            "uncertainty": [0.1, 0.3, 0.2],
            "organism": ["E. coli", "S. cerevisiae", "E. coli"],
            "batch": ["batch-1", "batch-3", "batch-3"],
            "leakage_cluster": ["c1", "c2", "c3"],
        }
    ).to_csv(processed / "test.csv", index=False)
    summary = {
        "artifact_dir": str(tmp_path),
        "params": {
            "target_col": "target",
            "prediction_col": "prediction",
            "uncertainty_col": "uncertainty",
            "id_col": "id",
            "sequence_col": "sequence",
            "metadata_cols": ["organism", "batch"],
        },
        "prediction_audit": _benchmark_report(),
        "split_diagnostics": {"num_clusters": 3, "split_sizes": {"train": 6, "val": 2, "test": 3}},
    }

    section = ui._benchmark_section(summary, standalone=True)
    assert {
        "benchmark-row-store",
        "benchmark-selection-store",
        "benchmark-fit-graph",
        "benchmark-residual-graph",
        "benchmark-uncertainty-graph",
        "benchmark-slice-container",
        "benchmark-inspector",
        "benchmark-clear-selection",
    }.issubset(_pattern_id_types(section))


def test_graph_config_keeps_box_and_lasso_selection_available() -> None:
    removed = set(ui.GRAPH_CONFIG.get("modeBarButtonsToRemove", []))
    assert {"select2d", "lasso2d"}.isdisjoint(removed)


def test_selected_row_keys_are_stable_unique_and_ignore_malformed_points() -> None:
    event = {
        "points": [
            {"customdata": ["row-b", "p005"]},
            {"customdata": ["row-a", "p001"]},
            {"customdata": ["row-b", "p005"]},
            {"curveNumber": 1, "customdata": ["row-aggregate", "bin"]},
            {"curveNumber": 0, "customdata": ["not-an-observation", "bin"]},
            {"customdata": []},
            {},
        ]
    }
    assert ui._selected_row_keys(event) == ["row-b", "row-a"]
    assert ui._selected_row_keys(None) == []
    assert ui._selected_row_keys({"points": [{"customdata": None}]}) == []


def test_payload_selected_frame_filters_by_stable_key_and_preserves_row_order() -> None:
    payload = {
        "rows": [
            {"stable_key": "row-a", "batch": "batch-1", "target": 1.0},
            {"stable_key": "row-b", "batch": "batch-2", "target": 2.0},
            {"stable_key": "row-c", "batch": "batch-3", "target": 3.0},
        ],
        "columns": {"row_key": "stable_key"},
    }

    selected = ui._payload_selected_frame(payload, ["row-c", "row-a", "row-c"])
    assert selected.to_dict("records") == [payload["rows"][0], payload["rows"][2]]
    assert ui._payload_selected_frame(payload, []).to_dict("records") == payload["rows"]
    assert ui._payload_selected_frame(payload, ["missing"]).empty


def test_observation_inspector_shows_sequence_values_and_metadata() -> None:
    payload = {
        "rows": [
            {
                "row_key": "row-a",
                "sequence_id": "p001",
                "sequence": "ACGT",
                "target": 1.0,
                "prediction": 1.1,
                "uncertainty": 0.1,
                "organism": "E. coli",
                "batch": "batch-1",
            },
            {
                "row_key": "row-b",
                "sequence_id": "p005",
                "sequence": "GATTACA",
                "target": 1.34,
                "prediction": 1.21,
                "uncertainty": 0.3,
                "organism": "S. cerevisiae",
                "batch": "batch-3",
            },
        ],
        "columns": {
            "row_key": "row_key",
            "id": "sequence_id",
            "target": "target",
            "prediction": "prediction",
            "uncertainty": "uncertainty",
            "sequence": "sequence",
            "metadata": ["organism", "batch"],
        },
        "slice_cols": ["organism", "batch"],
        "task_type": "regression",
        "uncertainty_audit": {"declared_type": "none"},
    }

    text = _component_text(ui._observation_inspector(payload, "row-b"))
    for value in ["p005", "GATTACA", "1.34", "1.21", "S. cerevisiae", "batch-3"]:
        assert value in text


def test_create_app_registers_linked_selection_and_inspection_dependencies() -> None:
    app = ui.create_app()

    def component_type(component_id: Any) -> str:
        parsed = component_id
        if isinstance(component_id, str) and component_id.startswith("{"):
            parsed = json.loads(component_id)
        if isinstance(parsed, dict):
            return str(parsed.get("type"))
        return str(parsed)

    edges: set[tuple[str, str, str, str]] = set()
    for callback in app.callback_map.values():
        raw_outputs = callback.get("output")
        outputs = raw_outputs if isinstance(raw_outputs, list) else [raw_outputs]
        for callback_input in callback.get("inputs") or []:
            input_type = component_type(callback_input.get("id"))
            input_property = str(callback_input.get("property"))
            for output in outputs:
                edges.add(
                    (
                        input_type,
                        input_property,
                        component_type(output.component_id),
                        str(output.component_property),
                    )
                )

    graph_types = {
        "benchmark-fit-graph",
        "benchmark-residual-graph",
        "benchmark-uncertainty-graph",
    }
    for graph_type in graph_types:
        assert (graph_type, "selectedData", "benchmark-selection-store", "data") in edges
        assert (graph_type, "clickData", "benchmark-inspector-body", "children") in edges

    assert {
        ("benchmark-selection-store", "data", "benchmark-fit-graph", "figure"),
        ("benchmark-selection-store", "data", "benchmark-residual-graph", "figure"),
        ("benchmark-selection-store", "data", "benchmark-uncertainty-graph", "figure"),
        ("benchmark-selection-store", "data", "benchmark-slice-container", "children"),
    }.issubset(edges)


def test_simulation_dropdown_defaults_to_available_scorer() -> None:
    page = ui.page_simulations()
    dropdowns = _components(page, dcc.Dropdown)
    model = next(component for component in dropdowns if component.id == "sim-model")
    values = {option["value"] for option in model.options}
    assert model.value in values
