from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import dash_ag_grid as dag
from dash import dcc, html

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
            "candidate_prioritization": {
                "ok": False,
                "status": "review_with_cautions",
                "reasons": ["no candidates"],
            },
        },
        "assurance_level": {
            "key": "self_declared",
            "label": "Self-declared audit",
            "basis": ["customer-supplied predictions"],
            "limitations": ["training independence was not independently verified"],
        },
        "evidence_level": {
            "key": "descriptive_audit_only",
            "label": "Descriptive audit only",
            "support": ["retrospective metric computed"],
            "limitations": ["prospective utility has not been demonstrated"],
        },
        "threshold_sensitivity": {
            "interpretation": "Diagnostic profiles only.",
            "profiles": [
                {
                    "name": "declared",
                    "thresholds": {"min_test_rows": 20},
                    "retrospective_criteria_met": True,
                }
            ],
            "stable_under_stricter_thresholds": False,
            "conclusion_changes_under_stricter_thresholds": True,
        },
        "similarity_sensitivity": {
            "interpretation": "Similarity policies are proxies.",
            "settings": [],
            "conclusion_changes_across_similarity_settings": False,
            "test_membership_changes_across_similarity_settings": False,
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
    assert list(figure.data[0].marker.color) == ["#5ac8ad", "#cbd5e1", "#94a3b8"]
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
    histogram = next(trace for trace in residual.data if trace.type == "histogram")
    assert histogram.nbinsy == 6
    assert residual.layout.bargap == pytest.approx(0.12)


def test_analyze_page_uses_progressive_workflow_and_compact_advanced_sections() -> None:
    page = ui.page_analyze()
    text = _component_text(page)
    headings = [_component_text(item) for item in _components(page, ui.html.H2)]
    assert headings[:4] == [
        "Choose objective",
        "Upload assay data",
        "Verify column mappings",
        "Review and run",
    ]
    assert "Evaluate model predictions" in text
    assert "Try with sample data" in text
    assert "Browse public benchmarks" in text
    assert "Independence boundary columns" not in text
    assert "Advanced analysis settings" in text

    cards = {
        item.id: item
        for item in _components(page, ui.html.Div)
        if getattr(item, "id", None) in {"objective-card", "data-card", "columns-card", "review-card"}
    }
    assert "completed-stage" in cards["objective-card"].className
    assert "active-stage" in cards["data-card"].className
    assert "locked-stage" in cards["columns-card"].className
    assert "locked-stage" in cards["review-card"].className

    confirm_mappings = next(
        item
        for item in _components(page, ui.html.Button)
        if getattr(item, "id", None) == "confirm-mappings-button"
    )
    assert confirm_mappings.disabled is True

    local_options = next(
        item
        for item in _components(page, ui.html.Div)
        if getattr(item, "id", None) == "local-model-options"
    )
    assert local_options.style == {"display": "none"}

    detail_labels = [_component_text(item) for item in _components(page, ui.html.Summary)]
    assert "Split and ranking settings" not in detail_labels
    assert "Evaluation context and provenance" not in detail_labels


def test_analysis_context_help_changes_with_objective_and_uploaded_data() -> None:
    empty_text = _component_text(ui.html.Div(ui._analysis_help_content(None, "prediction")))
    assert "No file uploaded" in empty_text
    assert "Model prediction" in empty_text
    assert "Upload an assay table to begin validation" in empty_text

    state = {
        "filename": "assay.csv",
        "rows": 120,
        "columns": ["sequence", "measured", "prediction"],
    }
    loaded_text = _component_text(
        ui.html.Div(
            ui._analysis_help_content(
                state,
                "prediction",
                sequence_col="sequence",
                target_col="measured",
                prediction_col="prediction",
                task_type="regression",
            )
        )
    )
    assert "assay.csv" in loaded_text
    assert "120 rows · 3 detected columns · Parsed" in loaded_text
    assert "No blockers detected" in loaded_text


def test_uploaded_data_preview_reports_parsing_and_renders_rows() -> None:
    preview = ui._analysis_preview_from_state(
        {
            "filename": "audit.csv",
            "rows": 2,
            "columns": ["sequence", "measured"],
            "preview_rows": [
                {"sequence": "ACGT", "measured": 1.2},
                {"sequence": "TGCA", "measured": 0.8},
            ],
        }
    )

    text = _component_text(preview)
    assert "audit.csv" in text
    assert "Parsing Complete" in text
    assert "sequence" in text
    assert len(_components(preview, dag.AgGrid)) == 1


def test_analysis_blocker_requires_mapping_confirmation_before_run() -> None:
    blocker = ui._analysis_blocker_message(
        assay_ready=True,
        columns_ready=True,
        prediction_ready=True,
        classification_ready=True,
        mappings_confirmed=False,
        analysis_type="prediction",
    )
    ready = ui._analysis_blocker_message(
        assay_ready=True,
        columns_ready=True,
        prediction_ready=True,
        classification_ready=True,
        mappings_confirmed=True,
        analysis_type="prediction",
    )

    assert "Confirm mappings to continue" in _component_text(blocker)
    assert "New analysis ready" in _component_text(ready)


def test_data_table_formats_floating_point_values_for_readability() -> None:
    rendered = ui._data_table(pd.DataFrame({"score": [0.08000000000000007], "label": ["a"]}))
    table = _components(rendered, dag.AgGrid)[0]
    score = next(column for column in table.columnDefs if column["field"] == "score")
    assert score["type"] == "numericColumn"
    assert "toPrecision(5)" in score["valueFormatter"]["function"]


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
    model_figure = simulation_landscape_figure(candidates, model_backed=True, target_label="measured_activity")
    assert model_figure.layout.title.text == "Model prediction and uncertainty"
    assert model_figure.layout.yaxis.title.text == "Ensemble disagreement (SD)"
    constraints = simulation_constraints_figure(candidates, gc_low=0.3, gc_high=0.7, max_homopolymer=6)
    assert len(constraints.data) == 2
    assert constraints.layout.xaxis.tickformat == ".0%"
    assert constraints.layout.xaxis2.dtick == 1
    assert constraints.layout.xaxis2.tickformat == "d"
    plate_figure = plate_layout_figure(plate, plate_size=24)
    assert len(plate_figure.data) == 1
    assert plate_figure.data[0].z.shape == (4, 6)
    assert plate_figure.data[0].text[3][5] == "D6<br>Empty"


def test_draft_well_layout_is_explicitly_unapproved() -> None:
    candidates = pd.DataFrame(
        {
            "design_id": ["d1", "d2"],
            "sequence": ["AAAA", "CCCC"],
            "rank": [1, 2],
            "acquisition_score": [1.0, 0.5],
            "passes_basic_filters": [True, True],
        }
    )
    layout = ui._build_plate_plan(candidates, plate_size=24, control_wells=2, seed=13)
    assert not layout.empty
    assert layout["approved_for_execution"].eq(False).all()
    assert layout["layout_limitations"].astype(str).str.len().gt(0).all()


def test_acquisition_utility_respects_objective_direction_and_exploration() -> None:
    maximize = ui._acquisition_utility(
        [2.0, 4.0], [0.5, 0.0], exploration_weight=2.0, objective_direction="maximize"
    )
    minimize = ui._acquisition_utility(
        [2.0, 4.0], [0.5, 0.0], exploration_weight=2.0, objective_direction="minimize"
    )

    assert maximize.tolist() == [3.0, 4.0]
    assert minimize.tolist() == [-1.0, -4.0]
    assert minimize.idxmax() == 0


def test_acquisition_utility_rejects_unknown_direction() -> None:
    with pytest.raises(ValueError, match="objective_direction"):
        ui._acquisition_utility(
            [1.0], [0.1], exploration_weight=1.0, objective_direction="sideways"
        )


def test_draft_well_layout_uses_stable_corner_control_anchors() -> None:
    candidates = pd.DataFrame(
        {
            "design_id": ["d1"],
            "sequence": ["ACGT"],
            "rank": [1],
            "acquisition_score": [1.0],
            "passes_basic_filters": [True],
        }
    )
    layout = ui._build_plate_plan(candidates, plate_size=96, control_wells=4, seed=13)
    controls = set(layout.loc[layout["role"].str.endswith("control"), "well"])
    assert controls == {"A1", "A12", "H1", "H12"}


def test_draft_well_layout_does_not_plate_failed_candidates() -> None:
    candidates = pd.DataFrame(
        {
            "design_id": ["eligible", "failed"],
            "sequence": ["ACGT", "AAAAAAAA"],
            "rank": [1, 2],
            "acquisition_score": [1.0, 99.0],
            "passes_basic_filters": [True, False],
        }
    )
    layout = ui._build_plate_plan(candidates, plate_size=24, control_wells=4, seed=13)
    assert "eligible" in set(layout["design_id"])
    assert "failed" not in set(layout["design_id"])


def test_draft_well_layout_preserves_canonical_prioritization_rank() -> None:
    candidates = pd.DataFrame(
        {
            "design_id": ["ranked_first", "raw_prediction_first"],
            "sequence": ["AAAA", "CCCC"],
            "rank": [1, 2],
            "assay_prediction": [1.0, 99.0],
            "passes_basic_filters": [True, True],
        }
    )
    layout = ui._build_plate_plan(candidates, plate_size=24, control_wells=23, seed=13)
    candidate = layout[layout["role"] == "prioritized_candidate"].iloc[0]
    assert candidate["design_id"] == "ranked_first"
    assert int(candidate["rank"]) == 1


def test_top_candidates_exposes_uncertainty_and_prioritization_policy(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "rank": 1,
                "display_id": "candidate-1",
                "sequence": "ACGT",
                "prediction": 0.8,
                "uncertainty": "",
                "uncertainty_status": "missing_or_invalid_prediction_only",
                "acquisition_policy": "prediction_only",
                "prioritization_status": "review_with_cautions",
                "evidence_level": "descriptive_audit_only",
                "diversity_policy": "kmer_cosine",
                "max_similarity_to_previous_selection": 0.0,
            }
        ]
    ).to_csv(tmp_path / "candidate_prioritization.csv", index=False)
    frame = ui._top_candidates(
        {"artifact_dir": str(tmp_path), "params": {"sequence_col": "sequence"}}
    )
    assert frame.loc[0, "acquisition_policy"] == "prediction_only"
    assert "missing_or_invalid" in frame.loc[0, "uncertainty_status"]
    assert frame.loc[0, "evidence_level"] == "descriptive_audit_only"


def test_similarity_panel_does_not_imply_stability_when_not_evaluated() -> None:
    panel = ui._similarity_sensitivity_panel(
        {
            "similarity_sensitivity": {
                "status": "not_evaluated",
                "reason": "Refitting was not performed.",
                "conclusion_changes_across_similarity_settings": False,
            }
        }
    )
    text = _component_text(panel)
    assert "Not evaluated" in text
    assert "conclusion changed: False" not in text


def test_benchmark_without_baselines_still_shows_evidence_classification(tmp_path: Path) -> None:
    report = _benchmark_report()
    report["baselines"] = {}
    report["evidence_level"] = {
        "key": "insufficient_evidence",
        "label": "Insufficient evidence",
        "support": [],
        "limitations": ["no usable held-out primary metric was produced"],
    }
    section = ui._benchmark_section(
        {"artifact_dir": str(tmp_path), "prediction_audit": report, "split_diagnostics": {}},
        standalone=True,
    )
    text = _component_text(section)
    assert "Insufficient evidence" in text
    assert "no usable held-out primary metric" in text


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
    text = _component_text(section)
    assert "Self-declared" in text
    assert "Descriptive audit" in text
    assert "PASSED DECLARED POLICY" in text
    assert "SUPPORTED" not in text
    assert "Recommendation" not in text


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


def test_summary_metrics_separate_uploaded_and_heldout_rows_and_gate_low_n() -> None:
    report = _benchmark_report()
    report["test_metrics"]["num_rows"] = 4
    summary = {
        "valid_prediction_rows": 20,
        "split_diagnostics": {"split_sizes": {"train": 12, "val": 4, "test": 4}},
        "prediction_audit": report,
    }

    metrics = {item["label"]: item for item in ui._summary_metrics(summary)}

    assert metrics["Valid uploaded rows"]["value"] == 20
    assert metrics["Held-out test rows"]["value"] == 4
    assert metrics["Reliability"]["value"] == "Insufficient evidence"
    assert "Observed R-squared: 0.72" in metrics["Reliability"]["note"]
    assert "only 4 held-out rows" in metrics["Reliability"]["note"]


def test_sha_provenance_is_short_by_default_and_full_on_disclosure() -> None:
    digest = "a" * 64
    component = ui._provenance_identifier(f"assayready-dnabert2@sha256:{digest}", kind="model")

    assert isinstance(component, html.Details)
    assert "DNABERT-2 - aaaaaaaa..." in _component_text(component)
    assert f"sha256:{digest}" in _component_text(component)
    clipboards = _components(component, dcc.Clipboard)
    assert len(clipboards) == 1
    assert clipboards[0].content == f"sha256:{digest}"


def test_benchmark_metadata_columns_exclude_invariant_slices() -> None:
    frame = pd.DataFrame(
        {
            "sequence": ["AAAA", "CCCC", "GGGG"],
            "target": [1.0, 2.0, 3.0],
            "prediction": [1.1, 1.9, 3.2],
            "assay": ["gfp", "gfp", "gfp"],
            "batch": ["b1", "b1", "b2"],
        }
    )
    summary = {"params": {"metadata_cols": ["assay", "batch"]}}

    metadata, slice_columns = ui._benchmark_metadata_columns(
        frame,
        summary,
        excluded={"sequence", "target", "prediction"},
    )

    assert metadata == ["assay", "batch"]
    assert slice_columns == ["batch"]


def test_duplicate_run_views_collapse_reindexed_copies() -> None:
    summary = {
        "prediction_audit": {
            "assurance": {"level": "provenance_verified"},
            "evidence": {"level": "descriptive_audit_only"},
        },
        "data_id": "GSE135464",
    }
    base = {
        "project": "gse135464_gpd_verified_holdout",
        "workflow": "prediction",
        "primary_metric": 0.115,
        "best_baseline_metric": 0.037,
        "test_rows": 225,
        "ranked_candidates": 0,
        "accepted_rows": 1500,
        "summary_json": json.dumps(summary),
    }
    runs = [
        {**base, "run_id": "first", "updated_at": "2026-08-09T09:00:00Z"},
        {**base, "run_id": "second", "updated_at": "2026-08-09T10:00:00Z"},
    ]

    unique = ui._deduplicate_run_views(runs)

    assert len(unique) == 1
    assert unique[0]["run_id"] == "first"


def test_evidence_package_contains_manifest_and_integrity_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report = run_dir / "report.md"
    report.write_text("# Evidence\n", encoding="utf-8")
    summary = run_dir / "summary.json"
    summary.write_text('{"ok": true}', encoding="utf-8")
    run = {
        "run_id": "run-1",
        "project": "example-project",
        "artifact_dir": str(run_dir),
        "report_name": report.name,
        "summary_path": str(summary),
    }
    monkeypatch.setattr(ui, "list_runs", lambda limit=1000: [run])
    monkeypatch.setattr(
        ui,
        "list_artifacts",
        lambda run_id: [
            {"path": str(report)},
            {"path": str(summary)},
        ],
    )

    payload, filename = ui._evidence_package_bytes("run-1")

    assert filename == "example-project_evidence_package.zip"
    with zipfile.ZipFile(__import__("io").BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {"report.md", "summary.json", "evidence_package_manifest.json"}
        manifest = json.loads(archive.read("evidence_package_manifest.json"))
        assert manifest["run_id"] == "run-1"
        assert {item["path"] for item in manifest["files"]} == {"report.md", "summary.json"}
        assert all(len(item["sha256"]) == 64 for item in manifest["files"])
