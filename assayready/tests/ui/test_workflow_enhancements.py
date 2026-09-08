from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from model_assessment import ui
from model_assessment.visualization import (
    ranking_diagnostics_figure,
    simulation_generation_figure,
    simulation_landscape_figure,
)


def _component_ids(component: object) -> set[str]:
    component_id = getattr(component, "id", None)
    found = {component_id} if isinstance(component_id, str) else set()
    children = getattr(component, "children", None)
    if children is None:
        return found
    for child in children if isinstance(children, (list, tuple)) else [children]:
        found.update(_component_ids(child))
    return found


def _walk_components(component: object) -> list[object]:
    found = [component]
    children = getattr(component, "children", None)
    if children is None:
        return found
    for child in children if isinstance(children, (list, tuple)) else [children]:
        found.extend(_walk_components(child))
    return found


def _component_text(component: object) -> str:
    if component is None:
        return ""
    if isinstance(component, (str, int, float)):
        return str(component)
    children = getattr(component, "children", None)
    if children is None:
        return ""
    values = children if isinstance(children, (list, tuple)) else [children]
    return " ".join(_component_text(child) for child in values)


def test_simulation_visual_presets_are_curated_and_return_independent_lists() -> None:
    compact = ui._simulation_visual_defaults("Mask-fill", "compact")
    recommended = ui._simulation_visual_defaults("Random mutagenesis", "recommended")
    full = ui._simulation_visual_defaults("Diversity sampling", "full")

    assert compact == ["landscape", "generation"]
    assert recommended == ["landscape", "generation", "constraints"]
    assert full == ["landscape", "generation", "constraints", "plate"]
    assert ui._simulation_visual_defaults("Mask-fill", "unknown") == recommended

    compact.append("plate")
    assert ui._simulation_visual_defaults("Mask-fill", "compact") == ["landscape", "generation"]


@pytest.mark.parametrize(
    ("mode", "generation_label"),
    [
        ("Mask-fill", "Masked-position nucleotide composition"),
        ("Random mutagenesis", "Mutation position and count profile"),
        ("Diversity sampling", "Observed sequence-spread diagnostics"),
    ],
)
def test_simulation_visual_options_explain_the_selected_generation_mode(
    mode: str,
    generation_label: str,
) -> None:
    options = ui._simulation_visual_options(mode)

    assert [option["value"] for option in options] == ["landscape", "generation", "constraints", "plate"]
    assert options[1]["label"] == generation_label


def test_mask_fill_generation_figure_reports_composition_only_at_masked_positions() -> None:
    frame = pd.DataFrame({"sequence": ["AACT", "ACGT", "AGTT", "ATAT"]})

    figure = simulation_generation_figure(frame, parent_sequence="ANNT", mode="Mask-fill")

    assert figure.layout.uirevision == "simulation-mask-composition"
    assert [trace.name for trace in figure.data] == ["A", "C", "G", "T"]
    assert all(list(trace.x) == [2, 3] for trace in figure.data)
    stacked_totals = [sum(float(trace.y[index]) for trace in figure.data) for index in range(2)]
    assert stacked_totals == pytest.approx([1.0, 1.0])
    assert figure.layout.yaxis.range == (0, 1)


def test_random_mutagenesis_figure_reports_mutation_counts_and_position_rates() -> None:
    frame = pd.DataFrame({"sequence": ["AAAA", "CAAA", "CCAA"]})

    figure = simulation_generation_figure(frame, parent_sequence="AAAA", mode="Random mutagenesis")

    assert figure.layout.uirevision == "simulation-mutation-profile"
    assert list(figure.data[0].x) == [0, 1, 2]
    assert list(figure.data[1].x) == [1, 2, 3, 4]
    assert list(figure.data[1].y) == pytest.approx([2 / 3, 1 / 3, 0.0, 0.0])
    assert figure.layout.yaxis2.range == (0, 1)


def test_diversity_generation_figure_uses_normalized_hamming_distances() -> None:
    frame = pd.DataFrame({"sequence": ["AAAA", "CAAA", "CCAA"]})

    figure = simulation_generation_figure(frame, parent_sequence="AAAA", mode="Diversity sampling")

    assert figure.layout.uirevision == "simulation-diversity-diagnostics"
    assert list(figure.data[0].x) == pytest.approx([0.0, 0.25, 0.5])
    assert list(figure.data[1].x) == pytest.approx([0.25, 0.25, 0.25])
    assert "Observed sequence spread" in figure.layout.title.text


def test_simulation_preview_filters_are_recomputed_without_mutating_saved_results() -> None:
    frame = pd.DataFrame(
        {
            "design_id": ["pass", "low-gc", "long-run"],
            "gc_fraction": [0.5, 0.2, 0.5],
            "max_homopolymer": [3, 3, 8],
            "passes_basic_filters": [False, True, True],
            "filter_result": ["saved-a", "saved-b", "saved-c"],
        }
    )

    preview = ui._simulation_preview_frame(frame, gc_low=0.3, gc_high=0.7, max_homopolymer=6)

    assert preview["passes_preview_filters"].tolist() == [True, False, False]
    assert preview["preview_filter_result"].tolist() == [
        "Eligible under sequence constraints",
        "GC below minimum",
        "Homopolymer exceeds limit",
    ]
    assert frame["passes_basic_filters"].tolist() == [False, True, True]
    assert frame["filter_result"].tolist() == ["saved-a", "saved-b", "saved-c"]


def test_simulation_candidate_id_accepts_plotly_and_grid_events() -> None:
    assert ui._simulation_candidate_id({"points": [{"customdata": ["d-plot", 0.5, 1.2]}]}) == "d-plot"
    assert ui._simulation_candidate_id({"data": {"design_id": "d-grid"}}) == "d-grid"
    assert ui._simulation_candidate_id({"points": []}) is None


def test_simulation_landscape_marks_selected_candidate_and_enables_point_selection() -> None:
    frame = pd.DataFrame(
        {
            "design_id": ["selected", "other"],
            "predicted_activity": [1.0, 1.5],
            "model_uncertainty": [0.2, 0.3],
            "acquisition_score": [1.2, 1.8],
            "mlm_plausibility": [0.8, 0.6],
            "gc_fraction": [0.4, 0.8],
            "passes_basic_filters": [True, True],
        }
    )

    figure = simulation_landscape_figure(frame, selected_ids={"selected"})

    assert list(figure.data[0].selectedpoints) == [0]
    assert figure.data[0].name == "✓ Eligible"
    assert figure.data[0].marker.symbol == "circle"
    assert figure.layout.clickmode == "event+select"
    assert figure.layout.dragmode == "lasso"
    assert figure.layout.template.layout.plot_bgcolor == "rgb(17,17,17)"
    assert figure.layout.plot_bgcolor == "#141922"


def test_simulation_page_exposes_truthful_indeterminate_run_status() -> None:
    page = ui.page_simulations()

    assert "simulation-running-status" in _component_ids(page)
    assert "simulation-sequence-summary" in _component_ids(page)
    assert "simulation-action-summary" in _component_ids(page)
    running_status = next(
        component
        for component in _walk_components(page)
        if getattr(component, "id", None) == "simulation-running-status"
    )
    assert running_status.hidden is True


def test_sequence_input_summary_surfaces_mask_and_validation_state() -> None:
    ready = _component_text(ui._sequence_input_summary("ACNNGT", "Mask-fill"))
    missing_mask = _component_text(ui._sequence_input_summary("ACGT", "Mask-fill"))
    invalid = _component_text(ui._sequence_input_summary("ACXG", "Random mutagenesis"))

    assert "Length 6 nt" in ready
    assert "Masked 2" in ready
    assert "Ready for mask-fill" in ready
    assert "Add at least one N or [MASK] position" in missing_mask
    assert "Invalid symbols: X" in invalid


def test_workspace_guide_is_collapsed_until_requested() -> None:
    shell = ui._app_shell()
    components = _walk_components(shell)
    frame = next(component for component in components if getattr(component, "id", None) == "workspace-frame")
    guide = next(component for component in components if getattr(component, "id", None) == "toolkit-panel")
    toggle = next(component for component in components if getattr(component, "id", None) == "toggle-toolkit-button")

    assert frame.className == "workspace-frame toolkit-collapsed"
    assert guide.className == "toolkit-panel is-collapsed"
    assert getattr(toggle, "aria-expanded") == "false"


def test_run_history_table_keeps_only_comparison_columns() -> None:
    frame = ui._runs_display_frame(
        [
            {
                "run_id": "run-1",
                "updated_at": "2026-08-12T20:00:00Z",
                "project": "example_project",
                "workflow": "prediction",
                "assurance_level": "self_declared",
                "evidence_level": "descriptive_audit",
                "primary_metric_name": "r2",
                "primary_metric": 0.927,
                "test_rows": 4,
                "ranked_candidates": 8,
            }
        ]
    )

    assert frame.columns.tolist() == [
        "Updated",
        "Project",
        "Recommended use",
        "Evaluation",
        "Evidence",
        "Held-out n",
        "Run ID",
    ]
    assert frame.loc[0, "Updated"].startswith("Aug 12,")
    assert frame.loc[0, "Recommended use"] == "Exploratory only"
    assert frame.loc[0, "Evaluation"] == "Descriptive audit"
    assert frame.loc[0, "Evidence"] == "Self-declared"
    assert frame.loc[0, "Held-out n"] == 4


def test_generation_results_lead_with_eligible_candidates_and_full_funnel() -> None:
    funnel = ui._generation_funnel(requested=96, generated=96, eligible=31)
    status = ui._planning_status_strip(eligible=31, generated=96, model_backed=False)

    assert "31 eligible candidates" in _component_text(funnel)
    assert "96 requested → 96 generated → 31 eligible → 65 excluded" in _component_text(funnel)
    assert "DRAFT · Heuristic scoring · 31/96 candidates eligible" in _component_text(status)
    assert "planning and ranking only" in _component_text(status)


def test_candidate_table_keeps_complete_sequences_and_styles_risk_chips() -> None:
    sequence = "ACGT" * 20
    frame = ui._candidate_display_frame(
        [
            {
                "workflow": "prediction",
                "project": "example",
                "rank": 1,
                "display_id": "candidate-1",
                "sequence": sequence,
                "prediction": 0.9,
                "uncertainty": 0.1,
                "training_distribution_status": "in_distribution",
                "risk_flags_json": '["high_uncertainty"]',
                "run_id": "run-1",
            }
        ]
    )

    assert frame.loc[0, "Sequence"] == sequence
    grid = ui._data_table(frame, table_id="candidate-index-table")
    definitions = {column["field"]: column for column in grid.columnDefs}
    assert definitions["Sequence"]["wrapText"] is True
    assert definitions["Sequence"]["autoHeight"] is True
    assert definitions["Primary risk"]["cellClass"] == "candidate-risk-cell"
    assert set(definitions["Primary risk"]["cellClassRules"]) == {
        "candidate-risk-cell-clear",
        "candidate-risk-cell-warning",
    }


def test_runs_page_uses_master_detail_evidence_drawer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui, "list_runs", lambda limit=500: [])

    content = ui._runs_content()
    classes = {getattr(component, "className", None) for component in _walk_components(content)}

    assert "runs-master-detail" in classes
    assert "runs-evidence-drawer" in classes
    assert "runs-run-detail" in _component_ids(content)


def test_sandbox_draft_reports_show_an_informative_empty_state() -> None:
    summary = {"run_id": "draft-1", "project": "sandbox", "simulation_report": {}}
    run = {"run_id": "draft-1", "project": "sandbox", "workflow": "simulation"}

    assert ui._is_non_audited_sandbox_run(summary, run) is True
    state = ui._sandbox_report_empty_state(summary, run)
    assert "No audit report is available" in _component_text(state)
    assert "held-out assay measurements" in _component_text(state)
    hrefs = {component.href for component in _walk_components(state) if getattr(component, "href", None)}
    assert {"/simulations", "/analyze", "/candidates?run_id=draft-1"} <= hrefs


def test_plotly_hints_are_disabled_because_the_ui_supplies_its_own_help() -> None:
    assert ui.GRAPH_CONFIG["showTips"] is False


def test_scorer_setup_is_a_complete_in_app_workflow() -> None:
    component_ids = _component_ids(ui.page_simulations())

    assert {
        "install-recommended-model-button",
        "scorer-training-upload",
        "scorer-training-sequence-col",
        "scorer-training-target-col",
        "scorer-training-task-type",
        "scorer-training-objective",
        "scorer-training-confirmation",
        "train-scorer-button",
        "browse-custom-model-button",
        "browse-custom-head-button",
        "verify-scorer-button",
    } <= component_ids


def test_forbidden_motif_constraints_check_both_strands_and_preserve_reasons() -> None:
    motifs = ui._parse_sequence_motifs("GGTCTC, AAAA; GGTCTC")
    assert motifs == ["GGTCTC", "AAAA"]

    scores = ui._heuristic_sequence_scores(
        "TTTGAGACCTTT",
        (0.0, 1.0),
        20,
        __import__("random").Random(13),
        forbidden_motifs=["GGTCTC"],
        check_reverse_complements=True,
    )

    assert scores["passes_basic_filters"] is False
    assert "GGTCTC reverse complement" in scores["filter_result"]
    assert scores["forbidden_motif_hits"] == "GGTCTC reverse complement"


def test_forbidden_motif_parser_rejects_ambiguous_rules() -> None:
    with pytest.raises(ValueError, match="only A, C, G, and T"):
        ui._parse_sequence_motifs("GGTNTC")


@pytest.mark.parametrize(
    ("objective_direction", "measured_ranks", "predicted_ranks"),
    [
        ("maximize", [3.0, 2.0, 1.0], [1.0, 2.0, 3.0]),
        ("minimize", [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]),
    ],
)
def test_ranking_diagnostics_respects_objective_direction_and_top_k_recovery(
    objective_direction: str,
    measured_ranks: list[float],
    predicted_ranks: list[float],
) -> None:
    frame = pd.DataFrame({"measured": [1.0, 2.0, 3.0], "predicted": [3.0, 2.0, 1.0]})

    figure = ranking_diagnostics_figure(
        frame,
        target_col="measured",
        prediction_col="predicted",
        objective_direction=objective_direction,
    )

    assert list(figure.data[0].x) == measured_ranks
    assert list(figure.data[0].y) == predicted_ranks
    assert list(figure.data[2].x) == [1, 2, 3]
    assert list(figure.data[2].y) == pytest.approx([0.0, 0.5, 1.0])
    assert figure.layout.xaxis.autorange == "reversed"
    assert figure.layout.yaxis.autorange == "reversed"


def test_config_ui_defaults_hydrates_nested_workflow_and_manifest_values() -> None:
    config = {
        "project": "screen-42",
        "task": {
            "type": "classification",
            "prediction_col": "score",
            "positive_label": "active",
        },
        "ingestion": {
            "low_n_threshold": 75,
            "column_aliases": {"sequence": ["DNA sequence"]},
        },
        "split": {
            "val_fraction": 0.2,
            "test_fraction": 0.25,
            "homology_threshold": 0.82,
            "homology_k": 6,
            "similarity_policy": "exact_sequence",
            "sensitivity_policies": ["canonical_kmer_jaccard"],
            "sensitivity_thresholds": [0.8, 0.9],
        },
        "training": {"seed": 91, "epochs": 12, "learning_rate": 0.002},
        "uq": {"ensemble_size": 7},
        "acquisition": {"method": "expected_improvement", "top_k": 24, "beta": 1.5, "diversity_penalty": 0.35},
        "generation": {"enabled": False, "num_proposals": 250},
    }
    manifest = {
        "objective_direction": "minimize",
        "units": "micromolar",
        "uncertainty_type": "ensemble_sd",
        "provenance": {"model_identifier": "model-7", "data_identifier": "data-9"},
    }

    defaults = ui._config_ui_defaults(config, evaluation_manifest=manifest)

    assert defaults == {
        "analysis_type": "prediction",
        "project": "screen-42",
        "task_type": "classification",
        "top_k": 24,
        "low_n": 75,
        "val_fraction": 0.2,
        "test_fraction": 0.25,
        "homology_threshold": 0.82,
        "homology_k": 6,
        "beta": 1.5,
        "diversity_penalty": 0.35,
        "seed": 91,
        "ensemble_size": 7,
        "epochs": 12,
        "learning_rate": 0.002,
        "num_proposals": 250,
        "evaluation_objective": "minimize",
        "evaluation_units": "micromolar",
        "evaluation_uncertainty_type": "ensemble_sd",
        "evaluation_model_id": "model-7",
        "evaluation_data_id": "data-9",
        "runtime": {
            "positive_label": "active",
            "column_aliases": {"sequence": ["DNA sequence"]},
            "similarity_policy": "exact_sequence",
            "similarity_sensitivity_policies": ["canonical_kmer_jaccard"],
            "similarity_sensitivity_thresholds": [0.8, 0.9],
            "acquisition_method": "expected_improvement",
            "generate_candidates": False,
        },
    }


def test_classification_label_options_come_from_the_selected_target() -> None:
    state = {
        "label_values": {
            "activity_class": ["active", "inactive"],
            "batch": ["one", "two"],
        }
    }

    assert ui._classification_label_options(state, "activity_class") == [
        {"label": "active", "value": "active"},
        {"label": "inactive", "value": "inactive"},
    ]
    assert ui._classification_label_options(state, "missing") == []
    assert ui._classification_label_key("Active") == ui._classification_label_key("active")
    assert ui._classification_label_key("1.0") == ui._classification_label_key(1)


def test_positive_class_control_is_on_analyze_not_design_sandbox() -> None:
    assert {"positive-label", "positive-label-wrapper"} <= _component_ids(ui.page_analyze())
    assert "positive-label" not in _component_ids(ui.page_simulations())


def test_config_preview_applies_normalized_aliases_without_mutating_uploaded_frame() -> None:
    source = pd.DataFrame({"DNA sequence": ["ACGT"], "Measured-Value": [1.2], "batch": ["b1"]})
    config = {
        "ingestion": {
            "column_aliases": {
                "sequence": ["dna_sequence"],
                "target": ["measured value"],
            }
        }
    }

    preview = ui._config_preview_frame(source, config)

    assert list(preview.columns) == ["sequence", "target", "batch"]
    assert list(source.columns) == ["DNA sequence", "Measured-Value", "batch"]


def test_config_preview_rejects_aliases_that_collapse_distinct_uploaded_columns() -> None:
    source = pd.DataFrame({"sequence": ["ACGT"], "DNA sequence": ["TGCA"]})
    config = {"column_aliases": {"sequence": ["DNA sequence"]}}

    with pytest.raises(ValueError, match="multiple uploaded columns"):
        ui._config_preview_frame(source, config)


def test_blank_analysis_output_uses_configured_root_and_project_slug_independent_of_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "canonical-output"
    unrelated_cwd = tmp_path / "elsewhere"
    unrelated_cwd.mkdir()
    monkeypatch.setenv(ui.OUTPUT_ROOT_ENV, str(output_root))
    monkeypatch.chdir(unrelated_cwd)

    assert ui._analysis_output_dir(None, "My Project") == (output_root / "my_project").resolve()
    assert ui._analysis_output_dir("   ", "My Project") == (output_root / "my_project").resolve()
    assert ui._safe_output_dir("") == output_root.resolve()


def test_native_picker_authorizes_only_the_exact_selected_external_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "canonical-output"
    external = tmp_path / "chosen-elsewhere"
    external.mkdir()
    sibling = tmp_path / "not-chosen"
    sibling.mkdir()
    monkeypatch.setenv(ui.OUTPUT_ROOT_ENV, str(output_root))

    assert ui._analysis_output_dir(
        str(external),
        "project",
        selected_output=ui._output_selection_authorization(external),
    ) == external.resolve()
    with pytest.raises(ValueError, match="outputs/assayready"):
        ui._analysis_output_dir(
            str(sibling),
            "project",
            selected_output=ui._output_selection_authorization(external),
        )

    forged = {"path": str(sibling), "signature": "0" * 64}
    with pytest.raises(ValueError, match="outputs/assayready"):
        ui._analysis_output_dir(str(sibling), "project", selected_output=forged)
