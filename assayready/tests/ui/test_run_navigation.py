from __future__ import annotations

import json
from typing import Any

import pandas as pd
from dash import dcc, html

from model_assessment import ui


def _components(component: Any, target_type: type) -> list[Any]:
    found = [component] if isinstance(component, target_type) else []
    children = getattr(component, "children", None)
    if children is None:
        return found
    for child in children if isinstance(children, (list, tuple)) else [children]:
        found.extend(_components(child, target_type))
    return found


def _component_id_key(component_id: Any) -> str:
    return component_id if isinstance(component_id, str) else json.dumps(component_id, sort_keys=True, default=str)


def test_run_href_round_trips_a_url_escaped_run_id() -> None:
    href = ui._run_href("/reports", " run id/with?chars ")

    assert href == "/reports?run_id=run+id%2Fwith%3Fchars"
    assert ui._requested_run_id(href.split("?", 1)[1]) == "run id/with?chars"
    assert ui._run_href("/reports", None) == "/reports"


def test_linked_run_selection_requires_the_target_route_and_known_option() -> None:
    options = [{"label": "Run one", "value": "run-1"}]

    assert ui._linked_run_selection("/reports", "?run_id=run-1", "/reports", options) == "run-1"
    assert ui._linked_run_selection("/candidates", "?run_id=run-1", "/reports", options) is None
    assert ui._linked_run_selection("/reports", "?run_id=unknown", "/reports", options) is None


def test_preferred_run_uses_fresh_link_then_current_then_fallback() -> None:
    options = [{"label": "Old", "value": "old"}, {"label": "New", "value": "new"}]

    assert ui._preferred_run_value("?run_id=new", options, "old", "old") == "new"
    assert ui._preferred_run_value(None, options, "old", "new") == "old"
    assert ui._preferred_run_value("?run_id=missing", options, "missing", "new") == "new"


def test_analysis_and_run_results_link_to_the_originating_run(monkeypatch) -> None:
    monkeypatch.setattr(ui, "_report_payload", lambda _summary: {"verdict": "Complete"})
    monkeypatch.setattr(ui, "_top_candidates", lambda _summary: pd.DataFrame([{"Candidate": "candidate-1"}]))
    monkeypatch.setattr(ui, "_artifact_options", lambda _summary, _report_name: [])
    monkeypatch.setattr(ui, "_benchmark_section", lambda *_args, **_kwargs: html.Div())
    monkeypatch.setattr(ui, "_candidate_result_frame", lambda frame: frame)
    monkeypatch.setattr(ui, "_candidate_eligibility", lambda _summary: (True, "Eligible", []))
    monkeypatch.setattr(ui, "_summary_metrics", lambda _summary: [])
    summary = {"run_id": "run id/1", "artifact_dir": ""}

    for surface in ["analysis", "runs"]:
        links = _components(ui._result_summary(summary, "readiness_report.md", surface=surface), dcc.Link)
        hrefs = {link.href for link in links}
        assert "/reports?run_id=run+id%2F1" in hrefs
        assert "/candidates?run_id=run+id%2F1" in hrefs
        assert "/benchmarks?run_id=run+id%2F1" in hrefs


def test_benchmark_selector_keeps_quick_audits_available(monkeypatch) -> None:
    runs = [
        {"run_id": "verified", "project": "verified-project", "workflow": "prediction"},
        {"run_id": "quick", "project": "quick-project", "workflow": "internal"},
    ]
    monkeypatch.setattr(ui, "list_runs", lambda limit=500: runs)
    monkeypatch.setattr(ui, "_deduplicate_run_views", lambda values: values)
    monkeypatch.setattr(
        ui,
        "_run_display_metadata",
        lambda run: {
            "verified": run["run_id"] == "verified",
            "assurance": "Verified" if run["run_id"] == "verified" else "Self-declared",
            "evidence": "Controlled evaluation" if run["run_id"] == "verified" else "Audit only",
        },
    )
    monkeypatch.setattr(ui, "_benchmark_run_priority", lambda run: (run["run_id"] == "verified",))

    page = ui.page_benchmarks()
    selector = next(component for component in _components(page, dcc.RadioItems) if component.id == "benchmark-run-select")

    assert [option["value"] for option in selector.options] == ["verified", "quick"]
    assert not any(component.id == "benchmark-run-select" for component in _components(page, dcc.Dropdown))


def test_navigation_snapshots_see_runs_added_after_initial_layout(monkeypatch) -> None:
    runs = [{"run_id": "old", "project": "old-project", "workflow": "prediction"}]
    candidates = [{"run_id": "old", "display_id": "old-1", "sequence_hash": "old-hash", "workflow": "prediction"}]
    monkeypatch.setattr(ui, "list_runs", lambda limit=500: list(runs))
    monkeypatch.setattr(ui, "list_candidates", lambda limit=5000: list(candidates))
    monkeypatch.setattr(ui, "_deduplicate_run_views", lambda values: values)
    monkeypatch.setattr(ui, "_benchmark_run_priority", lambda run: (run["run_id"] == "new",))

    assert [option["value"] for option in ui._report_navigation_snapshot()[1]] == ["old"]

    runs.append({"run_id": "new", "project": "new-project", "workflow": "internal"})
    candidates.append({"run_id": "new", "display_id": "new-1", "sequence_hash": "new-hash", "workflow": "internal"})

    candidate_rows, candidate_options, _default = ui._candidate_navigation_snapshot()
    assert {option["value"] for option in candidate_options} == {"old", "new"}
    assert {row["run_id"] for row in candidate_rows} == {"old", "new"}
    assert {option["value"] for option in ui._report_navigation_snapshot()[1]} == {"old", "new"}
    assert [option["value"] for option in ui._benchmark_navigation_snapshot()[1]] == ["new", "old"]


def test_app_refreshes_destination_snapshots_on_route_entry() -> None:
    app = ui.create_app()
    target_outputs = {
        ("candidate-project-filter", "options"),
        ("candidate-project-filter", "value"),
        ("candidate-index-store", "data"),
        ("reports-run-select", "options"),
        ("reports-run-select", "value"),
        ("benchmark-run-select", "options"),
        ("benchmark-run-select", "value"),
    }

    linked_callback = None
    output_counts = {target: 0 for target in target_outputs}
    for callback in app.callback_map.values():
        outputs = callback.get("output")
        outputs = outputs if isinstance(outputs, list) else [outputs]
        output_keys = {(_component_id_key(output.component_id), output.component_property) for output in outputs}
        for target in target_outputs & output_keys:
            output_counts[target] += 1
        if output_keys == target_outputs:
            linked_callback = callback

    assert linked_callback is not None
    assert set(output_counts.values()) == {1}
    inputs = {(item["id"], item["property"]) for item in linked_callback.get("inputs") or []}
    assert {("location", "pathname"), ("location", "search")} <= inputs

    edges = set()
    for callback in app.callback_map.values():
        outputs = callback.get("output")
        outputs = outputs if isinstance(outputs, list) else [outputs]
        for callback_input in callback.get("inputs") or []:
            for output in outputs:
                edges.add(
                    (
                        _component_id_key(callback_input["id"]),
                        callback_input["property"],
                        _component_id_key(output.component_id),
                        output.component_property,
                    )
                )

    assert ("location", "pathname", "runs-content", "children") in edges
    assert ("candidate-index-store", "data", "candidate-table-container", "children") in edges
    assert ("candidate-index-store", "data", "candidate-detail", "children") in edges
    assert ("reports-run-select", "options", "report-detail", "children") in edges
    assert ("benchmark-run-select", "options", "benchmark-detail", "children") in edges
