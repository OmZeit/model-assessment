"""Plotly figures for scientific diagnostics.

The functions in this module are deliberately presentation-only: every value
comes from a run report or a persisted row-level artifact.  They never invent
benchmark observations or recompute claim gates.
"""

from __future__ import annotations

import hashlib
import html as html_lib
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


INK = "#e8eef5"
MUTED = "#a8b3c2"
ACCENT = "#5ac8ad"
ACCENT_LIGHT = "#5eead4"
INDIGO = "#818cf8"
SKY = "#38bdf8"
DANGER = "#fb7185"
WARN = "#fb923c"
GRID = "#2b3542"
PALETTE = [ACCENT, INDIGO, SKY, WARN, "#a78bfa", "#f472b6"]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _label(value: str | None, *, fallback: str = "Row") -> str:
    if not value:
        return fallback
    label = str(value).replace("_", " ").replace("-", " ").strip().title()
    return label.replace("Id", "ID").replace("Gc", "GC").replace("Dna", "DNA")


def _selected_key_set(selected_row_keys: set[str] | list[str] | tuple[str, ...] | None) -> set[str]:
    return {str(value) for value in (selected_row_keys or [])}


def _observation_customdata(
    data: pd.DataFrame,
    *,
    id_col: str | None,
    hover_cols: list[str] | None,
    row_key_col: str | None,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Build one stable customdata schema shared by all row-level plots."""
    row_keys = data[row_key_col].astype(str) if row_key_col and row_key_col in data else data.index.astype(str)
    ids = data[id_col].astype(str) if id_col and id_col in data else data.index.astype(str)
    metadata = [
        column
        for column in dict.fromkeys(hover_cols or [])
        if column in data.columns
        and column not in {id_col, row_key_col, "_target", "_prediction", "_residual"}
        and not str(column).startswith("_")
    ][:6]
    values: list[np.ndarray] = [row_keys.to_numpy(dtype=object), ids.to_numpy(dtype=object)]
    for column in metadata:
        series = data[column].astype(object).where(data[column].notna(), "—")
        values.append(series.to_numpy(dtype=object))
    indices = {
        "target": len(values),
        "prediction": len(values) + 1,
        "residual": len(values) + 2,
    }
    values.extend(
        [
            data["_target"].to_numpy(dtype=float),
            data["_prediction"].to_numpy(dtype=float),
            data["_residual"].to_numpy(dtype=float),
        ]
    )
    return np.column_stack(values), metadata, indices


def _observation_hovertemplate(
    *,
    id_col: str | None,
    metadata: list[str],
    indices: dict[str, int],
    view: str,
) -> str:
    lines = [f"<b>{html_lib.escape(_label(id_col))}: %{{customdata[1]}}</b>"]
    for offset, column in enumerate(metadata, start=2):
        lines.append(f"{html_lib.escape(_label(column))}: %{{customdata[{offset}]}}")
    lines.extend(
        [
            f"Measured: %{{customdata[{indices['target']}]:.5g}}",
            f"Predicted: %{{customdata[{indices['prediction']}]:.5g}}",
        ]
    )
    if view in {"residual", "uncertainty"}:
        lines.append(f"Residual: %{{customdata[{indices['residual']}]:.5g}}")
    if view == "uncertainty":
        lines.extend(["Reported uncertainty: %{x:.5g}", "Absolute error: %{y:.5g}"])
    return "<br>".join(lines) + "<extra></extra>"


def _selected_point_indices(
    data: pd.DataFrame,
    *,
    row_key_col: str | None,
    selected_row_keys: set[str] | list[str] | tuple[str, ...] | None,
) -> list[int] | None:
    selected = _selected_key_set(selected_row_keys)
    if not selected:
        return None
    keys = data[row_key_col].astype(str) if row_key_col and row_key_col in data else data.index.astype(str)
    return [index for index, key in enumerate(keys) if key in selected]


def _enable_row_selection(
    figure: go.Figure,
    *,
    uirevision: str,
    selected_row_keys: set[str] | list[str] | tuple[str, ...] | None,
) -> None:
    selected = sorted(_selected_key_set(selected_row_keys))
    selection_digest = hashlib.sha1("\0".join(selected).encode("utf-8")).hexdigest()[:12] if selected else "none"
    figure.update_layout(
        clickmode="event+select",
        dragmode="select",
        uirevision=uirevision,
        selectionrevision=f"linked-row-selection-{selection_digest}",
        newselection={"line": {"color": ACCENT, "width": 2}},
        activeselection={"fillcolor": "rgba(15, 118, 110, 0.08)"},
    )


def _style_figure(
    figure: go.Figure,
    *,
    title: str,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int = 360,
) -> go.Figure:
    figure.update_layout(
        title={"text": title, "font": {"size": 18, "color": INK}, "x": 0.02, "xanchor": "left"},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#141922",
        font={"family": "Inter, ui-sans-serif, system-ui, sans-serif", "color": INK, "size": 13},
        colorway=PALETTE,
        height=height,
        margin={"l": 58, "r": 24, "t": 62, "b": 54},
        hovermode="closest",
        hoverlabel={"bgcolor": "#102421", "bordercolor": "#102421", "font": {"color": "white", "size": 13}, "namelength": -1},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1, "font": {"size": 12}},
    )
    axis_style = {
        "automargin": True,
        "gridcolor": GRID,
        "gridwidth": 1,
        "zerolinecolor": "#526071",
        "showline": False,
        "ticks": "outside",
        "ticklen": 4,
        "tickcolor": GRID,
        "tickfont": {"size": 12, "color": MUTED},
        "title_font": {"size": 13, "color": INK},
    }
    figure.update_xaxes(title_text=x_title, **axis_style)
    figure.update_yaxes(title_text=y_title, **axis_style)
    return figure


def empty_figure(title: str, message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"color": MUTED})
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return _style_figure(figure, title=title)


def benchmark_metric_figure(report: dict[str, Any]) -> go.Figure:
    """Model and actual fitted-baseline primary metrics.

    Only the model receives an interval: the current baseline API persists
    point metrics, not row-level baseline predictions suitable for paired CIs.
    """
    model_metrics = report.get("test_metrics") or report.get("task_head_metrics") or {}
    model_value = _finite(model_metrics.get("primary_metric", report.get("task_head_primary_metric")))
    raw_metric_name = str(report.get("primary_metric_name") or model_metrics.get("primary_metric_name") or "primary metric")
    metric_name = {
        "r2": "Held-out R-squared (higher is better)",
        "r_squared": "Held-out R-squared (higher is better)",
        "mae": "Held-out MAE (lower is better)",
        "rmse": "Held-out RMSE (lower is better)",
        "auroc": "Held-out AUROC (higher is better)",
        "average_precision": "Held-out average precision (higher is better)",
        "balanced_accuracy": "Held-out balanced accuracy (higher is better)",
        "spearman": "Held-out Spearman correlation (higher is better)",
    }.get(raw_metric_name.lower(), raw_metric_name.replace("_", " ").title())
    names: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    descriptions: list[str] = []
    if model_value is not None:
        names.append("Supplied model" if "test_metrics" in report else "Task model")
        values.append(model_value)
        colors.append(ACCENT)
        descriptions.append("Model evaluated on the persisted held-out split")
    baseline_descriptions = {
        "gc_length_baseline": "GC-content and sequence-length baseline",
        "kmer_ridge_or_logistic": "Regularized k-mer linear baseline",
        "kmer_ridge": "Regularized k-mer linear baseline",
        "kmer_random_forest": "Random-forest baseline using k-mer features",
        "kmer4_linear_probe": "Linear probe using 4-mer frequency features",
    }
    best_baseline_name = str((report.get("best_simple_baseline") or {}).get("name") or "")
    for name, metrics in (report.get("baselines") or {}).items():
        value = _finite((metrics or {}).get("primary_metric"))
        if value is not None:
            names.append(str(name).replace("_", " "))
            values.append(value)
            colors.append("#94a3b8" if str(name) == best_baseline_name else "#cbd5e1")
            descriptions.append(baseline_descriptions.get(str(name), "Fitted sequence baseline evaluated on the same held-out split"))
    if not values:
        return empty_figure("Model vs fitted baselines", "No comparable primary metrics were persisted for this run.")

    error_plus = [0.0] * len(values)
    error_minus = [0.0] * len(values)
    interval = (((report.get("claim_gate") or {}).get("lift_claim") or {}).get("model_metric_ci"))
    if model_value is not None and isinstance(interval, (list, tuple)) and len(interval) == 2:
        low, high = _finite(interval[0]), _finite(interval[1])
        if low is not None and high is not None:
            error_minus[0] = max(0.0, model_value - low)
            error_plus[0] = max(0.0, high - model_value)

    figure = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker={"color": colors},
            error_x={"type": "data", "array": error_plus, "arrayminus": error_minus, "visible": any(error_plus)},
            customdata=np.asarray(descriptions, dtype=object).reshape(-1, 1),
            hovertemplate=f"<b>%{{y}}</b><br>{html_lib.escape(metric_name)}: %{{x:.5g}}<br>%{{customdata[0]}}<extra></extra>",
        )
    )
    figure.update_yaxes(categoryorder="array", categoryarray=list(reversed(names)))
    _style_figure(figure, title="Model vs fitted sequence baselines", x_title=metric_name, y_title=None)
    figure.update_layout(bargap=0.38, barcornerradius=7)
    figure.update_yaxes(showgrid=False)
    return figure


def regression_fit_figure(
    frame: pd.DataFrame,
    *,
    target_col: str,
    prediction_col: str,
    id_col: str | None = None,
    hover_cols: list[str] | None = None,
    row_key_col: str | None = None,
    selected_row_keys: set[str] | list[str] | tuple[str, ...] | None = None,
) -> go.Figure:
    data = frame.copy()
    data["_target"] = pd.to_numeric(data.get(target_col), errors="coerce")
    data["_prediction"] = pd.to_numeric(data.get(prediction_col), errors="coerce")
    data = data.dropna(subset=["_target", "_prediction"])
    if data.empty:
        return empty_figure("Measured vs predicted", "Held-out target and prediction values are unavailable.")
    data["_residual"] = data["_prediction"] - data["_target"]
    low = float(min(data["_target"].min(), data["_prediction"].min()))
    high = float(max(data["_target"].max(), data["_prediction"].max()))
    padding = max((high - low) * 0.05, 1e-9)
    customdata, metadata, indices = _observation_customdata(
        data,
        id_col=id_col,
        hover_cols=hover_cols,
        row_key_col=row_key_col,
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=data["_target"],
            y=data["_prediction"],
            mode="markers",
            name=f"Held-out rows (n={len(data)})",
            customdata=customdata,
            selectedpoints=_selected_point_indices(data, row_key_col=row_key_col, selected_row_keys=selected_row_keys),
            marker={"color": ACCENT, "size": 8, "opacity": 0.76, "line": {"color": "white", "width": 0.8}},
            selected={"marker": {"color": ACCENT, "size": 10, "opacity": 0.98}},
            unselected={"marker": {"opacity": 0.14}},
            hovertemplate=_observation_hovertemplate(id_col=id_col, metadata=metadata, indices=indices, view="fit"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[low - padding, high + padding],
            y=[low - padding, high + padding],
            mode="lines",
            name="Ideal 1:1",
            line={"color": MUTED, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    figure.update_xaxes(range=[low - padding, high + padding])
    figure.update_yaxes(range=[low - padding, high + padding], scaleanchor="x", scaleratio=1)
    _style_figure(figure, title="Measured vs predicted on held-out rows", x_title="Measured", y_title="Predicted")
    _enable_row_selection(figure, uirevision="held-out-fit", selected_row_keys=selected_row_keys)
    return figure


def residual_figure(
    frame: pd.DataFrame,
    *,
    target_col: str,
    prediction_col: str,
    id_col: str | None = None,
    hover_cols: list[str] | None = None,
    row_key_col: str | None = None,
    selected_row_keys: set[str] | list[str] | tuple[str, ...] | None = None,
) -> go.Figure:
    data = frame.copy()
    data["_target"] = pd.to_numeric(data.get(target_col), errors="coerce")
    data["_prediction"] = pd.to_numeric(data.get(prediction_col), errors="coerce")
    data = data.dropna(subset=["_target", "_prediction"])
    if data.empty:
        return empty_figure("Residual diagnostics", "Held-out target and prediction values are unavailable.")
    data["_residual"] = data["_prediction"] - data["_target"]
    customdata, metadata, indices = _observation_customdata(
        data,
        id_col=id_col,
        hover_cols=hover_cols,
        row_key_col=row_key_col,
    )
    selected = _selected_key_set(selected_row_keys)
    histogram_data = data
    if selected:
        keys = data[row_key_col].astype(str) if row_key_col and row_key_col in data else data.index.astype(str)
        histogram_data = data[keys.isin(selected)]
    figure = make_subplots(rows=1, cols=2, column_widths=[0.68, 0.32], horizontal_spacing=0.12, subplot_titles=("Residual vs prediction", "Residual distribution"))
    figure.add_trace(
        go.Scatter(
            x=data["_prediction"],
            y=data["_residual"],
            customdata=customdata,
            mode="markers",
            selectedpoints=_selected_point_indices(data, row_key_col=row_key_col, selected_row_keys=selected_row_keys),
            marker={"color": ACCENT, "size": 8, "opacity": 0.76, "line": {"color": "white", "width": 0.8}},
            selected={"marker": {"color": ACCENT, "size": 10, "opacity": 0.98}},
            unselected={"marker": {"opacity": 0.14}},
            hovertemplate=_observation_hovertemplate(id_col=id_col, metadata=metadata, indices=indices, view="residual"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_hline(y=0, line_dash="dash", line_color=MUTED, row=1, col=1)
    figure.add_trace(
        go.Histogram(
            y=histogram_data["_residual"],
            nbinsy=max(6, min(20, int(len(histogram_data) ** 0.5 * 2))),
            marker={"color": INDIGO if selected else SKY, "line": {"color": "white", "width": 1}},
            opacity=0.82,
            hovertemplate="Residual %{y:.5g}<br>Count %{x}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    _style_figure(figure, title="Residual diagnostics (prediction - measured)", x_title=None, y_title=None)
    figure.update_layout(bargap=0.12)
    figure.update_xaxes(title_text="Prediction", row=1, col=1)
    figure.update_yaxes(title_text="Residual", row=1, col=1)
    figure.update_xaxes(title_text="Count", row=1, col=2)
    figure.update_yaxes(title_text="Residual", row=1, col=2)
    _enable_row_selection(figure, uirevision="held-out-residual", selected_row_keys=selected_row_keys)
    return figure


def classification_discrimination_figure(
    frame: pd.DataFrame,
    *,
    target_col: str,
    prediction_col: str,
    positive_label: Any,
) -> go.Figure:
    scores = pd.to_numeric(frame.get(prediction_col), errors="coerce")
    labels = frame.get(target_col)
    valid = scores.notna() & labels.notna()
    scores = scores[valid].to_numpy(dtype=float)
    labels = (labels[valid].astype(str).to_numpy() == str(positive_label)).astype(int)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if not len(labels) or positives == 0 or negatives == 0:
        return empty_figure("Classification discrimination", "ROC/PR requires both positive and negative held-out labels.")
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    tp = np.cumsum(ordered)
    fp = np.cumsum(1 - ordered)
    tpr = np.r_[0.0, tp / positives]
    fpr = np.r_[0.0, fp / negatives]
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / positives
    figure = make_subplots(rows=1, cols=2, subplot_titles=("ROC", "Precision–recall"), horizontal_spacing=0.12)
    figure.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line={"color": ACCENT, "width": 3}, name="Model", hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>"), row=1, col=1)
    figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"color": MUTED, "dash": "dash"}, name="Chance", hoverinfo="skip"), row=1, col=1)
    figure.add_trace(go.Scatter(x=np.r_[0.0, recall], y=np.r_[1.0, precision], mode="lines", line={"color": WARN, "width": 3}, name="Model PR", hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>"), row=1, col=2)
    figure.add_hline(y=positives / len(labels), line_dash="dash", line_color=MUTED, row=1, col=2)
    _style_figure(figure, title=f"Held-out classification discrimination (n={len(labels)})", height=370)
    figure.update_xaxes(title_text="False-positive rate", range=[0, 1], row=1, col=1)
    figure.update_yaxes(title_text="True-positive rate", range=[0, 1], row=1, col=1)
    figure.update_xaxes(title_text="Recall", range=[0, 1], row=1, col=2)
    figure.update_yaxes(title_text="Precision", range=[0, 1], row=1, col=2)
    return figure


def ranking_diagnostics_figure(
    frame: pd.DataFrame,
    *,
    target_col: str,
    prediction_col: str,
    objective_direction: str = "maximize",
) -> go.Figure:
    """Rank agreement and top-k recovery for persisted held-out observations."""
    data = frame.copy()
    data["_target"] = pd.to_numeric(data.get(target_col), errors="coerce")
    data["_prediction"] = pd.to_numeric(data.get(prediction_col), errors="coerce")
    data = data.dropna(subset=["_target", "_prediction"]).reset_index(drop=True)
    if len(data) < 2:
        return empty_figure("Held-out ranking diagnostics", "At least two comparable held-out rows are required.")

    ascending = str(objective_direction).strip().lower() == "minimize"
    data["_measured_rank"] = data["_target"].rank(method="average", ascending=ascending)
    data["_predicted_rank"] = data["_prediction"].rank(method="average", ascending=ascending)
    n_rows = len(data)
    predicted_order = data.sort_values("_prediction", ascending=ascending, kind="stable").index.to_list()
    measured_order = data.sort_values("_target", ascending=ascending, kind="stable").index.to_list()
    predicted_top: set[int] = set()
    measured_top: set[int] = set()
    overlap: list[float] = []
    for index in range(n_rows):
        predicted_top.add(int(predicted_order[index]))
        measured_top.add(int(measured_order[index]))
        overlap.append(len(predicted_top & measured_top) / float(index + 1))

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Measured rank vs predicted rank", "Top-k set recovery"),
        horizontal_spacing=0.12,
    )
    figure.add_trace(
        go.Scatter(
            x=data["_measured_rank"],
            y=data["_predicted_rank"],
            mode="markers",
            marker={"color": ACCENT, "size": 8, "opacity": 0.72, "line": {"color": "white", "width": 0.7}},
            customdata=np.stack([data["_target"], data["_prediction"]], axis=1),
            hovertemplate="Measured rank %{x:.3g}<br>Predicted rank %{y:.3g}<br>Measured %{customdata[0]:.5g}<br>Predicted %{customdata[1]:.5g}<extra></extra>",
            name="Held-out rows",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[1, n_rows],
            y=[1, n_rows],
            mode="lines",
            line={"color": MUTED, "dash": "dash"},
            hoverinfo="skip",
            name="Perfect agreement",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=list(range(1, n_rows + 1)),
            y=overlap,
            mode="lines",
            line={"color": INDIGO, "width": 3},
            hovertemplate="k=%{x}<br>Shared top-k fraction %{y:.1%}<extra></extra>",
            name="Shared top-k fraction",
        ),
        row=1,
        col=2,
    )
    _style_figure(figure, title=f"Held-out ranking diagnostics (n={n_rows})", height=390)
    figure.update_xaxes(title_text="Measured rank (1 = best)", autorange="reversed", row=1, col=1)
    figure.update_yaxes(title_text="Predicted rank (1 = best)", autorange="reversed", row=1, col=1)
    figure.update_xaxes(title_text="Top k", row=1, col=2)
    figure.update_yaxes(title_text="Shared fraction", range=[0, 1.02], tickformat=".0%", row=1, col=2)
    return figure


def uncertainty_figure(
    frame: pd.DataFrame,
    *,
    target_col: str,
    prediction_col: str,
    uncertainty_col: str,
    audit: dict[str, Any] | None = None,
    id_col: str | None = None,
    hover_cols: list[str] | None = None,
    row_key_col: str | None = None,
    selected_row_keys: set[str] | list[str] | tuple[str, ...] | None = None,
) -> go.Figure:
    data = frame.copy()
    data["_target"] = pd.to_numeric(data.get(target_col), errors="coerce")
    data["_prediction"] = pd.to_numeric(data.get(prediction_col), errors="coerce")
    data["_uncertainty"] = pd.to_numeric(data.get(uncertainty_col), errors="coerce")
    data = data.dropna(subset=["_target", "_prediction", "_uncertainty"])
    if data.empty:
        return empty_figure("Uncertainty usefulness", "No held-out uncertainty values were persisted.")
    data["_residual"] = data["_prediction"] - data["_target"]
    data["_abs_error"] = (data["_prediction"] - data["_target"]).abs()
    customdata, metadata, indices = _observation_customdata(
        data,
        id_col=id_col,
        hover_cols=hover_cols,
        row_key_col=row_key_col,
    )
    audit = audit or {}
    declared_type = str(audit.get("declared_type") or "none")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=data["_uncertainty"],
            y=data["_abs_error"],
            mode="markers",
            name="Held-out rows",
            customdata=customdata,
            selectedpoints=_selected_point_indices(data, row_key_col=row_key_col, selected_row_keys=selected_row_keys),
            marker={"color": SKY, "size": 8, "opacity": 0.62, "line": {"color": "white", "width": 0.8}},
            selected={"marker": {"color": ACCENT, "size": 10, "opacity": 0.98}},
            unselected={"marker": {"opacity": 0.14}},
            hovertemplate=_observation_hovertemplate(id_col=id_col, metadata=metadata, indices=indices, view="uncertainty"),
        )
    )
    bins = pd.DataFrame([] if _selected_key_set(selected_row_keys) else (audit.get("bins") or []))
    if not bins.empty and {"mean_uncertainty", "mean_abs_error"}.issubset(bins.columns):
        figure.add_trace(
            go.Scatter(
                x=bins["mean_uncertainty"],
                y=bins["mean_abs_error"],
                mode="lines+markers",
                name="Quantile-bin means",
                marker={"color": WARN, "size": 9},
                line={"color": WARN, "width": 2},
                customdata=np.asarray(bins.get("count", pd.Series([None] * len(bins)))).reshape(-1, 1),
                hovertemplate="Mean uncertainty %{x:.5g}<br>Mean absolute error %{y:.5g}<br>n=%{customdata[0]}<extra></extra>",
            )
        )
    if declared_type == "predicted_absolute_error":
        high = float(max(data["_uncertainty"].max(), data["_abs_error"].max()))
        figure.add_trace(go.Scatter(x=[0, high], y=[0, high], mode="lines", name="Equal error scale", line={"color": MUTED, "dash": "dash"}, hoverinfo="skip"))
        title = "Absolute-error calibration diagnostic"
    elif declared_type == "none":
        title = "Uncertainty usefulness (semantics undeclared)"
    else:
        title = "Uncertainty usefulness / error ordering"
    _style_figure(figure, title=title, x_title="Reported uncertainty", y_title="Absolute error")
    _enable_row_selection(figure, uirevision="held-out-uncertainty", selected_row_keys=selected_row_keys)
    return figure


def split_composition_figure(split_diagnostics: dict[str, Any]) -> go.Figure:
    sizes = split_diagnostics.get("split_sizes") or {}
    names = [name for name in ["train", "val", "test"] if name in sizes]
    values = [int(sizes.get(name) or 0) for name in names]
    if not names:
        return empty_figure("Leakage-aware split composition", "Split sizes were not persisted.")
    figure = go.Figure(go.Bar(x=names, y=values, marker={"color": [INDIGO, WARN, ACCENT]}, text=values, textposition="outside", hovertemplate="%{x}: %{y} rows<extra></extra>"))
    return _style_figure(figure, title=f"Leakage-aware split composition · {split_diagnostics.get('num_clusters', '—')} clusters", x_title="Split", y_title="Rows", height=330)


def regression_slice_summary(
    frame: pd.DataFrame,
    *,
    target_col: str,
    prediction_col: str,
    slice_cols: list[str],
    minimum_n: int = 5,
) -> pd.DataFrame:
    """Descriptive subgroup errors, explicitly flagged when sample sizes are small."""
    data = frame.copy()
    data["_target"] = pd.to_numeric(data.get(target_col), errors="coerce")
    data["_prediction"] = pd.to_numeric(data.get(prediction_col), errors="coerce")
    data = data.dropna(subset=["_target", "_prediction"])
    data["_error"] = data["_prediction"] - data["_target"]
    rows: list[dict[str, Any]] = []
    for column in slice_cols:
        if column not in data or data[column].nunique(dropna=True) > 50:
            continue
        for value, group in data.groupby(column, dropna=False):
            n = len(group)
            rows.append(
                {
                    "slice": column,
                    "level": "(missing)" if pd.isna(value) else str(value),
                    "n": n,
                    "mae": float(group["_error"].abs().mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(group["_error"])))),
                    "mean_bias": float(group["_error"].mean()),
                    "stability": "low n — descriptive only" if n < minimum_n else "descriptive",
                }
            )
    if not rows:
        return pd.DataFrame(columns=["slice", "level", "n", "mae", "rmse", "mean_bias", "stability"])
    return pd.DataFrame(rows).sort_values(["mae", "n"], ascending=[False, True]).reset_index(drop=True)


def simulation_landscape_figure(
    frame: pd.DataFrame,
    *,
    model_backed: bool = False,
    target_label: str | None = None,
    selected_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> go.Figure:
    prediction_col = "predicted_activity" if "predicted_activity" in frame else "assay_prediction"
    uncertainty_col = "model_uncertainty" if "model_uncertainty" in frame else "assay_uncertainty"
    title = "Model prediction and uncertainty" if model_backed else "Heuristic candidate landscape"
    if prediction_col not in frame or uncertainty_col not in frame:
        return empty_figure(title, "Candidate prediction and uncertainty columns are unavailable.")
    figure = go.Figure()
    selected = {str(value) for value in (selected_ids or [])}
    passes = frame.get("passes_basic_filters", pd.Series([True] * len(frame), index=frame.index)).astype(bool)
    for passed, label, color, symbol in [
        (True, "✓ Eligible", ACCENT, "circle"),
        (False, "× Excluded", DANGER, "x"),
    ]:
        subset = frame[passes == passed]
        if subset.empty:
            continue
        sizes = pd.to_numeric(subset.get("mlm_plausibility", 0.5), errors="coerce").fillna(0.5)
        sizes = 7 + 12 * sizes.clip(0, 1)
        custom = np.empty((len(subset), 3), dtype=object)
        custom[:, 0] = subset.get("design_id", subset.index).astype(str)
        custom[:, 1] = pd.to_numeric(subset.get("gc_fraction"), errors="coerce")
        custom[:, 2] = pd.to_numeric(subset.get("acquisition_score"), errors="coerce")
        prediction_name = _label(target_label, fallback="Model prediction") if model_backed else "Heuristic activity"
        uncertainty_name = "Ensemble disagreement (SD)" if model_backed else "Heuristic uncertainty"
        design_ids = subset.get("design_id", subset.index).astype(str).tolist()
        selected_points = [index for index, design_id in enumerate(design_ids) if design_id in selected]
        figure.add_trace(
            go.Scatter(
                x=subset[prediction_col],
                y=subset[uncertainty_col],
                mode="markers",
                name=label,
                customdata=custom,
                marker={"color": color, "symbol": symbol, "size": sizes, "opacity": 0.78, "line": {"color": "#0d1117", "width": 0.8}},
                selectedpoints=selected_points if selected else None,
                selected={"marker": {"opacity": 1.0, "size": 18}},
                unselected={"marker": {"opacity": 0.24}} if selected else None,
                hovertemplate=f"%{{customdata[0]}}<br>{prediction_name} %{{x:.4g}}<br>{uncertainty_name} %{{y:.4g}}<br>GC %{{customdata[1]:.1%}}<br>Acquisition %{{customdata[2]:.4g}}<extra></extra>",
            )
        )
    x_title = _label(target_label, fallback="Predicted assay activity") if model_backed else "Simulated activity proxy"
    y_title = "Ensemble disagreement (SD)" if model_backed else "Simulated uncertainty proxy"
    _style_figure(figure, title=title, x_title=x_title, y_title=y_title)
    figure.update_layout(
        uirevision="simulation-landscape",
        selectionrevision="simulation-candidate-selection",
        clickmode="event+select",
        dragmode="lasso",
    )
    return figure


def simulation_generation_figure(frame: pd.DataFrame, *, parent_sequence: str, mode: str) -> go.Figure:
    """Describe what the selected generator actually changed, without inferring biology."""
    if "sequence" not in frame or frame.empty:
        return empty_figure("Generation diagnostics", "No generated sequences are available.")
    parent = "".join(str(parent_sequence or "").upper().replace("[MASK]", "N").split())
    sequences = ["".join(str(value).upper().split()) for value in frame["sequence"]]
    comparable = [sequence for sequence in sequences if len(sequence) == len(parent)]
    if not parent or not comparable:
        return empty_figure("Generation diagnostics", "Generated sequences are not length-compatible with the parent.")

    if mode == "Mask-fill":
        masked = [index for index, base in enumerate(parent) if base == "N"]
        if not masked:
            return empty_figure("Masked-position composition", "The parent contains no N or [MASK] positions.")
        figure = go.Figure()
        for base, color in [("A", ACCENT), ("C", SKY), ("G", INDIGO), ("T", WARN)]:
            fractions = [sum(sequence[index] == base for sequence in comparable) / len(comparable) for index in masked]
            figure.add_trace(
                go.Bar(
                    x=[index + 1 for index in masked],
                    y=fractions,
                    name=base,
                    marker={"color": color},
                    hovertemplate=f"Position %{{x}}<br>{base} %{{y:.1%}}<extra></extra>",
                )
            )
        _style_figure(
            figure,
            title=f"Masked-position nucleotide composition (n={len(comparable)})",
            x_title="1-based parent position",
            y_title="Generated fraction",
            height=370,
        )
        figure.update_layout(barmode="stack", uirevision="simulation-mask-composition")
        figure.update_yaxes(range=[0, 1], tickformat=".0%")
        return figure

    mutation_counts = [sum(parent[index] != sequence[index] for index in range(len(parent)) if parent[index] != "N") for sequence in comparable]
    position_rates = [
        sum(sequence[index] != parent[index] for sequence in comparable) / len(comparable)
        for index in range(len(parent))
        if parent[index] != "N"
    ]
    positions = [index + 1 for index, base in enumerate(parent) if base != "N"]
    if mode == "Diversity sampling":
        sample = comparable[:250]
        nearest_distances: list[float] = []
        for index, sequence in enumerate(sample):
            distances = [
                sum(left != right for left, right in zip(sequence, other, strict=True)) / len(parent)
                for other_index, other in enumerate(sample)
                if other_index != index
            ]
            if distances:
                nearest_distances.append(min(distances))
        parent_distances = [sum(left != right for left, right in zip(sequence, parent, strict=True)) / len(parent) for sequence in sample]
        figure = make_subplots(rows=1, cols=2, subplot_titles=("Distance from parent", "Nearest-neighbor distance"), horizontal_spacing=0.12)
        figure.add_trace(go.Histogram(x=parent_distances, marker={"color": ACCENT}, opacity=0.82, showlegend=False), row=1, col=1)
        figure.add_trace(go.Histogram(x=nearest_distances, marker={"color": INDIGO}, opacity=0.82, showlegend=False), row=1, col=2)
        _style_figure(figure, title=f"Observed sequence spread (sample n={len(sample)})", height=370)
        figure.update_xaxes(title_text="Normalized Hamming distance", row=1, col=1)
        figure.update_yaxes(title_text="Candidates", row=1, col=1)
        figure.update_xaxes(title_text="Normalized Hamming distance", row=1, col=2)
        figure.update_yaxes(title_text="Candidates", row=1, col=2)
        figure.update_layout(uirevision="simulation-diversity-diagnostics")
        return figure

    figure = make_subplots(rows=1, cols=2, subplot_titles=("Mutations per candidate", "Mutation frequency by position"), horizontal_spacing=0.12)
    figure.add_trace(go.Histogram(x=mutation_counts, marker={"color": INDIGO}, opacity=0.82, showlegend=False), row=1, col=1)
    figure.add_trace(go.Bar(x=positions, y=position_rates, marker={"color": ACCENT}, showlegend=False), row=1, col=2)
    _style_figure(figure, title=f"Random-mutagenesis profile (n={len(comparable)})", height=370)
    figure.update_xaxes(title_text="Changed unmasked bases", row=1, col=1)
    figure.update_yaxes(title_text="Candidates", row=1, col=1)
    figure.update_xaxes(title_text="1-based parent position", row=1, col=2)
    figure.update_yaxes(title_text="Changed fraction", range=[0, 1], tickformat=".0%", row=1, col=2)
    figure.update_layout(uirevision="simulation-mutation-profile")
    return figure


def simulation_constraints_figure(frame: pd.DataFrame, *, gc_low: float, gc_high: float, max_homopolymer: int) -> go.Figure:
    figure = make_subplots(rows=1, cols=2, subplot_titles=("GC fraction", "Maximum homopolymer"), horizontal_spacing=0.18)
    if "gc_fraction" in frame:
        figure.add_trace(
            go.Histogram(
                x=frame["gc_fraction"],
                marker={"color": ACCENT},
                opacity=0.82,
                showlegend=False,
                hovertemplate="GC fraction: %{x:.1%}<br>Candidates: %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        figure.add_vline(x=float(gc_low), line_dash="dash", line_color=WARN, row=1, col=1)
        figure.add_vline(x=float(gc_high), line_dash="dash", line_color=WARN, row=1, col=1)
    if "max_homopolymer" in frame:
        homopolymer = pd.to_numeric(frame["max_homopolymer"], errors="coerce").dropna()
        figure.add_trace(
            go.Histogram(
                x=homopolymer,
                xbins={"start": 0.5, "end": max(1.5, float(homopolymer.max()) + 0.5) if not homopolymer.empty else 1.5, "size": 1},
                marker={"color": "#80aaa2"},
                opacity=0.82,
                showlegend=False,
                hovertemplate="Longest run: %{x:.0f} nt<br>Candidates: %{y}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        figure.add_vline(x=float(max_homopolymer), line_dash="dash", line_color=WARN, row=1, col=2)
    _style_figure(figure, title="Sequence constraint distributions", height=360)
    figure.update_layout(margin={"l": 68, "r": 36, "t": 68, "b": 62}, bargap=0.12)
    figure.update_xaxes(title_text="GC fraction", tickformat=".0%", nticks=7, row=1, col=1)
    figure.update_yaxes(title_text="Candidates", row=1, col=1)
    figure.update_xaxes(title_text="Longest run (nt)", dtick=1, tickformat="d", row=1, col=2)
    figure.update_yaxes(title_text="Candidates", row=1, col=2)
    return figure


def plate_layout_figure(plate_plan: pd.DataFrame, *, plate_size: int | None = None) -> go.Figure:
    required = {"plate_row", "plate_column", "role"}
    if plate_plan.empty or not required.issubset(plate_plan.columns):
        return empty_figure("Draft well layout", "No unapproved well-layout preview was generated.")
    data = plate_plan.copy()
    data["plate_column"] = pd.to_numeric(data["plate_column"], errors="coerce")
    data = data.dropna(subset=["plate_column"])
    layout_shapes = {24: (list("ABCD"), list(range(1, 7))), 48: (list("ABCDEF"), list(range(1, 9))), 96: (list("ABCDEFGH"), list(range(1, 13))), 384: (list("ABCDEFGHIJKLMNOP"), list(range(1, 25)))}
    if plate_size in layout_shapes:
        rows, cols = layout_shapes[int(plate_size)]
    else:
        observed_rows = sorted(data["plate_row"].astype(str).unique())
        observed_cols = sorted(data["plate_column"].astype(int).unique())
        inferred = 24 if len(observed_rows) <= 4 and max(observed_cols) <= 6 else 96 if len(observed_rows) <= 8 and max(observed_cols) <= 12 else 384
        rows, cols = layout_shapes[inferred]
    role_order = [
        "empty",
        "prioritized_candidate",
        "exploit_top_prediction",
        "explore_high_uncertainty",
        "diversity_representative",
        "backup_ranked_candidate",
        "positive_control",
        "negative_control",
        "blank_control",
        "process_control",
    ]
    observed_roles = set(data["role"].astype(str))
    roles = [role for role in role_order if role == "empty" or role in observed_roles]
    roles.extend(sorted(observed_roles - set(roles)))
    role_values = {role: index for index, role in enumerate(roles)}
    z = np.full((len(rows), len(cols)), role_values["empty"], dtype=float)
    hover = np.asarray([[f"{row}{col}<br>Empty" for col in cols] for row in rows], dtype=object)
    row_index = {value: index for index, value in enumerate(rows)}
    col_index = {value: index for index, value in enumerate(cols)}
    for _, item in data.iterrows():
        r = row_index[str(item["plate_row"])]
        c = col_index[int(item["plate_column"])]
        role = str(item["role"])
        z[r, c] = role_values[role]
        hover[r, c] = f"{item.get('well', '')}<br>{role}<br>{item.get('design_id', '')}"
    role_colors = {
        "empty": "#252e38",
        "prioritized_candidate": ACCENT,
        "exploit_top_prediction": ACCENT,
        "explore_high_uncertainty": SKY,
        "diversity_representative": INDIGO,
        "backup_ranked_candidate": "#64748b",
        "positive_control": "#2563eb",
        "negative_control": "#64748b",
        "blank_control": "#7c3aed",
        "process_control": WARN,
    }
    denom = max(1, len(roles) - 1)
    colorscale: list[list[Any]] = []
    for index, _role in enumerate(roles):
        start = max(0.0, (index - 0.49) / denom)
        end = min(1.0, (index + 0.49) / denom)
        color = role_colors.get(_role, "#db2777")
        colorscale.extend([[start, color], [end, color]])
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=cols,
            y=rows,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            colorscale=colorscale,
            zmin=0,
            zmax=max(1, len(roles) - 1),
            colorbar={"tickvals": list(range(len(roles))), "ticktext": [role.replace("_", " ") for role in roles], "title": "Role", "len": 0.86},
            xgap=2,
            ygap=2,
        )
    )
    figure.update_yaxes(autorange="reversed")
    figure.update_xaxes(dtick=1, range=[min(cols) - 0.5, max(cols) + 0.5], side="top")
    figure.update_yaxes(categoryorder="array", categoryarray=rows)
    return _style_figure(figure, title=f"Unapproved full {len(rows)} × {len(cols)} well-layout preview", x_title="Column", y_title="Row", height=500)
