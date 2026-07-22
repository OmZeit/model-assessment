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


INK = "#192124"
MUTED = "#617174"
ACCENT = "#0f766e"
ACCENT_LIGHT = "#5eead4"
INDIGO = "#4f46e5"
SKY = "#0284c7"
DANGER = "#b91c1c"
WARN = "#ea580c"
GRID = "#e7efed"
PALETTE = [ACCENT, INDIGO, SKY, WARN, "#7c3aed", "#db2777"]


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
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
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
        "zerolinecolor": "#cbd9d5",
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
    metric_name = str(report.get("primary_metric_name") or model_metrics.get("primary_metric_name") or "primary metric")
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
    baseline_colors = [SKY, "#7c3aed", "#64748b", "#db2777"]
    baseline_index = 0
    for name, metrics in (report.get("baselines") or {}).items():
        value = _finite((metrics or {}).get("primary_metric"))
        if value is not None:
            names.append(str(name).replace("_", " "))
            values.append(value)
            colors.append(INDIGO if str(name) == best_baseline_name else baseline_colors[baseline_index % len(baseline_colors)])
            baseline_index += 1
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
            marker={"color": INDIGO if selected else SKY},
            opacity=0.78,
            hovertemplate="Residual %{y:.5g}<br>Count %{x}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    _style_figure(figure, title="Residual diagnostics (prediction − measured)", x_title=None, y_title=None)
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


def simulation_landscape_figure(frame: pd.DataFrame) -> go.Figure:
    prediction_col = "predicted_activity" if "predicted_activity" in frame else "assay_prediction"
    uncertainty_col = "model_uncertainty" if "model_uncertainty" in frame else "assay_uncertainty"
    if prediction_col not in frame or uncertainty_col not in frame:
        return empty_figure("Heuristic candidate landscape", "Candidate prediction and uncertainty columns are unavailable.")
    figure = go.Figure()
    passes = frame.get("passes_basic_filters", pd.Series([True] * len(frame), index=frame.index)).astype(bool)
    for passed, label, color in [(True, "Passes simple filters", ACCENT), (False, "Fails simple filters", DANGER)]:
        subset = frame[passes == passed]
        if subset.empty:
            continue
        sizes = pd.to_numeric(subset.get("mlm_plausibility", 0.5), errors="coerce").fillna(0.5)
        sizes = 7 + 12 * sizes.clip(0, 1)
        custom = np.empty((len(subset), 3), dtype=object)
        custom[:, 0] = subset.get("design_id", subset.index).astype(str)
        custom[:, 1] = pd.to_numeric(subset.get("gc_fraction"), errors="coerce")
        custom[:, 2] = pd.to_numeric(subset.get("acquisition_score"), errors="coerce")
        figure.add_trace(
            go.Scatter(
                x=subset[prediction_col],
                y=subset[uncertainty_col],
                mode="markers",
                name=label,
                customdata=custom,
                marker={"color": color, "size": sizes, "opacity": 0.72, "line": {"color": "white", "width": 0.7}},
                hovertemplate="%{customdata[0]}<br>Heuristic activity %{x:.4g}<br>Heuristic uncertainty %{y:.4g}<br>GC %{customdata[1]:.1%}<br>Acquisition %{customdata[2]:.4g}<extra></extra>",
            )
        )
    return _style_figure(figure, title="Heuristic candidate landscape", x_title="Simulated activity proxy", y_title="Simulated uncertainty proxy")


def simulation_constraints_figure(frame: pd.DataFrame, *, gc_low: float, gc_high: float, max_homopolymer: int) -> go.Figure:
    figure = make_subplots(rows=1, cols=2, subplot_titles=("GC fraction", "Maximum homopolymer"), horizontal_spacing=0.12)
    if "gc_fraction" in frame:
        figure.add_trace(go.Histogram(x=frame["gc_fraction"], marker={"color": ACCENT}, opacity=0.82, showlegend=False), row=1, col=1)
        figure.add_vline(x=float(gc_low), line_dash="dash", line_color=WARN, row=1, col=1)
        figure.add_vline(x=float(gc_high), line_dash="dash", line_color=WARN, row=1, col=1)
    if "max_homopolymer" in frame:
        figure.add_trace(go.Histogram(x=frame["max_homopolymer"], marker={"color": "#80aaa2"}, opacity=0.82, showlegend=False), row=1, col=2)
        figure.add_vline(x=float(max_homopolymer), line_dash="dash", line_color=WARN, row=1, col=2)
    _style_figure(figure, title="Sequence constraint distributions", height=360)
    figure.update_xaxes(title_text="GC fraction", row=1, col=1)
    figure.update_yaxes(title_text="Candidates", row=1, col=1)
    figure.update_xaxes(title_text="Longest run (nt)", row=1, col=2)
    figure.update_yaxes(title_text="Candidates", row=1, col=2)
    return figure


def plate_layout_figure(plate_plan: pd.DataFrame) -> go.Figure:
    required = {"plate_row", "plate_column", "role"}
    if plate_plan.empty or not required.issubset(plate_plan.columns):
        return empty_figure("Draft plate layout", "No plate plan was generated.")
    data = plate_plan.copy()
    data["plate_column"] = pd.to_numeric(data["plate_column"], errors="coerce")
    data = data.dropna(subset=["plate_column"])
    rows = sorted(data["plate_row"].astype(str).unique())
    cols = sorted(data["plate_column"].astype(int).unique())
    roles = list(dict.fromkeys(data["role"].astype(str)))
    role_values = {role: index for index, role in enumerate(roles)}
    z = np.full((len(rows), len(cols)), np.nan)
    hover = np.full((len(rows), len(cols)), "Empty", dtype=object)
    row_index = {value: index for index, value in enumerate(rows)}
    col_index = {value: index for index, value in enumerate(cols)}
    for _, item in data.iterrows():
        r = row_index[str(item["plate_row"])]
        c = col_index[int(item["plate_column"])]
        role = str(item["role"])
        z[r, c] = role_values[role]
        hover[r, c] = f"{item.get('well', '')}<br>{role}<br>{item.get('design_id', '')}"
    palette = [ACCENT, "#2563eb", WARN, "#7c3aed", "#64748b", "#db2777", "#0891b2"]
    denom = max(1, len(roles) - 1)
    colorscale: list[list[Any]] = []
    for index, _role in enumerate(roles):
        start = max(0.0, (index - 0.49) / denom)
        end = min(1.0, (index + 0.49) / denom)
        color = palette[index % len(palette)]
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
            colorbar={"tickvals": list(range(len(roles))), "ticktext": [role.replace("_", " ") for role in roles], "title": "Role"},
            xgap=2,
            ygap=2,
        )
    )
    figure.update_yaxes(autorange="reversed")
    return _style_figure(figure, title="Draft randomized plate layout", x_title="Column", y_title="Row", height=420)
