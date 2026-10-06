"""Plotly chart builders shared by the dashboard.

Every function takes plain data (``DataFrame`` / ``list``) and returns a
``plotly.graph_objects.Figure`` so the dashboard stays declarative and free of
chart-configuration code.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import ML_STATE_LABELS, STRATEGY_ADAPTIVE, STRATEGY_ADAPTIVE_ML, STRATEGY_SHORTEST

STRATEGY_COLORS = {
    STRATEGY_SHORTEST: "#dc2626",
    STRATEGY_ADAPTIVE: "#2563eb",
    STRATEGY_ADAPTIVE_ML: "#16a34a",
}
STRATEGY_ORDER = [STRATEGY_SHORTEST, STRATEGY_ADAPTIVE, STRATEGY_ADAPTIVE_ML]

CONGESTION_COLORS = {0: "#16a34a", 1: "#f59e0b", 2: "#dc2626"}

# Transparent paper + plot backgrounds let the animated dashboard background
# show through the chart, so the figures blend in instead of sitting on a
# white rectangle.
BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0"),
    margin=dict(l=40, r=20, t=70, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)


def empty_figure(message: str = "No data yet - run an experiment first.") -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=15, color="#64748b"),
    )
    figure.update_layout(height=320, **BASE_LAYOUT)
    return figure


def _apply(figure: go.Figure, title: str, height: int = 420) -> go.Figure:
    figure.update_layout(title=dict(text=title, font=dict(size=17)), height=height, **BASE_LAYOUT)
    return figure


# ---------------------------------------------------------------------------
# Experiment / comparison charts
# ---------------------------------------------------------------------------

def metric_by_scenario(
    frame: pd.DataFrame, metric: str, title: str, yaxis: str = ""
) -> go.Figure:
    """Grouped bar chart: one group per scenario, one bar per strategy."""
    if frame.empty or metric not in frame.columns:
        return empty_figure()

    working = frame.copy()
    working["strategy"] = pd.Categorical(working["strategy"], STRATEGY_ORDER, ordered=True)
    working = working.sort_values("strategy")

    figure = go.Figure()
    for strategy in STRATEGY_ORDER:
        subset = working[working["strategy"] == strategy]
        if subset.empty:
            continue
        figure.add_trace(go.Bar(
            x=subset["scenario"],
            y=subset[metric],
            name=strategy,
            marker_color=STRATEGY_COLORS.get(strategy, "#64748b"),
            text=[f"{value:.1f}" for value in subset[metric]],
            textposition="outside",
        ))
    figure.update_layout(barmode="group")
    figure.update_yaxes(title_text=yaxis or metric)
    return _apply(figure, title)


def summary_metric_bar(summary: pd.DataFrame, metric: str, title: str, yaxis: str = "") -> go.Figure:
    """Bar chart of one averaged metric across the three routing methods."""
    if summary.empty or metric not in summary.columns:
        return empty_figure()

    working = summary.copy()
    working["strategy"] = pd.Categorical(working["strategy"], STRATEGY_ORDER, ordered=True)
    working = working.sort_values("strategy")

    figure = go.Figure(go.Bar(
        x=working["strategy"].astype(str),
        y=working[metric],
        marker_color=[STRATEGY_COLORS.get(s, "#64748b") for s in working["strategy"]],
        text=[f"{value:.2f}" for value in working[metric]],
        textposition="outside",
    ))
    figure.update_yaxes(title_text=yaxis or metric)
    return _apply(figure, title, height=380)


def delivery_rate_chart(summary: pd.DataFrame) -> go.Figure:
    return summary_metric_bar(
        summary, "delivery_rate_pct",
        "Successful packet delivery rate (%)", yaxis="Delivery rate (%)",
    )


def latency_chart(summary: pd.DataFrame) -> go.Figure:
    return summary_metric_bar(
        summary, "avg_latency_ms",
        "Average end-to-end latency (ms)", yaxis="Latency (ms)",
    )


# ---------------------------------------------------------------------------
# Live simulation charts
# ---------------------------------------------------------------------------

def route_timeline(history: pd.DataFrame, title: str = "Route cost over simulation steps") -> go.Figure:
    """Line chart of the dynamic route cost with the selected route annotated."""
    if history.empty:
        return empty_figure("No simulation steps recorded yet.")

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=history["step"], y=history["cost"],
        mode="lines+markers+text",
        line=dict(color="#2563eb", width=3),
        marker=dict(size=10, color="#2563eb"),
        text=history["route"].str.replace(" -> ", "\u2192", regex=False),
        textposition="top center",
        textfont=dict(size=9),
        name="Route cost",
    ))
    figure.update_xaxes(title_text="Simulation step", dtick=1)
    figure.update_yaxes(title_text="Dynamic cost (lower is better)")
    return _apply(figure, title, height=400)


def congested_links_timeline(history: pd.DataFrame) -> go.Figure:
    """Stacked area/line showing how many links were congested per step."""
    if history.empty:
        return empty_figure("No simulation steps recorded yet.")

    figure = go.Figure(go.Bar(
        x=history["step"],
        y=history["congested_links"],
        marker_color="#dc2626",
        text=history["condition"],
        textposition="outside",
        name="Congested links",
    ))
    figure.update_xaxes(title_text="Simulation step", dtick=1)
    figure.update_yaxes(title_text="Number of congested links", rangemode="tozero")
    return _apply(figure, "Network congestion over time", height=360)


def utilization_timeline(history: pd.DataFrame) -> go.Figure:
    if history.empty:
        return empty_figure("No simulation steps recorded yet.")

    figure = go.Figure(go.Scatter(
        x=history["step"], y=history["route_utilization_pct"],
        mode="lines+markers",
        line=dict(color="#16a34a", width=3),
        marker=dict(size=9),
        name="Route utilisation",
        fill="tozeroy",
    ))
    figure.update_xaxes(title_text="Simulation step", dtick=1)
    figure.update_yaxes(title_text="Average utilisation of the chosen links (%)",
                        range=[0, 100])
    return _apply(figure, "Utilisation of the selected route", height=340)


def latency_histogram(packets: Sequence, title: str = "End-to-end delay distribution") -> go.Figure:
    rows = [{"delay_ms": p.delay_ms, "status": p.status} for p in packets if p.delivered]
    if not rows:
        return empty_figure("No delivered packets yet.")
    frame = pd.DataFrame(rows)
    figure = px.histogram(
        frame, x="delay_ms", color="status", nbins=30,
        color_discrete_map={"Delivered": "#16a34a"},
        labels={"delay_ms": "Delay (ms)"},
    )
    figure.update_layout(bargap=0.15)
    return _apply(figure, title, height=360)


# ---------------------------------------------------------------------------
# ML charts
# ---------------------------------------------------------------------------

def confusion_matrix_figure(matrix: Sequence[Sequence[int]],
                            labels: Sequence[str] | None = None) -> go.Figure:
    labels = list(labels) or [ML_STATE_LABELS[i] for i in range(3)]
    size = len(matrix)
    annotations = []
    for i in range(size):
        for j in range(size):
            value = matrix[i][j]
            annotations.append(dict(
                x=labels[j], y=labels[i],
                text=str(value), showarrow=False,
                font=dict(color="white" if value > max(map(max, matrix)) * 0.55 else "#0f172a",
                          size=15),
            ))
    figure = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels,
        colorscale="Blues", showscale=True,
        xgap=3, ygap=3,
    ))
    figure.update_layout(
        annotations=annotations,
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    return _apply(figure, "Confusion matrix (test set)", height=420)


def feature_importance_figure(importances: Mapping[str, float]) -> go.Figure:
    if not importances:
        return empty_figure("Train the model to see feature importances.")
    ordered = dict(sorted(importances.items(), key=lambda kv: kv[1]))
    figure = go.Figure(go.Bar(
        x=list(ordered.values()),
        y=list(ordered.keys()),
        orientation="h",
        marker_color="#2563eb",
        text=[f"{value:.3f}" for value in ordered.values()],
        textposition="outside",
    ))
    figure.update_xaxes(title_text="Importance")
    figure.update_yaxes(title_text="")
    return _apply(figure, "Random Forest feature importance", height=380)


def class_distribution_figure(counts: Mapping[int, int]) -> go.Figure:
    if not counts:
        return empty_figure("Generate the dataset to see its class distribution.")
    labels = [ML_STATE_LABELS.get(int(k), str(k)) for k in counts]
    values = list(counts.values())
    figure = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.45,
        marker=dict(colors=[CONGESTION_COLORS.get(int(k), "#64748b") for k in counts]),
        textinfo="label+percent",
    ))
    return _apply(figure, "Dataset class distribution", height=360)


def dataset_histogram(frame: pd.DataFrame, column: str) -> go.Figure:
    if frame.empty or column not in frame.columns:
        return empty_figure("Dataset not available.")
    figure = px.histogram(frame, x=column, nbins=40, color_discrete_sequence=["#2563eb"])
    figure.update_layout(bargap=0.1)
    return _apply(figure, f"Distribution of {column}", height=340)


def current_prediction_bars(predictions: Sequence[Mapping]) -> go.Figure:
    if not predictions:
        return empty_figure("No predictions available.")
    frame = pd.DataFrame([{
        "link": p["link"],
        "level": p["level"],
        "state": p["state"],
        "confidence": p["confidence"],
        "penalty": p["penalty"],
    } for p in predictions]).sort_values("link")

    figure = go.Figure(go.Bar(
        x=frame["link"],
        y=frame["confidence"],
        marker_color=[CONGESTION_COLORS.get(int(level), "#64748b") for level in frame["level"]],
        text=frame["state"],
        textposition="outside",
        hovertemplate="%{x}<br>%{text}<br>confidence %{y:.2f}<extra></extra>",
    ))
    figure.update_yaxes(title_text="Model confidence", range=[0, 1.15])
    figure.update_xaxes(title_text="Link")
    return _apply(figure, "Current congestion prediction per link", height=380)
