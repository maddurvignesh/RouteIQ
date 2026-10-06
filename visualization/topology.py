"""Interactive network-topology figure built with Plotly.

The drawing deliberately uses a small number of traces so that Streamlit can
redraw it quickly:

===========================  ================================================
trace                        content
===========================  ================================================
``Links (healthy)``          grey links
``Links (congested)``        red links (congestion >= threshold)
``Active route``             blue links currently selected by the router
``Active + congested``       amber links that are on the route *and* bad
``Routers``                  node markers (source / destination highlighted)
``Link info``                invisible markers placed at link mid-points that
                             show the full condition summary on hover
===========================  ================================================
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

from config import CONGESTION_HIGH_THRESHOLD
from core.topology import Topology

# Colours tuned for a dark dashboard background: nodes are light enough that
# the dark label text below stays readable.
COLOR_HEALTHY = "#7d8db5"
COLOR_CONGESTED = "#ef4444"
COLOR_ROUTE = "#3b82f6"
COLOR_ROUTE_CONGESTED = "#f59e0b"
COLOR_NODE = "#cbd5e1"
COLOR_SOURCE = "#22c55e"
COLOR_DESTINATION = "#f97316"
COLOR_NODE_ON_ROUTE = "#60a5fa"
COLOR_NODE_TEXT = "#0b1220"
COLOR_NODE_RING = "#f8fafc"

LINK_CATEGORY_ORDER = ("healthy", "congested", "route", "route_congested")


def categorize_links(
    topology: Topology,
    active_route: Sequence[str] | None = None,
    threshold: float = CONGESTION_HIGH_THRESHOLD,
) -> dict[str, list[tuple[str, str]]]:
    """Split the links into the four drawing categories."""
    route_links = set()
    if active_route:
        route_links = {tuple(sorted(e)) for e in zip(active_route, list(active_route)[1:])}

    buckets: dict[str, list[tuple[str, str]]] = {name: [] for name in LINK_CATEGORY_ORDER}
    for u, v in topology.links:
        key = tuple(sorted((u, v)))
        congested = topology.is_congested(u, v, threshold)
        on_route = key in route_links
        if on_route and congested:
            buckets["route_congested"].append((u, v))
        elif on_route:
            buckets["route"].append((u, v))
        elif congested:
            buckets["congested"].append((u, v))
        else:
            buckets["healthy"].append((u, v))
    return buckets


def _link_hover_text(topology: Topology, u: str, v: str, on_route: bool, ml_state: str | None) -> str:
    attrs = topology.link_attributes(u, v)
    utilization = topology.utilization(u, v)
    parts = [
        f"<b>{u} \u2194 {v}</b>",
        f"Latency: {attrs['latency']:.1f} ms",
        f"Bandwidth: {attrs['bandwidth']:.0f} Mbps",
        f"Traffic: {attrs['traffic']:.0f} Mbps (utilisation {utilization:.0%})",
        f"Packet loss: {attrs['packet_loss'] * 100:.2f} %",
        f"Congestion: {attrs['congestion']:.0%}",
        f"Reliability: {attrs['reliability'] * 100:.1f} %",
    ]
    if ml_state:
        parts.append(f"ML prediction: <b>{ml_state}</b>")
    parts.append("Route: " + ("active" if on_route else "inactive"))
    return "<br>".join(parts)


def build_topology_figure(
    topology: Topology,
    active_route: Sequence[str] | None = None,
    src: str | None = None,
    dst: str | None = None,
    ml_states: Mapping[tuple[str, str], str] | None = None,
    title: str = "Network topology",
    height: int = 560,
) -> go.Figure:
    """Render the network with its live link conditions and the active route."""
    positions = topology.position_map()
    route_set = {tuple(sorted(e)) for e in zip(active_route, list(active_route)[1:])} if active_route else set()
    buckets = categorize_links(topology, active_route)
    colors = {
        "healthy": COLOR_HEALTHY,
        "congested": COLOR_CONGESTED,
        "route": COLOR_ROUTE,
        "route_congested": COLOR_ROUTE_CONGESTED,
    }
    labels = {
        "healthy": "Healthy links",
        "congested": "Congested links",
        "route": "Active route",
        "route_congested": "Active route (congested)",
    }

    figure = go.Figure()

    # -- links -------------------------------------------------------------
    for category in LINK_CATEGORY_ORDER:
        xs: list[float | None] = []
        ys: list[float | None] = []
        for u, v in buckets[category]:
            x0, y0 = positions[u]
            x1, y1 = positions[v]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        if not xs:
            continue
        figure.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(
                color=colors[category],
                width=5 if category.startswith("route") else 2.5,
            ),
            name=labels[category],
            hoverinfo="skip",
            showlegend=True,
        ))

    # -- invisible hover markers at the link mid-points ---------------------
    mid_x: list[float] = []
    mid_y: list[float] = []
    mid_text: list[str] = []
    for u, v in topology.links:
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        key = tuple(sorted((u, v)))
        ml_state = ml_states.get(key) or ml_states.get((u, v)) if ml_states else None
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        mid_text.append(_link_hover_text(topology, u, v, key in route_set, ml_state))
    figure.add_trace(go.Scatter(
        x=mid_x, y=mid_y,
        mode="markers",
        marker=dict(size=14, color="rgba(0,0,0,0)"),
        text=mid_text,
        hoverinfo="text",
        hoverlabel=dict(bgcolor="#0f172a", font_color="#f8fafc", align="left"),
        name="Link details",
        showlegend=False,
    ))

    # -- routers ------------------------------------------------------------
    route_nodes = set(active_route) if active_route else set()
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in topology.routers:
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        role = "Router"
        color = COLOR_NODE
        if node == src:
            role, color = "SOURCE", COLOR_SOURCE
        elif node == dst:
            role, color = "DESTINATION", COLOR_DESTINATION
        elif node in route_nodes:
            role, color = "on active route", COLOR_NODE_ON_ROUTE
        degree = len(topology.neighbours(node))
        node_text.append(f"<b>{node}</b><br>{role}<br>links: {degree}")
        node_color.append(color)
        node_size.append(26 if node in (src, dst) else 20)

    figure.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=node_size, color=node_color,
                    line=dict(width=2, color=COLOR_NODE_RING)),
        text=list(topology.routers),
        textposition="top center",
        textfont=dict(size=13, color=COLOR_NODE_TEXT),
        hovertext=node_text,
        hoverinfo="text",
        name="Routers",
        showlegend=False,
    ))

    route_title = " -> ".join(active_route) if active_route else "no route selected"
    figure.update_layout(
        title=dict(text=f"{title}<br><sup>Active route: {route_title}</sup>",
                   font=dict(size=18)),
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.6, 5.0]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 2.4]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=18, color="#e2e8f0"),
        margin=dict(l=20, r=20, t=90, b=20),
        hoverlabel=dict(font_size=12),
    )
    return figure


def build_link_table(
    topology: Topology,
    active_route: Sequence[str] | None = None,
    threshold: float = CONGESTION_HIGH_THRESHOLD,
) -> pd.DataFrame:
    """Tabular view of every link with its condition and route status."""
    route_set = {tuple(sorted(e)) for e in zip(active_route, list(active_route)[1:])} if active_route else set()
    rows = []
    for u, v, data in sorted(topology.graph.edges(data=True)):
        key = tuple(sorted((u, v)))
        congested = topology.is_congested(u, v, threshold)
        rows.append({
            "Link": f"{u}-{v}",
            "Latency (ms)": round(data["latency"], 1),
            "Bandwidth (Mbps)": round(data["bandwidth"], 0),
            "Traffic (Mbps)": round(data["traffic"], 0),
            "Utilisation": round(topology.utilization(u, v), 3),
            "Packet loss": round(data["packet_loss"], 4),
            "Congestion": round(data["congestion"], 3),
            "Reliability": round(data["reliability"], 3),
            "Status": "Congested" if congested else "Healthy",
            "Active route": "Yes" if key in route_set else "",
        })
    return pd.DataFrame(rows)
