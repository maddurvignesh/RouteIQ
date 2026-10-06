"""Generate the figures used by the report and the presentation.

Run::

    python -m docs.diagrams

Outputs (all written next to this file):

========================  ==========================================
``architecture.png``      the layered architecture of the system
``routing_decision.png``  what happens when a route is computed
``topology.png``          the 8-router / 13-link demo topology
``results_summary.png``   key results of the 18 experiment runs
``ml_confusion.png``      classifier confusion matrix
``ml_importance.png``     Random Forest feature importances
========================  ==========================================

The script is deliberately dependency-light (matplotlib only) so it can be
re-run at any time without a browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:          # allows `python docs/diagrams.py`
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK = "#1f2933"
ACCENT = "#2563eb"
MUTED = "#64748b"
GOOD = "#16a34a"
BAD = "#dc2626"
PANEL = "#f1f5f9"


# ---------------------------------------------------------------------------
# 1. layered architecture
# ---------------------------------------------------------------------------

LAYERS = [
    ("Presentation", ["app.py", "dashboard.py (6 pages)", "charts.py", "topology.py"], "#dbeafe"),
    ("Orchestration", ["engine.py", "experiments.py", "scenarios.py", "metrics.py"], "#e0e7ff"),
    ("Routing", ["routing.py (baseline)", "adaptive_routing.py (cost + Dijkstra)"], "#fef3c7"),
    ("Network model", ["topology.py", "network_conditions.py", "packet_simulator.py"], "#dcfce7"),
    ("Machine learning", ["dataset.py", "model.py", "train.py", "predict.py"], "#fae8ff"),
    ("Configuration", ["config.py - every constant in one place"], "#e2e8f0"),
]


def architecture(path: Path = DOCS_DIR / "architecture.png") -> Path:
    fig, ax = plt.subplots(figsize=(11, 7.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(LAYERS) * 1.18 + 0.6)
    ax.axis("off")

    height, gap = 0.86, 0.32
    for index, (name, items, colour) in enumerate(LAYERS):
        bottom = (len(LAYERS) - index - 1) * (height + gap) + 0.35
        box = FancyBboxPatch(
            (0.35, bottom), 9.3, height,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.2, edgecolor=INK, facecolor=colour,
        )
        ax.add_patch(box)
        ax.text(0.65, bottom + height / 2, name, va="center", ha="left",
                fontsize=12, fontweight="bold", color=INK)
        ax.text(2.9, bottom + height / 2, "  |  ".join(items), va="center",
                ha="left", fontsize=10, color=INK)

        if index < len(LAYERS) - 1:
            ax.add_patch(FancyArrowPatch(
                (5.0, bottom), (5.0, bottom - gap + 0.02),
                arrowstyle="-|>", mutation_scale=16, linewidth=1.4, color=MUTED,
            ))

    ax.text(5.0, len(LAYERS) * (height + gap) + 0.42,
            "Adaptive Routing - layered architecture",
            ha="center", fontsize=14, fontweight="bold", color=INK)
    ax.text(5.0, 0.05,
            "ML reaches the router only through a penalty(u, v, attrs) -> float callback",
            ha="center", fontsize=9.5, style="italic", color=MUTED)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. routing decision flow
# ---------------------------------------------------------------------------

def routing_decision(path: Path = DOCS_DIR / "routing_decision.png") -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 8.4))
    ax.set_xlim(0, 10.7)
    ax.set_ylim(0, 11.4)
    ax.axis("off")

    def step(x, y, w, h, text, colour, bold=False, fontsize=10.5):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.14",
            linewidth=1.3, edgecolor=INK, facecolor=colour,
        ))
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight="bold" if bold else "normal", color=INK)

    def arrow(x1, y1, x2, y2, label="", side="right"):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
            linewidth=1.4, color=INK,
            connectionstyle="arc3,rad=0.0",
        ))
        if label:
            offset = 0.16 if side == "right" else -0.16
            ax.text((x1 + x2) / 2 + offset, (y1 + y2) / 2, label,
                    ha="left" if side == "right" else "right",
                    va="center", fontsize=9, color=MUTED)

    step(5, 10.6, 8.6, 0.72, "One routing decision begins\n(live link state at time t)", PANEL, True, 11)

    step(5, 9.35, 6.6, 0.78, "Which strategy?", "#dbeafe", True)
    step(1.9, 7.85, 3.1, 0.9, "Shortest Path\nDijkstra on hop count", "#fef3c7")
    step(8.1, 7.85, 3.1, 0.9, "Adaptive Routing\n/ Adaptive + ML", "#fef3c7")

    arrow(4.0, 9.0, 2.4, 8.35)
    arrow(6.0, 9.0, 7.6, 8.35)

    step(8.1, 6.35, 3.6, 0.82, "Normalise every link\nto [0, 1]", "#dcfce7")
    arrow(8.1, 7.4, 8.1, 6.8)

    step(8.1, 5.0, 4.6, 0.98,
         "cost = 0.35*L + 0.30*C\n+ 0.25*P + 0.10*B\n[+ ML penalty]", "#dcfce7")
    arrow(8.1, 5.9, 8.1, 5.5)

    step(8.1, 3.55, 3.8, 0.82, "Dijkstra with this cost\nas the edge weight", "#e0e7ff")
    arrow(8.1, 4.5, 8.1, 4.0)

    step(5, 2.2, 8.6, 0.86,
         "RouteResult { path, dynamic cost, reason, 4 rejected alternatives }",
         "#dbeafe", True, 10.5)
    arrow(6.6, 3.35, 5.8, 2.7)
    arrow(1.9, 7.4, 3.4, 2.7)

    step(5, 0.95, 8.6, 0.86,
         "PacketSimulator: 80 packets -> latency / loss / throughput",
         "#dcfce7", True, 10.5)
    arrow(5, 1.75, 5, 1.4)

    ax.text(5, 0.18,
            "Penalties are frozen once per decision, for every link, so the choice is stable and explainable",
            ha="center", fontsize=9, style="italic", color=MUTED)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. topology
# ---------------------------------------------------------------------------

def topology(path: Path = DOCS_DIR / "topology.png") -> Path:
    from core.topology import build_predefined_topology

    topo = build_predefined_topology()
    positions = topo.position_map()
    route = ["S", "A", "C", "T"]

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    pad_x, pad_y = 0.9, 0.8
    width_units = (max(xs) - min(xs)) + 2 * pad_x
    height_units = (max(ys) - min(ys)) + 2 * pad_y

    # Match the figure ratio to the data ratio so the axes are not letterboxed.
    width_in = 10.0
    height_in = max(3.2, width_in * height_units / width_units)

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.axis("off")

    route_links = set()
    for u, v in zip(route, route[1:]):
        route_links.add(frozenset((u, v)))

    for u, v in topo.links:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        on_route = frozenset((u, v)) in route_links
        congested = topo.is_congested(u, v)
        colour = BAD if congested else (ACCENT if on_route else MUTED)
        width = 3.2 if on_route else 1.6
        ax.plot([x1, x2], [y1, y2], color=colour, linewidth=width,
                solid_capstyle="round", zorder=1)

        midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
        label = (f"{int(topo.get(u, v, 'latency'))} ms\n"
                 f"{int(topo.get(u, v, 'bandwidth'))} Mbps")
        ax.text(midpoint[0], midpoint[1] + 0.14, label, fontsize=7.6,
                ha="center", va="bottom", color=INK, zorder=4,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))

    for node, (x, y) in positions.items():
        is_endpoint = node in ("S", "T")
        ax.scatter([x], [y], s=1500 if is_endpoint else 1100,
                   c=GOOD if node == "S" else (ACCENT if node == "T" else "#ffffff"),
                   edgecolors=INK, linewidths=1.6, zorder=3)
        ax.text(x, y, node, ha="center", va="center", fontsize=13,
                fontweight="bold", color="white" if is_endpoint else INK, zorder=4)

    ax.set_title("Demo topology - 8 routers, 13 links "
                 "(blue: 3-hop shortest path S-A-C-T)",
                 fontsize=13, fontweight="bold", color=INK, pad=14)
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(max(ys) + pad_y, min(ys) - pad_y)   # inverted: y grows downwards

    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 4. results summary
# ---------------------------------------------------------------------------

def results_summary(path: Path = DOCS_DIR / "results_summary.png") -> Path:
    import pandas as pd

    from config import COMPARISON_SUMMARY_CSV

    frame = pd.read_csv(COMPARISON_SUMMARY_CSV)
    strategies = frame["strategy"].tolist()
    colours = ["#94a3b8", "#2563eb", "#16a34a"]

    panels = [
        ("Avg latency (ms)", "avg_latency_ms", False),
        ("Packet loss (%)", "packet_loss_pct", False),
        ("Delivery rate (%)", "delivery_rate_pct", True),
        ("Steps on a congested path (%)", "congestion_frequency_pct", False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.4))
    for ax, (title, column, annotate) in zip(axes, panels):
        values = frame[column].tolist()
        bars = ax.bar(strategies, values, color=colours, width=0.62)
        ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, pad=10)
        ax.tick_params(axis="x", rotation=18, labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        top = max(values)
        ax.set_ylim(0, top * 1.22 if top > 0 else 1)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.04,
                    f"{value:.2f}", ha="center", va="bottom", fontsize=9, color=INK)

    fig.suptitle("Six-scenario average - 18 runs, 7 200 simulated packets, seed 42",
                 fontsize=13, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 5. ML evaluation figures
# ---------------------------------------------------------------------------

def ml_confusion(path: Path = DOCS_DIR / "ml_confusion.png") -> Path:
    import numpy as np

    from ml.train import load_training_report

    report = load_training_report()
    matrix = np.array(report["confusion_matrix"])
    labels = ["Normal", "Warning", "Congested"]

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion matrix - {report['accuracy']:.2%} test accuracy",
                 fontsize=13, fontweight="bold", color=INK, pad=12)
    ax.text(0.0, -0.13, f"{report['test_size']} held-out samples",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, color=MUTED)

    threshold = matrix.max() / 2.0
    for i in range(3):
        for j in range(3):
            value = matrix[i, j]
            ax.text(j, i, str(value), ha="center", va="center", fontsize=13,
                    fontweight="bold", color="white" if value > threshold else INK)

    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def ml_importance(path: Path = DOCS_DIR / "ml_importance.png") -> Path:
    from ml.train import load_training_report

    importance = load_training_report()["feature_importances"]
    items = sorted(importance.items(), key=lambda kv: kv[1])
    names = [name.replace("_", " ") for name, _ in items]
    values = [100.0 * value for _, value in items]

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    bars = ax.barh(names, values, color="#7c3aed", height=0.62)
    ax.set_xlabel("Importance (%)", fontsize=11)
    ax.set_title("Random Forest feature importances", fontsize=12,
                 fontweight="bold", color=INK, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(values) * 1.2)

    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f} %", va="center", fontsize=9.5, color=INK)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    for builder in (architecture, routing_decision, topology, results_summary,
                    ml_confusion, ml_importance):
        print("wrote", builder())


if __name__ == "__main__":
    main()
