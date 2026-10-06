"""Streamlit dashboard – all six pages of the *Adaptive Routing* project.

``app.py`` only calls :func:`run`; every page lives here so that the routing,
ML and experiment code stay free of any UI concern.

Page map
--------
=====  =========================  ==========================================
page   name                       purpose
=====  =========================  ==========================================
1      Overview                   project summary + live network status
2      Network Topology           interactive graph, link conditions, route
3      Adaptive Routing           pick src/dst/strategy, see why a route won
4      Network Simulation         inject faults and watch the route change
5      ML Prediction              dataset, model metrics, live predictions
6      Performance Comparison     Shortest vs Adaptive vs Adaptive + ML
=====  =========================  ==========================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

import visualization.charts as charts
from config import ALL_STRATEGIES, ML_COST_PENALTY, ML_FEATURES, ML_STATE_LABELS
from core.network_conditions import ConditionType
from ml.dataset import class_distribution, ensure_dataset
from ml.train import load_training_report
from simulation.engine import SimulationEngine
from simulation.experiments import load_results, summarise
from simulation.scenarios import SCENARIOS
from visualization.theme import DEFAULT_THEME, THEME_NAMES, inject
from visualization.topology import build_link_table, build_topology_figure

ENGINE_KEY = "adaptive_routing_engine"
EXPERIMENTS_KEY = "experiment_frame"
THEME_KEY = "theme_name"

PAGES = (
    "Overview",
    "Network Topology",
    "Adaptive Routing",
    "Network Simulation",
    "ML Prediction",
    "Performance Comparison",
)

# Buttons of the simulation page: label -> (condition, description)
CONDITION_ACTIONS: tuple[tuple[str, ConditionType, str], ...] = (
    ("Increase traffic", ConditionType.TRAFFIC_SPIKE,
     "Offered load jumps until the link is nearly saturated."),
    ("Increase congestion", ConditionType.CONGESTION,
     "The router queue fills up: congestion, load, delay and loss all rise."),
    ("Increase packet loss", ConditionType.HIGH_PACKET_LOSS,
     "Loss probability rises sharply (buffer overflow / bit errors)."),
    ("Increase latency", ConditionType.HIGH_LATENCY,
     "Propagation and processing delay of the link increase."),
    ("Reduce bandwidth", ConditionType.LOW_BANDWIDTH,
     "Link capacity is throttled, so the same traffic now congests it."),
    ("Multiple congested links", ConditionType.MULTI_CONGESTION,
     "Several links degrade at the same time."),
)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def get_engine() -> SimulationEngine:
    """Create the simulation engine once and reuse it across reruns."""
    if ENGINE_KEY not in st.session_state:
        st.session_state[ENGINE_KEY] = SimulationEngine()
    return st.session_state[ENGINE_KEY]


def get_experiments(force_rerun: bool = False) -> pd.DataFrame:
    """Return the experiment matrix, generating it on first use."""
    if force_rerun or st.session_state.get(EXPERIMENTS_KEY) is None:
        engine = get_engine()
        with st.spinner("Running 6 scenarios x 3 routing strategies ..."):
            frame = engine.run_experiments(save=True)
        st.session_state[EXPERIMENTS_KEY] = frame
    return st.session_state[EXPERIMENTS_KEY]


def _metric_card(columns, label: str, value, help_text: str = "") -> None:
    columns.metric(label, value, help=help_text or None)


def _status_badge(state: str) -> str:
    return {"Normal": "🟢 Normal", "Warning": "🟠 Warning", "Congested": "🔴 Congested"}.get(state, state)


# ---------------------------------------------------------------------------
# Page 1 - Overview
# ---------------------------------------------------------------------------

def page_overview() -> None:
    engine = get_engine()
    status = engine.status()

    st.title("Adaptive Routing Based on Real-Time Network Conditions")
    st.caption(
        "Network simulation and adaptive routing decision system – a Computer "
        "Networks + AIML academic project. It is **not** a real router implementation."
    )

    st.markdown(
        """
Traditional routing protocols compute the shortest path from a **static**
metric (hop count or configured cost) and keep using it even when the network
becomes congested. This project builds a simulation in which every link is
continuously monitored (latency, bandwidth, packet loss, congestion,
reliability) and routes are re-evaluated from a **dynamic cost**. An optional
Random-Forest model predicts upcoming congestion and adds an extra penalty to
at-risk links.
"""
    )

    st.subheader("Live network status")
    left, right = st.columns(2)
    with left:
        cols = st.columns(2)
        _metric_card(cols[0], "Routers", status["routers"])
        _metric_card(cols[1], "Links", status["links"])
        cols = st.columns(2)
        _metric_card(cols[0], "Source", status["source"])
        _metric_card(cols[1], "Destination", status["destination"])
        cols = st.columns(2)
        _metric_card(cols[1], "Active route hops", status["hops"])
        _metric_card(cols[0], "Route cost (dynamic)", f"{status['route_cost']:.3f}")
    with right:
        cols = st.columns(2)
        _metric_card(cols[0], "Congested links", f"{status['congested_links']} / {status['links']}",
                     status["congested_link_labels"])
        _metric_card(cols[1], "Route changes", status["route_changes"])
        cols = st.columns(2)
        _metric_card(cols[0], "Routing strategy", status["strategy"])
        _metric_card(cols[1], "ML module", status["ml_state"])

    st.markdown(f"**Current route:** `{status['route']}`")

    state = status["overall_congestion"]
    if state == "Normal":
        st.success(f"Network condition: {_status_badge(state)} – all links healthy.")
    elif state == "Warning":
        st.warning(f"Network condition: {_status_badge(state)} – some links are loading up.")
    else:
        st.error(f"Network condition: {_status_badge(state)} – several links are saturated.")

    st.subheader("Traffic results of the current run")
    cols = st.columns(4)
    _metric_card(cols[0], "Packets sent", status["packets_sent"])
    _metric_card(cols[1], "Packets delivered", status["packets_delivered"])
    _metric_card(cols[2], "Average latency (ms)", f"{status['avg_latency_ms']:.1f}")
    _metric_card(cols[3], "Packet loss (%)", f"{status['packet_loss_pct']:.2f}")

    cols = st.columns(3)
    _metric_card(cols[0], "Delivery rate (%)", f"{status['delivery_rate_pct']:.1f}")
    metrics = engine.metrics()
    _metric_card(cols[1], "Throughput (Mbps)", f"{metrics.throughput_mbps:.2f}")
    _metric_card(cols[2], "Avg route cost", f"{metrics.avg_route_cost:.3f}")

    with st.expander("Project modules (architecture at a glance)"):
        st.markdown(
            """
| component | module | responsibility |
|---|---|---|
| A. Topology generator | `core/topology.py` | build routers and links |
| B. Condition simulator | `core/network_conditions.py` | change latency/loss/load over time |
| C. Routing engine | `core/routing.py` | traditional shortest path |
| D. Adaptive cost calculator | `core/adaptive_routing.py` | dynamic link cost + route choice |
| E. Packet simulator | `core/packet_simulator.py` | delay, loss, retransmission |
| F. ML prediction | `ml/` | dataset, training, live predictions |
| G. Performance evaluation | `simulation/metrics.py` | latency, loss, throughput, ... |
| H. Visualization | `visualization/` | topology graph, charts, dashboard |
| I. Experiment manager | `simulation/experiments.py` | 6 scenarios x 3 strategies |
| J. Results storage | `results/*.csv` | detailed + summary tables |
"""
        )


# ---------------------------------------------------------------------------
# Page 2 - Network topology
# ---------------------------------------------------------------------------

def page_topology() -> None:
    engine = get_engine()
    st.title("Network Topology")
    st.caption("Interactive view of routers, links, their live conditions and the active route.")

    route = engine.route
    ml_states = None
    if engine.ml_enabled:
        ml_states = {}
        for prediction in engine.ml_predictions():
            u, v = prediction["link"].split("-")
            ml_states[(u, v)] = prediction["state"]

    figure = build_topology_figure(
        engine.topology,
        active_route=route.path,
        src=engine.src,
        dst=engine.dst,
        ml_states=ml_states,
        title=f"Topology '{engine.topology_name}'",
    )
    st.plotly_chart(figure, use_container_width=True)

    left, right = st.columns([2, 1])
    with left:
        st.markdown(f"**Active route:** `{route.as_text}`  \n"
                    f"**Strategy:** {route.strategy}  \n"
                    f"**Dynamic cost:** {route.cost:.4f}")
        st.dataframe(build_link_table(engine.topology, route.path),
                     use_container_width=True, hide_index=True)
    with right:
        st.markdown("##### Link status")
        status = engine.status()
        st.write(f"Congested links: **{status['congested_link_labels']}**")
        st.write(f"Network condition: {_status_badge(status['overall_congestion'])}")
        st.markdown("##### Legend")
        st.markdown(
            "- 🔵 blue – active route\n"
            "- 🔴 red – congested link (≥ 70 %)\n"
            "- 🟠 amber – active **and** congested\n"
            "- ⚪ grey – healthy, not used\n"
            "- 🟢 source router  🟠 destination router"
        )
        st.info("Hover a link to see its latency, bandwidth, loss, congestion, "
                "reliability and the ML prediction.")


# ---------------------------------------------------------------------------
# Page 3 - Adaptive routing
# ---------------------------------------------------------------------------

def page_adaptive_routing() -> None:
    engine = get_engine()
    st.title("Adaptive Routing")
    st.caption("Choose the endpoints and the routing strategy, then compare the decisions.")

    col_a, col_b, col_c = st.columns([2, 2, 2])
    with col_a:
        src = st.selectbox("Source router", engine.routers,
                           index=engine.routers.index(engine.src))
    with col_b:
        dst = st.selectbox("Destination router", engine.routers,
                           index=engine.routers.index(engine.dst))
    with col_c:
        strategy = st.selectbox("Routing strategy", list(ALL_STRATEGIES),
                                index=list(ALL_STRATEGIES).index(engine.strategy))

    changed = (src != engine.src) or (dst != engine.dst)
    if src == dst:
        st.error("Source and destination must be different routers.")
        return
    if changed:
        engine.set_endpoints(src, dst)
    if strategy != engine.strategy:
        engine.set_strategy(strategy)

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### Selected route")
        st.code(engine.route.as_text, language=None)
        st.markdown(f"**Reason:** {engine.route.reason}")

        st.markdown("#### Alternative routes")
        if engine.route.alternatives:
            alt_frame = pd.DataFrame([
                {
                    "Alternative route": " -> ".join(alt.path),
                    "Hops": alt.hops,
                    "Dynamic cost": round(alt.cost, 4),
                    "Extra cost vs selected": round(alt.cost - engine.route.cost, 4),
                }
                for alt in engine.route.alternatives
            ])
            st.dataframe(alt_frame, use_container_width=True, hide_index=True)
        else:
            st.info("No alternative loop-free route exists for this pair.")

    with right:
        st.markdown("#### All strategies, same live conditions")
        for name, result in engine.strategy_comparison().items():
            icon = "🟢" if result.path == engine.route.path else "⚪"
            st.write(f"{icon} **{name}**")
            st.write(f"`{result.as_text}` — cost `{result.cost:.3f}`")

    st.divider()
    st.markdown("#### Adaptive cost formula")
    st.latex(
        r"cost = w_1\cdot\overline{latency} + w_2\cdot congestion"
        r" + w_3\cdot\overline{loss} + w_4\cdot bandwidth_{penalty}"
        r" + penalty_{ML}"
    )
    weights = engine.weights
    st.markdown(
        f"Current weights: `w1 (latency) = {weights.latency}`, "
        f"`w2 (congestion) = {weights.congestion}`, "
        f"`w3 (packet loss) = {weights.packet_loss}`, "
        f"`w4 (bandwidth) = {weights.bandwidth}`. "
        f"ML penalties: " + ", ".join(
            f"{ML_STATE_LABELS[level]} → +{value:.2f}"
            for level, value in sorted(ML_COST_PENALTY.items())
        ) + "."
    )

    st.markdown("#### Link-by-link cost breakdown")
    costs = pd.DataFrame(engine.link_costs())
    if not costs.empty:
        display = costs.drop(columns=["total"]).rename(columns={
            "link": "Link", "cost": "Total cost",
            "latency term": "Latency term", "congestion term": "Congestion term",
            "packet-loss term": "Loss term", "bandwidth term": "Bandwidth term",
            "latency_ms": "Latency (ms)", "congestion": "Congestion",
            "packet_loss": "Packet loss", "bandwidth_mbps": "Bandwidth (Mbps)",
        })
        assert not display.columns.duplicated().any()
        st.dataframe(display.round(4), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page 4 - Network simulation
# ---------------------------------------------------------------------------

def page_simulation() -> None:
    engine = get_engine()
    st.title("Network Simulation")
    st.caption("Inject faults into the network and watch the routing decision react.")

    st.markdown(f"**Current route before the next action:** `{engine.route.as_text}`")

    col_ctrl, col_info = st.columns([3, 2])
    with col_ctrl:
        severity = st.slider("Severity of the injected condition", 0.1, 1.0, 1.0, 0.1)
        target = st.radio(
            "Target of the condition",
            ["Current shortest path (the route a naive protocol keeps using)",
             "Single link of my choice"],
            help="Targeting the shortest path is how the experiments place a fault "
                 "exactly where load-oblivious routing would suffer.",
        )
        link_choice = None
        if target.startswith("Single"):
            options = [f"{u}-{v}" for u, v in sorted(engine.topology.links)]
            picked = st.selectbox("Link", options)
            link_choice = tuple(picked.split("-"))

        st.markdown("**Inject a condition**")
        for label, condition, description in CONDITION_ACTIONS:
            if st.button(label, use_container_width=True, help=description):
                before = engine.route.as_text
                engine.apply_condition(
                    condition,
                    severity=severity,
                    links=[link_choice] if link_choice else None,
                    target_shortest_path=link_choice is None,
                )
                after = engine.route.as_text
                if before == after:
                    st.info(f"{condition.value} applied on {engine.conditions.last_event.link_labels} "
                            f"— the selected route did **not** change.")
                else:
                    st.success(f"{condition.value} applied — route changed!\n\n"
                               f"`{before}` → `{after}`")
                st.rerun()

        if st.button("Restore normal conditions", use_container_width=True):
            before = engine.route.as_text
            engine.restore_normal()
            st.info(f"Network restored — `{before}` → `{engine.route.as_text}`")
            st.rerun()

    with col_info:
        st.markdown("#### Event log")
        events = list(reversed(engine.events))
        if events:
            st.text_area("History", "\n".join(events), height=260, label_visibility="collapsed")
        else:
            st.info("No events yet.")

    st.divider()
    st.subheader("Route evolution")
    history = engine.history_frame()
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.route_timeline(history), use_container_width=True)
    with right:
        st.plotly_chart(charts.congested_links_timeline(history), use_container_width=True)

    st.markdown("#### Step-by-step routing decisions")
    st.dataframe(history, use_container_width=True, hide_index=True)

    st.subheader("Resulting packet statistics")
    metrics = engine.metrics()
    cols = st.columns(5)
    _metric_card(cols[0], "Avg latency (ms)", f"{metrics.avg_latency_ms:.1f}")
    _metric_card(cols[1], "Packet loss (%)", f"{metrics.packet_loss_pct:.2f}")
    _metric_card(cols[2], "Throughput (Mbps)", f"{metrics.throughput_mbps:.2f}")
    _metric_card(cols[3], "Delivery rate (%)", f"{metrics.delivery_rate_pct:.1f}")
    _metric_card(cols[4], "Route changes", metrics.route_changes)

    if engine.packets:
        st.plotly_chart(charts.latency_histogram(engine.packets), use_container_width=True)


# ---------------------------------------------------------------------------
# Page 5 - ML prediction
# ---------------------------------------------------------------------------

def page_ml() -> None:
    engine = get_engine()
    st.title("Machine Learning – Congestion Prediction")
    st.caption(
        "Random Forest classifier that predicts the congestion state of a link "
        "and adds a routing penalty when trouble is expected."
    )

    predictor = engine.predictor
    if not engine.use_ml:
        st.warning("The ML module is **switched off** in the sidebar. Routing continues "
                   "with the plain adaptive cost function.")
    elif not predictor.active:
        st.error(f"Model not usable: {predictor.error}")
    else:
        st.success("Model loaded – the *Adaptive + ML* strategy is fully operational.")

    st.divider()
    st.subheader("1. Dataset")
    try:
        dataset = ensure_dataset()
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Could not load the dataset: {exc}")
        dataset = pd.DataFrame()

    if not dataset.empty:
        cols = st.columns(4)
        _metric_card(cols[0], "Samples", len(dataset))
        _metric_card(cols[1], "Features", len(ML_FEATURES))
        _metric_card(cols[2], "Link streams", dataset["link_id"].nunique())
        _metric_card(cols[3], "Observations/link", int(dataset["timestep"].nunique()))

        left, right = st.columns(2)
        with left:
            st.plotly_chart(charts.class_distribution_figure(class_distribution(dataset)),
                            use_container_width=True)
        with right:
            column = st.selectbox("Feature to visualise", ML_FEATURES, index=2)
            st.plotly_chart(charts.dataset_histogram(dataset, column), use_container_width=True)

        st.caption(
            "**Labels** come from a documented rule over utilisation, latency and "
            "packet loss – the dataset is synthetic and generated by `ml/dataset.py`; "
            "no external data is used."
        )

    st.divider()
    st.subheader("2. Model evaluation")
    report = load_training_report()
    if report is None:
        st.info("No training report found. Run `python -m ml.train` to train the model.")
    else:
        cols = st.columns(4)
        _metric_card(cols[0], "Accuracy", f"{report['accuracy']:.2%}")
        _metric_card(cols[1], "Train samples", report["train_size"])
        _metric_card(cols[2], "Test samples", report["test_size"])
        _metric_card(cols[3], "Algorithm", "Random Forest")

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                charts.confusion_matrix_figure(report["confusion_matrix"],
                                               labels=report["classes"]),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(charts.feature_importance_figure(report["feature_importances"]),
                            use_container_width=True)

        with st.expander("Per-class precision / recall / F1"):
            per_class = pd.DataFrame(
                {k: v for k, v in report["classification_report"].items()
                 if isinstance(v, dict)}
            ).T
            st.dataframe(per_class.round(3), use_container_width=True)

    st.divider()
    st.subheader("3. Current predictions")
    predictions = engine.ml_predictions()
    prediction_frame = pd.DataFrame([
        {
            "Link": p["link"],
            "Predicted state": p["state"],
            "Confidence": round(p["confidence"], 3),
            "Routing penalty": round(p["penalty"], 2),
            "Latency (ms)": round(p["features"]["latency_ms"], 1),
            "Packet loss": round(p["features"]["packet_loss"], 4),
            "Utilisation": round(p["features"]["bandwidth_utilization"], 3),
            "Queue length": round(p["features"]["queue_length"], 1),
        }
        for p in predictions
    ])
    st.dataframe(prediction_frame, use_container_width=True, hide_index=True)
    st.plotly_chart(charts.current_prediction_bars(predictions), use_container_width=True)

    st.markdown(
        """
**How the ML output is used**

1. Every routing interval the model reads the six features of each link.
2. It predicts *Normal / Warning / Congested*.
3. The prediction is converted into an extra cost: `+0.00 / +0.35 / +0.90`.
4. Dijkstra re-computes the cheapest path **with the normal adaptive cost plus
   this penalty**.

The ML module only *assists* the routing algorithm – it never replaces it, and
the project runs normally with the module disabled.
"""
    )


# ---------------------------------------------------------------------------
# Page 6 - Performance comparison
# ---------------------------------------------------------------------------

def page_performance() -> None:
    st.title("Performance Comparison")
    st.caption("Shortest Path vs Adaptive Routing vs Adaptive + ML – all numbers "
               "come from real simulation runs.")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        if st.button("Run all experiments (6 scenarios x 3 strategies)", type="primary"):
            frame = get_experiments(force_rerun=True)
            st.success(f"Finished – {len(frame)} experiment runs recorded in `results/`.")
    with col_right:
        if st.button("Reload saved results"):
            st.session_state.pop(EXPERIMENTS_KEY, None)
            st.rerun()

    frame = st.session_state.get(EXPERIMENTS_KEY)
    if frame is None:
        frame = load_results()
    if frame is None or frame.empty:
        st.info("No results available yet – press **Run all experiments**.")
        st.stop()

    summary = summarise(frame)

    st.subheader("Summary table (averaged over all scenarios)")
    display = summary.copy()
    display = display.rename(columns={
        "strategy": "Method",
        "scenarios": "Scenarios",
        "avg_latency_ms": "Avg latency (ms)",
        "packet_loss_pct": "Packet loss (%)",
        "throughput_mbps": "Throughput (Mbps)",
        "delivery_rate_pct": "Delivery rate (%)",
        "route_changes": "Route changes",
        "route_utilization_pct": "Route utilisation (%)",
        "congestion_frequency_pct": "Congestion frequency (%)",
        "avg_route_cost": "Avg route cost",
    })
    st.dataframe(display.round(2), use_container_width=True, hide_index=True)

    cols = st.columns(3)
    _metric_card(cols[0], "Best avg latency",
                 f"{summary['avg_latency_ms'].min():.1f} ms",
                 summary.loc[summary['avg_latency_ms'].idxmin(), 'strategy'])
    _metric_card(cols[1], "Best delivery rate",
                 f"{summary['delivery_rate_pct'].max():.1f} %",
                 summary.loc[summary['delivery_rate_pct'].idxmax(), 'strategy'])
    _metric_card(cols[2], "Highest throughput",
                 f"{summary['throughput_mbps'].max():.2f} Mbps",
                 summary.loc[summary['throughput_mbps'].idxmax(), 'strategy'])

    st.subheader("Key charts")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.latency_chart(summary), use_container_width=True)
        st.plotly_chart(charts.summary_metric_bar(
            summary, "throughput_mbps", "Throughput (Mbps)", "Mbps"),
            use_container_width=True)
    with right:
        st.plotly_chart(charts.delivery_rate_chart(summary), use_container_width=True)
        st.plotly_chart(charts.summary_metric_bar(
            summary, "packet_loss_pct", "Packet loss (%)", "Loss (%)"),
            use_container_width=True)

    st.subheader("Scenario-by-scenario results")
    metric_labels = {
        "avg_latency_ms": "Average latency (ms)",
        "packet_loss_pct": "Packet loss (%)",
        "throughput_mbps": "Throughput (Mbps)",
        "delivery_rate_pct": "Delivery rate (%)",
        "route_changes": "Number of route changes",
        "route_utilization_pct": "Route utilisation (%)",
        "congestion_frequency_pct": "Congestion frequency (%)",
        "avg_route_cost": "Average route cost",
    }
    chosen = st.selectbox("Metric", list(metric_labels), index=0)
    st.plotly_chart(
        charts.metric_by_scenario(frame, chosen, metric_labels[chosen], metric_labels[chosen]),
        use_container_width=True,
    )

    st.subheader("Detailed results")
    st.dataframe(
        frame[["scenario", "strategy", "avg_latency_ms", "packet_loss_pct",
               "throughput_mbps", "delivery_rate_pct", "route_changes",
               "distinct_routes", "avg_route_cost"]].round(2),
        use_container_width=True, hide_index=True,
    )

    st.subheader("What the experiments show")
    for scenario_id, scenario in SCENARIOS.items():
        with st.expander(scenario.name):
            st.markdown(f"**Setup:** {scenario.description}")
            st.markdown(f"**Expected outcome:** {scenario.expected_outcome}")
            rows = frame[frame["scenario"] == scenario_id]
            st.dataframe(
                rows[["strategy", "avg_latency_ms", "packet_loss_pct",
                      "throughput_mbps", "delivery_rate_pct", "route_changes"]]
                .round(2),
                use_container_width=True, hide_index=True,
            )
            routes = rows[["strategy", "routes"]].copy()
            routes.columns = ["Strategy", "Routes seen per step"]
            st.dataframe(routes, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    st.sidebar.title("Adaptive Routing")
    st.sidebar.caption("Computer Networks + AIML project")
    page = st.sidebar.radio("Navigation", PAGES)

    engine = get_engine()

    st.sidebar.divider()
    st.sidebar.subheader("Simulation settings")
    topology_names = ["predefined", "random"]
    topology_name = st.sidebar.selectbox(
        "Topology", topology_names,
        index=topology_names.index(engine.topology_name)
        if engine.topology_name in topology_names else 0,
        help="predefined = the 8-router demo network, random = generated graph.",
    )
    if topology_name != engine.topology_name:
        engine.reset(topology_name)
        st.sidebar.info(f"Switched to the '{topology_name}' topology.")

    use_ml = st.sidebar.checkbox(
        "Enable ML module",
        value=engine.use_ml,
        help="Disables the Random-Forest penalty; routing still works normally.",
    )
    if use_ml != engine.use_ml:
        engine.set_ml_enabled(use_ml)

    ml_summary = engine.predictor.summary()
    if ml_summary["active"]:
        st.sidebar.success(f"ML model: active ({Path(ml_summary['model_path']).name})")
    elif not ml_summary["enabled"]:
        st.sidebar.warning("ML model: disabled by the user")
    else:
        st.sidebar.error(f"ML model: {ml_summary['error']}")

    st.sidebar.divider()
    if st.sidebar.button("Reset simulation", use_container_width=True):
        engine.reset()
        st.session_state.pop(EXPERIMENTS_KEY, None)
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Appearance")
    theme = st.sidebar.selectbox(
        "Theme", THEME_NAMES,
        index=THEME_NAMES.index(st.session_state.get(THEME_KEY, DEFAULT_THEME)),
        help="Animated background and glass styling. 'Minimal' turns the motion off.",
    )
    st.session_state[THEME_KEY] = theme

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Run the project**\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "python -m ml.dataset\n"
        "python -m ml.train\n"
        "streamlit run app.py\n"
        "```"
    )
    return page


def run() -> None:
    """Entry point called by ``app.py``."""
    page = render_sidebar()
    inject(st.session_state.get(THEME_KEY, DEFAULT_THEME))
    handlers: dict[str, Callable[[], None]] = {
        "Overview": page_overview,
        "Network Topology": page_topology,
        "Adaptive Routing": page_adaptive_routing,
        "Network Simulation": page_simulation,
        "ML Prediction": page_ml,
        "Performance Comparison": page_performance,
    }
    handlers[page]()
