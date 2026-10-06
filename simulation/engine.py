"""High-level facade that ties every module together.

The dashboard and the tests talk to this class instead of wiring the topology,
the condition simulator, the router, the predictor and the packet transmitter
by hand.

Typical life-cycle::

    engine = SimulationEngine()          # healthy 8-router network
    engine.current_route()               # S -> A -> C -> T
    engine.apply_condition(CONGESTION)   # backbone congests
    engine.current_route()               # S -> B -> D -> F -> T  (switched!)
    engine.run_traffic()                 # simulate packets on the new route
    engine.metrics()                     # latency / loss / throughput ...

Every call that changes state ends with :meth:`refresh`, which recomputes the
route, appends a :class:`~simulation.metrics.StepRecord` and (optionally)
transmits a batch of packets.  The list of step records is what the dashboard
plots and what :meth:`metrics` aggregates.
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from config import (
    ALL_STRATEGIES,
    DEFAULT_DST,
    DEFAULT_PACKET_CONFIG,
    DEFAULT_SIM_STEPS,
    DEFAULT_SRC,
    DEFAULT_TOPOLOGY_NAME,
    DEFAULT_WEIGHTS,
    EXPERIMENT_SEED,
    STRATEGY_ADAPTIVE,
    CostWeights,
    PacketConfig,
)
from core.adaptive_routing import AdaptiveRouter, PenaltyFunction, path_cost
from core.network_conditions import ConditionEvent, ConditionSimulator, ConditionType
from core.packet_simulator import Packet, PacketSimulator, TransmissionReport
from core.routing import RouteResult, shortest_path_route
from core.topology import Topology, build_topology
from ml.predict import CongestionPredictor, load_predictor
from simulation.experiments import run_all_experiments
from simulation.metrics import SimulationMetrics, StepRecord, build_step_record, compute_metrics


class SimulationEngine:
    """Orchestrates topology, conditions, routing, ML and packet simulation."""

    def __init__(
        self,
        topology_name: str = DEFAULT_TOPOLOGY_NAME,
        src: str = DEFAULT_SRC,
        dst: str = DEFAULT_DST,
        strategy: str = STRATEGY_ADAPTIVE,
        use_ml: bool = True,
        seed: int = EXPERIMENT_SEED,
        weights: CostWeights = DEFAULT_WEIGHTS,
        packet_config: PacketConfig = DEFAULT_PACKET_CONFIG,
    ) -> None:
        self.topology_name = topology_name
        self.weights = weights
        self.packet_config = packet_config
        self.seed = seed
        self.use_ml = use_ml

        self.src = src
        self.dst = dst
        self.strategy = strategy

        self.topology: Topology = build_topology(topology_name)
        self.conditions = ConditionSimulator(self.topology)
        self.predictor: CongestionPredictor = load_predictor(enabled=use_ml)
        self._rebuild_router()

        self._packets: list[Packet] = []
        self._history: list[StepRecord] = []
        self._events: list[str] = []
        self._step_index = 0
        self._route: RouteResult = self.current_route()
        # Record the initial healthy state and send the first batch of packets.
        self.refresh(run_traffic=True)
        self._events = [f"[step 0] Simulation started on a healthy '{self.topology_name}' topology"]

    # -- configuration -----------------------------------------------------
    def _rebuild_router(self) -> None:
        penalty: PenaltyFunction | None = (
            self.predictor.link_penalty
            if (self.use_ml and self.predictor.active)
            else None
        )
        self.router = AdaptiveRouter(self.topology, weights=self.weights, penalty_fn=penalty)

    @property
    def penalty_fn(self) -> PenaltyFunction | None:
        return self.router.penalty_fn

    @property
    def ml_enabled(self) -> bool:
        return self.router.penalty_fn is not None

    @property
    def strategies(self) -> tuple[str, ...]:
        return tuple(ALL_STRATEGIES)

    @property
    def routers(self) -> list[str]:
        return self.topology.routers

    @property
    def history(self) -> list[StepRecord]:
        return list(self._history)

    @property
    def events(self) -> list[str]:
        return list(self._events)

    @property
    def packets(self) -> list[Packet]:
        return list(self._packets)

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def max_steps(self) -> int:
        return DEFAULT_SIM_STEPS

    def set_endpoints(self, src: str, dst: str) -> None:
        if src == dst:
            raise ValueError("Source and destination must be different routers")
        self.src, self.dst = src, dst
        self.refresh(run_traffic=False)

    def set_strategy(self, strategy: str) -> None:
        if strategy not in ALL_STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {ALL_STRATEGIES}")
        self.strategy = strategy
        self.refresh(run_traffic=False)

    def set_ml_enabled(self, enabled: bool) -> None:
        self.use_ml = enabled
        self._rebuild_router()
        self.refresh(run_traffic=False)

    # -- routing -----------------------------------------------------------
    def current_route(self, include_alternatives: bool = True) -> RouteResult:
        """Route for the currently selected endpoints and strategy."""
        return self.router.route(
            self.src, self.dst, strategy=self.strategy,
            include_alternatives=include_alternatives,
        )

    def naive_route(self) -> RouteResult:
        """The hop-count route a load-oblivious protocol would use."""
        return shortest_path_route(self.topology, self.src, self.dst)

    def strategy_comparison(self) -> dict[str, RouteResult]:
        """Side-by-side decision of all three strategies (same live conditions)."""
        return {
            name: self.router.route(self.src, self.dst, strategy=name)
            for name in ALL_STRATEGIES
        }

    def link_costs(self) -> list[dict]:
        return self.router.link_costs(strategy=self.strategy)

    # -- conditions --------------------------------------------------------
    def apply_condition(
        self,
        condition: ConditionType,
        severity: float = 1.0,
        links: Sequence[tuple[str, str]] | None = None,
        count: int | None = None,
        target_shortest_path: bool = True,
    ) -> ConditionEvent:
        """Apply a simulated event; ``target_shortest_path`` aims it at the naive route."""
        prefer = self.naive_route().path if (target_shortest_path and links is None) else None
        event = self.conditions.apply(
            condition, links=links, prefer_path=prefer, severity=severity, count=count
        )
        self._log(f"{event.condition.value} (severity {severity:.0%}) on {event.link_labels}")
        self.refresh(run_traffic=True)
        return event

    def restore_normal(self) -> None:
        self.conditions.restore_normal()
        self._log("All links restored to healthy baseline values")
        self.refresh(run_traffic=True)

    def reset(self, topology_name: str | None = None) -> None:
        """Rebuild everything from scratch (fresh topology, empty history)."""
        if topology_name:
            self.topology_name = topology_name
        self.topology = build_topology(self.topology_name)
        self.conditions = ConditionSimulator(self.topology)
        self.predictor.reset()
        self._rebuild_router()
        self._packets = []
        self._history = []
        self._events = []
        self._step_index = 0
        self._route = self.current_route()
        self.refresh(run_traffic=True)
        self._events = [f"[step 0] Simulation reset on a healthy '{self.topology_name}' topology"]

    # -- traffic -----------------------------------------------------------
    def run_traffic(self, num_packets: int | None = None) -> TransmissionReport:
        """Transmit a batch of packets on the currently selected route."""
        report = PacketSimulator(self.topology, config=self.packet_config, seed=self.seed).send(
            self.src,
            self.dst,
            self._route.path,
            num_packets=num_packets,
            strategy=self.strategy,
            step=self._step_index,
            start_time_ms=self._step_index * 1000.0,
        )
        self._packets.extend(report.packets)
        return report

    # -- state -------------------------------------------------------------
    def refresh(self, run_traffic: bool = True) -> RouteResult:
        """Recompute the route, record a step and optionally send packets."""
        self._route = self.current_route()
        record = build_step_record(
            step=self._step_index,
            route=self._route.path,
            # Dynamic weighted cost of the selected route, regardless of the
            # strategy, so the timeline stays comparable across methods.
            cost=path_cost(self.topology, self._route.path, self.weights),
            condition=self.conditions.last_event.condition.value
            if self.conditions.last_event else ConditionType.NORMAL.value,
            topology=self.topology,
        )
        self._history.append(record)
        if run_traffic:
            self.run_traffic()
        self._step_index += 1
        return self._route

    @property
    def route(self) -> RouteResult:
        return self._route

    # -- reporting ---------------------------------------------------------
    def metrics(self) -> SimulationMetrics:
        return compute_metrics(
            self._packets, self._history, self.topology,
            scenario="live", strategy=self.strategy,
        )

    def history_frame(self) -> pd.DataFrame:
        rows = []
        for record in self._history:
            rows.append({
                "step": record.step,
                "route": record.route_text,
                "hops": max(len(record.route) - 1, 0),
                "cost": round(record.cost, 4),
                "condition": record.condition,
                "congested_links": record.congested_links,
                "route_utilization_pct": round(record.route_utilization * 100, 2),
                "route_congested": record.route_has_congestion,
            })
        return pd.DataFrame(rows)

    def status(self) -> dict[str, Any]:
        """Overview block of the dashboard."""
        metrics = self.metrics()
        route = self._route
        congested = self.topology.congested_links()
        ml_summary = self.predictor.summary()
        return {
            "topology": self.topology_name,
            "routers": self.topology.router_count,
            "links": self.topology.link_count,
            "congested_links": len(congested),
            "congested_link_labels": ", ".join(f"{u}-{v}" for u, v in congested) or "none",
            "source": self.src,
            "destination": self.dst,
            "strategy": self.strategy,
            "route": route.as_text,
            "hops": route.hops,
            "route_cost": round(route.cost, 4),
            "ml_enabled": self.ml_enabled,
            "ml_state": "active" if ml_summary["active"] else "disabled",
            "packets_sent": metrics.packets_sent,
            "packets_delivered": metrics.packets_delivered,
            "avg_latency_ms": round(metrics.avg_latency_ms, 2),
            "packet_loss_pct": round(metrics.packet_loss_pct, 2),
            "delivery_rate_pct": round(metrics.delivery_rate_pct, 2),
            "route_changes": metrics.route_changes,
            "overall_congestion": self.overall_congestion(),
        }

    def overall_congestion(self) -> str:
        """Coarse network-wide congestion label used by the overview page."""
        if not self.topology.links:
            return "Unknown"
        average = sum(self.topology.get(u, v, "congestion")
                      for u, v in self.topology.links) / self.topology.link_count
        if average >= 0.55:
            return "Congested"
        if average >= 0.35:
            return "Warning"
        return "Normal"

    def link_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.topology.to_edge_records())

    def ml_predictions(self) -> list[dict]:
        """Predicted state of every link (read-only, does not alter routing)."""
        rows = []
        for u, v in sorted(self.topology.links):
            rows.append(self.predictor.inspect_link(u, v, self.topology.link_attributes(u, v)))
        return rows

    # -- experiments -------------------------------------------------------
    def run_experiments(self, save: bool = True, progress=None) -> pd.DataFrame:
        """Run the full 6 x 3 experiment matrix using this engine's ML hook."""
        return run_all_experiments(
            penalty_fn=self.penalty_fn, save=save, progress_callback=progress
        )

    # -- internals ---------------------------------------------------------
    def _log(self, message: str) -> None:
        self._events.append(f"[step {self._step_index}] {message}")
