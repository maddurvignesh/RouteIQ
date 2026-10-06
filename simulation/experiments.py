"""Experiment framework: runs every scenario with every routing strategy.

``run_experiment`` executes one *(scenario x strategy)* combination:

1. build a **fresh** topology (so runs never contaminate each other)
2. walk through the scenario timeline
3. at each step apply the scheduled network condition
4. ask the router for a route using the strategy under test
5. transmit simulated packets along that route
6. record a :class:`~simulation.metrics.StepRecord`
7. aggregate everything into a :class:`SimulationMetrics`

``run_all_experiments`` repeats this for the 6 scenarios x 3 strategies and
returns a pandas ``DataFrame`` which is written to ``results/``.

The ML module is injected as a plain ``penalty_fn`` callback.  When it is not
supplied, the *Adaptive + ML* strategy silently degrades to *Adaptive*, which
is exactly the required "project still works with ML disabled" behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from config import (
    ALL_STRATEGIES,
    COMPARISON_SUMMARY_CSV,
    DEFAULT_PACKET_CONFIG,
    DEFAULT_WEIGHTS,
    EXPERIMENT_METRICS_CSV,
    EXPERIMENT_SEED,
    STRATEGY_ADAPTIVE_ML,
    CostWeights,
    PacketConfig,
)
from core.adaptive_routing import AdaptiveRouter, PenaltyFunction, path_cost
from core.network_conditions import ConditionSimulator
from core.packet_simulator import Packet, PacketSimulator
from core.routing import shortest_path_route
from core.topology import Topology, build_predefined_topology
from simulation.metrics import (
    SimulationMetrics,
    StepRecord,
    build_step_record,
    compute_metrics,
)
from simulation.scenarios import Scenario, get_scenario, scenario_ids

# Order in which the aggregate table is printed (matches config.ALL_STRATEGIES).
_STRATEGY_ORDER = tuple(ALL_STRATEGIES)

# Columns of the compact "Method vs metrics" summary table.
SUMMARY_COLUMNS = [
    "strategy",
    "scenarios",
    "avg_latency_ms",
    "packet_loss_pct",
    "throughput_mbps",
    "delivery_rate_pct",
    "route_changes",
    "route_utilization_pct",
    "congestion_frequency_pct",
    "avg_route_cost",
]


@dataclass
class RunResult:
    """Everything produced by a single experiment run."""

    scenario: Scenario
    strategy: str
    metrics: SimulationMetrics
    steps: list[StepRecord] = field(default_factory=list)
    packets: list[Packet] = field(default_factory=list)

    @property
    def explanation(self) -> str:
        return self.scenario.expected_outcome


def run_experiment(
    scenario: Scenario,
    strategy: str,
    penalty_fn: PenaltyFunction | None = None,
    seed: int = EXPERIMENT_SEED,
    weights: CostWeights = DEFAULT_WEIGHTS,
    packet_config: PacketConfig = DEFAULT_PACKET_CONFIG,
    topology: Topology | None = None,
    include_alternatives: bool = False,
) -> RunResult:
    """Run one scenario with one routing strategy and return all its outputs."""
    topo = topology.copy() if topology is not None else build_predefined_topology()
    conditions = ConditionSimulator(topo)
    router = AdaptiveRouter(
        topo,
        weights=weights,
        penalty_fn=penalty_fn if strategy == STRATEGY_ADAPTIVE_ML else None,
    )
    packets_engine = PacketSimulator(topo, config=packet_config, seed=seed)

    steps: list[StepRecord] = []
    all_packets: list[Packet] = []

    for action in scenario.steps:
        # Target the route a load-oblivious protocol would keep using.
        naive_route = shortest_path_route(topo, scenario.src, scenario.dst).path
        links = action.links or None
        event = conditions.apply(
            action.condition,
            links=links,
            prefer_path=None if links else naive_route,
            severity=action.severity,
        )

        route_result = router.route(scenario.src, scenario.dst, strategy=strategy,
                                    include_alternatives=include_alternatives)

        steps.append(
            build_step_record(
                step=action.step,
                route=route_result.path,
                # Always the *dynamic* weighted cost, whatever the strategy, so
                # avg_route_cost is comparable across methods (see metrics.py).
                cost=path_cost(topo, route_result.path, weights),
                condition=event.condition.value,
                topology=topo,
            )
        )

        report = packets_engine.send(
            scenario.src,
            scenario.dst,
            route_result.path,
            strategy=strategy,
            step=action.step,
        )
        all_packets.extend(report.packets)

    metrics = compute_metrics(
        all_packets, steps, topo, scenario=scenario.id, strategy=strategy
    )
    return RunResult(scenario=scenario, strategy=strategy, metrics=metrics,
                     steps=steps, packets=all_packets)


def run_all_experiments(
    penalty_fn: PenaltyFunction | None = None,
    scenarios: list[str] | None = None,
    strategies: tuple[str, ...] = _STRATEGY_ORDER,
    seed: int = EXPERIMENT_SEED,
    save: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Run every scenario with every strategy; optionally persist to ``results/``."""
    ids = scenarios if scenarios is not None else scenario_ids()
    rows: list[dict] = []
    total = len(ids) * len(strategies)
    done = 0

    for scenario_id in ids:
        scenario = get_scenario(scenario_id)
        for strategy in strategies:
            result = run_experiment(scenario, strategy, penalty_fn=penalty_fn, seed=seed)
            rows.append(result.metrics.as_dict())
            done += 1
            if progress_callback is not None:
                progress_callback(done, total)

    frame = pd.DataFrame(rows)
    if save:
        save_results(frame)
    return frame


def save_results(frame: pd.DataFrame,
                 metrics_path: Path = EXPERIMENT_METRICS_CSV,
                 summary_path: Path = COMPARISON_SUMMARY_CSV) -> None:
    """Write the detailed table and the compact per-method summary."""
    frame = frame.copy()
    frame["routes"] = frame["routes"].apply(lambda r: " | ".join(r) if isinstance(r, (list, tuple)) else r)
    frame.to_csv(metrics_path, index=False)
    summarise(frame).to_csv(summary_path, index=False)


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the *Method vs metrics* table averaged over all scenarios."""
    if frame.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    numeric = [
        "avg_latency_ms", "packet_loss_pct", "throughput_mbps",
        "delivery_rate_pct", "route_changes", "network_utilization_pct",
        "route_utilization_pct", "avg_route_cost", "congestion_frequency_pct",
        "avg_hops",
    ]
    grouped = frame.groupby("strategy", sort=False)[numeric].mean().reset_index()
    counts = frame.groupby("strategy", sort=False)["scenario"].nunique().reset_index(name="scenarios")
    summary = grouped.merge(counts, on="strategy")
    summary["strategy"] = pd.Categorical(summary["strategy"], categories=_STRATEGY_ORDER, ordered=True)
    summary = summary.sort_values("strategy").reset_index(drop=True)

    ordered = [c for c in SUMMARY_COLUMNS if c in summary.columns]
    extra = [c for c in summary.columns if c not in ordered]
    return summary[ordered + extra]


def load_results(metrics_path: Path = EXPERIMENT_METRICS_CSV) -> pd.DataFrame | None:
    """Read previously generated results (``None`` when the file is missing)."""
    if not metrics_path.exists():
        return None
    return pd.read_csv(metrics_path)


def load_summary(summary_path: Path = COMPARISON_SUMMARY_CSV) -> pd.DataFrame | None:
    if not summary_path.exists():
        return None
    return pd.read_csv(summary_path)


def run_strategy_comparison(
    scenario_id: str,
    penalty_fn: PenaltyFunction | None = None,
    seed: int = EXPERIMENT_SEED,
) -> dict[str, RunResult]:
    """Convenience helper used by tests and by the dashboard."""
    scenario = get_scenario(scenario_id)
    return {
        strategy: run_experiment(scenario, strategy, penalty_fn=penalty_fn, seed=seed)
        for strategy in _STRATEGY_ORDER
    }
