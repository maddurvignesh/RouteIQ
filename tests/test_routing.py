"""Unit tests for baseline shortest-path and adaptive routing."""

from __future__ import annotations

import pytest

from config import (
    DEFAULT_WEIGHTS,
    ML_COST_PENALTY,
    STRATEGY_ADAPTIVE,
    STRATEGY_ADAPTIVE_ML,
    STRATEGY_SHORTEST,
    CostWeights,
)
from core.adaptive_routing import (
    AdaptiveRouter,
    adaptive_route,
    link_cost,
    link_cost_breakdown,
    normalised_terms,
    path_cost,
)
from core.network_conditions import ConditionSimulator, ConditionType
from core.routing import (
    RoutingError,
    enumerate_routes,
    metric_weight,
    shortest_path_route,
)
from core.topology import build_predefined_topology


@pytest.fixture()
def topology():
    return build_predefined_topology()


@pytest.fixture()
def router(topology):
    return AdaptiveRouter(topology)


# ---------------------------------------------------------------------------
# Baseline shortest-path routing
# ---------------------------------------------------------------------------

def test_shortest_path_uses_hop_count(topology) -> None:
    result = shortest_path_route(topology, "S", "T")
    assert result.path == ("S", "A", "C", "T")
    assert result.cost == 3.0
    assert result.hops == 3
    assert result.strategy == STRATEGY_SHORTEST


def test_shortest_path_reports_alternatives(topology) -> None:
    result = shortest_path_route(topology, "S", "T")
    assert len(result.alternatives) >= 1
    # alternatives must be ordered from cheap to expensive
    costs = [a.cost for a in result.alternatives]
    assert costs == sorted(costs)
    assert all(a.path != result.path for a in result.alternatives)


def test_shortest_path_static_latency_metric(topology) -> None:
    result = shortest_path_route(topology, "S", "T", metric="static_latency")
    assert result.cost == pytest.approx(30.0)
    assert result.path == ("S", "A", "C", "T")


def test_unknown_metric_raises() -> None:
    with pytest.raises(RoutingError):
        metric_weight("random_jitter")


def test_unknown_source_raises(topology) -> None:
    with pytest.raises(RoutingError):
        shortest_path_route(topology, "Z", "T")


def test_same_source_and_destination_raises(topology) -> None:
    with pytest.raises(RoutingError):
        shortest_path_route(topology, "S", "S")


def test_disconnected_pair_raises() -> None:
    from core.topology import Topology

    topo = Topology.empty()
    for node in ("P", "Q", "R"):
        topo.add_router(node)
    topo.add_link("P", "Q")
    with pytest.raises(RoutingError):
        shortest_path_route(topo, "P", "R")


def test_enumerate_routes_orders_by_cost(topology) -> None:
    routes = enumerate_routes(topology, "S", "T", weight=None, limit=5)
    assert len(routes) == 5
    costs = [c for _, c in routes]
    assert costs == sorted(costs)


def test_enumerate_routes_produces_distinct_paths(topology) -> None:
    routes = enumerate_routes(topology, "S", "T", limit=6)
    paths = [p for p, _ in routes]
    assert len(paths) == len(set(paths))


# ---------------------------------------------------------------------------
# Adaptive cost maths
# ---------------------------------------------------------------------------

def test_normalised_terms_are_bounded(topology) -> None:
    for u, v in topology.links:
        terms = normalised_terms(topology.link_attributes(u, v))
        assert all(0.0 <= value <= 1.0 for value in terms.values())


def test_normalised_terms_extreme_values() -> None:
    worst = {"latency": 1000, "congestion": 1.0, "packet_loss": 1.0, "bandwidth": 1.0}
    terms = normalised_terms(worst)
    assert terms["latency"] == 1.0
    assert terms["packet_loss"] == 1.0
    assert terms["bandwidth"] == pytest.approx(0.999)


def test_link_cost_increases_with_congestion() -> None:
    calm = {"latency": 10, "congestion": 0.1, "packet_loss": 0.005, "bandwidth": 1000}
    busy = dict(calm, congestion=0.9)
    assert link_cost(busy) > link_cost(calm)


def test_link_cost_penalises_low_bandwidth() -> None:
    fast = {"latency": 10, "congestion": 0.2, "packet_loss": 0.01, "bandwidth": 1000}
    slow = dict(fast, bandwidth=100)
    assert link_cost(slow) > link_cost(fast)


def test_link_cost_breakdown_sums_to_total() -> None:
    attrs = {"latency": 42, "congestion": 0.7, "packet_loss": 0.08, "bandwidth": 250}
    parts = link_cost_breakdown(attrs, DEFAULT_WEIGHTS, ml_penalty=0.35)
    assert parts["total"] == pytest.approx(sum(v for k, v in parts.items() if k != "total"))
    assert parts["ML penalty"] == 0.35


def test_ml_penalty_adds_to_cost() -> None:
    attrs = {"latency": 10, "congestion": 0.2, "packet_loss": 0.005, "bandwidth": 1000}
    assert link_cost(attrs, DEFAULT_WEIGHTS, ML_COST_PENALTY[2]) == pytest.approx(
        link_cost(attrs) + ML_COST_PENALTY[2]
    )


def test_cost_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        CostWeights(latency=0.9, congestion=0.9, packet_loss=0.1, bandwidth=0.1)


def test_path_cost_is_sum_of_link_costs(topology) -> None:
    path = ["S", "A", "C", "T"]
    total = path_cost(topology, path)
    manual = sum(
        link_cost(topology.link_attributes(u, v)) for u, v in topology.link_on_path(path)
    )
    assert total == pytest.approx(manual)


# ---------------------------------------------------------------------------
# Adaptive route selection
# ---------------------------------------------------------------------------

def test_adaptive_picks_backbone_when_network_is_healthy(router) -> None:
    result = router.route("S", "T", strategy=STRATEGY_ADAPTIVE)
    assert result.path == ("S", "A", "C", "T")
    assert result.cost < 0.5


def test_adaptive_switches_away_from_congested_shortest_path(topology, router) -> None:
    ConditionSimulator(topology).apply(
        ConditionType.CONGESTION, prefer_path=["S", "A", "C", "T"], severity=1.0
    )
    baseline = shortest_path_route(topology, "S", "T")
    adaptive = router.route("S", "T", strategy=STRATEGY_ADAPTIVE)

    assert baseline.path == ("S", "A", "C", "T")      # static routing does not move
    assert adaptive.path != baseline.path             # adaptive routing does
    assert adaptive.path == ("S", "B", "D", "F", "T")
    # the adaptive choice must really be cheaper under the live conditions
    assert adaptive.cost < path_cost(topology, baseline.path)


def test_adaptive_reason_explains_the_decision(topology, router) -> None:
    ConditionSimulator(topology).apply(
        ConditionType.HIGH_LATENCY, prefer_path=["S", "A", "C", "T"], severity=1.0
    )
    result = router.route("S", "T", strategy=STRATEGY_ADAPTIVE)
    assert "dynamic cost" in result.reason
    assert "hop-count path" in result.reason
    assert "worst link" in result.reason
    assert "S-A" in result.reason


def test_adaptive_reason_names_avoided_congestion(topology, router) -> None:
    ConditionSimulator(topology).apply(
        ConditionType.CONGESTION, prefer_path=["S", "A", "C", "T"], severity=1.0
    )
    result = router.route("S", "T", strategy=STRATEGY_ADAPTIVE)
    assert "Avoids congested link" in result.reason


def test_adaptive_returns_alternatives(router) -> None:
    result = router.route("S", "T", strategy=STRATEGY_ADAPTIVE)
    assert len(result.alternatives) >= 1
    assert all(a.cost > result.cost for a in result.alternatives)


def test_unknown_strategy_raises(router) -> None:
    with pytest.raises(RoutingError):
        router.route("S", "T", strategy="Quantum Routing")


def test_adaptive_route_wrapper(topology) -> None:
    result = adaptive_route(topology, "S", "T")
    assert result.strategy == STRATEGY_ADAPTIVE


def test_link_costs_covers_every_link(router) -> None:
    rows = router.link_costs()
    assert len(rows) == router.topology.link_count
    assert rows == sorted(rows, key=lambda r: r["cost"])
    # The dashboard feeds these rows straight into a DataFrame, so the keys
    # must be unique and the two independent totals must agree.
    for row in rows:
        assert len(row) == len(set(row)), f"duplicate keys in {sorted(row)}"
        assert row["total"] == pytest.approx(row["cost"])


def test_compare_strategies_returns_all_strategies(router) -> None:
    results = router.compare_strategies("S", "T")
    assert set(results) == {STRATEGY_SHORTEST, STRATEGY_ADAPTIVE, STRATEGY_ADAPTIVE_ML}


# ---------------------------------------------------------------------------
# ML-assisted routing (penalty function injected from outside)
# ---------------------------------------------------------------------------

def _fake_penalty(u: str, v: str, attrs: dict) -> float:
    """Mark every link on the backbone as 'congested' in the ML model's eyes."""
    backbone = {("S", "A"), ("A", "C"), ("C", "T")}
    key = (u, v) if (u, v) in backbone else (v, u)
    return ML_COST_PENALTY[2] if key in backbone else 0.0


def test_ml_penalty_can_flip_the_route_without_real_congestion(topology) -> None:
    router = AdaptiveRouter(topology, penalty_fn=_fake_penalty)
    plain = router.route("S", "T", strategy=STRATEGY_ADAPTIVE)
    guided = router.route("S", "T", strategy=STRATEGY_ADAPTIVE_ML)
    assert plain.path == ("S", "A", "C", "T")
    assert guided.path != plain.path


def test_ml_strategy_falls_back_to_adaptive_without_predictor(router) -> None:
    """Project must keep working when the ML module is disabled."""
    result = router.route("S", "T", strategy=STRATEGY_ADAPTIVE_ML)
    assert result.strategy == STRATEGY_ADAPTIVE_ML
    assert result.path  # a route is still produced
    assert router.ml_enabled is False


def test_compute_penalties_is_empty_for_plain_adaptive(router) -> None:
    assert router.compute_penalties(STRATEGY_ADAPTIVE) == {}
    assert router.compute_penalties(STRATEGY_ADAPTIVE_ML) == {}
