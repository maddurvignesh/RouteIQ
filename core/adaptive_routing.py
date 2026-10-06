"""Adaptive routing driven by live link conditions (and optionally by ML).

Adaptive cost formula
---------------------

For every link the following **normalised** terms are computed:

.. math::

    L = \\min\\left(\\frac{latency_{ms}}{100}, 1\\right)
        \\qquad
        C = congestion \\in [0,1]
        \\qquad
        P = \\min\\left(\\frac{packet\\_loss}{0.20}, 1\\right)
        \\qquad
        B = 1 - \\min\\left(\\frac{bandwidth_{Mbps}}{1000}, 1\\right)

and the link cost is

.. math::

    cost = w_1 L + w_2 C + w_3 P + w_4 B + penalty_{ML}

with default weights ``(w1, w2, w3, w4) = (0.35, 0.30, 0.25, 0.10)``.
``penalty_{ML}`` is ``0.0`` unless the ML strategy is selected; the ML module
then adds ``0.0 / 0.35 / 0.90`` for a predicted *Normal / Warning / Congested*
link state.

A path is simply the loop-free route whose **total cost** (sum of link costs)
is smallest.  Dijkstra's algorithm is used with this dynamic cost as edge
weight, which is exactly how a link-state protocol computes its routing table –
only the metric is now measured in real time instead of being static.

Design note: the router never imports the ML package.  It only accepts an
optional ``penalty_fn(u, v, attributes) -> float`` callback, so the routing
algorithm keeps working unchanged when the ML module is switched off.
"""

from __future__ import annotations

from typing import Callable, Iterable

import networkx as nx

from config import (
    BANDWIDTH_NORMALISATION_MBPS,
    DEFAULT_WEIGHTS,
    LATENCY_NORMALISATION_MS,
    MAX_ALTERNATIVE_ROUTES,
    ML_COST_PENALTY,
    ML_STATE_LABELS,
    PACKET_LOSS_NORMALISATION,
    STRATEGY_ADAPTIVE,
    STRATEGY_ADAPTIVE_ML,
    STRATEGY_SHORTEST,
    CostWeights,
)
from core.routing import (
    Alternative,
    RouteResult,
    RoutingError,
    enumerate_routes,
    shortest_path_route,
    validate_endpoints,
)
from core.topology import Topology

# ``(u, v, attributes) -> extra cost`` ; injected by the ML module.
PenaltyFunction = Callable[[str, str, dict], float]

ADAPTIVE_STRATEGIES = (STRATEGY_ADAPTIVE, STRATEGY_ADAPTIVE_ML)


# ---------------------------------------------------------------------------
# Cost maths
# ---------------------------------------------------------------------------

def normalised_terms(attributes: dict) -> dict[str, float]:
    """Return the four normalised link terms of the adaptive cost formula."""
    return {
        "latency": min(float(attributes["latency"]) / LATENCY_NORMALISATION_MS, 1.0),
        "congestion": min(max(float(attributes["congestion"]), 0.0), 1.0),
        "packet_loss": min(float(attributes["packet_loss"]) / PACKET_LOSS_NORMALISATION, 1.0),
        "bandwidth": 1.0 - min(float(attributes["bandwidth"]) / BANDWIDTH_NORMALISATION_MBPS, 1.0),
    }


def link_cost(
    attributes: dict,
    weights: CostWeights = DEFAULT_WEIGHTS,
    ml_penalty: float = 0.0,
) -> float:
    """Dynamic cost of a single link (lower is better)."""
    terms = normalised_terms(attributes)
    return (
        weights.latency * terms["latency"]
        + weights.congestion * terms["congestion"]
        + weights.packet_loss * terms["packet_loss"]
        + weights.bandwidth * terms["bandwidth"]
        + ml_penalty
    )


def link_cost_breakdown(
    attributes: dict,
    weights: CostWeights = DEFAULT_WEIGHTS,
    ml_penalty: float = 0.0,
) -> dict[str, float]:
    """Per-term contribution of the cost formula – used by the dashboard."""
    terms = normalised_terms(attributes)
    parts = {
        "latency term": weights.latency * terms["latency"],
        "congestion term": weights.congestion * terms["congestion"],
        "packet-loss term": weights.packet_loss * terms["packet_loss"],
        "bandwidth term": weights.bandwidth * terms["bandwidth"],
        "ML penalty": float(ml_penalty),
    }
    parts["total"] = sum(parts.values())
    return parts


def path_cost(
    topology: Topology,
    path: Iterable[str],
    weights: CostWeights = DEFAULT_WEIGHTS,
    penalties: dict[tuple[str, str], float] | None = None,
) -> float:
    """Sum of the dynamic costs of all links on ``path``."""
    nodes = list(path)
    total = 0.0
    for u, v in zip(nodes, nodes[1:]):
        penalty = 0.0
        if penalties:
            penalty = penalties.get(_canonical(u, v), 0.0)
        total += link_cost(topology.link_attributes(u, v), weights, penalty)
    return total


def _canonical(u: str, v: str) -> tuple[str, str]:
    return (u, v) if u <= v else (v, u)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class AdaptiveRouter:
    """Selects routes using live conditions instead of a static metric.

    Parameters
    ----------
    topology:
        The network this router computes paths for.
    weights:
        Weights of the adaptive cost formula.
    penalty_fn:
        Optional callback returning the extra ML penalty for a link.  When it
        is ``None`` the router behaves as pure adaptive routing, so the project
        stays fully functional with the ML module disabled.
    """

    def __init__(
        self,
        topology: Topology,
        weights: CostWeights = DEFAULT_WEIGHTS,
        penalty_fn: PenaltyFunction | None = None,
    ) -> None:
        self.topology = topology
        self.weights = weights
        self.penalty_fn = penalty_fn

    # -- public helpers ----------------------------------------------------
    @property
    def ml_enabled(self) -> bool:
        return self.penalty_fn is not None

    def compute_penalties(self, strategy: str) -> dict[tuple[str, str], float]:
        """Evaluate the ML penalty for every link (once per routing decision)."""
        if strategy != STRATEGY_ADAPTIVE_ML or self.penalty_fn is None:
            return {}
        penalties: dict[tuple[str, str], float] = {}
        for u, v, data in self.topology.graph.edges(data=True):
            penalties[_canonical(u, v)] = float(self.penalty_fn(u, v, dict(data)))
        return penalties

    def weight_function(
        self, strategy: str, penalties: dict[tuple[str, str], float]
    ) -> Callable[[str, str, dict], float]:
        """Build the NetworkX edge-weight callable for a strategy."""

        def weight(u: str, v: str, data: dict) -> float:
            penalty = penalties.get(_canonical(u, v), 0.0)
            return link_cost(data, self.weights, penalty)

        return weight

    # -- main entry point ---------------------------------------------------
    def route(
        self,
        src: str,
        dst: str,
        strategy: str = STRATEGY_ADAPTIVE,
        include_alternatives: bool = True,
    ) -> RouteResult:
        """Return the :class:`RouteResult` for ``src -> dst`` under ``strategy``."""
        if strategy == STRATEGY_SHORTEST:
            return shortest_path_route(self.topology, src, dst, metric="hops")
        if strategy not in ADAPTIVE_STRATEGIES:
            raise RoutingError(
                f"Unknown strategy '{strategy}'. Choose from "
                f"{(STRATEGY_SHORTEST, *ADAPTIVE_STRATEGIES)}"
            )
        return self._adaptive_route(src, dst, strategy, include_alternatives)

    # -- internals ----------------------------------------------------------
    def _adaptive_route(
        self, src: str, dst: str, strategy: str, include_alternatives: bool
    ) -> RouteResult:
        validate_endpoints(self.topology, src, dst)
        penalties = self.compute_penalties(strategy)
        weight = self.weight_function(strategy, penalties)

        path = list(nx.shortest_path(self.topology.graph, src, dst, weight=weight))
        cost = path_cost(self.topology, path, self.weights, penalties)

        alternatives: tuple[Alternative, ...] = ()
        if include_alternatives:
            candidates = enumerate_routes(
                self.topology, src, dst, weight=weight, limit=MAX_ALTERNATIVE_ROUTES + 1
            )
            alternatives = tuple(
                Alternative(path=p, cost=c, hops=len(p) - 1)
                for p, c in candidates[1:] if p != tuple(path)
            )

        reason = self._explain(path, cost, strategy, penalties)
        return RouteResult(
            strategy=strategy,
            path=tuple(path),
            cost=cost,
            reason=reason,
            alternatives=alternatives,
        )

    def _explain(
        self,
        path: list[str],
        cost: float,
        strategy: str,
        penalties: dict[tuple[str, str], float],
    ) -> str:
        """Build a human-readable justification of the routing decision."""
        summary = self.topology.path_summary(path)
        parts = [
            f"Lowest dynamic cost {cost:.3f} over {summary['hops']} hop(s): "
            f"avg latency {summary['latency_ms']:.1f} ms, "
            f"avg congestion {summary['avg_congestion']:.0%}, "
            f"avg loss {summary['avg_packet_loss']:.2%}, "
            f"bottleneck {summary['min_bandwidth_mbps']:.0f} Mbps."
        ]

        # Compare with the classical hop-count choice.
        baseline = shortest_path_route(self.topology, path[0], path[-1],
                                       metric="hops", include_alternatives=False)
        if baseline.path != tuple(path):
            baseline_cost = path_cost(self.topology, baseline.path, self.weights, penalties)
            worst_link, worst_cost = self._workest_link(baseline.path, penalties)
            parts.append(
                f"Chosen instead of the hop-count path {' -> '.join(baseline.path)}, "
                f"whose live cost is {baseline_cost:.3f} versus {cost:.3f} here "
                f"(worst link {worst_link} costs {worst_cost:.3f})."
            )

        congested = {_canonical(u, v) for u, v in self.topology.congested_links()}
        route_links = [_canonical(u, v) for u, v in self.topology.link_on_path(path)]
        on_route = [f"{u}-{v}" for u, v in route_links if (u, v) in congested]
        avoided = [f"{u}-{v}" for u, v in sorted(congested) if (u, v) not in set(route_links)]
        if avoided:
            parts.append(f"Avoids congested link(s): {', '.join(avoided)}.")
        if on_route and not avoided:
            parts.append(f"Every alternative is at least as congested as {', '.join(on_route)}.")

        if strategy == STRATEGY_ADAPTIVE_ML and penalties:
            flagged = [
                f"{u}-{v} ({ML_STATE_LABELS.get(_level(penalties, self.topology, u, v), '?')})"
                for u, v in self.topology.link_on_path(path)
                if penalties.get(_canonical(u, v), 0.0) > 0
            ]
            if flagged:
                parts.append(f"ML penalty applied on this route: {', '.join(flagged)}.")
            else:
                parts.append("ML module predicts every link on this route is healthy.")

        return " ".join(parts)

    def _workest_link(
        self, path: list[str] | tuple[str, ...], penalties: dict[tuple[str, str], float]
    ) -> tuple[str, float]:
        """Return ``(link label, cost)`` of the most expensive link on ``path``."""
        worst_label, worst_cost = "-", float("-inf")
        for u, v in zip(path, path[1:]):
            penalty = penalties.get(_canonical(u, v), 0.0)
            cost = link_cost(self.topology.link_attributes(u, v), self.weights, penalty)
            if cost > worst_cost:
                worst_label, worst_cost = f"{u}-{v}", cost
        return worst_label, worst_cost

    # -- inspection helpers -------------------------------------------------
    def link_costs(self, strategy: str = STRATEGY_ADAPTIVE) -> list[dict]:
        """Cost + breakdown of every link – rendered as a table on the dashboard."""
        penalties = self.compute_penalties(strategy)
        rows = []
        for u, v, data in sorted(self.topology.graph.edges(data=True)):
            penalty = penalties.get(_canonical(u, v), 0.0)
            row = {
                "link": f"{u}-{v}",
                "cost": link_cost(data, self.weights, penalty),
                **link_cost_breakdown(data, self.weights, penalty),
                "latency_ms": data["latency"],
                "congestion": data["congestion"],
                "packet_loss": data["packet_loss"],
                "bandwidth_mbps": data["bandwidth"],
            }
            rows.append(row)
        return sorted(rows, key=lambda r: r["cost"])

    def compare_strategies(
        self, src: str, dst: str, strategies: Iterable[str] = (STRATEGY_SHORTEST, *ADAPTIVE_STRATEGIES)
    ) -> dict[str, RouteResult]:
        """Convenience helper: same endpoints, several strategies side by side."""
        return {name: self.route(src, dst, strategy=name) for name in strategies}


def _level(penalties: dict, topology: Topology, u: str, v: str) -> int:
    """Reverse the ML penalty back into the predicted congestion level."""
    value = penalties.get(_canonical(u, v), 0.0)
    for level, penalty in sorted(ML_COST_PENALTY.items(), reverse=True):
        if abs(value - penalty) < 1e-9:
            return level
    return 0


def adaptive_route(
    topology: Topology,
    src: str,
    dst: str,
    strategy: str = STRATEGY_ADAPTIVE,
    penalty_fn: PenaltyFunction | None = None,
    weights: CostWeights = DEFAULT_WEIGHTS,
) -> RouteResult:
    """One-shot convenience wrapper around :class:`AdaptiveRouter`."""
    return AdaptiveRouter(topology, weights=weights, penalty_fn=penalty_fn).route(
        src, dst, strategy=strategy
    )
