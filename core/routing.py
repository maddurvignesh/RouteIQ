"""Traditional (load-oblivious) shortest-path routing.

This module is the **baseline** of the project.  It reproduces what a classic
distance-vector / link-state protocol such as RIP or OSPF does: it computes the
best path from a *static* metric (hop count or configured link cost) and then
keeps using that path until the topology itself changes.

It deliberately ignores live congestion, latency spikes and packet loss – the
whole point of the project is to show what happens when you do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import networkx as nx

from config import MAX_ALTERNATIVE_ROUTES
from core.topology import Topology

# Weight functions accepted by :func:`enumerate_routes`.
WeightFunction = Callable[[str, str, dict], float]

HOP_METRIC = "hops"
STATIC_LATENCY_METRIC = "static_latency"
SUPPORTED_METRICS = (HOP_METRIC, STATIC_LATENCY_METRIC)

BASELINE_STRATEGY = "Shortest Path"


class RoutingError(ValueError):
    """Raised when no route exists between two routers."""


@dataclass(frozen=True)
class Alternative:
    """A candidate path that was *not* selected, together with its cost."""

    path: tuple[str, ...]
    cost: float
    hops: int

    @property
    def as_text(self) -> str:
        return " -> ".join(self.path)


@dataclass(frozen=True)
class RouteResult:
    """Everything the caller needs to know about a routing decision."""

    strategy: str
    path: tuple[str, ...]
    cost: float
    reason: str
    alternatives: tuple[Alternative, ...] = field(default_factory=tuple)

    @property
    def hops(self) -> int:
        return max(len(self.path) - 1, 0)

    @property
    def exists(self) -> bool:
        return bool(self.path)

    @property
    def as_text(self) -> str:
        return " -> ".join(self.path)

    @property
    def links(self) -> list[tuple[str, str]]:
        return list(zip(self.path, self.path[1:]))


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def validate_endpoints(topology: Topology, src: str, dst: str) -> None:
    """Raise :class:`RoutingError` if the endpoints cannot be routed between."""
    if not topology.has_router(src):
        raise RoutingError(f"Unknown source router '{src}'")
    if not topology.has_router(dst):
        raise RoutingError(f"Unknown destination router '{dst}'")
    if src == dst:
        raise RoutingError("Source and destination must be different routers")
    if not nx.has_path(topology.graph, src, dst):
        raise RoutingError(f"No physical path exists between {src} and {dst}")


def metric_weight(metric: str) -> WeightFunction | None:
    """Return the edge-weight function for a *static* routing metric.

    ``None`` means "unweighted", which NetworkX interprets as hop count.
    """
    if metric == HOP_METRIC:
        return None
    if metric == STATIC_LATENCY_METRIC:
        return lambda _u, _v, data: float(data["base_latency"])
    raise RoutingError(f"Unknown routing metric '{metric}'. Choose from {SUPPORTED_METRICS}")


def enumerate_routes(
    topology: Topology,
    src: str,
    dst: str,
    weight: WeightFunction | None = None,
    limit: int = MAX_ALTERNATIVE_ROUTES,
) -> list[tuple[tuple[str, ...], float]]:
    """Return up to ``limit`` loop-free paths ordered from cheapest to costliest.

    ``weight`` is a NetworkX edge-weight callable ``(u, v, data) -> float``.
    Passing ``None`` orders the paths by hop count.
    """
    validate_endpoints(topology, src, dst)
    results: list[tuple[tuple[str, ...], float]] = []
    generator = nx.shortest_simple_paths(topology.graph, src, dst, weight=weight)
    for index, path in enumerate(generator):
        if index >= limit:
            break
        results.append((tuple(path), _path_weight(topology, path, weight)))
    return results


def _path_weight(topology: Topology, path: list[str] | tuple[str, ...], weight: WeightFunction | None) -> float:
    if weight is None:
        return float(max(len(path) - 1, 0))
    total = 0.0
    for u, v in zip(path, path[1:]):
        total += float(weight(u, v, topology.graph.edges[u, v]))
    return total


# ---------------------------------------------------------------------------
# Baseline routing algorithm
# ---------------------------------------------------------------------------

def shortest_path_route(
    topology: Topology,
    src: str,
    dst: str,
    metric: str = HOP_METRIC,
    strategy: str = BASELINE_STRATEGY,
    include_alternatives: bool = True,
) -> RouteResult:
    """Compute the traditional shortest path and explain the decision.

    Parameters
    ----------
    topology:
        Current network.
    src, dst:
        Source and destination routers.
    metric:
        ``"hops"`` (classic hop-count routing) or ``"static_latency"``
        (OSPF-style cost derived from a fixed link metric).
    strategy:
        Label stored in the result (``Shortest Path`` by default).
    include_alternatives:
        Also compute up to ``MAX_ALTERNATIVE_ROUTES`` runner-up paths.
    """
    weight = metric_weight(metric)
    validate_endpoints(topology, src, dst)

    path = list(nx.shortest_path(topology.graph, src, dst, weight=weight))
    cost = _path_weight(topology, path, weight)
    hops = len(path) - 1

    alternatives: tuple[Alternative, ...] = ()
    if include_alternatives:
        candidates = enumerate_routes(topology, src, dst, weight=weight,
                                      limit=MAX_ALTERNATIVE_ROUTES + 1)
        alternatives = tuple(
            Alternative(path=p, cost=c, hops=len(p) - 1)
            for p, c in candidates[1:] if p != tuple(path)
        )

    metric_text = "hop count" if metric == HOP_METRIC else "static link cost (base latency)"
    reason = (
        f"Traditional Dijkstra on {metric_text}: {hops} hop(s), "
        f"static cost {cost:.2f}. This route is computed once from the "
        f"configured topology and does **not** react to live load."
    )
    return RouteResult(
        strategy=strategy,
        path=tuple(path),
        cost=cost,
        reason=reason,
        alternatives=alternatives,
    )
