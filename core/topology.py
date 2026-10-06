"""Network topology representation.

A *topology* is a set of routers (nodes) connected by communication links
(edges).  Every link carries the physical/network parameters that the routing
engine needs:

===============  =========================================================
attribute        meaning
===============  =========================================================
bandwidth        maximum data rate of the link in Mbps
latency          fixed propagation+processing delay of one hop in ms
packet_loss      probability that a packet is lost on this link (0..1)
congestion       queue occupancy / load of the link (0..1)
reliability      probability that the link hardware is up (0..1)
traffic          currently offered load on the link in Mbps; kept equal to
                 ``congestion * bandwidth`` so utilisation == congestion
===============  =========================================================

Each edge also stores the ``base_*`` copy of these values so that a scenario
can always restore "normal" conditions.

The class wraps :class:`networkx.Graph`; NetworkX is used because it already
provides correct, well-tested implementations of Dijkstra, simple-path
enumeration and graph layout, which lets this project focus on the *routing
logic* instead of on graph plumbing.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Iterable

import networkx as nx

from config import RANDOM_TOPOLOGY_SEED

# Default (healthy) values used when a link does not specify them explicitly.
LINK_DEFAULTS: dict[str, float] = {
    "bandwidth": 500.0,     # Mbps
    "latency": 15.0,        # ms
    "packet_loss": 0.01,    # 1 %
    "congestion": 0.20,     # 20 % queue occupancy
    "reliability": 0.99,    # 99 % uptime
    "traffic": 100.0,       # Mbps (= congestion x bandwidth)
}

# Attributes that are simulated and therefore restored from ``base_<name>``.
VOLATILE_ATTRIBUTES = ("bandwidth", "latency", "packet_loss", "congestion", "traffic")


class TopologyError(ValueError):
    """Raised when a topology operation is not possible."""


class Topology:
    """Undirected network of routers and links."""

    def __init__(self, graph: nx.Graph, name: str = "custom", *, allow_empty: bool = False) -> None:
        self.name = name
        self.graph = graph
        self._validate(allow_empty=allow_empty)

    # -- construction ------------------------------------------------------
    @classmethod
    def empty(cls, name: str = "custom") -> "Topology":
        """Return an empty topology that routers/links can be added to."""
        return cls(nx.Graph(name=name), name=name, allow_empty=True)

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[str, str, dict[str, float]]],
        name: str = "custom",
    ) -> "Topology":
        """Build a topology from an iterable of ``(u, v, attributes)``."""
        topo = cls.empty(name)
        for u, v, attrs in edges:
            topo.add_link(u, v, **attrs)
        return topo

    def add_router(self, router_id: str, label: str | None = None, pos: tuple[float, float] | None = None) -> None:
        """Add a router node.  ``pos`` is an optional (x, y) drawing position."""
        attrs: dict[str, Any] = {}
        if label is not None:
            attrs["label"] = label
        if pos is not None:
            attrs["pos"] = pos
        self.graph.add_node(router_id, **attrs)

    def add_link(
        self,
        u: str,
        v: str,
        **attrs: float,
    ) -> None:
        """Add (or update) a link between two routers and store its base values."""
        merged = {**LINK_DEFAULTS, **attrs}
        for key in VOLATILE_ATTRIBUTES:
            merged[f"base_{key}"] = merged[key]
        self.graph.add_edge(u, v, **merged)

    # -- basic accessors ---------------------------------------------------
    @property
    def routers(self) -> list[str]:
        return list(self.graph.nodes)

    @property
    def links(self) -> list[tuple[str, str]]:
        return [(u, v) for u, v in self.graph.edges]

    @property
    def router_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def link_count(self) -> int:
        return self.graph.number_of_edges()

    def has_router(self, router_id: str) -> bool:
        return router_id in self.graph

    def has_link(self, u: str, v: str) -> bool:
        return self.graph.has_edge(u, v)

    def link_attributes(self, u: str, v: str) -> dict[str, float]:
        """Return a *copy* of the attributes of link ``u -- v``."""
        if not self.has_link(u, v):
            raise TopologyError(f"No link between {u} and {v}")
        return dict(self.graph.edges[u, v])

    def get(self, u: str, v: str, key: str) -> float:
        if not self.has_link(u, v):
            raise TopologyError(f"No link between {u} and {v}")
        return float(self.graph.edges[u, v][key])

    def set_link(self, u: str, v: str, **values: float) -> None:
        """Update live attributes of a link (``base_*`` values are untouched)."""
        if not self.has_link(u, v):
            raise TopologyError(f"No link between {u} and {v}")
        for key, value in values.items():
            self.graph.edges[u, v][key] = value

    def neighbours(self, router_id: str) -> list[str]:
        return list(self.graph.neighbors(router_id))

    def link_on_path(self, path: list[str]) -> list[tuple[str, str]]:
        """Convert ``['S', 'A', 'C']`` into ``[('S', 'A'), ('A', 'C')]``."""
        return list(zip(path, path[1:]))

    def path_attributes(self, path: list[str]) -> list[dict[str, float]]:
        return [self.link_attributes(u, v) for u, v in self.link_on_path(path)]

    def path_summary(self, path: list[str]) -> dict[str, float]:
        """Aggregate statistics of all links that make up ``path``."""
        if len(path) < 2:
            return {"hops": 0, "latency_ms": 0.0, "min_bandwidth_mbps": 0.0,
                    "avg_packet_loss": 0.0, "avg_congestion": 0.0}
        attrs = self.path_attributes(path)
        return {
            "hops": len(path) - 1,
            "latency_ms": sum(a["latency"] for a in attrs),
            "min_bandwidth_mbps": min(a["bandwidth"] for a in attrs),
            "avg_packet_loss": sum(a["packet_loss"] for a in attrs) / len(attrs),
            "avg_congestion": sum(a["congestion"] for a in attrs) / len(attrs),
        }

    # -- condition management ---------------------------------------------
    def reset_conditions(self) -> None:
        """Restore every link to its recorded healthy ("base") values."""
        for u, v in self.graph.edges:
            data = self.graph.edges[u, v]
            for key in VOLATILE_ATTRIBUTES:
                data[key] = data[f"base_{key}"]

    def is_congested(self, u: str, v: str, threshold: float = 0.70) -> bool:
        return self.get(u, v, "congestion") >= threshold

    def congested_links(self, threshold: float = 0.70) -> list[tuple[str, str]]:
        return [(u, v) for u, v in self.links if self.is_congested(u, v, threshold)]

    def utilization(self, u: str, v: str) -> float:
        """Offered load divided by capacity, clipped to [0, 1]."""
        bandwidth = max(self.get(u, v, "bandwidth"), 1e-9)
        return min(self.get(u, v, "traffic") / bandwidth, 1.0)

    # -- misc --------------------------------------------------------------
    def copy(self) -> "Topology":
        return Topology(copy.deepcopy(self.graph), self.name)

    def position_map(self) -> dict[str, tuple[float, float]]:
        """Drawing positions: stored ``pos`` when available, spring layout otherwise."""
        positions = {}
        for node, data in self.graph.nodes(data=True):
            if "pos" in data:
                pos = data["pos"]
                positions[node] = (float(pos[0]), float(pos[1]))
        if len(positions) == self.router_count:
            return positions
        # Deterministic fallback layout for graphs without stored positions.
        layout = nx.spring_layout(self.graph, seed=RANDOM_TOPOLOGY_SEED, k=1.2)
        for node, (x, y) in layout.items():
            positions[node] = (float(x), float(y))
        return positions

    def to_edge_records(self) -> list[dict[str, Any]]:
        """One flat record per link – handy for pandas / CSV export."""
        records = []
        for u, v, data in self.graph.edges(data=True):
            record = {"source": u, "target": v, **{k: data[k] for k in data if not k.startswith("base_")}}
            record["utilization"] = self.utilization(u, v)
            records.append(record)
        return records

    def _validate(self, allow_empty: bool = False) -> None:
        if self.graph.is_directed():
            raise TopologyError("This project models undirected links only.")
        if self.graph.number_of_nodes() == 0 and not allow_empty:
            raise TopologyError("Topology must contain at least one router.")


# ---------------------------------------------------------------------------
# Predefined demonstration topology
# ---------------------------------------------------------------------------
#
#   S --10ms-- A --10ms-- C --10ms-- T      <- 3 hops, 30 ms  (BACKBONE)
#   |           |           |    ^  \
#  30ms       35ms        28ms   |   30ms
#   |           |           |    |    \
#   B --20ms-- D --12ms-- F ----+     E
#              |           |
#             25ms        18ms
#              |           |
#              +---- E ----+
#
# The unique hop-count shortest path S -> A -> C -> T is intentionally built
# on a high-capacity backbone.  The alternative route S -> B -> D -> F -> T
# has one more hop but runs over lighter-loaded links, so it becomes the
# better choice as soon as the backbone congests.

DEMO_LINKS: list[tuple[str, str, dict[str, float]]] = [
    # backbone (shortest, 3 hops)
    ("S", "A", dict(latency=10, bandwidth=1000, packet_loss=0.005, congestion=0.20, reliability=0.995, traffic=200)),
    ("A", "C", dict(latency=10, bandwidth=1000, packet_loss=0.005, congestion=0.20, reliability=0.995, traffic=200)),
    ("C", "T", dict(latency=10, bandwidth=1000, packet_loss=0.005, congestion=0.20, reliability=0.995, traffic=200)),
    # lower-capacity upper detour
    ("S", "B", dict(latency=30, bandwidth=500, packet_loss=0.010, congestion=0.25, reliability=0.990, traffic=125)),
    ("B", "D", dict(latency=20, bandwidth=500, packet_loss=0.010, congestion=0.25, reliability=0.990, traffic=125)),
    # mid-section cross links
    ("A", "D", dict(latency=35, bandwidth=300, packet_loss=0.020, congestion=0.30, reliability=0.980, traffic=90)),
    ("C", "D", dict(latency=28, bandwidth=400, packet_loss=0.015, congestion=0.28, reliability=0.985, traffic=112)),
    ("C", "E", dict(latency=40, bandwidth=300, packet_loss=0.020, congestion=0.30, reliability=0.980, traffic=90)),
    ("D", "E", dict(latency=25, bandwidth=500, packet_loss=0.010, congestion=0.25, reliability=0.990, traffic=125)),
    # low-load southern corridor (the adaptive "escape" route)
    ("D", "F", dict(latency=12, bandwidth=800, packet_loss=0.005, congestion=0.15, reliability=0.995, traffic=120)),
    ("F", "E", dict(latency=18, bandwidth=600, packet_loss=0.008, congestion=0.20, reliability=0.990, traffic=120)),
    ("E", "T", dict(latency=30, bandwidth=500, packet_loss=0.010, congestion=0.25, reliability=0.990, traffic=125)),
    ("F", "T", dict(latency=14, bandwidth=800, packet_loss=0.005, congestion=0.15, reliability=0.995, traffic=120)),
]

# Hand-tuned drawing positions so the demo graph always looks the same.
DEMO_POSITIONS: dict[str, tuple[float, float]] = {
    "S": (0.0, 1.0),
    "A": (1.0, 1.6),
    "B": (1.0, 0.3),
    "C": (2.2, 1.6),
    "D": (2.2, 0.3),
    "E": (3.3, 0.4),
    "F": (3.3, -0.7),
    "T": (4.4, 1.0),
}

DEMO_LABELS: dict[str, str] = {"S": "S (Src)", "T": "T (Dst)"}


def build_predefined_topology() -> Topology:
    """Return the 8-router demonstration network used throughout the project."""
    topo = Topology.empty("predefined")
    for node, pos in DEMO_POSITIONS.items():
        topo.add_router(node, label=DEMO_LABELS.get(node, node), pos=pos)
    for u, v, attrs in DEMO_LINKS:
        topo.add_link(u, v, **attrs)
    return topo


def build_random_topology(
    num_routers: int = 10,
    connection_probability: float = 0.45,
    seed: int = RANDOM_TOPOLOGY_SEED,
) -> Topology:
    """Generate a random connected topology (used to show the generator works).

    A connected Erdős–Rényi graph is produced; isolated routers are re-wired
    until the graph is fully connected, because a routing demo needs at least
    one path between every pair of routers.
    """
    if num_routers < 2:
        raise TopologyError("A random topology needs at least 2 routers.")

    rng = random.Random(seed)
    nodes = [f"R{i + 1}" for i in range(num_routers)]

    while True:
        graph = nx.gnp_random_graph(num_routers, connection_probability, seed=seed)
        nx.relabel_nodes(graph, dict(enumerate(nodes)), copy=False)
        if nx.is_connected(graph):
            break
        # Force connectivity by chaining every node in a ring.
        ring = list(graph.nodes)
        for i in range(len(ring)):
            graph.add_edge(ring[i], ring[(i + 1) % len(ring)])

    topo = Topology.empty("random")
    angle_step = 2 * math.pi / num_routers
    for index, node in enumerate(graph.nodes):
        topo.add_router(
            node,
            pos=(2.5 * math.cos(index * angle_step), 2.5 * math.sin(index * angle_step)),
        )

    for u, v in graph.edges:
        attrs = {
            "latency": rng.uniform(5, 45),
            "bandwidth": float(rng.choice([100, 200, 300, 500, 800, 1000])),
            "packet_loss": rng.uniform(0.002, 0.03),
            "congestion": rng.uniform(0.10, 0.45),
            "reliability": rng.uniform(0.97, 0.999),
        }
        # keep the invariant traffic == congestion x bandwidth
        attrs["traffic"] = attrs["congestion"] * attrs["bandwidth"]
        topo.add_link(u, v, **attrs)
    return topo


TOPOLOGY_BUILDERS = {
    "predefined": build_predefined_topology,
    "random": build_random_topology,
}


def build_topology(name: str) -> Topology:
    """Factory used by the dashboard / experiments."""
    if name not in TOPOLOGY_BUILDERS:
        raise TopologyError(f"Unknown topology '{name}'. Choose from {list(TOPOLOGY_BUILDERS)}")
    return TOPOLOGY_BUILDERS[name]()
