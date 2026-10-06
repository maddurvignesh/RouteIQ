"""Unit tests for the network topology generator (core/topology.py)."""

from __future__ import annotations

import networkx as nx
import pytest

from core.topology import (
    LINK_DEFAULTS,
    Topology,
    TopologyError,
    build_predefined_topology,
    build_random_topology,
    build_topology,
)


def test_predefined_topology_shape() -> None:
    topo = build_predefined_topology()
    assert topo.router_count == 8
    assert topo.link_count == 13
    assert topo.name == "predefined"


def test_predefined_topology_is_connected() -> None:
    topo = build_predefined_topology()
    assert nx.is_connected(topo.graph)


def test_predefined_topology_has_alternative_routes() -> None:
    """Adaptive routing is pointless unless several distinct paths exist."""
    topo = build_predefined_topology()
    paths = list(nx.all_simple_paths(topo.graph, "S", "T", cutoff=6))
    assert len(paths) >= 5


def test_predefined_shortest_path_is_unique_three_hops() -> None:
    """The hop-count shortest path must be unique so the baseline is stable."""
    topo = build_predefined_topology()
    paths = list(nx.all_simple_paths(topo.graph, "S", "T", cutoff=6))
    three_hop = [p for p in paths if len(p) - 1 == 3]
    assert three_hop == [["S", "A", "C", "T"]]


def test_add_link_records_base_values() -> None:
    topo = Topology.empty()
    topo.add_router("A")
    topo.add_router("B")
    topo.add_link("A", "B", latency=20.0, bandwidth=800.0)
    attrs = topo.link_attributes("A", "B")
    assert attrs["latency"] == 20.0
    assert attrs["base_latency"] == 20.0
    assert attrs["bandwidth"] == 800.0
    assert attrs["base_bandwidth"] == 800.0
    # unspecified attributes fall back to the documented defaults
    assert attrs["packet_loss"] == LINK_DEFAULTS["packet_loss"]


def test_reset_conditions_restores_original_values() -> None:
    topo = build_predefined_topology()
    original = topo.link_attributes("S", "A")
    topo.set_link("S", "A", latency=95.0, congestion=0.95, packet_loss=0.3)
    assert topo.get("S", "A", "latency") == 95.0
    topo.reset_conditions()
    restored = topo.link_attributes("S", "A")
    for key in ("latency", "congestion", "packet_loss", "bandwidth", "traffic"):
        assert restored[key] == original[key]


def test_link_attributes_returns_a_copy() -> None:
    topo = build_predefined_topology()
    attrs = topo.link_attributes("S", "A")
    attrs["latency"] = -1
    assert topo.get("S", "A", "latency") != -1


def test_missing_link_raises() -> None:
    topo = build_predefined_topology()
    with pytest.raises(TopologyError):
        topo.link_attributes("S", "T")


def test_path_summary_and_link_list() -> None:
    topo = build_predefined_topology()
    path = ["S", "A", "C", "T"]
    assert topo.link_on_path(path) == [("S", "A"), ("A", "C"), ("C", "T")]
    summary = topo.path_summary(path)
    assert summary["hops"] == 3
    assert summary["latency_ms"] == 30.0


def test_utilization_and_congested_links() -> None:
    topo = build_predefined_topology()
    util = topo.utilization("S", "A")  # 200 Mbps on a 1000 Mbps link
    assert 0.19 <= util <= 0.21
    topo.set_link("S", "A", congestion=0.90)
    assert topo.is_congested("S", "A")
    assert ("S", "A") in topo.congested_links()


def test_random_topology_is_connected_and_has_base_values() -> None:
    topo = build_random_topology(num_routers=10, seed=3)
    assert topo.router_count == 10
    assert nx.is_connected(topo.graph)
    for u, v in topo.links:
        assert "base_latency" in topo.graph.edges[u, v]
        assert "base_bandwidth" in topo.graph.edges[u, v]


def test_topology_factory_rejects_unknown_name() -> None:
    with pytest.raises(TopologyError):
        build_topology("does-not-exist")


def test_directed_graph_rejected() -> None:
    with pytest.raises(TopologyError):
        Topology(nx.DiGraph())


def test_positions_available_for_every_router() -> None:
    topo = build_predefined_topology()
    positions = topo.position_map()
    assert set(positions) == set(topo.routers)
    assert all(len(p) == 2 for p in positions.values())


def test_traffic_equals_congestion_times_bandwidth() -> None:
    """Documented invariant: utilisation == congestion on every link."""
    for topo in (build_predefined_topology(), build_random_topology(seed=4)):
        for u, v in topo.links:
            expected = topo.get(u, v, "congestion") * topo.get(u, v, "bandwidth")
            assert topo.get(u, v, "traffic") == pytest.approx(expected)
