"""Unit tests for the real-time network-condition simulator."""

from __future__ import annotations

import pytest

from core.network_conditions import ConditionSimulator, ConditionType
from core.topology import build_predefined_topology


@pytest.fixture()
def simulator() -> ConditionSimulator:
    return ConditionSimulator(build_predefined_topology())


def test_normal_condition_restores_everything(simulator: ConditionSimulator) -> None:
    before = [simulator.topology.link_attributes(*e) for e in simulator.topology.links]
    simulator.apply(ConditionType.CONGESTION, links=[("S", "A")], severity=1.0)
    simulator.apply(ConditionType.NORMAL)
    after = [simulator.topology.link_attributes(*e) for e in simulator.topology.links]
    for b, a in zip(before, after):
        assert b == a


def test_congestion_raises_congestion_and_latency(simulator: ConditionSimulator) -> None:
    before = simulator.topology.link_attributes("S", "A")
    simulator.apply(ConditionType.CONGESTION, links=[("S", "A")], severity=1.0)
    after = simulator.topology.link_attributes("S", "A")
    assert after["congestion"] > before["congestion"]
    assert after["latency"] > before["latency"]
    assert after["traffic"] > before["traffic"]
    # base values must not be touched
    assert after["base_latency"] == before["base_latency"]


def test_high_packet_loss_only_touches_loss(simulator: ConditionSimulator) -> None:
    before = simulator.topology.link_attributes("A", "C")
    simulator.apply(ConditionType.HIGH_PACKET_LOSS, links=[("A", "C")], severity=1.0)
    after = simulator.topology.link_attributes("A", "C")
    assert after["packet_loss"] > before["packet_loss"]
    assert after["latency"] == before["latency"]
    assert after["bandwidth"] == before["bandwidth"]


def test_high_latency_only_touches_latency(simulator: ConditionSimulator) -> None:
    before = simulator.topology.link_attributes("C", "T")
    simulator.apply(ConditionType.HIGH_LATENCY, links=[("C", "T")], severity=0.8)
    after = simulator.topology.link_attributes("C", "T")
    assert after["latency"] > before["latency"]
    assert after["packet_loss"] == before["packet_loss"]


def test_low_bandwidth_never_drops_below_minimum(simulator: ConditionSimulator) -> None:
    simulator.apply(ConditionType.LOW_BANDWIDTH, links=[("S", "A")], severity=1.0)
    assert simulator.topology.get("S", "A", "bandwidth") >= 50.0
    assert simulator.topology.get("S", "A", "congestion") > 0.20


def test_prefer_path_degrades_every_link_of_the_route(simulator: ConditionSimulator) -> None:
    event = simulator.apply(
        ConditionType.CONGESTION, prefer_path=["S", "A", "C", "T"], severity=1.0
    )
    assert event.links == (("S", "A"), ("A", "C"), ("C", "T"))
    for u, v in event.links:
        assert simulator.topology.is_congested(u, v)


def test_prefer_path_with_count_limits_the_number_of_links(
    simulator: ConditionSimulator,
) -> None:
    event = simulator.apply(
        ConditionType.CONGESTION, prefer_path=["S", "A", "C", "T"], severity=1.0, count=1
    )
    assert event.links == (("S", "A"),)


def test_multi_congestion_hits_three_links(simulator: ConditionSimulator) -> None:
    event = simulator.apply(ConditionType.MULTI_CONGESTION, severity=1.0)
    assert len(event.links) == 3
    for u, v in event.links:
        assert simulator.topology.get(u, v, "congestion") >= 0.5


def test_traffic_spike_keeps_traffic_below_capacity(simulator: ConditionSimulator) -> None:
    simulator.apply(ConditionType.TRAFFIC_SPIKE, links=[("D", "F")], severity=1.0)
    attrs = simulator.topology.link_attributes("D", "F")
    assert attrs["traffic"] <= attrs["bandwidth"]
    assert attrs["congestion"] > attrs["base_congestion"]


def test_severity_is_validated(simulator: ConditionSimulator) -> None:
    with pytest.raises(ValueError):
        simulator.apply(ConditionType.CONGESTION, severity=2.0)


def test_unknown_link_raises(simulator: ConditionSimulator) -> None:
    with pytest.raises(ValueError):
        simulator.apply(ConditionType.CONGESTION, links=[("S", "T")])


def test_event_history_is_recorded(simulator: ConditionSimulator) -> None:
    simulator.apply(ConditionType.HIGH_LATENCY, links=[("A", "C")])
    simulator.apply(ConditionType.NORMAL)
    assert len(simulator.history) == 2
    assert simulator.last_event.condition is ConditionType.NORMAL
