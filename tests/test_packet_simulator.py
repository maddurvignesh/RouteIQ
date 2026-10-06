"""Unit tests for the simulated packet transmitter."""

from __future__ import annotations

import pytest

from config import DEFAULT_PACKET_CONFIG, PacketConfig
from core.network_conditions import ConditionSimulator, ConditionType
from core.packet_simulator import (
    STATUS_DELIVERED,
    STATUS_DROPPED,
    PacketSimulator,
    packets_to_records,
)
from core.topology import build_predefined_topology

BACKBONE = ["S", "A", "C", "T"]


@pytest.fixture()
def topology():
    return build_predefined_topology()


def test_all_packets_delivered_on_a_healthy_network(topology) -> None:
    sim = PacketSimulator(topology, seed=11)
    report = sim.send("S", "T", BACKBONE, num_packets=100)
    assert report.sent == 100
    assert report.delivered == 100
    assert report.dropped == 0


def test_packet_record_contains_required_fields(topology) -> None:
    sim = PacketSimulator(topology, seed=11)
    report = sim.send("S", "T", BACKBONE, num_packets=1, strategy="Shortest Path", step=2)
    packet = report.packets[0]
    assert packet.packet_id == 1
    assert packet.source == "S"
    assert packet.destination == "T"
    assert packet.route == ("S", "A", "C", "T")
    assert packet.sent_at_ms == 0.0
    assert packet.delay_ms > 0
    assert packet.status in (STATUS_DELIVERED, STATUS_DROPPED)
    assert packet.strategy == "Shortest Path"
    assert packet.step == 2
    assert packet.hops == 3


def test_congestion_increases_delay(topology) -> None:
    healthy = PacketSimulator(topology, seed=5).send("S", "T", BACKBONE, num_packets=50)
    ConditionSimulator(topology).apply(
        ConditionType.CONGESTION, prefer_path=BACKBONE, severity=1.0
    )
    congested = PacketSimulator(topology, seed=5).send("S", "T", BACKBONE, num_packets=50)

    healthy_delay = sum(p.delay_ms for p in healthy.packets) / healthy.sent
    congested_delay = sum(p.delay_ms for p in congested.packets) / congested.sent
    assert congested_delay > healthy_delay * 2


def test_congestion_can_cause_packet_loss(topology) -> None:
    ConditionSimulator(topology).apply(
        ConditionType.HIGH_PACKET_LOSS, prefer_path=BACKBONE, severity=1.0
    )
    report = PacketSimulator(topology, seed=7).send("S", "T", BACKBONE, num_packets=200)
    assert report.dropped > 0
    dropped = [p for p in report.packets if not p.delivered]
    assert all(p.dropped_at is not None for p in dropped)


def test_retransmissions_are_counted(topology) -> None:
    ConditionSimulator(topology).apply(
        ConditionType.HIGH_PACKET_LOSS, prefer_path=BACKBONE, severity=1.0
    )
    report = PacketSimulator(topology, seed=7).send("S", "T", BACKBONE, num_packets=200)
    # some packets must have needed more than one attempt
    assert any(p.attempts > 1 for p in report.packets)
    assert all(p.attempts <= DEFAULT_PACKET_CONFIG.max_retries + 1 for p in report.packets)


def test_clean_route_beats_congested_route(topology) -> None:
    ConditionSimulator(topology).apply(
        ConditionType.CONGESTION, prefer_path=BACKBONE, severity=1.0
    )
    sim = PacketSimulator(topology, seed=9)
    worst = sim.send("S", "T", BACKBONE, num_packets=150)
    best = sim.send("S", "T", ["S", "B", "D", "F", "T"], num_packets=150)

    assert best.delivered >= worst.delivered
    worst_delays = [p.delay_ms for p in worst.packets if p.delivered]
    best_delays = [p.delay_ms for p in best.packets if p.delivered]
    assert sum(best_delays) / len(best_delays) < sum(worst_delays) / len(worst_delays)


def test_same_seed_is_reproducible(topology) -> None:
    first = PacketSimulator(topology, seed=3).send("S", "T", BACKBONE, num_packets=40)
    second = PacketSimulator(topology, seed=3).send("S", "T", BACKBONE, num_packets=40)
    assert [p.status for p in first.packets] == [p.status for p in second.packets]
    assert [p.delay_ms for p in first.packets] == [p.delay_ms for p in second.packets]


def test_different_seed_changes_jitter(topology) -> None:
    first = PacketSimulator(topology, seed=1).send("S", "T", BACKBONE, num_packets=40)
    second = PacketSimulator(topology, seed=2).send("S", "T", BACKBONE, num_packets=40)
    assert [p.delay_ms for p in first.packets] != [p.delay_ms for p in second.packets]


def test_inter_arrival_spacing(topology) -> None:
    report = PacketSimulator(topology, seed=1).send("S", "T", BACKBONE, num_packets=10)
    times = [p.sent_at_ms for p in report.packets]
    gaps = [round(b - a, 6) for a, b in zip(times, times[1:])]
    assert gaps == [DEFAULT_PACKET_CONFIG.inter_arrival_ms] * 9


def test_route_must_match_endpoints(topology) -> None:
    sim = PacketSimulator(topology)
    with pytest.raises(ValueError):
        sim.send("S", "T", ["S", "A", "C"])


def test_route_must_exist(topology) -> None:
    sim = PacketSimulator(topology)
    with pytest.raises(ValueError):
        sim.send("S", "T", ["S", "T"])


def test_route_needs_two_routers(topology) -> None:
    sim = PacketSimulator(topology)
    with pytest.raises(ValueError):
        sim.send("S", "S", ["S"])


def test_packet_records_are_tabular(topology) -> None:
    report = PacketSimulator(topology, seed=1).send("S", "T", BACKBONE, num_packets=5)
    records = packets_to_records(report.packets)
    assert len(records) == 5
    assert {"packet_id", "route", "status", "delay_ms", "strategy"} <= set(records[0])


def test_custom_packet_config(topology) -> None:
    cfg = PacketConfig(num_packets=7, packet_size_kb=1, max_retries=0)
    report = PacketSimulator(topology, config=cfg, seed=1).send("S", "T", BACKBONE)
    assert report.sent == 7
    assert all(p.attempts == 1 for p in report.packets)
