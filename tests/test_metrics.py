"""Unit tests for metric computation (simulation/metrics.py)."""

from __future__ import annotations

import pytest

from config import CONGESTION_HIGH_THRESHOLD
from core.packet_simulator import STATUS_DELIVERED, STATUS_DROPPED, Packet
from core.topology import build_predefined_topology
from simulation.metrics import (
    SimulationMetrics,
    StepRecord,
    build_step_record,
    compute_metrics,
    count_congested_links,
)


def _packet(pid: int, delay: float, delivered: bool, size: int = 1024) -> Packet:
    return Packet(
        packet_id=pid,
        source="S",
        destination="T",
        route=("S", "A", "T"),
        size_bytes=size,
        status=STATUS_DELIVERED if delivered else STATUS_DROPPED,
        delay_ms=delay,
        sent_at_ms=0.0,
        delivered_at_ms=delay,
        attempts=1,
        dropped_at=None if delivered else "A-T",
    )


def _steps(routes: list[tuple[str, ...]], costs: list[float] | None = None) -> list[StepRecord]:
    costs = costs or [1.0] * len(routes)
    return [
        StepRecord(
            step=i, route=r, cost=c, condition="x",
            congested_links=0, total_links=10,
            route_utilization=0.5, network_utilization=0.4,
            route_has_congestion=False,
        )
        for i, (r, c) in enumerate(zip(routes, costs))
    ]


def test_latency_statistics() -> None:
    packets = [_packet(1, 10.0, True), _packet(2, 30.0, True), _packet(3, 0.0, False)]
    metrics = compute_metrics(packets, _steps([("S", "T")]), build_predefined_topology())
    assert metrics.avg_latency_ms == pytest.approx(20.0)
    assert metrics.min_latency_ms == pytest.approx(10.0)
    assert metrics.max_latency_ms == pytest.approx(30.0)


def test_packet_loss_and_delivery_rate() -> None:
    packets = [_packet(i, 5.0, i <= 7) for i in range(1, 11)]
    metrics = compute_metrics(packets, _steps([("S", "T")]), build_predefined_topology())
    assert metrics.packets_sent == 10
    assert metrics.packets_delivered == 7
    assert metrics.packet_loss_pct == pytest.approx(30.0)
    assert metrics.delivery_rate_pct == pytest.approx(70.0)


def test_throughput_is_delivered_bits_over_window() -> None:
    packets = [
        Packet(packet_id=1, source="S", destination="T", route=("S", "T"),
               size_bytes=1000, status=STATUS_DELIVERED, delay_ms=100.0,
               sent_at_ms=0.0, delivered_at_ms=100.0, attempts=1),
        Packet(packet_id=2, source="S", destination="T", route=("S", "T"),
               size_bytes=1000, status=STATUS_DELIVERED, delay_ms=100.0,
               sent_at_ms=0.0, delivered_at_ms=200.0, attempts=1),
    ]
    metrics = compute_metrics(packets, _steps([("S", "T")]), build_predefined_topology())
    # 2000 bytes = 16000 bits delivered in 0.2 s -> 0.08 Mbit/s
    assert metrics.throughput_mbps == pytest.approx(0.08, rel=1e-3)


def test_throughput_zero_when_nothing_delivered() -> None:
    packets = [_packet(1, 10.0, False)]
    metrics = compute_metrics(packets, _steps([("S", "T")]), build_predefined_topology())
    assert metrics.throughput_mbps == 0.0


def test_route_changes_are_counted() -> None:
    routes = [("S", "A", "T"), ("S", "A", "T"), ("S", "B", "T"), ("S", "B", "T")]
    metrics = compute_metrics([], _steps(routes), build_predefined_topology())
    assert metrics.route_changes == 1
    assert metrics.distinct_routes == 2


def test_no_route_change_when_route_is_stable() -> None:
    metrics = compute_metrics([], _steps([("S", "T")] * 5), build_predefined_topology())
    assert metrics.route_changes == 0
    assert metrics.distinct_routes == 1


def test_avg_route_cost_and_hops() -> None:
    routes = [("S", "A", "T"), ("S", "A", "B", "T")]
    metrics = compute_metrics([], _steps(routes, [0.5, 1.5]), build_predefined_topology())
    assert metrics.avg_route_cost == pytest.approx(1.0)
    assert metrics.avg_hops == pytest.approx(2.5)


def test_congestion_frequency_reflects_selected_route() -> None:
    congested = StepRecord(step=0, route=("S", "T"), cost=1.0, condition="x",
                           congested_links=3, total_links=10, route_has_congestion=True)
    calm = StepRecord(step=1, route=("S", "T"), cost=1.0, condition="x",
                      congested_links=0, total_links=10, route_has_congestion=False)
    metrics = compute_metrics([], [congested, calm], build_predefined_topology())
    assert metrics.congestion_frequency_pct == pytest.approx(50.0)


def test_route_and_network_utilization_come_from_step_records() -> None:
    metrics = compute_metrics([], _steps([("S", "T")] * 3), build_predefined_topology())
    assert metrics.route_utilization_pct == pytest.approx(50.0)
    assert metrics.network_utilization_pct == pytest.approx(40.0)


def test_metrics_serialise_to_dict() -> None:
    metrics = compute_metrics([], _steps([("S", "T")]), build_predefined_topology(),
                              scenario="exp1", strategy="Shortest Path")
    data = metrics.as_dict()
    assert data["scenario"] == "exp1"
    assert data["strategy"] == "Shortest Path"
    assert isinstance(data, dict)


def test_empty_run_is_safe() -> None:
    metrics = compute_metrics([], [], build_predefined_topology())
    assert metrics.packets_sent == 0
    assert metrics.avg_latency_ms == 0.0
    assert metrics.packet_loss_pct == 0.0
    assert metrics.route_changes == 0
    assert metrics.throughput_mbps == 0.0


def test_final_route_property() -> None:
    metrics = SimulationMetrics(scenario="x", strategy="y",
                                routes=("S -> A -> T", "S -> B -> T"))
    assert metrics.final_route == "S -> B -> T"


def test_build_step_record_snapshots_link_state() -> None:
    topo = build_predefined_topology()
    topo.set_link("S", "A", congestion=0.95, traffic=990.0)
    record = build_step_record(1, ["S", "A", "C", "T"], 0.4, "Congestion", topo)

    assert record.route == ("S", "A", "C", "T")
    assert record.congested_links == 1
    assert record.total_links == topo.link_count
    assert record.route_has_congestion is True
    assert record.network_utilization > 0
    assert 0 <= record.route_utilization <= 1
    assert record.route_text == "S -> A -> C -> T"


def test_count_congested_links() -> None:
    topo = build_predefined_topology()
    assert count_congested_links(topo) == 0
    topo.set_link("S", "A", congestion=CONGESTION_HIGH_THRESHOLD)
    assert count_congested_links(topo) == 1
