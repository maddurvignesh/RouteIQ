"""Performance metrics computed from an actual simulation run.

Every number produced here comes from packets that were really simulated – no
value in this project is hard-coded or fabricated.

Metric definitions in plain Computer Networks language
------------------------------------------------------
===============  ==============================================================
metric           meaning
===============  ==============================================================
avg latency      average end-to-end delay of the packets that were delivered
min / max        fastest and slowest delivered packet
packet loss %    percentage of packets that could not be delivered even after
                 all retransmissions
throughput       useful data actually delivered per second, in Mbit/s
delivery rate %  percentage of packets delivered successfully
route changes    how many times the routing decision changed between steps
avg route cost   average dynamic cost of the routes that were selected - the
                 four weighted terms only, so the value stays comparable even
                 when one strategy carries an ML penalty
network util.    average of (traffic / capacity) over **all** links, in %
route util.      same average, but only over the links the strategy selected
congestion freq. percentage of steps in which the *selected* route crossed at
                 least one congested link
===============  ==============================================================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

from config import CONGESTION_HIGH_THRESHOLD
from core.packet_simulator import Packet
from core.topology import Topology


@dataclass
class StepRecord:
    """What happened at one time step of a simulation run."""

    step: int
    route: tuple[str, ...]
    cost: float
    condition: str
    congested_links: int
    total_links: int
    route_utilization: float = 0.0
    network_utilization: float = 0.0
    route_has_congestion: bool = False

    @property
    def congested_fraction(self) -> float:
        if self.total_links == 0:
            return 0.0
        return self.congested_links / self.total_links

    @property
    def route_text(self) -> str:
        return " -> ".join(self.route)


@dataclass
class SimulationMetrics:
    """Complete metric bundle of one (scenario x strategy) run."""

    scenario: str
    strategy: str
    packets_sent: int = 0
    packets_delivered: int = 0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    packet_loss_pct: float = 0.0
    throughput_mbps: float = 0.0
    delivery_rate_pct: float = 0.0
    network_utilization_pct: float = 0.0
    route_utilization_pct: float = 0.0
    route_changes: int = 0
    avg_route_cost: float = 0.0
    congestion_frequency_pct: float = 0.0
    avg_hops: float = 0.0
    distinct_routes: int = 1
    routes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def final_route(self) -> str:
        return self.routes[-1] if self.routes else ""


def compute_metrics(
    packets: Sequence[Packet],
    steps: Sequence[StepRecord],
    topology: Topology,
    scenario: str = "",
    strategy: str = "",
) -> SimulationMetrics:
    """Derive all performance metrics from packets, steps and the topology."""
    sent = list(packets)
    delivered = [p for p in sent if p.delivered]
    dropped = [p for p in sent if not p.delivered]

    delays = [p.delay_ms for p in delivered]
    metrics = SimulationMetrics(
        scenario=scenario,
        strategy=strategy,
        packets_sent=len(sent),
        packets_delivered=len(delivered),
        avg_latency_ms=_mean(delays),
        min_latency_ms=min(delays) if delays else 0.0,
        max_latency_ms=max(delays) if delays else 0.0,
        packet_loss_pct=(len(dropped) / len(sent) * 100.0) if sent else 0.0,
        delivery_rate_pct=(len(delivered) / len(sent) * 100.0) if sent else 0.0,
        throughput_mbps=_throughput(sent, delivered),
        network_utilization_pct=_mean([s.network_utilization for s in steps]) * 100.0,
        route_utilization_pct=_mean([s.route_utilization for s in steps]) * 100.0,
        route_changes=_route_changes(steps),
        avg_route_cost=_mean([s.cost for s in steps]),
        congestion_frequency_pct=_mean(
            [100.0 if s.route_has_congestion else 0.0 for s in steps]
        ),
        avg_hops=_mean([len(s.route) - 1 for s in steps if len(s.route) >= 2]),
        distinct_routes=len({s.route for s in steps}),
        routes=tuple(s.route_text for s in steps),
    )
    return metrics


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def build_step_record(
    step: int,
    route: Sequence[str],
    cost: float,
    condition: str,
    topology: Topology,
    threshold: float = CONGESTION_HIGH_THRESHOLD,
) -> StepRecord:
    """Snapshot the routing decision **and** the link state at one time step."""
    nodes = list(route)
    route_links = list(zip(nodes, nodes[1:]))
    utils = [topology.utilization(u, v) for u, v in route_links]
    network_utils = [topology.utilization(u, v) for u, v in topology.links]
    congested = topology.congested_links(threshold)
    congested_set = {(u, v) for u, v in congested} | {(v, u) for u, v in congested}

    return StepRecord(
        step=step,
        route=tuple(nodes),
        cost=cost,
        condition=condition,
        congested_links=len(congested),
        total_links=topology.link_count,
        route_utilization=_mean(utils),
        network_utilization=_mean(network_utils),
        route_has_congestion=any((u, v) in congested_set for u, v in route_links),
    )


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _throughput(sent: Sequence[Packet], delivered: Sequence[Packet]) -> float:
    """Useful bits delivered divided by the measurement window, in Mbit/s."""
    if not sent or not delivered:
        return 0.0
    start_ms = min(p.sent_at_ms for p in sent)
    end_ms = max(p.delivered_at_ms for p in delivered)
    duration_s = max((end_ms - start_ms) / 1000.0, 1e-6)
    bits = sum(p.size_bytes * 8 for p in delivered)
    return bits / duration_s / 1e6


def _route_changes(steps: Sequence[StepRecord]) -> int:
    """Number of times the selected route differed from the previous step."""
    changes = 0
    previous: tuple[str, ...] | None = None
    for record in steps:
        if previous is not None and record.route != previous:
            changes += 1
        previous = record.route
    return changes


def count_congested_links(topology: Topology, threshold: float = CONGESTION_HIGH_THRESHOLD) -> int:
    return len(topology.congested_links(threshold))
