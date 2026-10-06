"""Simulated packet transmission over a chosen route.

This is **not** a TCP/IP stack re-implementation.  It is a lightweight, fully
deterministic model that reproduces the three phenomena a routing protocol
cares about:

1. **Transmission delay** – every hop spends ``size / bandwidth`` milliseconds
   pushing the bits onto the wire (serialization delay).
2. **Queueing delay** – a congested link makes packets wait in a buffer.  The
   model uses the classic non-linear queueing curve

   .. math::

       d_{queue} = k \\cdot \\frac{c}{1 - c}

   where ``c`` is the link congestion in ``[0, 1)`` and ``k`` is
   ``queue_scale_ms``.  Delay explodes as the link approaches saturation,
   exactly like a real router buffer.
3. **Packet loss** – a link drops a packet with probability

   .. math::

       p_{drop} = 1 - (1 - loss_{link}) \\cdot reliability_{link}

   which combines protocol-level loss (buffer overflow / bit errors) with the
   probability that the link hardware is down.

After a drop the packet is retransmitted up to ``max_retries`` times, waiting
``retransmission_timeout_ms`` between attempts – the same stop-and-wait style
behaviour used to explain reliability in Computer Networks courses.

Every packet is stored as a :class:`Packet` record so the dashboard and the
metric module can inspect exactly what happened.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from config import DEFAULT_PACKET_CONFIG, PacketConfig
from core.topology import Topology

STATUS_DELIVERED = "Delivered"
STATUS_DROPPED = "Dropped"


@dataclass
class Packet:
    """Record of a single simulated packet."""

    packet_id: int
    source: str
    destination: str
    route: tuple[str, ...]
    size_bytes: int
    status: str
    delay_ms: float
    sent_at_ms: float
    attempts: int
    dropped_at: str | None = None
    strategy: str = ""
    step: int = 0
    delivered_at_ms: float = 0.0

    @property
    def delivered(self) -> bool:
        return self.status == STATUS_DELIVERED

    @property
    def hops(self) -> int:
        return max(len(self.route) - 1, 0)

    @property
    def retransmissions(self) -> int:
        return self.attempts - 1

    @property
    def route_text(self) -> str:
        return " -> ".join(self.route)


@dataclass
class TransmissionReport:
    """Aggregate outcome of one ``send()`` call."""

    packets: list[Packet] = field(default_factory=list)
    strategy: str = ""
    step: int = 0

    @property
    def sent(self) -> int:
        return len(self.packets)

    @property
    def delivered(self) -> int:
        return sum(1 for p in self.packets if p.delivered)

    @property
    def dropped(self) -> int:
        return self.sent - self.delivered


class PacketSimulator:
    """Transmits packets along a route using the current link conditions."""

    def __init__(
        self,
        topology: Topology,
        config: PacketConfig = DEFAULT_PACKET_CONFIG,
        seed: int = 42,
    ) -> None:
        self.topology = topology
        self.config = config
        self.seed = seed

    # -- delay model -------------------------------------------------------
    def hop_delay_ms(self, u: str, v: str, size_bytes: int, rng: random.Random) -> float:
        """Delay introduced by one hop: latency + serialization + queueing."""
        attrs = self.topology.link_attributes(u, v)

        serialization_ms = (size_bytes * 8) / (max(attrs["bandwidth"], 1.0) * 1000.0)

        congestion = min(attrs["congestion"], self.config.max_congestion_for_delay)
        congestion = max(congestion, 0.0)
        queue_ms = self.config.queue_scale_ms * congestion / max(1.0 - congestion, 1e-6)

        jitter = 1.0 + rng.uniform(-self.config.jitter_fraction, self.config.jitter_fraction)
        return max((attrs["latency"] + serialization_ms + queue_ms) * jitter, 0.0)

    def hop_success_probability(self, u: str, v: str) -> float:
        """Probability that a packet survives one hop."""
        attrs = self.topology.link_attributes(u, v)
        return (1.0 - attrs["packet_loss"]) * attrs["reliability"]

    # -- transmission ------------------------------------------------------
    def send(
        self,
        src: str,
        dst: str,
        route: Sequence[str],
        num_packets: int | None = None,
        strategy: str = "",
        step: int = 0,
        start_time_ms: float = 0.0,
    ) -> TransmissionReport:
        """Send ``num_packets`` from ``src`` to ``dst`` along ``route``."""
        nodes = list(route)
        if len(nodes) < 2:
            raise ValueError("A route needs at least two routers")
        if nodes[0] != src or nodes[-1] != dst:
            raise ValueError(f"Route {' -> '.join(nodes)} does not run {src} -> {dst}")
        for u, v in zip(nodes, nodes[1:]):
            if not self.topology.has_link(u, v):
                raise ValueError(f"Route contains non-existent link {u}-{v}")

        count = num_packets if num_packets is not None else self.config.num_packets
        size_bytes = max(int(self.config.packet_size_kb * 1024), 1)
        rng = random.Random(self.seed + step * 7919)
        links = list(zip(nodes, nodes[1:]))

        packets: list[Packet] = []
        for index in range(count):
            sent_at = start_time_ms + index * self.config.inter_arrival_ms
            packet = self._transmit_one(
                packet_id=index + 1,
                src=src,
                dst=dst,
                route=tuple(nodes),
                links=links,
                size_bytes=size_bytes,
                sent_at_ms=sent_at,
                rng=rng,
                strategy=strategy,
                step=step,
            )
            packets.append(packet)

        return TransmissionReport(packets=packets, strategy=strategy, step=step)

    def _transmit_one(
        self,
        packet_id: int,
        src: str,
        dst: str,
        route: tuple[str, ...],
        links: list[tuple[str, str]],
        size_bytes: int,
        sent_at_ms: float,
        rng: random.Random,
        strategy: str,
        step: int,
    ) -> Packet:
        """Attempt delivery, retransmitting after every drop."""
        total_delay = 0.0
        attempts = 0
        max_attempts = self.config.max_retries + 1
        last_dropped_at: str | None = None

        while attempts < max_attempts:
            attempts += 1
            attempt_delay = 0.0
            lost_on: str | None = None

            for u, v in links:
                attempt_delay += self.hop_delay_ms(u, v, size_bytes, rng)
                if rng.random() > self.hop_success_probability(u, v):
                    lost_on = f"{u}-{v}"
                    break

            total_delay += attempt_delay
            if lost_on is None:
                return Packet(
                    packet_id=packet_id,
                    source=src,
                    destination=dst,
                    route=route,
                    size_bytes=size_bytes,
                    status=STATUS_DELIVERED,
                    delay_ms=total_delay,
                    sent_at_ms=sent_at_ms,
                    delivered_at_ms=sent_at_ms + total_delay,
                    attempts=attempts,
                    strategy=strategy,
                    step=step,
                )

            last_dropped_at = lost_on
            if attempts < max_attempts:
                total_delay += self.config.retransmission_timeout_ms

        return Packet(
            packet_id=packet_id,
            source=src,
            destination=dst,
            route=route,
            size_bytes=size_bytes,
            status=STATUS_DROPPED,
            delay_ms=total_delay,
            sent_at_ms=sent_at_ms,
            delivered_at_ms=sent_at_ms + total_delay,
            attempts=attempts,
            dropped_at=last_dropped_at,
            strategy=strategy,
            step=step,
        )


def packets_to_records(packets: Sequence[Packet]) -> list[dict]:
    """Flatten packets into dictionaries (pandas / CSV friendly)."""
    return [
        {
            "packet_id": p.packet_id,
            "source": p.source,
            "destination": p.destination,
            "route": p.route_text,
            "status": p.status,
            "delay_ms": round(p.delay_ms, 3),
            "sent_at_ms": round(p.sent_at_ms, 3),
            "hops": p.hops,
            "attempts": p.attempts,
            "retransmissions": p.retransmissions,
            "dropped_at": p.dropped_at or "",
            "size_bytes": p.size_bytes,
            "strategy": p.strategy,
            "step": p.step,
        }
        for p in packets
    ]
