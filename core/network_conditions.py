"""Real-time network-condition simulator.

The simulator mutates the *live* attributes of links (congestion, latency,
packet loss, bandwidth, traffic) while keeping the ``base_*`` copies intact so
that ``Topology.reset_conditions()`` can always bring the network back to a
healthy state.

Seven condition types are supported – six degradations plus ``NORMAL``:

======================  ====================================================
condition               effect on the affected links
======================  ====================================================
``NORMAL``              restore every link to its base values
``CONGESTION``          queue fills up -> congestion, traffic and latency rise
``TRAFFIC_SPIKE``       offered load suddenly doubles/triples
``HIGH_PACKET_LOSS``    loss probability rises sharply
``HIGH_LATENCY``        propagation/processing delay rises
``LOW_BANDWIDTH``       link capacity is throttled, congestion follows
``MULTI_CONGESTION``    several links degrade at the same time
======================  ====================================================

Every application returns a :class:`ConditionEvent` which the dashboard shows
in its event log and the experiment runner stores alongside its metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from config import CONGESTION_HIGH_THRESHOLD
from core.topology import Topology


class ConditionType(str, Enum):
    """Names of the simulated network events."""

    NORMAL = "Normal conditions"
    CONGESTION = "Link congestion"
    TRAFFIC_SPIKE = "Sudden traffic spike"
    HIGH_PACKET_LOSS = "High packet loss"
    HIGH_LATENCY = "High latency"
    LOW_BANDWIDTH = "Reduced bandwidth"
    MULTI_CONGESTION = "Multiple congested links"


@dataclass(frozen=True)
class ConditionEvent:
    """Record of one applied network condition."""

    condition: ConditionType
    severity: float
    links: tuple[tuple[str, str], ...]
    description: str

    @property
    def link_labels(self) -> str:
        return ", ".join(f"{u}-{v}" for u, v in self.links) or "whole network"


# How many links each condition touches when the caller does not choose them.
DEFAULT_LINK_COUNT: dict[ConditionType, int] = {
    ConditionType.NORMAL: 0,
    ConditionType.CONGESTION: 1,
    ConditionType.TRAFFIC_SPIKE: 1,
    ConditionType.HIGH_PACKET_LOSS: 1,
    ConditionType.HIGH_LATENCY: 1,
    ConditionType.LOW_BANDWIDTH: 1,
    ConditionType.MULTI_CONGESTION: 3,
}

# Which live attributes each condition is allowed to modify.
_AFFECTED_ATTRIBUTES: dict[ConditionType, tuple[str, ...]] = {
    ConditionType.NORMAL: (),
    ConditionType.CONGESTION: ("congestion", "traffic", "latency", "packet_loss"),
    ConditionType.TRAFFIC_SPIKE: ("traffic", "congestion", "latency"),
    ConditionType.HIGH_PACKET_LOSS: ("packet_loss",),
    ConditionType.HIGH_LATENCY: ("latency",),
    ConditionType.LOW_BANDWIDTH: ("bandwidth", "congestion", "traffic"),
    ConditionType.MULTI_CONGESTION: ("congestion", "traffic", "latency", "packet_loss"),
}

# Human-readable explanation shown on the dashboard for each condition.
_DESCRIPTIONS: dict[ConditionType, str] = {
    ConditionType.NORMAL: "All links restored to healthy baseline values.",
    ConditionType.CONGESTION: "Queue on the link filled up: congestion, load and delay increased.",
    ConditionType.TRAFFIC_SPIKE: "Offered traffic suddenly jumped, pushing the link towards saturation.",
    ConditionType.HIGH_PACKET_LOSS: "Loss probability rose (errors / buffer overflows), delivery rate drops.",
    ConditionType.HIGH_LATENCY: "Propagation and processing delay on the link increased.",
    ConditionType.LOW_BANDWIDTH: "Link capacity throttled, so the same traffic now causes congestion.",
    ConditionType.MULTI_CONGESTION: "Several links degraded at the same time.",
}

MIN_BANDWIDTH_MBPS = 50.0
MAX_CONGESTION = 0.97
MAX_PACKET_LOSS = 0.50


class ConditionSimulator:
    """Applies and restores simulated conditions on a :class:`Topology`."""

    def __init__(self, topology: Topology) -> None:
        self.topology = topology
        self.history: list[ConditionEvent] = []

    # -- helpers -----------------------------------------------------------
    def choose_links(
        self,
        condition: ConditionType,
        prefer_path: Sequence[str] | None = None,
        count: int | None = None,
    ) -> list[tuple[str, str]]:
        """Pick the links a condition should hit.

        If ``prefer_path`` is given (e.g. *the current shortest path*) every
        link of that path is targeted – this is how the experiments place
        congestion exactly on the route a naive protocol would keep using.
        Passing ``count`` limits how many links are touched.
        """
        if count is None:
            if prefer_path:
                wanted = len(self.topology.link_on_path(list(prefer_path)))
            else:
                wanted = DEFAULT_LINK_COUNT[condition]
        else:
            wanted = count
        if wanted <= 0:
            return []

        chosen: list[tuple[str, str]] = []

        if prefer_path:
            chosen = self.topology.link_on_path(list(prefer_path))[:wanted]
            if len(chosen) == wanted:
                return chosen

        # Deterministic fallback: busiest links first (no random seed needed).
        ranked = sorted(
            self.topology.links,
            key=lambda e: (-self.topology.get(e[0], e[1], "traffic"), e[0], e[1]),
        )
        remaining = wanted - len(chosen)
        for link in ranked:
            if remaining == 0:
                break
            if link not in chosen:
                chosen.append(link)
                remaining -= 1
        return chosen

    # -- main API ----------------------------------------------------------
    def apply(
        self,
        condition: ConditionType,
        links: Iterable[tuple[str, str]] | None = None,
        prefer_path: Sequence[str] | None = None,
        severity: float = 1.0,
        count: int | None = None,
    ) -> ConditionEvent:
        """Apply ``condition`` and return the resulting :class:`ConditionEvent`."""
        if not 0.0 <= severity <= 1.0:
            raise ValueError("severity must be inside [0, 1]")

        if condition is ConditionType.NORMAL:
            self.restore_normal()
            event = ConditionEvent(condition, 0.0, (), _DESCRIPTIONS[condition])
            self.history.append(event)
            return event

        targets = [tuple(e) for e in links] if links is not None else self.choose_links(
            condition, prefer_path=prefer_path, count=count
        )
        for u, v in targets:
            if not self.topology.has_link(u, v):
                raise ValueError(f"Cannot apply {condition}: no link {u}-{v}")
            self._degrade_link(u, v, condition, severity)

        event = ConditionEvent(
            condition=condition,
            severity=severity,
            links=tuple(targets),
            description=_DESCRIPTIONS[condition],
        )
        self.history.append(event)
        return event

    def restore_normal(self) -> None:
        """Bring the whole network back to its healthy baseline."""
        self.topology.reset_conditions()

    # -- degradation rules -------------------------------------------------
    def _degrade_link(self, u: str, v: str, condition: ConditionType, severity: float) -> None:
        """Translate a condition + severity into concrete attribute changes.

        Order matters: capacity is throttled first, then the target queue
        occupancy (``congestion``) is computed, and finally the offered load is
        set to ``congestion * bandwidth`` so that

            utilisation (traffic / bandwidth) == congestion

        which is the first-order definition of congestion used throughout the
        project and by the ML feature extractor.
        """
        affected = _AFFECTED_ATTRIBUTES[condition]
        current = self.topology.link_attributes(u, v)

        if "bandwidth" in affected:
            base = current["base_bandwidth"]
            self.topology.set_link(
                u, v, bandwidth=max(MIN_BANDWIDTH_MBPS, base * (1.0 - 0.70 * severity))
            )

        if "congestion" in affected:
            base = current["base_congestion"]
            if condition in (ConditionType.CONGESTION, ConditionType.MULTI_CONGESTION):
                value = base + severity * 0.75
            elif condition is ConditionType.TRAFFIC_SPIKE:
                value = base + severity * 0.75
            else:  # LOW_BANDWIDTH -> congestion follows the reduced capacity
                value = base + severity * 0.50
            self.topology.set_link(u, v, congestion=min(MAX_CONGESTION, value))

        if "traffic" in affected:
            live_bandwidth = self.topology.get(u, v, "bandwidth")
            live_congestion = self.topology.get(u, v, "congestion")
            self.topology.set_link(
                u, v, traffic=min(live_bandwidth * 0.99, live_congestion * live_bandwidth)
            )

        if "latency" in affected:
            base = current["base_latency"]
            if condition is ConditionType.HIGH_LATENCY:
                factor = 1.0 + 5.0 * severity
            elif condition is ConditionType.TRAFFIC_SPIKE:
                factor = 1.0 + 1.0 * severity
            else:  # congestion / multi-congestion: queueing delay
                factor = 1.0 + 2.0 * severity
            self.topology.set_link(u, v, latency=base * factor)

        if "packet_loss" in affected:
            base = current["base_packet_loss"]
            extra = 0.15 if condition in (ConditionType.CONGESTION, ConditionType.MULTI_CONGESTION) else 0.30
            self.topology.set_link(
                u, v, packet_loss=min(MAX_PACKET_LOSS, base + extra * severity)
            )

        if "bandwidth" in affected:
            base = current["base_bandwidth"]
            self.topology.set_link(
                u, v, bandwidth=max(MIN_BANDWIDTH_MBPS, base * (1.0 - 0.70 * severity))
            )

    # -- reporting ---------------------------------------------------------
    def congested_links(self, threshold: float = CONGESTION_HIGH_THRESHOLD) -> list[tuple[str, str]]:
        return self.topology.congested_links(threshold)

    @property
    def last_event(self) -> ConditionEvent | None:
        return self.history[-1] if self.history else None
