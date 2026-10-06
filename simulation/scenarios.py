"""Predefined experiment scenarios.

A :class:`Scenario` is a *timeline*: at every simulation step one network
condition is applied and every routing strategy then re-computes its route.

When :attr:`StepAction.links` is empty the runner automatically targets the
**current hop-count shortest path**.  That is exactly the behaviour we want to
demonstrate: "congest the route a naive protocol would keep using, and watch
whether the adaptive strategies move away from it".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from config import DEFAULT_SIM_STEPS
from core.network_conditions import ConditionType


@dataclass(frozen=True)
class StepAction:
    """Condition to apply at a specific simulation step."""

    step: int
    condition: ConditionType
    severity: float = 1.0
    # Empty -> the runner targets the current hop-count shortest path.
    links: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Scenario:
    """One complete experiment definition."""

    id: str
    name: str
    description: str
    src: str
    dst: str
    steps: tuple[StepAction, ...]
    expected_outcome: str = ""

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def conditions(self) -> tuple[ConditionType, ...]:
        return tuple(s.condition for s in self.steps)


def build_timeline(
    condition: ConditionType,
    severity: float = 1.0,
    links: Sequence[tuple[str, str]] = (),
    normal_steps: int = 1,
    total_steps: int = DEFAULT_SIM_STEPS,
) -> tuple[StepAction, ...]:
    """``normal_steps`` healthy steps followed by ``condition`` until the end."""
    if normal_steps < 0 or total_steps <= 0 or normal_steps >= total_steps:
        raise ValueError("normal_steps must be >= 0 and < total_steps")
    actions = [
        StepAction(step=i, condition=ConditionType.NORMAL) for i in range(normal_steps)
    ]
    actions += [
        StepAction(step=i, condition=condition, severity=severity, links=tuple(links))
        for i in range(normal_steps, total_steps)
    ]
    return tuple(actions)


# ---------------------------------------------------------------------------
# The six experiments of the project
# ---------------------------------------------------------------------------

_EXPERIMENT_LINKS_6 = (
    ("S", "A"), ("A", "C"), ("C", "T"),   # the hop-count shortest path
    ("S", "B"), ("B", "D"), ("D", "F"), ("F", "T"),  # its main alternative
)

# Experiment 5 deliberately stresses a *single* link: the spike is not yet bad
# enough for the weighted cost formula to move, but the ML model already
# predicts a "Warning" state and pushes the traffic away one step earlier.
_EXPERIMENT_LINKS_5 = (("S", "A"),)

SCENARIOS: dict[str, Scenario] = {}


def _register(scenario: Scenario) -> Scenario:
    SCENARIOS[scenario.id] = scenario
    return scenario


_register(Scenario(
    id="exp1_normal",
    name="Experiment 1 - Normal network conditions",
    description=(
        "All links run at their healthy baseline values. Every strategy should "
        "find the same 3-hop route and packet delivery should be excellent."
    ),
    src="S", dst="T",
    steps=tuple(StepAction(step=i, condition=ConditionType.NORMAL)
                for i in range(DEFAULT_SIM_STEPS)),
    expected_outcome=(
        "All three strategies select S -> A -> C -> T; delivery rate is high "
        "and latency is low, so adaptive routing has nothing to gain yet."
    ),
))

_register(Scenario(
    id="exp2_congestion",
    name="Experiment 2 - Congestion on the shortest path",
    description=(
        "From step 1 onwards the backbone links S-A, A-C and C-T become "
        "congested (queue fills up, latency and loss rise). The hop-count "
        "route does not react; adaptive routing should switch to the "
        "southern alternative S -> B -> D -> F -> T."
    ),
    src="S", dst="T",
    steps=build_timeline(ConditionType.CONGESTION),
    expected_outcome=(
        "Shortest Path stays on S-A-C-T and suffers high delay/loss, while "
        "Adaptive Routing switches to a 4-hop route with much better metrics."
    ),
))

_register(Scenario(
    id="exp3_packet_loss",
    name="Experiment 3 - High packet loss on one route",
    description=(
        "The backbone suffers from a sharp rise in packet-loss probability "
        "(buffer overflow / bit errors). Retransmissions keep some packets "
        "alive but end-to-end delay grows."
    ),
    src="S", dst="T",
    steps=build_timeline(ConditionType.HIGH_PACKET_LOSS),
    expected_outcome=(
        "Adaptive routing leaves the lossy route, raising the delivery rate "
        "and lowering the average latency compared with Shortest Path."
    ),
))

_register(Scenario(
    id="exp4_latency",
    name="Experiment 4 - High latency on one route",
    description=(
        "Propagation/processing delay on the backbone increases several "
        "times over while loss stays low."
    ),
    src="S", dst="T",
    steps=build_timeline(ConditionType.HIGH_LATENCY),
    expected_outcome=(
        "Adaptive routing selects the longer-hop but faster route, showing "
        "that hop count is a poor proxy for real delay."
    ),
))

_register(Scenario(
    id="exp5_traffic_spike",
    name="Experiment 5 - Sudden traffic spike",
    description=(
        "A traffic burst hits a single backbone link (S-A). Its utilisation "
        "jumps towards saturation while the other links are still healthy, so "
        "the weighted cost formula alone does not yet consider the route bad "
        "enough to abandon."
    ),
    src="S", dst="T",
    steps=build_timeline(ConditionType.TRAFFIC_SPIKE, links=_EXPERIMENT_LINKS_5),
    expected_outcome=(
        "Shortest Path and plain Adaptive Routing both stay on the backbone, "
        "whereas Adaptive + ML already predicts a Warning state on S-A, adds a "
        "routing penalty and moves the traffic one step earlier - showing the "
        "early-warning value of the ML module."
    ),
))

_register(Scenario(
    id="exp6_multi_congestion",
    name="Experiment 6 - Multiple congested links",
    description=(
        "Seven links degrade at the same time - both the backbone and its "
        "main alternative. No perfect path remains, so the router must pick "
        "the least-bad option."
    ),
    src="S", dst="T",
    steps=build_timeline(ConditionType.MULTI_CONGESTION, links=_EXPERIMENT_LINKS_6),
    expected_outcome=(
        "The system still finds a route, but every strategy now pays a higher "
        "price: this shows the limits of routing when the whole network is "
        "under stress."
    ),
))


def get_scenario(scenario_id: str) -> Scenario:
    if scenario_id not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{scenario_id}'. Available: {list(SCENARIOS)}")
    return SCENARIOS[scenario_id]


def scenario_ids() -> list[str]:
    return list(SCENARIOS)
