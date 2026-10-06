"""Tests for the high-level :class:`SimulationEngine` facade."""

from __future__ import annotations

import pytest

from config import (
    STRATEGY_ADAPTIVE,
    STRATEGY_ADAPTIVE_ML,
    STRATEGY_SHORTEST,
)
from core.network_conditions import ConditionType
from simulation.engine import SimulationEngine


@pytest.fixture()
def engine() -> SimulationEngine:
    return SimulationEngine()


def test_initial_state_is_healthy(engine: SimulationEngine) -> None:
    status = engine.status()
    assert status["routers"] == 8
    assert status["links"] == 13
    assert status["route"] == "S -> A -> C -> T"
    assert status["overall_congestion"] == "Normal"
    assert status["packets_sent"] > 0
    assert status["route_changes"] == 0


def test_applying_congestion_switches_the_route(engine: SimulationEngine) -> None:
    engine.apply_condition(ConditionType.CONGESTION, severity=1.0)
    status = engine.status()
    assert status["route"] == "S -> B -> D -> F -> T"
    assert status["route_changes"] == 1
    assert status["congested_links"] == 3


def test_restore_normal_switches_back(engine: SimulationEngine) -> None:
    engine.apply_condition(ConditionType.CONGESTION, severity=1.0)
    engine.restore_normal()
    assert engine.current_route().as_text == "S -> A -> C -> T"
    assert engine.status()["route_changes"] == 2


def test_shortest_path_strategy_never_moves(engine: SimulationEngine) -> None:
    engine.set_strategy(STRATEGY_SHORTEST)
    engine.apply_condition(ConditionType.CONGESTION, severity=1.0)
    assert engine.status()["route"] == "S -> A -> C -> T"
    assert engine.status()["route_changes"] == 0


def test_strategy_comparison_uses_live_conditions(engine: SimulationEngine) -> None:
    engine.apply_condition(ConditionType.CONGESTION, severity=1.0)
    comparison = engine.strategy_comparison()
    assert set(comparison) == {STRATEGY_SHORTEST, STRATEGY_ADAPTIVE, STRATEGY_ADAPTIVE_ML}
    assert comparison[STRATEGY_SHORTEST].path == ("S", "A", "C", "T")
    assert comparison[STRATEGY_ADAPTIVE].path != comparison[STRATEGY_SHORTEST].path


def test_changing_endpoints(engine: SimulationEngine) -> None:
    engine.set_endpoints("S", "E")
    status = engine.status()
    assert status["source"] == "S" and status["destination"] == "E"
    assert engine.route.path[0] == "S" and engine.route.path[-1] == "E"


def test_same_endpoint_rejected(engine: SimulationEngine) -> None:
    with pytest.raises(ValueError):
        engine.set_endpoints("S", "S")


def test_unknown_strategy_rejected(engine: SimulationEngine) -> None:
    with pytest.raises(ValueError):
        engine.set_strategy("OSPFv4")


def test_history_frame_records_every_step(engine: SimulationEngine) -> None:
    engine.apply_condition(ConditionType.HIGH_LATENCY, severity=0.8)
    frame = engine.history_frame()
    assert len(frame) >= 2
    assert {"step", "route", "cost", "condition", "route_congested"} <= set(frame.columns)


def test_metrics_accumulate_over_all_steps(engine: SimulationEngine) -> None:
    engine.apply_condition(ConditionType.CONGESTION, severity=1.0)
    metrics = engine.metrics()
    assert metrics.packets_sent == 80 * 2
    assert 0.0 <= metrics.delivery_rate_pct <= 100.0
    assert metrics.route_changes == 1


def test_ml_can_be_disabled_at_runtime() -> None:
    engine = SimulationEngine(use_ml=True)
    assert engine.ml_enabled is True
    engine.set_ml_enabled(False)
    assert engine.ml_enabled is False
    # routing must still work with the ML module switched off
    assert engine.current_route().path


def test_link_frame_has_one_row_per_link(engine: SimulationEngine) -> None:
    frame = engine.link_frame()
    assert len(frame) == engine.topology.link_count
    assert {"source", "target", "latency", "congestion", "utilization"} <= set(frame.columns)


def test_ml_predictions_cover_all_links(engine: SimulationEngine) -> None:
    predictions = engine.ml_predictions()
    assert len(predictions) == engine.topology.link_count
    assert all(p["state"] in {"Normal", "Warning", "Congested"} for p in predictions)


def test_reset_clears_history_and_packets() -> None:
    engine = SimulationEngine()
    engine.apply_condition(ConditionType.CONGESTION, severity=1.0)
    engine.reset()
    assert len(engine.history) == 1
    assert engine.status()["route_changes"] == 0
    assert engine.status()["congested_links"] == 0


def test_event_log_records_actions(engine: SimulationEngine) -> None:
    engine.apply_condition(ConditionType.TRAFFIC_SPIKE, severity=1.0)
    engine.restore_normal()
    assert any("traffic spike" in e.lower() for e in engine.events)
    assert any("restored" in e.lower() for e in engine.events)


def test_condition_targets_shortest_path_by_default(engine: SimulationEngine) -> None:
    engine.apply_condition(ConditionType.CONGESTION, severity=1.0, target_shortest_path=True)
    for link in (("S", "A"), ("A", "C"), ("C", "T")):
        assert engine.topology.is_congested(*link)


def test_explicit_links_are_respected(engine: SimulationEngine) -> None:
    engine.apply_condition(
        ConditionType.HIGH_PACKET_LOSS,
        severity=1.0,
        links=[("D", "F")],
        target_shortest_path=True,
    )
    assert engine.topology.get("D", "F", "packet_loss") > 0.01
    assert not engine.topology.is_congested("S", "A")
