"""Integration tests: conditions -> routing -> packet simulation -> metrics."""

from __future__ import annotations

import pytest

from config import (
    STRATEGY_ADAPTIVE,
    STRATEGY_ADAPTIVE_ML,
    STRATEGY_SHORTEST,
)
from simulation.experiments import (
    load_results,
    load_summary,
    run_all_experiments,
    run_experiment,
    run_strategy_comparison,
    save_results,
    summarise,
)
from simulation.scenarios import build_timeline, get_scenario, scenario_ids


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def test_six_experiments_are_defined() -> None:
    assert len(scenario_ids()) == 6
    assert set(scenario_ids()) == {
        "exp1_normal", "exp2_congestion", "exp3_packet_loss",
        "exp4_latency", "exp5_traffic_spike", "exp6_multi_congestion",
    }


def test_unknown_scenario_raises() -> None:
    with pytest.raises(KeyError):
        get_scenario("exp9_does_not_exist")


def test_timeline_shape() -> None:
    from core.network_conditions import ConditionType

    timeline = build_timeline(ConditionType.CONGESTION, normal_steps=2, total_steps=5)
    assert len(timeline) == 5
    assert [s.step for s in timeline] == [0, 1, 2, 3, 4]
    assert timeline[0].condition is ConditionType.NORMAL
    assert timeline[1].condition is ConditionType.NORMAL
    assert all(s.condition is ConditionType.CONGESTION for s in timeline[2:])


def test_invalid_timeline_rejected() -> None:
    from core.network_conditions import ConditionType

    with pytest.raises(ValueError):
        build_timeline(ConditionType.CONGESTION, normal_steps=5, total_steps=5)


def test_normal_scenario_stays_normal() -> None:
    scenario = get_scenario("exp1_normal")
    assert all(c.value == "Normal conditions" for c in scenario.conditions)


# ---------------------------------------------------------------------------
# Single experiment runs
# ---------------------------------------------------------------------------

def test_normal_experiment_all_strategies_agree() -> None:
    results = run_strategy_comparison("exp1_normal")
    paths = {r.metrics.routes[0] for r in results.values()}
    assert len(paths) == 1
    for result in results.values():
        assert result.metrics.delivery_rate_pct >= 95.0
        assert result.metrics.route_changes == 0


def test_congestion_experiment_shortest_path_does_not_move() -> None:
    scenario = get_scenario("exp2_congestion")
    result = run_experiment(scenario, STRATEGY_SHORTEST)
    assert result.metrics.route_changes == 0
    assert result.metrics.distinct_routes == 1
    assert result.metrics.routes[0].startswith("S -> A -> C -> T")


def test_congestion_experiment_adaptive_switches_route() -> None:
    scenario = get_scenario("exp2_congestion")
    result = run_experiment(scenario, STRATEGY_ADAPTIVE)
    assert result.metrics.route_changes >= 1
    assert result.metrics.distinct_routes >= 2
    # step 0 is healthy -> the backbone is still the cheapest route
    assert result.metrics.routes[0].startswith("S -> A -> C -> T")
    # from step 1 onwards the router has moved away from the congested backbone
    assert not result.metrics.routes[-1].startswith("S -> A -> C -> T")
    assert result.metrics.routes[-1].startswith("S -> B -> D -> F -> T")


def test_adaptive_beats_shortest_path_under_congestion() -> None:
    results = run_strategy_comparison("exp2_congestion")
    shortest = results[STRATEGY_SHORTEST].metrics
    adaptive = results[STRATEGY_ADAPTIVE].metrics

    assert adaptive.avg_latency_ms < shortest.avg_latency_ms
    assert adaptive.packet_loss_pct < shortest.packet_loss_pct
    assert adaptive.throughput_mbps > shortest.throughput_mbps
    assert adaptive.delivery_rate_pct >= shortest.delivery_rate_pct


def test_every_experiment_produces_metrics() -> None:
    for scenario_id in scenario_ids():
        scenario = get_scenario(scenario_id)
        result = run_experiment(scenario, STRATEGY_ADAPTIVE)
        m = result.metrics
        assert m.packets_sent > 0
        assert 0.0 <= m.delivery_rate_pct <= 100.0
        assert 0.0 <= m.packet_loss_pct <= 100.0
        assert m.avg_latency_ms >= 0.0
        assert m.throughput_mbps >= 0.0
        assert len(result.steps) == scenario.step_count
        assert len(result.packets) == m.packets_sent


def test_ml_strategy_works_without_a_model() -> None:
    """Adaptive + ML must degrade gracefully when no penalty function exists."""
    scenario = get_scenario("exp2_congestion")
    with_ml = run_experiment(scenario, STRATEGY_ADAPTIVE_ML)
    without = run_experiment(scenario, STRATEGY_ADAPTIVE)
    assert with_ml.metrics.routes == without.metrics.routes
    assert with_ml.metrics.avg_latency_ms == pytest.approx(without.metrics.avg_latency_ms)


def test_runs_are_reproducible() -> None:
    scenario = get_scenario("exp3_packet_loss")
    first = run_experiment(scenario, STRATEGY_SHORTEST, seed=99)
    second = run_experiment(scenario, STRATEGY_SHORTEST, seed=99)
    assert first.metrics.avg_latency_ms == pytest.approx(second.metrics.avg_latency_ms)
    assert first.metrics.delivery_rate_pct == pytest.approx(second.metrics.delivery_rate_pct)


def test_experiment_uses_a_fresh_topology() -> None:
    """Running an experiment must not leave conditions on the shared builder."""
    scenario = get_scenario("exp2_congestion")
    run_experiment(scenario, STRATEGY_ADAPTIVE)
    again = run_experiment(scenario, STRATEGY_ADAPTIVE)
    assert again.metrics.delivery_rate_pct == run_experiment(scenario, STRATEGY_ADAPTIVE).metrics.delivery_rate_pct


# ---------------------------------------------------------------------------
# Full suite + persistence
# ---------------------------------------------------------------------------

def test_run_all_experiments_frame(tmp_path) -> None:
    metrics_file = tmp_path / "metrics.csv"
    summary_file = tmp_path / "summary.csv"
    frame = run_all_experiments(save=False)

    assert len(frame) == 6 * 3
    assert set(frame["strategy"]) == {STRATEGY_SHORTEST, STRATEGY_ADAPTIVE, STRATEGY_ADAPTIVE_ML}
    assert set(frame["scenario"]) == set(scenario_ids())

    save_results(frame, metrics_path=metrics_file, summary_path=summary_file)
    assert metrics_file.exists() and summary_file.exists()

    reloaded = load_results(metrics_file)
    assert reloaded is not None and len(reloaded) == len(frame)

    summary = load_summary(summary_file)
    assert summary is not None and len(summary) == 3


def test_summary_averages_over_scenarios() -> None:
    frame = run_all_experiments(save=False)
    summary = summarise(frame)
    assert list(summary["strategy"]) == [
        STRATEGY_SHORTEST, STRATEGY_ADAPTIVE, STRATEGY_ADAPTIVE_ML
    ]
    assert (summary["scenarios"] == 6).all()
    # adaptive routing must be strictly better on latency than shortest path
    assert summary.iloc[1]["avg_latency_ms"] < summary.iloc[0]["avg_latency_ms"]


def test_result_files_exist_after_a_saved_run(tmp_path) -> None:
    # Write into tmp_path on purpose: a test must never overwrite the results
    # that ship in results/, which are produced by SimulationEngine.run_experiments()
    # *with* the ML hook and are the numbers quoted in docs/experiments.md.
    frame = run_all_experiments(save=False)
    metrics_file = tmp_path / "experiment_metrics.csv"
    summary_file = tmp_path / "comparison_summary.csv"
    save_results(frame, metrics_path=metrics_file, summary_path=summary_file)
    assert metrics_file.exists()
    assert summary_file.exists()
    assert len(frame) == 18


def test_ml_penalty_changes_the_traffic_spike_decision() -> None:
    """exp5 is the scenario built to isolate the ML module's contribution."""
    scenario = get_scenario("exp5_traffic_spike")

    def penalty(_u: str, _v: str, attributes: dict) -> float:
        # Pretend the model calls the spiking S-A link "Congested".
        return 0.90 if float(attributes["congestion"]) > 0.5 else 0.0

    plain = run_experiment(scenario, STRATEGY_ADAPTIVE)
    guided = run_experiment(scenario, STRATEGY_ADAPTIVE_ML, penalty_fn=penalty)

    assert plain.steps[0].route == guided.steps[0].route, "both start on the same route"
    assert plain.steps[-1].route == ("S", "A", "C", "T"), "reactive rule should stay put"
    assert guided.steps[-1].route == ("S", "B", "D", "F", "T"), "ML should move away"
    assert guided.metrics.avg_latency_ms < plain.metrics.avg_latency_ms


def test_experiment_explanation_is_available() -> None:
    result = run_experiment(get_scenario("exp2_congestion"), STRATEGY_ADAPTIVE)
    assert isinstance(result.explanation, str)
    assert len(result.explanation) > 20
