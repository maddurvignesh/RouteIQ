"""Headless integration tests for the Streamlit dashboard.

``streamlit.testing.v1.AppTest`` executes the real ``app.py`` without opening a
browser, so these tests catch broken imports, bad widget state and exceptions
in any of the six pages.

Each test gets a **fresh** app instance because Streamlit pages are stateful:
the sidebar navigation and the page bodies must be re-read from scratch or the
widgets of a previously visited page leak into the next assertion.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = [
    "Overview",
    "Network Topology",
    "Adaptive Routing",
    "Network Simulation",
    "ML Prediction",
    "Performance Comparison",
]


def make_app() -> AppTest:
    application = AppTest.from_file("app.py", default_timeout=120)
    application.run()
    assert not application.exception, f"app.py raised: {application.exception}"
    return application


def open_page(app: AppTest, page: str) -> AppTest:
    """Select ``page`` in the sidebar navigation and re-run the script."""
    nav = next(r for r in app.radio if r.label == "Navigation")
    nav.set_value(page)
    app.run()
    assert not app.exception, f"page '{page}' raised: {app.exception}"
    assert [r for r in app.radio if r.label == "Navigation"][0].value == page
    return app


def tables(app: AppTest) -> list[pd.DataFrame]:
    return [t.value for t in app.dataframe if isinstance(t.value, pd.DataFrame)]


def test_app_starts_without_errors() -> None:
    app = make_app()
    assert app.title, "the app should render a title"
    nav = next(r for r in app.radio if r.label == "Navigation")
    assert nav.options == list(PAGES)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page: str) -> None:
    app = open_page(make_app(), page)
    assert app.title, f"page '{page}' produced no title"


def test_overview_shows_key_status() -> None:
    app = open_page(make_app(), "Overview")
    labels = [m.label for m in app.metric]
    for expected in ("Routers", "Links", "Source", "Destination",
                     "Congested links", "Route changes"):
        assert expected in labels, f"missing metric '{expected}' on the overview, got {labels}"


def test_topology_page_shows_link_table() -> None:
    app = open_page(make_app(), "Network Topology")
    assert tables(app), "the topology page should render a link table"
    link_table = tables(app)[0]
    assert len(link_table) == 13, "the demo topology has 13 links"


def test_adaptive_page_lists_alternatives() -> None:
    app = open_page(make_app(), "Adaptive Routing")
    rendered = tables(app)
    assert rendered, "the adaptive page should render at least one table"
    combined = " ".join(" ".join(map(str, frame.values.ravel())) for frame in rendered)
    # The selected route is rendered as code, the alternatives as a table.
    selected = [c.value for c in app.code]
    assert "S -> A -> C -> T" in selected, f"selected route missing, got {selected}"
    assert "S -> B -> D -> F -> T" in combined, "the main alternate route should be listed"
    # Every row of the cost breakdown must be numeric-friendly and unique per link.
    breakdown = next(f for f in rendered if "Link" in f.columns)
    assert len(breakdown) == 13


def test_simulation_page_has_condition_buttons() -> None:
    app = open_page(make_app(), "Network Simulation")
    labels = [b.label for b in app.button]
    for label in ("Increase congestion", "Increase traffic", "Increase packet loss",
                  "Increase latency", "Reduce bandwidth", "Restore normal conditions"):
        assert label in labels, f"missing button '{label}', got {labels}"


def test_simulation_page_switches_route_after_congestion() -> None:
    app = open_page(make_app(), "Network Simulation")
    target = next(b for b in app.button if b.label == "Increase congestion")
    target.click()
    app.run()
    assert not app.exception

    history = next((f for f in tables(app) if "route" in f.columns), None)
    assert history is not None, "the step-by-step routing table should be rendered"
    assert len(set(history["route"])) >= 2, "the adaptive router should have changed route"


def test_ml_page_shows_model_metrics() -> None:
    app = open_page(make_app(), "ML Prediction")
    labels = [m.label for m in app.metric]
    assert "Accuracy" in labels, f"missing Accuracy metric, got {labels}"
    assert "Samples" in labels, f"missing Samples metric, got {labels}"


def test_performance_page_shows_summary_table() -> None:
    app = open_page(make_app(), "Performance Comparison")
    rendered = tables(app)
    assert rendered, "the performance page should render a summary table"
    combined = " ".join(" ".join(map(str, frame.values.ravel())) for frame in rendered)
    assert "Shortest Path" in combined
    assert "Adaptive Routing" in combined
