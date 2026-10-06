"""Tests for the dashboard theme module.

The CSS is injected into Streamlit, so a typo would not raise - it would just
leave the app unstyled or, worse, break a page. These tests pin the contract:
every theme builds a complete stylesheet, the decorative layers are present,
and switching themes never breaks page rendering.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from visualization.theme import (  # noqa: E402
    DEFAULT_THEME,
    THEMES,
    THEME_NAMES,
    build_css,
)

PLACEHOLDERS = ("@theme-vars@", "@noise-uri@", "@speed@")


# ---------------------------------------------------------------------------
# Stylesheet construction
# ---------------------------------------------------------------------------

def test_theme_names_match_the_palette_table() -> None:
    assert set(THEME_NAMES) == set(THEMES)
    assert DEFAULT_THEME in THEMES


@pytest.mark.parametrize("name", sorted(THEMES))
def test_every_theme_builds_complete_css(name: str) -> None:
    css = build_css(name)
    for placeholder in PLACEHOLDERS:
        assert placeholder not in css, f"{placeholder} was not substituted in {name}"
    # The pieces the design depends on.
    for fragment in ("arx-bg", "arx-aurora", "arx-orb", "arx-grid",
                     "arx-noise", "backdrop-filter", "@keyframes",
                     "prefers-reduced-motion", "--arx-speed"):
        assert fragment in css, f"{name} is missing '{fragment}'"
    assert len(css) > 5000, f"{name} stylesheet looks truncated ({len(css)} chars)"


@pytest.mark.parametrize("name", sorted(THEMES))
def test_css_braces_are_balanced(name: str) -> None:
    """An unbalanced brace silently kills every rule after the typo."""
    css = build_css(name)
    # Ignore braces inside the inlined SVG data URI (there are none) and in
    # strings; a plain count is enough for a stylesheet we control.
    assert css.count("{") == css.count("}"), f"{name} has unbalanced braces"


@pytest.mark.parametrize("name", sorted(THEMES))
def test_theme_declares_every_css_variable_it_uses(name: str) -> None:
    css = build_css(name)
    declared = {
        line.split(":")[0].strip()
        for line in css.splitlines()
        if line.strip().startswith("--") and ":" in line
    }
    used = set(_css_variable_uses(css))
    missing = used - declared
    assert not missing, f"{name} uses undeclared variables: {sorted(missing)}"


def _css_variable_uses(css: str) -> list[str]:
    import re

    return re.findall(r"var\(\s*(--[a-z0-9-]+)", css)


def test_themes_are_visually_distinct() -> None:
    accents = {name: theme["accent"] for name, theme in THEMES.items()}
    assert len(set(accents.values())) == len(accents), "two themes share an accent colour"


def test_minimal_theme_disables_motion() -> None:
    minimal = THEMES["Minimal"]
    assert minimal["speed"] == "0"
    assert minimal["noise"] == "0"
    animated = THEMES["Aurora"]
    assert animated["speed"] == "1"


def test_all_themes_use_light_heading_text() -> None:
    """A dark gradient end would make the H1 unreadable on a dark background."""

    def luminance(hex_colour: str) -> float:
        r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255

    for name, theme in THEMES.items():
        for key in ("text", "accent", "grad"):
            assert luminance(theme[key]) > 0.55, f"{name}.{key} is too dark for a heading"


def test_background_layers_markup_has_no_placeholders() -> None:
    from visualization.theme import _BACKGROUND_LAYERS

    assert _BACKGROUND_LAYERS.startswith('<div class="arx-bg"')
    assert _BACKGROUND_LAYERS.count("<div") == _BACKGROUND_LAYERS.count("</div>")
    for placeholder in PLACEHOLDERS:
        assert placeholder not in _BACKGROUND_LAYERS


# ---------------------------------------------------------------------------
# Integration: switching theme must not break any page
# ---------------------------------------------------------------------------

def _open_theme(app: AppTest, name: str) -> AppTest:
    box = next(s for s in app.selectbox if s.label == "Theme")
    box.set_value(name)
    app.run()
    assert not app.exception, f"theme '{name}' raised: {app.exception}"
    assert [s for s in app.selectbox if s.label == "Theme"][0].value == name
    return app


@pytest.mark.parametrize("name", sorted(THEMES))
def test_app_renders_with_every_theme(name: str) -> None:
    app = AppTest.from_file("app.py", default_timeout=120)
    app.run()
    assert not app.exception
    _open_theme(app, name)
    assert app.title, f"no title rendered with theme '{name}'"


def test_theme_survives_navigation() -> None:
    app = AppTest.from_file("app.py", default_timeout=120)
    app.run()
    _open_theme(app, "Sunset")
    for page in ("Network Topology", "ML Prediction", "Overview"):
        nav = next(r for r in app.radio if r.label == "Navigation")
        nav.set_value(page)
        app.run()
        assert not app.exception, f"'{page}' failed with the Sunset theme"
