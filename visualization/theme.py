"""Visual theme for the dashboard: animated background + glassmorphism styling.

Streamlit ships a single flat colour scheme, so all of the visual personality
is injected as CSS.  ``inject(theme)`` writes a ``<style>`` block once per
script run; the background is built from ``position: fixed`` layers (an aurora
gradient, drifting blurred orbs, a panning grid and a noise overlay) so it
never scrolls with the content and never intercepts clicks.

Four themes are provided:

============  ==============================================================
Aurora        deep navy with violet / cyan / magenta orbs  (default)
Sunset        warm amber and coral on a deep plum base
Arctic        cool whites and ice blue on a near-black base
Minimal       flat, no animation - for projectors and print
============  ==============================================================

If the CSS fails to load for any reason the app still renders perfectly; the
styling is purely additive.
"""

from __future__ import annotations

import os

import streamlit as st

# ---------------------------------------------------------------------------
# Theme palettes
# ---------------------------------------------------------------------------

THEMES: dict[str, dict[str, str]] = {
    "Aurora": {
        "base": "#05070f",
        "base2": "#0a1024",
        "glow1": "#7c3aed",
        "glow2": "#06b6d4",
        "glow3": "#ec4899",
        "glow4": "#2563eb",
        "accent": "#a78bfa",
        "grad": "#22d3ee",
        "text": "#eef2ff",
        "muted": "#9aa6c9",
        "card": "rgba(255,255,255,0.045)",
        "card_border": "rgba(255,255,255,0.10)",
        "grid": "rgba(148,163,216,0.07)",
        "noise": "0.035",
        "speed": "1",
    },
    "Sunset": {
        "base": "#0d0510",
        "base2": "#1c0a1b",
        "glow1": "#f97316",
        "glow2": "#ec4899",
        "glow3": "#f59e0b",
        "glow4": "#be185d",
        "accent": "#fb923c",
        "grad": "#f472b6",
        "text": "#fff1ec",
        "muted": "#d8a9a4",
        "card": "rgba(255,255,255,0.05)",
        "card_border": "rgba(255,190,170,0.14)",
        "grid": "rgba(253,186,116,0.07)",
        "noise": "0.035",
        "speed": "1",
    },
    "Arctic": {
        "base": "#030712",
        "base2": "#0b1220",
        "glow1": "#38bdf8",
        "glow2": "#818cf8",
        "glow3": "#22d3ee",
        "glow4": "#0ea5e9",
        "accent": "#7dd3fc",
        "grad": "#a5b4fc",
        "text": "#e6f4ff",
        "muted": "#93b4cc",
        "card": "rgba(255,255,255,0.05)",
        "card_border": "rgba(186,230,253,0.14)",
        "grid": "rgba(125,211,252,0.08)",
        "noise": "0.03",
        "speed": "1",
    },
    "Minimal": {
        "base": "#0e1117",
        "base2": "#0e1117",
        "glow1": "#1f2937",
        "glow2": "#111827",
        "glow3": "#1f2937",
        "glow4": "#0b0f19",
        "accent": "#d1d5db",
        "grad": "#9ca3af",
        "text": "#e5e7eb",
        "muted": "#9ca3af",
        "card": "rgba(255,255,255,0.03)",
        "card_border": "rgba(255,255,255,0.08)",
        "grid": "rgba(255,255,255,0.03)",
        "noise": "0",
        "speed": "0",
    },
}

THEME_NAMES = list(THEMES)


def _resolve_default() -> str:
    """Theme used on a fresh session.

    Override with the ``ARX_THEME`` environment variable so a deployment (or a
    screenshot script) can start in a given theme without touching the code.
    """
    requested = os.environ.get("ARX_THEME", "").strip()
    return requested if requested in THEMES else "Aurora"


DEFAULT_THEME = _resolve_default()


def _variables(theme: dict[str, str]) -> str:
    return f"""
    :root {{
        --base: {theme['base']};
        --base2: {theme['base2']};
        --glow1: {theme['glow1']};
        --glow2: {theme['glow2']};
        --glow3: {theme['glow3']};
        --glow4: {theme['glow4']};
        --accent: {theme['accent']};
        --grad: {theme['grad']};
        --txt: {theme['text']};
        --muted: {theme['muted']};
        --card: {theme['card']};
        --edge: {theme['card_border']};
        --grid: {theme['grid']};
        --noise: {theme['noise']};
        --speed: {theme['speed']};
    }}"""


# The SVG noise is inlined as a data URI so there is no extra HTTP request.
_NOISE_URI = (
    "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' "
    "width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' "
    "baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E"
    "%3Crect width='160' height='160' filter='url(%23n)'/%3E%3C/svg%3E\")"
)

_CSS = """
@theme-vars@

html { --arx-speed: @speed@; }

/* ------------------------------------------------------------------ *
 * 1. Let the custom background show through Streamlit's own surfaces
 * ------------------------------------------------------------------ */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stMainBlockContainer"], .block-container,
[data-testid="stMain"] {
    background-color: transparent !important;
}
body { background: var(--base); color: var(--txt); }

/* Content sits above the decorative layers. */
[data-testid="stAppViewContainer"] .main,
[data-testid="stMainBlockContainer"],
.block-container {
    position: relative;
    z-index: 2;
}

/* ------------------------------------------------------------------ *
 * 2. The animated background
 * ------------------------------------------------------------------ */
.arx-bg {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
    background:
        radial-gradient(120% 90% at 50% -10%, var(--base2) 0%, var(--base) 62%);
}
/* A soft scrim keeps text readable without flattening the colour. */
.arx-bg::after {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(85% 65% at 50% 34%, rgba(5, 7, 15, 0.05) 0%, rgba(5, 7, 15, 0.48) 100%),
        linear-gradient(180deg, rgba(5, 7, 15, 0.40) 0%, rgba(5, 7, 15, 0.05) 45%,
                               rgba(5, 7, 15, 0.55) 100%);
}
.arx-aurora {
    position: absolute;
    left: 50%;
    top: 55%;
    width: 175vmax;
    height: 175vmax;
    transform: translate(-50%, -50%);
    background: conic-gradient(from 0deg,
        transparent 0deg, var(--glow1) 55deg, transparent 130deg,
        var(--glow2) 190deg, transparent 260deg,
        var(--glow3) 320deg, transparent 360deg);
    filter: blur(90px);
    opacity: 0.24;
    animation: arx-spin 68s linear infinite;
    will-change: transform;
}
.arx-orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(72px);
    opacity: 0.42;
    will-change: transform;
}
.arx-orb.a { width: 46vmax; height: 46vmax; top: -10vmax; left: -8vmax;
             background: var(--glow1); animation: arx-drift-a 34s ease-in-out infinite alternate; }
.arx-orb.b { width: 40vmax; height: 40vmax; bottom: -12vmax; right: -6vmax;
             background: var(--glow2); animation: arx-drift-b 42s ease-in-out infinite alternate; }
.arx-orb.c { width: 32vmax; height: 32vmax; top: 38%; right: 16%;
             background: var(--glow3); opacity: 0.30;
             animation: arx-drift-c 50s ease-in-out infinite alternate; }
.arx-grid {
    position: absolute;
    inset: -50%;
    background-image:
        linear-gradient(to right, var(--grid) 1px, transparent 1px),
        linear-gradient(to bottom, var(--grid) 1px, transparent 1px);
    background-size: 54px 54px;
    animation: arx-pan 26s linear infinite;
    opacity: 0.55;
}
.arx-noise {
    position: absolute;
    inset: 0;
    background-image: @noise-uri@;
    opacity: var(--noise);
    mix-blend-mode: overlay;
}
.arx-vignette {
    position: absolute;
    inset: 0;
    background: radial-gradient(120% 80% at 50% 40%, transparent 45%, rgba(0,0,0,0.55) 100%);
}

/* Animated hairline along the very top of the viewport. */
body::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 2px;
    z-index: 5;
    background: linear-gradient(90deg, transparent, var(--glow2), var(--glow1),
                var(--glow3), var(--glow4), transparent);
    background-size: 300% 100%;
    animation: arx-slide 9s linear infinite;
    opacity: 0.9;
    pointer-events: none;
}

@keyframes arx-spin   { to { transform: translate(-50%, -50%) rotate(360deg); } }
@keyframes arx-pan    { to { background-position: 54px 54px, 54px 54px; } }
@keyframes arx-slide  { to { background-position: 300% 0; } }
@keyframes arx-drift-a {
    to { transform: translate3d(14vmax, 8vmax, 0) scale(1.18); }
}
@keyframes arx-drift-b {
    to { transform: translate3d(-12vmax, -9vmax, 0) scale(1.14); }
}
@keyframes arx-drift-c {
    to { transform: translate3d(-10vmax, 12vmax, 0) scale(0.82); }
}

/* The "Minimal" theme and users who ask for less motion get a static scene. */
html[data-arx-speed="0"] .arx-aurora,
html[data-arx-speed="0"] .arx-orb,
html[data-arx-speed="0"] .arx-grid,
html[data-arx-speed="0"] body::before { animation: none !important; }
html[data-arx-speed="0"] .arx-aurora { opacity: 0.12; }
html[data-arx-speed="0"] .arx-orb { opacity: 0.18; }
html[data-arx-speed="0"] body::before { opacity: 0.35; }

@media (prefers-reduced-motion: reduce) {
    .arx-aurora, .arx-orb, .arx-grid, body::before { animation: none !important; }
}

/* ------------------------------------------------------------------ *
 * 3. Typography
 * ------------------------------------------------------------------ */
h1, h2, h3, h4 {
    color: var(--txt) !important;
    letter-spacing: -0.02em;
    font-weight: 700;
}
h1, h2 {
    background: linear-gradient(96deg, var(--txt) 10%, var(--accent) 55%, var(--grad) 95%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: none;
}
/* Minimal has no colour accents, so keep the heading plain and readable. */
html[data-arx-speed="0"] h1,
html[data-arx-speed="0"] h2 {
    background: none;
    -webkit-text-fill-color: var(--txt);
}
p, li, label, span, div { color: var(--txt); }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
code {
    background: rgba(127, 127, 255, 0.13) !important;
    border: 1px solid var(--edge);
    border-radius: 7px;
    color: var(--accent) !important;
}
pre code { background: rgba(0, 0, 0, 0.30) !important; }
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--edge), transparent);
}

/* ------------------------------------------------------------------ *
 * 4. Glass surfaces
 * ------------------------------------------------------------------ */
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--edge);
    border-radius: 16px;
    padding: 14px 16px 12px;
    backdrop-filter: blur(14px) saturate(140%);
    -webkit-backdrop-filter: blur(14px) saturate(140%);
    box-shadow: 0 8px 26px rgba(0, 0, 0, 0.30);
    transition: transform .28s ease, box-shadow .28s ease, border-color .28s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: var(--accent);
    box-shadow: 0 14px 38px rgba(0, 0, 0, 0.42), 0 0 0 1px var(--accent);
}
[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-weight: 600; }
[data-testid="stMetricValue"] {
    color: var(--txt) !important;
    font-weight: 700;
}
/* Streamlit puts overflow/ellipsis on an inner emotion div; override it so
   long values such as "Adaptive Routing" are never clipped. */
[data-testid="stMetricValue"] > div {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    line-height: 1.2;
}
[data-testid="stMetricDelta"] { color: var(--accent) !important; }

[data-testid="stExpander"] {
    background: var(--card);
    border: 1px solid var(--edge);
    border-radius: 16px;
    overflow: hidden;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--edge);
    background: rgba(6, 9, 20, 0.55);
}
[data-testid="stDataFrame"] * { border-color: var(--edge) !important; }
[data-testid="stDataFrameResizable"] > div { background: rgba(0, 0, 0, 0.16) !important; }
/* Streamlit draws its own scrollbars inside dataframes - keep them subtle. */
[data-testid="stDataFrame"] ::-webkit-scrollbar { width: 8px; height: 8px; }
[data-testid="stDataFrame"] ::-webkit-scrollbar-thumb {
    background: var(--edge) !important;
    background-clip: padding-box !important;
    border: none !important;
}
[data-testid="stDataFrame"] ::-webkit-scrollbar-thumb:hover { background: var(--accent) !important; }

/* ------------------------------------------------------------------ *
 * 5. Sidebar
 * ------------------------------------------------------------------ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10, 12, 26, 0.90), rgba(5, 7, 15, 0.94));
    border-right: 1px solid var(--edge);
    backdrop-filter: blur(20px) saturate(150%);
    -webkit-backdrop-filter: blur(20px) saturate(150%);
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .35rem; }

/* Radio navigation rendered as glass pills. */
[data-testid="stSidebar"] [data-testid="stRadio"] label,
.stApp [data-testid="stRadio"] label {
    padding: 6px 10px;
    border-radius: 10px;
    border: 1px solid transparent;
    transition: background .2s ease, border-color .2s ease, transform .2s ease;
}
[data-testid="stRadio"] label:hover {
    background: rgba(255, 255, 255, 0.06) !important;
    transform: translateX(3px);
}
[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(96deg, rgba(124, 58, 237, .34), rgba(6, 182, 212, .22)) !important;
    border-color: var(--accent);
    box-shadow: 0 0 18px rgba(124, 58, 237, 0.28);
}

/* Tabs as pills. */
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--edge);
    border-radius: 999px;
    margin-right: 6px;
    transition: background .2s ease, border-color .2s ease;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover { background: rgba(255, 255, 255, 0.10); }
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(96deg, rgba(124, 58, 237, .38), rgba(6, 182, 212, .26)) !important;
    border-color: var(--accent);
}

/* ------------------------------------------------------------------ *
 * 6. Inputs and buttons
 * ------------------------------------------------------------------ */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
.stTextInput input, .stNumberInput input {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid var(--edge) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}
[data-baseweb="select"] > div:hover { border-color: var(--accent) !important; }

.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    background: linear-gradient(96deg,
        rgba(124, 58, 237, 0.85) 0%,
        rgba(99, 102, 241, 0.78) 45%,
        rgba(6, 182, 212, 0.80) 100%);
    border: 1px solid var(--edge);
    border-radius: 12px;
    color: #ffffff;
    font-weight: 650;
    letter-spacing: .01em;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.28);
    transition: transform .2s ease, box-shadow .25s ease, border-color .2s ease,
                filter .2s ease;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    border-color: var(--accent);
    filter: brightness(1.12) saturate(1.1);
    box-shadow: 0 12px 34px rgba(124, 58, 237, 0.45);
}
.stButton > button:active { transform: translateY(0); filter: brightness(0.98); }

/* ------------------------------------------------------------------ *
 * 7. Status messages
 * ------------------------------------------------------------------ */
[data-testid="stAlert"] {
    background: var(--card);
    border: 1px solid var(--edge);
    border-radius: 14px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

/* ------------------------------------------------------------------ *
 * 8. Plotly figures sit on glass
 * ------------------------------------------------------------------ */
/* Streamlit 1.35 uses the legacy .stPlotlyChart class, not a testid. */
.stPlotlyChart, .js-plotly-plot, .plot-container {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid var(--edge);
    background: var(--card);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}
/* Belt and braces: Plotly writes background: white on its <svg> inline. */
.stPlotlyChart svg, .js-plotly-plot svg, .js-plotly-plot .main-svg {
    background: transparent !important;
}
.js-plotly-plot .modebar-btn path { fill: var(--muted) !important; }
.js-plotly-plot .modebar-btn:hover path { fill: var(--accent) !important; }

/* ------------------------------------------------------------------ *
 * 9. Scrollbar
 * ------------------------------------------------------------------ */
::-webkit-scrollbar { width: 11px; height: 11px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--glow1), var(--glow2));
    border-radius: 99px;
    border: 3px solid transparent;
    background-clip: padding-box;
}
::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, var(--glow2), var(--glow3)); background-clip: padding-box; }

/* Plotly's own scrollbars */
::-webkit-scrollbar-thumb:horizontal { background: var(--glow4); background-clip: padding-box; }
"""


def build_css(theme_name: str = DEFAULT_THEME) -> str:
    """Return the complete CSS for ``theme_name``."""
    theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME])
    return (
        _CSS
        .replace("@theme-vars@", _variables(theme))
        .replace("@noise-uri@", _NOISE_URI)
        .replace("@speed@", str(theme["speed"]))
    )


# The decorative layers. Kept separate from the CSS because Streamlit's markdown
# renderer mangles a <style> block that shares a call with other HTML.
_BACKGROUND_LAYERS = (
    '<div class="arx-bg" aria-hidden="true">'
    '<div class="arx-aurora"></div>'
    '<div class="arx-orb a"></div>'
    '<div class="arx-orb b"></div>'
    '<div class="arx-orb c"></div>'
    '<div class="arx-grid"></div>'
    '<div class="arx-noise"></div>'
    '<div class="arx-vignette"></div>'
    "</div>"
)


def inject(theme_name: str = DEFAULT_THEME) -> None:
    """Write the theme and its background layers into the current page."""
    # A lone <style> element survives Streamlit's markdown sanitiser; a <style>
    # mixed with other markup does not, hence the two separate calls.
    st.markdown(f"<style>{build_css(theme_name)}</style>", unsafe_allow_html=True)
    st.markdown(_BACKGROUND_LAYERS, unsafe_allow_html=True)
