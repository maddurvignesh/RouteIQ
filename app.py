"""Streamlit entry point for the *Adaptive Routing* project.

Run it with::

    streamlit run app.py

All page logic lives in :mod:`visualization.dashboard`; this file only configures
the page and delegates, so the UI layer stays thin and the core simulation code
remains importable from tests and scripts without Streamlit.
"""

from __future__ import annotations

import streamlit as st

from visualization.dashboard import run

st.set_page_config(
    page_title="Adaptive Routing | Real-Time Network Conditions",
    page_icon="\U0001F310",
    layout="wide",
    initial_sidebar_state="expanded",
)

run()
