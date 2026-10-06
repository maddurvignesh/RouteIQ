"""Deployment descriptor tests.

Streamlit is a long-running server that upgrades to WebSockets, so it must be
deployed as a service (Community Cloud, Render, a container) and never as a
serverless function. These tests pin the descriptors that make that work, so a
later edit cannot quietly break the deploy.

The read-only-filesystem test is the important one: a container may mount the
source read-only, and ``config.py`` created directories at import time.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

def test_dockerfile_runs_streamlit_on_all_interfaces() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'CMD ["streamlit", "run", "app.py"' in dockerfile, "wrong entrypoint"
    assert '"0.0.0.0"' in dockerfile, "the container must not bind to localhost only"
    assert "EXPOSE 8501" in dockerfile
    assert "_stcore/health" in dockerfile, "PaaS platforms need a health check"
    assert "USER appuser" in dockerfile, "the image should not run as root"
    assert "python:3.12" in dockerfile


def test_dockerignore_keeps_caches_out_of_the_build_context() -> None:
    text = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    for entry in (".git", "__pycache__/", "docs/_screens/", ".venv/"):
        assert entry in text, f"{entry} is not excluded from the build context"


# ---------------------------------------------------------------------------
# PaaS descriptors
# ---------------------------------------------------------------------------

def test_render_yaml_starts_the_app() -> None:
    render = (PROJECT_ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "type: web" in render
    assert "streamlit run app.py" in render
    assert "$PORT" in render, "Render injects the listen port via $PORT"
    assert "0.0.0.0" in render, "Render needs the server bound to all interfaces"
    assert "healthCheckPath: /_stcore/health" in render


def test_streamlit_config_binds_all_interfaces() -> None:
    config = (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "[server]" in config
    assert 'address = "0.0.0.0"' in config
    assert "headless = true" in config
    assert "base = \"dark\"" in config, "a dark base avoids a white flash pre-paint"


def test_vercelignore_blocks_the_serverless_misdetection() -> None:
    """Vercel sees a root app.py and assumes a function - that build always fails."""
    ignore = (PROJECT_ROOT / ".vercelignore").read_text(encoding="utf-8")
    entries = {
        line.strip()
        for line in ignore.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert "app.py" in entries, (
        "app.py must be listed in .vercelignore, otherwise Vercel treats it as "
        "a serverless function and fails the build"
    )


def test_readme_explains_why_vercel_cannot_host_the_app() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Deploying" in readme
    assert "Vercel" in readme, "the README should explain the platform mismatch"
    assert "streamlit.app" in readme or "share.streamlit.io" in readme
    assert "render.com" in readme.lower()


# ---------------------------------------------------------------------------
# Robustness of the import-time setup
# ---------------------------------------------------------------------------

def test_config_survives_a_read_only_filesystem() -> None:
    """Simulate a read-only mount: `mkdir` raises, `import config` must not."""
    source = (PROJECT_ROOT / "config.py").read_text(encoding="utf-8")
    target = "        _dir.mkdir(parents=True, exist_ok=True)"
    assert target in source, "the mkdir call moved - update this test"

    patched = source.replace(
        target,
        "        raise OSError(13, 'Permission denied')  # simulated read-only mount",
    )
    assert patched != source

    workdir = Path(tempfile.mkdtemp())
    try:
        (workdir / "config.py").write_text(patched, encoding="utf-8")
        shutil.copytree(PROJECT_ROOT / "ml", workdir / "ml")
        shutil.copytree(PROJECT_ROOT / "core", workdir / "core")
        result = subprocess.run(
            [sys.executable, "-c", "import config; print(config.ML_MODEL_PATH.name)"],
            cwd=workdir, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            "config.py still crashes on a read-only filesystem:\n" + result.stderr[-500:]
        )
        assert "congestion_model.joblib" in result.stdout
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
