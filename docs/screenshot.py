"""Screenshot the running dashboard - used to review the visual design.

Start the app first::

    streamlit run app.py

Then::

    python docs/screenshot.py                      # the Overview page
    python docs/screenshot.py "ML Prediction" "Network Topology"

To capture a specific theme, set ``ARX_THEME`` **before** starting the server
(the theme selectbox then opens on that theme)::

    $env:ARX_THEME="Minimal"     # PowerShell
    streamlit run app.py

Images are written to ``docs/_screens/``. This is a development utility and
needs ``selenium`` plus a local Chrome install; nothing in the app depends on
it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent / "_screens"
URL = "http://localhost:8501"
PAGES = ("Overview", "Network Topology", "Adaptive Routing",
         "Network Simulation", "ML Prediction", "Performance Comparison")


def capture(pages: list[str], url: str = URL, out: Path = OUT) -> list[Path]:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By

    out.mkdir(exist_ok=True)
    options = Options()
    for flag in ("--headless=new", "--disable-gpu", "--no-sandbox",
                 "--window-size=1600,1000", "--force-device-scale-factor=1"):
        options.add_argument(flag)

    driver = webdriver.Chrome(options=options)
    saved: list[Path] = []
    try:
        driver.get(url)
        time.sleep(12)                     # Streamlit boot + first render

        for page in pages:
            labels = driver.find_elements(
                By.CSS_SELECTOR, '[data-testid="stRadio"] label')
            target = next((l for l in labels if page.lower() in l.text.lower()), None)
            if target is None:
                print(f"skip '{page}': navigation option not found")
                continue
            driver.execute_script("arguments[0].click();", target)
            time.sleep(5)
            name = page.lower().replace(" ", "_").replace("+", "plus")
            shot = out / f"{name}.png"
            driver.save_screenshot(str(shot))
            saved.append(shot)
            print("saved", shot)
    finally:
        driver.quit()
    return saved


def main(argv: list[str]) -> int:
    pages = argv[1:] or ["Overview"]
    unknown = [p for p in pages if p not in PAGES]
    if unknown:
        print(f"unknown page(s): {unknown}\navailable: {list(PAGES)}")
        return 1
    capture(pages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
