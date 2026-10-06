"""Build the project presentation as a real ``.pptx`` file.

Run::

    python docs/build_presentation.py

The content mirrors ``docs/presentation.md``; this script turns it into
16:9 slides with native (editable) tables, the figures from ``docs/diagrams.py``
and the speaker notes attached to each slide's notes pane.

Requires ``python-pptx`` (``pip install python-pptx``) and the figures to
exist — run ``python docs/diagrams.py`` first if they are missing.

Fill in ``STUDENT`` / ``GUIDE`` below before printing the deck.
"""

from __future__ import annotations

import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:          # allows `python docs/build_presentation.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

# ---------------------------------------------------------------------------
# Presentation metadata - edit these before printing
# ---------------------------------------------------------------------------

TITLE = "Adaptive Routing Based on Real-Time Network Conditions"
SUBTITLE = "A simulation study of reactive and ML-assisted routing"
STUDENT = "Your Name  |  Roll No. 00  |  III Year"
COURSE = "Computer Networks + Machine Learning"
GUIDE = "Guide: Prof. Guide Name"

OUTPUT = DOCS_DIR / "Adaptive_Routing_Presentation.pptx"

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

INK = RGBColor(0x1F, 0x29, 0x33)
MUTED = RGBColor(0x64, 0x74, 0x8B)
ACCENT = RGBColor(0x25, 0x63, 0xEB)
GOOD = RGBColor(0x16, 0xA3, 0x4A)
BAD = RGBColor(0xDC, 0x26, 0x26)
PANEL = RGBColor(0xF1, 0xF5, 0xF9)
BAND = RGBColor(0xDB, 0xEA, 0xFE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(0.62)
BODY_W = SLIDE_W - 2 * MARGIN


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def blank_slide(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def textbox(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    return frame


def style_run(run, size, *, bold=False, color=INK, italic=False, font="Calibri"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font


def add_title(slide, text, subtitle=None):
    """Standard slide heading with a thin accent rule underneath."""
    frame = textbox(slide, MARGIN, Inches(0.42), BODY_W, Inches(0.85))
    paragraph = frame.paragraphs[0]
    style_run(paragraph.add_run(), 30, bold=True, color=INK)
    paragraph.runs[0].text = text

    if subtitle:
        sub = textbox(slide, MARGIN, Inches(1.16), BODY_W, Inches(0.42))
        para = sub.paragraphs[0]
        style_run(para.add_run(), 15, color=MUTED, italic=True)
        para.runs[0].text = subtitle

    rule = slide.shapes.add_shape(1, MARGIN, Inches(1.62), Inches(1.5), Pt(4))
    rule.fill.solid()
    rule.fill.fore_color.rgb = ACCENT
    rule.line.fill.background()
    rule.shadow.inherit = False


def add_bullets(slide, items, left=MARGIN, top=Inches(2.0), width=None,
                height=None, size=19, gap=10):
    """``items`` are strings; a leading '- ' marks a sub-bullet.

    A hanging indent is applied so that wrapped lines line up under the text
    rather than under the bullet glyph.
    """
    width = width or BODY_W
    height = height or (SLIDE_H - top - Inches(0.6))
    frame = textbox(slide, left, top, width, height)

    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.space_after = Pt(gap)
        sub = item.startswith("- ")
        text = item[2:] if sub else item
        bullet = "\u2013  " if sub else "\u25aa  "

        hang = Pt(15 if sub else 20)
        p_pr = paragraph._p.get_or_add_pPr()
        p_pr.set("marL", str(int(hang + Pt(12))))
        p_pr.set("indent", str(-int(hang)))

        run = paragraph.add_run()
        run.text = bullet
        style_run(run, size - 2 if sub else size,
                  color=MUTED if sub else ACCENT, bold=not sub)
        body = paragraph.add_run()
        body.text = text
        style_run(body, size - 1 if sub else size, color=MUTED if sub else INK)
        paragraph.level = 1 if sub else 0
    return frame


def add_note(slide, text):
    frame = slide.notes_slide.notes_text_frame
    frame.text = text


def add_picture(slide, filename, left, top, max_w, max_h):
    """Insert a figure scaled to fit the given box, centred inside it."""
    path = DOCS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing - run `python docs/diagrams.py` first."
        )
    picture = slide.shapes.add_picture(str(path), left, top, width=max_w)
    if picture.height > max_h:
        ratio = max_h / picture.height
        picture.width = Emu(int(picture.width * ratio))
        picture.height = max_h
    picture.left = Emu(int(left + (max_w - picture.width) / 2))
    picture.top = Emu(int(top + (max_h - picture.height) / 2))
    return picture


def add_table(slide, headers, rows, left, top, width, *, size=14,
              highlight_last=False, col_widths=None, row_height=Inches(0.36),
              left_align=False):
    """Render a native, editable PowerPoint table.

    ``left_align=True`` is used for key/value tables, where every cell is prose
    and centring short values next to long ones looks accidental. Data tables
    keep numeric cells centred.
    """
    shape = slide.shapes.add_table(len(rows) + 1, len(headers),
                                   left, top, width, row_height * (len(rows) + 1))
    table = shape.table

    if col_widths:
        total = sum(col_widths)
        for index, share in enumerate(col_widths):
            table.columns[index].width = Emu(int(width * share / total))

    def paint(cell, text, *, bold, color, fill, align=PP_ALIGN.LEFT):
        cell.text = ""
        cell.margin_left = Inches(0.12)
        cell.margin_right = Inches(0.08)
        cell.margin_top = cell.margin_bottom = Inches(0.03)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
        paragraph = cell.text_frame.paragraphs[0]
        paragraph.alignment = align
        run = paragraph.add_run()
        run.text = str(text)
        style_run(run, size, bold=bold, color=color)

    for index, header in enumerate(headers):
        paint(table.cell(0, index), header, bold=True, color=WHITE, fill=ACCENT)

    for r, row in enumerate(rows, start=1):
        last = highlight_last and r == len(rows)
        fill = BAND if last else (WHITE if r % 2 else PANEL)
        for c, value in enumerate(row):
            # Numeric-looking cells read better centred; prose cells left-aligned.
            text = str(value)
            if left_align or c == 0 or len(text) > 14:
                align = PP_ALIGN.LEFT
            else:
                align = PP_ALIGN.CENTER
            paint(table.cell(r, c), value, bold=last, color=INK, fill=fill,
                  align=align)
    return shape


def add_callout(slide, text, top, *, color=BAND, size=16):
    box = slide.shapes.add_shape(1, MARGIN, top, BODY_W, Inches(0.62))
    box.fill.solid()
    box.fill.fore_color.rgb = color
    box.line.color.rgb = ACCENT
    box.line.width = Pt(1)
    box.shadow.inherit = False
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = Inches(0.2)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    run = paragraph.add_run()
    run.text = text
    style_run(run, size, bold=True, color=INK)
    return box


def add_footer(slide, number):
    frame = textbox(slide, MARGIN, SLIDE_H - Inches(0.42), BODY_W, Inches(0.3))
    paragraph = frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.RIGHT
    run = paragraph.add_run()
    run.text = f"{number}"
    style_run(run, 11, color=MUTED)


def image_right(slide, filename, bullets, *, width=None, note=None):
    """Two-column layout: bullets on the left, figure on the right."""
    image_w = width or Inches(5.6)
    add_bullets(slide, bullets, left=MARGIN, top=Inches(2.0),
                width=BODY_W - image_w - Inches(0.4), size=17, gap=8)
    add_picture(slide, filename, MARGIN + BODY_W - image_w, Inches(1.9),
                image_w, Inches(4.9))
    if note:
        add_note(slide, note)


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------

def build() -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
    number = 0

    def numbered(slide):
        nonlocal number
        number += 1
        add_footer(slide, number)
        return slide

    # -- 1. Title ----------------------------------------------------------
    slide = numbered(blank_slide(prs))
    band = slide.shapes.add_shape(1, 0, Inches(2.0), SLIDE_W, Pt(5))
    band.fill.solid()
    band.fill.fore_color.rgb = ACCENT
    band.line.fill.background()
    band.shadow.inherit = False

    frame = textbox(slide, MARGIN, Inches(2.45), BODY_W, Inches(1.5))
    paragraph = frame.paragraphs[0]
    style_run(paragraph.add_run(), 40, bold=True, color=INK)
    paragraph.runs[0].text = TITLE

    frame = textbox(slide, MARGIN, Inches(3.85), BODY_W, Inches(0.6))
    para = frame.paragraphs[0]
    style_run(para.add_run(), 19, color=MUTED, italic=True)
    para.runs[0].text = SUBTITLE

    frame = textbox(slide, MARGIN, Inches(5.15), BODY_W, Inches(1.2))
    for index, line in enumerate([STUDENT, COURSE, GUIDE]):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.space_after = Pt(4)
        run = para.add_run()
        run.text = line
        style_run(run, 15, color=INK if index == 0 else MUTED,
                  bold=index == 0)
    add_note(slide, "One-line hook: classical routing asks 'what is shortest?' - "
                    "this project asks 'what is fastest right now?'")

    # -- 2. Problem --------------------------------------------------------
    slide = numbered(blank_slide(prs))
    add_title(slide, "The problem", "Static metrics ignore the state of the network")
    add_bullets(slide, [
        "OSPF and RIP choose a path from a static metric - normally hop count",
        "The path is recomputed only when a link goes up or down",
        "Networks rarely fail - they degrade: buffers fill, delay rises, packets drop",
        "Result: all traffic stays pinned to \u201cthe shortest path\u201d "
        "while that path saturates",
    ], top=Inches(2.05), size=20, gap=14)
    add_callout(slide, "Cost and congestion are simply not the same quantity.",
                Inches(5.6))
    add_note(slide, "This is the standard congestion-collapse story: the metric and "
                    "the objective have drifted apart. 803 ms vs 86 ms in experiment 2.")

    # -- 3. Objectives -----------------------------------------------------
    slide = numbered(blank_slide(prs))
    add_title(slide, "Goal and objectives")
    add_bullets(slide, [
        "Build a routing decision that reacts to live link conditions",
        "Measure it against a load-oblivious baseline with a real packet simulation",
        "Add an ML congestion predictor and test whether it adds value",
        "Report the results honestly - including where the ML module does NOT help",
    ], top=Inches(2.05), size=20, gap=16)
    add_note(slide, "The last bullet matters: the ML module only changes the outcome "
                    "in one of six scenarios, and we say so.")

    # -- 4. Architecture ---------------------------------------------------
    slide = numbered(blank_slide(prs))
    add_title(slide, "System architecture", "Five layers, one callback boundary")
    add_picture(slide, "architecture.png", MARGIN, Inches(1.85), BODY_W, Inches(4.5))
    add_callout(slide,
                "ML reaches the router only through penalty(u, v, attrs) -> float "
                "- so core never imports ml", Inches(6.35))
    add_note(slide, "The callback boundary is the key design decision: it is why the "
                    "system still runs if models/congestion_model.joblib is deleted.")

    # -- 5. Network --------------------------------------------------------
    slide = numbered(blank_slide(prs))
    add_title(slide, "The network", "8 routers, 13 links")
    image_right(
        slide, "topology.png",
        [
            "Routers S, A, B, C, D, E, F, T",
            "Hop-count shortest path: S \u2192 A \u2192 C \u2192 T (3 hops)",
            "Main alternate: S \u2192 B \u2192 D \u2192 F \u2192 T (4 hops)",
            "Each link carries bandwidth, latency, loss, congestion, traffic, reliability",
            "Live values plus base_* copies - so any fault can be undone exactly",
            "Invariant: congestion = traffic / bandwidth",
        ],
        note="Point out that the backbone is faster on paper but is exactly the route "
             "every naive protocol keeps using.",
    )

    # -- 6. Conditions -----------------------------------------------------
    slide = numbered(blank_slide(prs))
    add_title(slide, "Simulating real-time conditions", "Six degradations plus a way back to health")
    add_table(
        slide,
        ["Condition", "Effect on the affected links"],
        [
            ["CONGESTION", "queue fills; congestion, traffic and latency rise"],
            ["TRAFFIC_SPIKE", "utilisation jumps on a single link"],
            ["HIGH_LATENCY", "delay \u00d7 6, loss untouched"],
            ["HIGH_PACKET_LOSS", "loss driven to 5 - 20 %"],
            ["LOW_BANDWIDTH", "capacity cut by up to 70 %"],
            ["NORMAL", "restore every link to its baseline"],
        ],
        MARGIN, Inches(2.05), BODY_W, size=15, col_widths=[1.2, 3.0], left_align=True,
    )
    add_bullets(slide, [
        "A severity parameter in (0, 1] scales every effect",
        "Congestion is clamped at 0.97 so the queueing curve stays finite",
        "Default target: the current shortest path - \u201ccongest the route a naive "
        "protocol would keep using\u201d",
    ], top=Inches(4.85), size=16, gap=8)
    add_note(slide, "Severity is what lets one scenario be milder than another; exp5 "
                    "uses a gentle spike while exp2 saturates the whole backbone.")

    # -- 7. Baseline -------------------------------------------------------
    slide = numbered(blank_slide(prs))
    add_title(slide, "Strategy 1 - baseline", "Exactly what OSPF / RIP do")
    add_bullets(slide, [
        "Dijkstra on hop count",
        "Never looks at latency, loss or congestion",
        "The control group every other strategy is measured against",
    ], top=Inches(2.1), width=Inches(5.6), size=20, gap=16)
    add_picture(slide, "routing_decision.png", Inches(6.9), Inches(1.9),
                Inches(6.0), Inches(4.7))
    add_note(slide, "Walk the flowchart once here, then reuse it on the next two slides.")

    # -- 8. Adaptive -------------------------------------------------------
    slide = numbered(blank_slide(prs))
    add_title(slide, "Strategy 2 - adaptive cost", "Every term normalised to [0, 1]")
    add_callout(
        slide,
        "cost = 0.35\u00b7min(lat/100,1) + 0.30\u00b7congestion\n"
        "        + 0.25\u00b7min(loss/0.20,1) + 0.10\u00b7(1 \u2212 min(bw/1000,1))",
        Inches(2.0), size=19,
    )
    add_bullets(slide, [
        "Normalisation makes the four metrics comparable - ms, % and Mbps cannot be added raw",
        "Weights are validated to sum to 1.0 and live in config.py",
        "Dijkstra with this cost as the edge weight \u2192 minimum cost under current conditions",
        "The four cheapest rejected alternatives are listed too - the decision is explainable",
    ], top=Inches(3.6), size=18, gap=14)
    add_note(slide, "Weights are a priori engineering judgement. A sensitivity sweep is "
                    "the first item of future work.")

    # -- 9. ML strategy ----------------------------------------------------
    slide = numbered(blank_slide(prs))
    add_title(slide, "Strategy 3 - ML-assisted", "A proactive signal added to the same cost")
    add_bullets(slide, [
        "6 features: latency, loss, utilisation, traffic, queue length, previous_level",
        "Random Forest - 5 000 samples, 75/25 split, seed 42",
        "Labels come from an explicit rule:",
        "- severity = 0.40\u00b7util + 0.30\u00b7min(lat/100,1) + 0.30\u00b7min(loss/0.15,1)",
        "- severity \u2265 0.65 \u2192 Congested,  \u2265 0.40 \u2192 Warning, else Normal",
        "Predicted class \u2192 additive penalty of 0.00 / 0.35 / 0.90",
        "Penalties are computed once per decision, for every link",
    ], top=Inches(2.05), size=18, gap=11)
    add_note(slide, "previous_level is the previous prediction for the same link - a "
                    "router always knows it, so it is not leakage. It is what makes the "
                    "prediction temporal rather than per-sample.")

    # -- 10. ML results ----------------------------------------------------
    slide = numbered(blank_slide(prs))
    add_title(slide, "Classifier evaluation", "98.48 % on 1 250 held-out samples")
    add_picture(slide, "ml_confusion.png", MARGIN, Inches(1.9), Inches(5.9), Inches(4.6))
    add_picture(slide, "ml_importance.png", Inches(6.9), Inches(1.9),
                Inches(6.0), Inches(4.6))
    add_callout(slide, "Accuracy measures consistency with our own labelling rule - "
                       "the end-to-end proof is slide 13", Inches(6.2), size=14)
    add_note(slide, "Macro F1 0.983. Importances: previous_level 26 %, packet_loss 25 %, "
                    "utilisation 22 % - congestion persists, and the two direct symptoms "
                    "follow. Say the caveat out loud before anyone asks.")

    # -- 11. Design --------------------------------------------------------
    slide = numbered(blank_slide(prs))
    add_title(slide, "Experimental design", "6 scenarios \u00d7 3 strategies = 18 runs")
    add_table(
        slide,
        ["Dimension", "Value"],
        [
            ["Runs", "18 (6 scenarios \u00d7 3 strategies)"],
            ["Time steps per run", "5"],
            ["Packets per step", "80  \u2192  400 per run, 7 200 in total"],
            ["Seed", "42 (conditions, packets, ML split)"],
            ["Topology", "predefined 8-router / 13-link graph"],
            ["Source \u2192 destination", "S \u2192 T"],
        ],
        MARGIN, Inches(2.05), BODY_W, size=16, col_widths=[1.0, 2.4], left_align=True,
    )
    add_bullets(slide, [
        "Every strategy sees the identical timeline on an identical fresh topology",
        "exp1 is a control: with no faults all three must agree",
        "exp5 targets only link S-A - the scenario built to isolate the ML contribution",
    ], top=Inches(4.95), size=16, gap=9)
    add_note(slide, "A fresh topology per run means runs cannot contaminate each other. "
                    "The condition targets the hop-count shortest path, so the experiment "
                    "is not rigged in favour of the adaptive router.")

    # -- 12. Headline results ---------------------------------------------
    slide = numbered(blank_slide(prs))
    add_title(slide, "Headline results", "Averaged over the six scenarios")
    add_table(
        slide,
        ["Strategy", "Latency (ms)", "Loss (%)", "Delivery (%)", "Congested steps (%)"],
        [
            ["Shortest Path", "354.74", "5.83", "94.17", "40.00"],
            ["Adaptive Routing", "151.05", "0.29", "99.71", "26.67"],
            ["Adaptive + ML", "125.58", "0.29", "99.71", "13.33"],
        ],
        MARGIN, Inches(2.0), BODY_W, size=16, highlight_last=True,
        col_widths=[1.5, 1.0, 0.8, 0.9, 1.3],
    )
    add_picture(slide, "results_summary.png", MARGIN, Inches(3.75), BODY_W, Inches(2.5))
    add_callout(slide, "vs hop count:  \u221257 % latency,  \u221295 % packet loss,  "
                       "+43 % throughput", Inches(6.3), size=15)
    add_note(slide, "Network utilisation is identical (31.32 %) for all three - the same "
                    "traffic is offered every run, only the route differs. Route "
                    "utilisation 43.3 % \u2192 23.2 % is the meaningful figure.")

    # -- 13. Where the win comes from -------------------------------------
    slide = numbered(blank_slide(prs))
    add_title(slide, "Where the win comes from", "Two experiments carry the result")
    add_table(
        slide,
        ["exp2 - congestion on the shortest path", "Route", "Latency"],
        [
            ["Shortest Path", "S \u2192 A \u2192 C \u2192 T  (stuck)", "803 ms"],
            ["Adaptive (+ ML)", "S \u2192 B \u2192 D \u2192 F \u2192 T", "86 ms"],
        ],
        MARGIN, Inches(2.0), BODY_W, size=15, col_widths=[1.6, 1.7, 0.7],
    )
    add_table(
        slide,
        ["exp5 - traffic spike on a single link", "Route", "Latency"],
        [
            ["Shortest Path", "S \u2192 A \u2192 C \u2192 T", "238 ms"],
            ["Adaptive Routing", "S \u2192 A \u2192 C \u2192 T  (does not move)", "238 ms"],
            ["Adaptive + ML", "S \u2192 B \u2192 D \u2192 F \u2192 T", "86 ms"],
        ],
        MARGIN, Inches(3.35), BODY_W, size=15, highlight_last=True,
        col_widths=[1.6, 1.7, 0.7],
    )
    add_callout(slide, "In exp5 the ML route's current cost is higher (0.608 vs 0.512) - "
                       "the model is buying a future improvement, not a present one",
                Inches(5.3), size=14)
    add_note(slide, "This is the defining difference between a reactive and a proactive "
                    "signal, and it is the honest answer to 'why would you pay more now?'")

    # -- 14. When it does not help ----------------------------------------
    slide = numbered(blank_slide(prs))
    add_title(slide, "When it does not help", "Reported, not hidden")
    add_bullets(slide, [
        "Healthy network (exp1): all three strategies pick the identical route - "
        "adaptation costs nothing, but gains nothing either",
        "Whole neighbourhood degraded (exp6): every strategy falls to ~369 ms - "
        "adaptive routing finds the least-bad path but cannot create capacity",
        "The ML module changes the outcome in 1 of 6 scenarios, not all of them",
        "Results are deterministic (one seed, one topology): reproducible, "
        "but not a statistical sample",
    ], top=Inches(2.15), size=19, gap=20)
    add_note(slide, "Volunteering the limitations is the strongest thing you can do here. "
                    "Exp6 shows the technique's boundary: finding the least-bad path is "
                    "not the same as having capacity.")

    # -- 15. Limitations ---------------------------------------------------
    slide = numbered(blank_slide(prs))
    add_title(slide, "Limitations")
    add_bullets(slide, [
        "Simulation, not a testbed - no NS-3, Mininet or real routers",
        "Synthetic dataset, so accuracy reflects our own labelling rule",
        "Small fixed topology (8 routers); the random-topology builder is not yet swept",
        "Only three strategies - no ECMP, QoS or SDN baseline",
        "Hand-chosen weights, no sensitivity analysis yet",
    ], top=Inches(2.15), size=20, gap=20)
    add_note(slide, "Each of these is a concrete next step, not a dead end.")

    # -- 16. Conclusion ----------------------------------------------------
    slide = numbered(blank_slide(prs))
    add_title(slide, "Conclusion and future work")
    add_bullets(slide, [
        "Reacting to real-time conditions halves latency and removes 95 % of packet loss",
        "A small explainable classifier adds a proactive step that reactive routing "
        "cannot take",
        "- learned or swept cost weights",
        "- real public traffic traces instead of generated data",
        "- 20 - 100 router topologies, repeated over many seeds",
        "- ECMP and QoS baselines; closed-loop feedback",
        "- deploy the cost function as an OSPF metric or an SDN controller app",
    ], top=Inches(2.05), size=19, gap=11)
    add_callout(slide, "Thank you - questions?", Inches(6.35), size=20)
    add_note(slide, "Close on the one-line framing: hop count is cheap to compute and "
                    "wrong exactly when it matters. Thank the panel and invite questions.")

    prs.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    target = build()
    print(f"Wrote {len(Presentation(target).slides._sldIdLst)} slides to {target}")
