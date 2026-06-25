"""Live-UI smoke test for the new evidence figures (heatmap + funnel).

Drives a real browser through one ask cycle on the running CastleRAG UI and
asserts that two recent figures actually render in the DOM:

* the cross-camera agreement heatmap inside ``#evidence-figure``
  (two subplot titles: ``Camera match scores`` + ``Cross-camera agreement``);
* the per-thread retrieval pipeline funnel (``#funnel-<group_id>``) with the
  canonical stage labels ``Retrieved / Reranked / Candidates / Displayed``.

Pairs with the unit-level pin in ``tests/unit/test_figures.py``: that test
locks the figure construction; this one pins that the figures actually reach
the rendered page.

Usage
-----
# Install the runtime dep (one-time):
    .venv/bin/python -m pip install playwright
    .venv/bin/python -m playwright install chromium

# Run against a live UI:
    CASTLERAG_UI_BASIC_AUTH=demo:<pw> python scripts/ui_smoke_figures.py

Environment
-----------
CASTLERAG_UI_BASIC_AUTH   required, ``user:password`` for HTTP basic auth
UI_URL                    optional, defaults to ``http://localhost:8050/``
UI_SMOKE_OUT_DIR          optional, screenshot directory (default outputs/ui_smoke)

Exit code 0 = both subplot titles found in #evidence-figure AND a #funnel-<gid>
node with the expected stage labels in the rendered thread card.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

UI_URL = os.environ.get("UI_URL", "http://localhost:8050/")
OUT = Path(os.environ.get("UI_SMOKE_OUT_DIR", "outputs/ui_smoke"))
QUESTION = "What book was Allie reading?"
EXPECTED_STAGES = {"Retrieved", "Reranked", "Candidates", "Displayed"}


def log(msg: str) -> None:
    print(f"[figures] {msg}", flush=True)


def main() -> int:
    try:
        auth = os.environ["CASTLERAG_UI_BASIC_AUTH"]
    except KeyError:
        sys.stderr.write(
            "ERROR: CASTLERAG_UI_BASIC_AUTH not set (expected user:password)\n"
        )
        return 2
    user, _, password = auth.partition(":")
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1600, "height": 1400},
            http_credentials={"username": user, "password": password},
        )
        page = ctx.new_page()
        page.set_default_timeout(60_000)

        log(f"navigating {UI_URL}")
        page.goto(UI_URL, wait_until="networkidle", timeout=60_000)

        log(f"asking: {QUESTION!r}")
        page.locator("#new-question-input").fill(QUESTION)
        page.locator("#ask-new-button").click()

        log("waiting for camera tiles (RAG cold-path can be slow)")
        deadline = time.time() + 240
        tiles = 0
        while time.time() < deadline:
            tiles = page.evaluate(
                "() => document.querySelectorAll("
                "'#camera-grid .camera-tile, #camera-grid > *').length"
            )
            if tiles >= 2:
                log(f"  camera-grid has {tiles} tiles")
                break
            time.sleep(2)
        else:
            log("ERROR: camera grid never populated")
            return 1
        page.screenshot(path=str(OUT / "figures_01_after_ask.png"), full_page=True)

        # ----- heatmap probe -----
        # camera_match_figure renders a 2-row subplot into #evidence-figure with
        # titles "Camera match scores" + "Cross-camera agreement" when n >= 2.
        labels = page.evaluate(
            """
            () => {
                const out = [];
                document.querySelectorAll(
                    '#evidence-figure .annotation-text, '
                    + '#evidence-figure g.annotation text, '
                    + '#evidence-figure text'
                ).forEach(el => out.push(el.textContent));
                return out;
            }
            """
        )
        has_scores_title = any("Camera match scores" in lbl for lbl in labels)
        has_heatmap_title = any("Cross-camera agreement" in lbl for lbl in labels)
        log(f"  scores subplot title:  {has_scores_title}  (expected True)")
        log(f"  heatmap subplot title: {has_heatmap_title}  (expected True)")

        # Drive the thread card to render so the pipeline funnel appears.
        # Three Confirms gets us a converged thread without extra rounds.
        log("recording 3x Confirm so a thread card renders")
        review_cols = page.locator("#review-row .review")
        for idx in range(3):
            btn = review_cols.nth(idx).locator("button:has-text('✓ Confirm')").first
            btn.scroll_into_view_if_needed()
            btn.click()
            time.sleep(0.6)
        page.locator("#submit-reviews-button").wait_for(state="visible", timeout=30_000)
        page.locator("#submit-reviews-button").click()

        log("waiting for #funnel-<gid> in the rendered thread")
        funnel_ids: list[str] = []
        deadline = time.time() + 60
        while time.time() < deadline:
            funnel_ids = page.evaluate(
                "() => Array.from(document.querySelectorAll('[id^=\"funnel-\"]'))"
                ".map(e => e.id)"
            )
            if funnel_ids:
                break
            time.sleep(1)
        log(f"  funnel ids: {funnel_ids}")

        page.screenshot(
            path=str(OUT / "figures_02_after_converge.png"), full_page=True
        )

        funnel_svg_text = page.evaluate(
            """
            () => {
                const out = [];
                document.querySelectorAll('[id^="funnel-"] text').forEach(
                    el => out.push(el.textContent)
                );
                return out;
            }
            """
        )
        stages_seen = {t for t in funnel_svg_text if t in EXPECTED_STAGES}
        missing = EXPECTED_STAGES - stages_seen
        log(f"  funnel stages found: {sorted(stages_seen)}")
        if missing:
            log(f"  missing funnel stages: {sorted(missing)}")

        ctx.close()
        browser.close()

        ok = (
            has_scores_title
            and has_heatmap_title
            and bool(funnel_ids)
            and not missing
        )
        log(f"RESULT: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
