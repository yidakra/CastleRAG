"""Live-UI smoke test for clickable time-evidence citations.

Drives a real browser through one ask → record verdicts → submit cycle, then
finds a ``.cite-chip`` inside the rendered thread answer and clicks it.  The
click should focus the cited moment and autoplay that camera's embed in place —
asserted via the ``.playing-badge`` that ``_render_camera_grid`` only adds when
``autoplay_camera`` matches a tile's camera id.

Pairs with the unit-level pins in ``tests/unit/test_ui.py`` /
``tests/unit/test_rag_engine.py``: those tests lock chip rendering + citation
embedding into answer text; this one pins that a click actually drives the
seeked-autoplay path end to end.

Usage
-----
# Install the runtime dep (one-time):
    .venv/bin/python -m pip install playwright
    .venv/bin/python -m playwright install chromium

# Run against a live UI:
    CASTLERAG_UI_BASIC_AUTH=demo:<pw> python scripts/ui_smoke_citation.py

Environment
-----------
CASTLERAG_UI_BASIC_AUTH   required, ``user:password`` for HTTP basic auth
UI_URL                    optional, defaults to ``http://localhost:8050/``
UI_SMOKE_OUT_DIR          optional, screenshot directory (default outputs/ui_smoke)

Exit code 0 = at least one .cite-chip found, click promotes its camera's tile
to .playing-badge AND the iframe src for that tile carries ``autoplay=1``.
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


def log(msg: str) -> None:
    print(f"[citation] {msg}", flush=True)


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
        page.screenshot(path=str(OUT / "citation_01_after_ask.png"), full_page=True)

        # Confirm all three so the thread card (and its answer + chips) renders.
        log("recording 3x Confirm so a thread card renders")
        review_cols = page.locator("#review-row .review")
        for idx in range(3):
            btn = review_cols.nth(idx).locator("button:has-text('✓ Confirm')").first
            btn.scroll_into_view_if_needed()
            btn.click()
            time.sleep(0.6)
        page.locator("#submit-reviews-button").wait_for(state="visible", timeout=30_000)
        page.locator("#submit-reviews-button").click()

        log("waiting for .cite-chip nodes in the rendered thread")
        chips_info: list[dict] = []
        deadline = time.time() + 60
        while time.time() < deadline:
            chips_info = page.evaluate(
                """
                () => Array.from(document.querySelectorAll('.cite-chip')).map(
                    el => ({ text: el.textContent.trim(), title: el.title || '' })
                )
                """
            )
            if chips_info:
                break
            time.sleep(1)
        log(f"  cite-chips found: {len(chips_info)}")
        for c in chips_info[:5]:
            log(f"    chip: text={c['text']!r}  title={c['title']!r}")
        if not chips_info:
            log("ERROR: no citation chips appeared in the answer")
            page.screenshot(
                path=str(OUT / "citation_02_no_chips.png"), full_page=True
            )
            ctx.close()
            browser.close()
            return 1
        page.screenshot(path=str(OUT / "citation_02_with_chips.png"), full_page=True)

        # Parse the camera id out of the first chip's "Play <cam> at <label>"
        # title (more robust than scraping the "▶ <cam> · <label>" body).
        first = chips_info[0]
        title = first["title"]
        target_cam = ""
        if title.startswith("Play ") and " at " in title:
            target_cam = title[len("Play ") :].split(" at ", 1)[0].strip()
        if not target_cam:
            # Fallback: body text is "▶ <cam> · <label>"
            body = first["text"].lstrip("▶ ").strip()
            target_cam = body.split("·", 1)[0].strip()
        log(f"  clicking chip targeting camera {target_cam!r}")
        page.locator(".cite-chip").first.click()

        log("waiting for .playing-badge to appear in #camera-grid")
        playing_seen = False
        playing_camera = ""
        autoplay_src = ""
        deadline = time.time() + 20
        while time.time() < deadline:
            state = page.evaluate(
                """
                () => {
                    const tiles = Array.from(
                        document.querySelectorAll('#camera-grid .camera-tile, '
                            + '#camera-grid > *')
                    );
                    for (const t of tiles) {
                        const badge = t.querySelector('.playing-badge');
                        if (!badge) continue;
                        // Camera id is the first .mantine-Text in the tile header.
                        const camEl = t.querySelector('.mantine-Text-root');
                        const iframe = t.querySelector('iframe.camera-frame');
                        return {
                            cam: camEl ? camEl.textContent.trim() : '',
                            src: iframe ? iframe.src : '',
                        };
                    }
                    return null;
                }
                """
            )
            if state:
                playing_seen = True
                playing_camera = state["cam"]
                autoplay_src = state["src"]
                break
            time.sleep(0.5)
        log(f"  playing badge present: {playing_seen}")
        log(
            f"  playing camera:        {playing_camera!r}  "
            f"(chip pointed at {target_cam!r})"
        )
        log(f"  autoplay iframe src:   {autoplay_src[:120]!r}")

        page.screenshot(
            path=str(OUT / "citation_03_after_click.png"), full_page=True
        )

        ctx.close()
        browser.close()

        ok = (
            playing_seen
            and playing_camera == target_cam
            and "autoplay=1" in autoplay_src
        )
        log(f"RESULT: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
