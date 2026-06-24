"""Live-UI smoke test for the '— Ignore' verdict convergence path.

Drives a real browser through one ask → record verdicts (Confirm/Confirm/Ignore)
→ submit-reviews cycle against the running CastleRAG UI, then asserts that the
investigation reaches the converged state: ``#converged-banner`` becomes visible
and ``#compose-wrap`` stays hidden.

Pairs with the engine-side test in ``tests/unit/test_ui.py``: that test pins the
``_should_converge`` predicate; this one pins the end-to-end behaviour the
predicate is supposed to drive.

Usage
-----
# Install the runtime dep (one-time):
    .venv/bin/python -m pip install playwright
    .venv/bin/python -m playwright install chromium

# Run against a live UI:
    CASTLERAG_UI_BASIC_AUTH=demo:<pw> python scripts/ui_smoke_ignore.py

Environment
-----------
CASTLERAG_UI_BASIC_AUTH   required, ``user:password`` for HTTP basic auth
UI_URL                    optional, defaults to ``http://localhost:8050/``
UI_SMOKE_OUT_DIR          optional, screenshot directory (default outputs/ui_smoke)

Exit code 0 = converged banner visible AND compose-wrap hidden AND banner
text contains 'converged'.
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
    print(f"[ignore] {msg}", flush=True)


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
            viewport={"width": 1600, "height": 1100},
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
        while time.time() < deadline:
            n = page.evaluate(
                "() => document.querySelectorAll("
                "'#camera-grid .camera-tile, #camera-grid > *').length"
            )
            if n >= 3:
                log(f"  camera-grid has {n} tiles")
                break
            time.sleep(2)
        else:
            log("ERROR: camera grid never populated")
            return 1
        page.screenshot(path=str(OUT / "ignore_01_after_ask.png"), full_page=True)

        # All three verdicts in the "resolved" bucket: Confirm / Confirm / Ignore.
        # _should_converge() (callbacks.py) requires every verdict to be confirmed
        # or ignored — no outstanding refines.
        log("recording verdicts: Confirm / Confirm / Ignore")
        review_cols = page.locator("#review-row .review")
        actions = ["✓ Confirm", "✓ Confirm", "— Ignore"]
        for idx in range(3):
            col = review_cols.nth(idx)
            btn = col.locator(f"button:has-text('{actions[idx]}')").first
            btn.scroll_into_view_if_needed()
            btn.click()
            time.sleep(1.0)
        page.screenshot(path=str(OUT / "ignore_02_verdicts.png"), full_page=True)

        log("clicking submit-reviews-button (should converge, not open compose)")
        page.locator("#submit-reviews-button").wait_for(state="visible", timeout=30_000)
        page.locator("#submit-reviews-button").click()

        # Poll Dash-managed visibility props. The convergence branch toggles
        # both #converged-banner (hidden -> False) and #compose-wrap (hidden -> True).
        log("waiting for converged-banner to unhide")
        deadline = time.time() + 30
        converged_hidden = True
        compose_hidden = True
        while time.time() < deadline:
            converged_hidden = page.evaluate(
                "() => document.getElementById('converged-banner')?.hidden ?? true"
            )
            compose_hidden = page.evaluate(
                "() => document.getElementById('compose-wrap')?.hidden ?? true"
            )
            if not converged_hidden:
                break
            time.sleep(0.5)

        log(f"  converged-banner hidden: {converged_hidden}  (expected False)")
        log(f"  compose-wrap     hidden: {compose_hidden}  (expected True)")

        banner_text = page.evaluate(
            "() => document.getElementById('converged-banner')?.innerText || ''"
        )
        log(f"  banner text: {banner_text[:160]!r}")

        page.screenshot(path=str(OUT / "ignore_03_converged.png"), full_page=True)

        ok = (
            (not converged_hidden)
            and compose_hidden
            and "converged" in banner_text.lower()
        )
        log(f"RESULT: {'PASS' if ok else 'FAIL'}")

        ctx.close()
        browser.close()
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
