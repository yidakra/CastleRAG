/* Draggable splitter for the chat | evidence-viewer columns.
 *
 * Drags the `.app-gutter` to set `--thread-w` on `.app-body` (which the CSS grid
 * uses for the left column width). Width is clamped and persisted to
 * localStorage so it survives reloads. Pure vanilla JS, auto-loaded by Dash from
 * the assets/ folder — no extra component library.
 */
(function () {
  "use strict";

  var MIN_LEFT = 240; // px — keep the chat usable
  var MIN_RIGHT = 440; // px — keep the 3-column camera/button grid usable
  var KEY = "castlerag.threadWidth";

  function appBody() {
    return document.querySelector(".app-body");
  }

  function setWidth(px) {
    var b = appBody();
    if (b) b.style.setProperty("--thread-w", Math.round(px) + "px");
  }

  // Restore a saved width once the body has rendered (Dash renders client-side).
  var tries = 0;
  var restore = setInterval(function () {
    var b = appBody();
    if (b) {
      var saved = parseFloat(localStorage.getItem(KEY));
      if (saved && saved >= MIN_LEFT) setWidth(saved);
      clearInterval(restore);
    } else if (tries++ > 50) {
      clearInterval(restore);
    }
  }, 100);

  var dragging = false;
  var gutter = null;

  document.addEventListener("mousedown", function (e) {
    var g = e.target.closest ? e.target.closest(".app-gutter") : null;
    if (!g) return;
    dragging = true;
    gutter = g;
    g.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    e.preventDefault();
  });

  document.addEventListener("mousemove", function (e) {
    if (!dragging) return;
    var b = appBody();
    if (!b) return;
    var rect = b.getBoundingClientRect();
    var w = e.clientX - rect.left;
    var max = rect.width - MIN_RIGHT;
    if (w < MIN_LEFT) w = MIN_LEFT;
    if (w > max) w = max;
    setWidth(w);
  });

  document.addEventListener("mouseup", function () {
    if (!dragging) return;
    dragging = false;
    if (gutter) gutter.classList.remove("dragging");
    gutter = null;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    var b = appBody();
    if (b) {
      var w = parseFloat(b.style.getPropertyValue("--thread-w"));
      if (w) localStorage.setItem(KEY, Math.round(w));
    }
  });

  // Double-click the gutter resets to the default width.
  document.addEventListener("dblclick", function (e) {
    var g = e.target.closest ? e.target.closest(".app-gutter") : null;
    if (!g) return;
    var b = appBody();
    if (b) b.style.removeProperty("--thread-w");
    localStorage.removeItem(KEY);
  });
})();
