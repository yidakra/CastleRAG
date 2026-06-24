/* Auto-expanding query textarea.
 *
 * Uses document-level event delegation so the listeners survive Dash/React
 * re-renders that recreate the textarea DOM node.
 *
 *   Enter        → click the Send button (submit)
 *   Shift+Enter  → insert a newline
 *   input event  → grow the textarea height up to MAX_H, then scroll
 */
(function () {
  "use strict";

  var MAX_H = 120; // px — textarea stops growing and scrolls after this

  document.addEventListener("input", function (e) {
    if (e.target.id !== "new-question-input") return;
    var ta = e.target;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, MAX_H) + "px";
  });

  document.addEventListener("keydown", function (e) {
    if (e.target.id !== "new-question-input") return;
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      var btn = document.getElementById("ask-new-button");
      if (btn && !btn.disabled) btn.click();
    }
  });
})();
