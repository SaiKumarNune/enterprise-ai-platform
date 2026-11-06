(function () {
  function indicator(form, on) {
    const el = form.querySelector(".loading");
    if (el) el.hidden = !on;
  }

  async function swap(targetSelector, html, swap) {
    const target = targetSelector ? document.querySelector(targetSelector) : null;
    if (!target) {
      document.body.innerHTML = html;
      bindPolling(document);
      return;
    }
    if (swap === "outerHTML") {
      target.outerHTML = html;
    } else {
      target.innerHTML = html;
    }
    bindPolling(document);
  }

  document.addEventListener("submit", async (event) => {
    const form = event.target;
    if (!(form instanceof HTMLFormElement) || !form.hasAttribute("hx-post")) {
      return;
    }
    event.preventDefault();
    indicator(form, true);
    try {
      const response = await fetch(form.getAttribute("hx-post"), {
        method: "POST",
        body: new FormData(form),
        headers: { "HX-Request": "true" },
      });
      const html = await response.text();
      await swap(form.getAttribute("hx-target"), html, form.getAttribute("hx-swap") || "innerHTML");
    } catch (_error) {
      await swap(
        form.getAttribute("hx-target"),
        '<p class="state error" role="alert">The request failed.</p>',
        form.getAttribute("hx-swap") || "innerHTML"
      );
    } finally {
      indicator(form, false);
    }
  });

  const timers = new Map();

  function bindPolling(root) {
    root.querySelectorAll("[hx-get][hx-trigger]").forEach((el) => {
      if (timers.has(el)) {
        return;
      }
      const trigger = el.getAttribute("hx-trigger") || "";
      const match = trigger.match(/every\s+(\d+(?:\.\d+)?)s/);
      if (!match) {
        return;
      }
      const url = el.getAttribute("hx-get");
      const ms = Math.max(1, Number(match[1])) * 1000;
      const id = window.setInterval(async () => {
        if (!document.body.contains(el)) {
          window.clearInterval(id);
          timers.delete(el);
          return;
        }
        try {
          const response = await fetch(url, { headers: { "HX-Request": "true" } });
          const html = await response.text();
          const swapInto = el.getAttribute("hx-target");
          const dest = swapInto ? document.querySelector(swapInto) : el;
          if (!dest) {
            return;
          }
          dest.outerHTML = html;
          window.clearInterval(id);
          timers.delete(el);
          bindPolling(document);
        } catch (_error) {
          /* keep polling */
        }
      }, ms);
      timers.set(el, id);
    });
  }

  document.addEventListener("DOMContentLoaded", () => bindPolling(document));
})();
