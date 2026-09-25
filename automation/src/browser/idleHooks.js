/**
 * In-page XHR/fetch pending counter (ADR-0001).
 * Loaded via Playwright addInitScript({ path }) so tsx/esbuild never rewrites it.
 */
(() => {
  const w = window;
  if (w.__efHooked) return;
  w.__efHooked = true;
  w.__efPending = 0;

  const bump = (delta) => {
    w.__efPending = Math.max(0, (w.__efPending || 0) + delta);
  };

  const originalFetch = window.fetch.bind(window);
  window.fetch = async (...args) => {
    bump(1);
    try {
      return await originalFetch(...args);
    } finally {
      bump(-1);
    }
  };

  const XO = XMLHttpRequest.prototype;
  const open = XO.open;
  const send = XO.send;
  XO.open = function (method, url, ...rest) {
    this.__efTrack = true;
    return open.apply(this, [method, url, ...rest]);
  };
  XO.send = function (body) {
    if (this.__efTrack) {
      bump(1);
      const done = () => {
        this.removeEventListener("loadend", done);
        bump(-1);
      };
      this.addEventListener("loadend", done);
    }
    return send.call(this, body);
  };
})();
