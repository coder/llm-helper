// Shared by every page: talking to the API, and the chrome that is the same
// everywhere. Page-specific behaviour lives in machine.js / models.js.
"use strict";

const $ = (sel) => document.querySelector(sel);

// The token arrives once in the URL and is exchanged for a cookie by the
// server. Strip it from the address bar so it isn't left in history or
// pasted into a chat along with a screenshot.
const tokenFromURL = new URLSearchParams(location.search).get("token");
if (tokenFromURL) {
  history.replaceState({}, "", location.pathname);
}

// The cookie is HttpOnly, so the page cannot read the token back out of it.
// After a reload -- or a navigation to another page, which is why this
// matters more now -- the URL no longer carries it either, so ask the server,
// which will answer a request the cookie already authenticated.
let cachedToken = tokenFromURL || "";
async function token() {
  if (!cachedToken) {
    const res = await fetch("/api/token");
    if (!res.ok) {
      throw new Error(
        "This tab is not authenticated. Reopen the URL printed by 'llm-helper serve'.",
      );
    }
    cachedToken = (await res.json()).token;
  }
  return cachedToken;
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.method && options.method !== "GET") {
    // Mutations must carry the token in a header. A cookie alone would let
    // any other site POST here with the browser attaching it.
    headers["X-VLLM-Helper-Token"] = await token();
  }
  const res = await fetch(path, { ...options, headers });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(body.error || `${res.status} ${res.statusText}`);
    // Carry the rest of the payload: a failed source build explains itself in
    // pip's last lines, and a browser has no journal to go and read.
    err.log = body.log;
    err.status = res.status;
    throw err;
  }
  return body;
}

function gb(mb) {
  return `${Math.round(mb / 1024)} GB`;
}

function fmtCount(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${Math.round(n / 1e3)}k`;
  return String(n);
}

// Both pages carry the machine's identity in the bar, so it is filled from one
// place. It is also the cheapest check that the page is talking to the server
// it thinks it is.
//
// Cached because host and build do not change between pages: without this,
// navigating blanks them for the length of a fetch and the header jitters.
const CHROME_KEY = "vllm-helper:chrome";

function paintChrome(c) {
  const host = $("[data-host]");
  const build = $("[data-build]");
  if (host && c.host !== undefined) host.textContent = c.host || "";
  if (build && c.build !== undefined) build.textContent = c.build || "";
}

async function fillChrome() {
  try {
    paintChrome(JSON.parse(sessionStorage.getItem(CHROME_KEY) || "{}"));
  } catch {
    // A malformed cache is not worth reporting: the fetch below is the
    // source of truth and will repaint in a moment.
  }
  try {
    const s = await api("/api/status");
    paintChrome(s);
    sessionStorage.setItem(
      CHROME_KEY,
      JSON.stringify({ host: s.host || "", build: s.build || "" }),
    );
    return s;
  } catch {
    return null;
  }
}
