// The add-model page: find something this machine can actually run.
//
// Everything here is shaped around one question -- will it fit, and will it
// start -- because that is what separates a model you can use from a 100 GB
// download that OOMs. Search results cannot answer it (the Hub's listing has
// no sizes), so picking one measures it, and the answer gets the whole right
// column rather than a line of small print.
"use strict";

const el = {
  budgetLine: $("[data-budget-line]"),
  form: $("[data-search-form]"),
  input: $("[data-search-input]"),
  btn: $("[data-search-btn]"),
  error: $("[data-search-error]"),
  hits: $("[data-hits]"),
  resultsNote: $("[data-results-note]"),

  empty: $("[data-detail-empty]"),
  detail: $("[data-detail]"),
  title: $("[data-detail-title]"),
  link: $("[data-detail-link]"),
  loading: $("[data-detail-loading]"),
  loadingText: $("[data-detail-loading-text]"),
  body: $("[data-detail-body]"),
  verdict: $("[data-verdict]"),
  facts: $("[data-facts]"),
  quants: $("[data-quants]"),
  quantsCount: $("[data-quants-count]"),
  quantSelect: $("[data-quant-select]"),
  quantNote: $("[data-quant-note]"),
  flags: $("[data-flags]"),
  thinking: $("[data-thinking]"),
  thinkingNote: $("[data-thinking-note]"),
  thinkingSelect: $("[data-thinking-select]"),
  dropped: $("[data-dropped]"),
  detailError: $("[data-detail-error]"),
  gguf: $("[data-gguf]"),
  ggufCmd: $("[data-gguf-cmd]"),
  ggufCaveats: $("[data-gguf-caveats]"),
  ggufInstall: $("[data-gguf-install]"),
  ggufStatus: $("[data-gguf-status]"),
  ggufLog: $("[data-gguf-log]"),
  pull: $("[data-pull]"),
  pullNote: $("[data-pull-note]"),
};

// The constraint is stated before any results, so the numbers below are read
// in context instead of being judged against nothing.
async function fillBudget() {
  try {
    const m = await api("/api/machine");
    if (!m.budget_known) return;
    // A measured 0 is the most important thing this line can say, not a
    // reason to stay quiet.
    el.budgetLine.textContent =
      m.budget_bytes > 0
        ? `This machine can serve models up to about ${m.budget_gb} GB of weights — ${m.budget_reason}`
        : `This machine has no room for model weights — ${m.budget_reason}`;
    el.budgetLine.hidden = false;
  } catch {
    // Not knowing the budget is survivable; the detail view says so too.
  }
}

// ---------- results ----------

function renderHits(results, query) {
  el.hits.innerHTML = "";
  if (!results.length) {
    el.resultsNote.textContent = `Nothing matched “${query}”. Try different terms.`;
    el.resultsNote.hidden = false;
    return;
  }
  el.resultsNote.textContent = `${results.length} results for “${query}”, most downloaded first.`;
  el.resultsNote.hidden = false;

  for (const r of results) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.className = "hit";
    btn.type = "button";
    btn.dataset.repo = r.id;

    const id = document.createElement("span");
    id.className = "hit-id";
    id.textContent = r.id;

    const meta = document.createElement("span");
    meta.className = "hit-meta";
    meta.textContent = `${fmtCount(r.downloads)} downloads · ${fmtCount(r.likes)} likes · ${r.license}`;

    btn.append(id, meta);
    if (r.quant_tags?.length) {
      const q = document.createElement("span");
      q.className = "hit-quant";
      q.textContent = r.quant_tags.join(" ");
      btn.append(q);
    }
    btn.addEventListener("click", () => select(btn, r.id));
    li.append(btn);
    el.hits.append(li);
  }
}

el.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = el.input.value.trim();
  if (!q) return;
  el.error.hidden = true;
  el.btn.disabled = true;
  el.btn.innerHTML = '<span class="spinner"></span>Searching…';
  el.hits.innerHTML = "";
  el.resultsNote.hidden = true;
  try {
    const { results } = await api(`/api/search?q=${encodeURIComponent(q)}`);
    renderHits(results, q);
  } catch (err) {
    el.error.textContent = err.message;
    el.error.hidden = false;
  } finally {
    el.btn.disabled = false;
    el.btn.textContent = "Search";
  }
});

// ---------- detail ----------

let selected = null;

// Whether the service can be controlled from here, filled by the status
// request the header already makes. Assumed available until told otherwise,
// so a slow status never leaves a usable button disabled.
let serviceControl = { ok: true, reason: "" };

function select(btn, repo) {
  document
    .querySelectorAll(".hit[aria-current]")
    .forEach((b) => b.removeAttribute("aria-current"));
  btn.setAttribute("aria-current", "true");
  inspect(repo);
}

function fact(term, value) {
  const dt = document.createElement("dt");
  dt.textContent = term;
  const dd = document.createElement("dd");
  dd.textContent = value;
  el.facts.append(dt, dd);
}

// The verdict repeats the CLI's wording on purpose. Someone who has read
// 'inspect' output in a terminal should not have to learn a second vocabulary
// for the same judgement, and the error direction matters: over-reporting a
// size is an annoyance, under-reporting it gets a huge download and an OOM.
function renderVerdict(d) {
  const of =
    d.variants > 1 ? `, smallest of ${d.variants} quants in this repo` : "";
  const set = (fit, headline, sub) => {
    el.verdict.dataset.fit = fit;
    el.verdict.innerHTML = "";
    const h = document.createElement("strong");
    h.textContent = headline;
    el.verdict.append(h);
    if (sub) {
      const p = document.createElement("span");
      p.textContent = sub;
      el.verdict.append(p);
    }
    // "Fits" on its own is a weaker answer than people read it as: a model can
    // fit and still have room for only a fraction of one request. Say what the
    // remaining memory actually buys.
    if (fit === "yes" && d.context_tokens) {
      const c = document.createElement("span");
      c.className = "verdict-context";
      const full = d.max_context
        ? ` — ${(d.context_tokens / d.max_context).toFixed(1)}× its ${fmtCount(d.max_context)} maximum`
        : "";
      c.textContent = `Room for about ${fmtCount(d.context_tokens)} tokens of context in ${d.kv_cache_gb.toFixed(1)} GB of KV cache${full}.`;
      el.verdict.append(c);
    }
  };

  if (d.size_error) {
    set("unknown", "Size could not be measured", d.size_error);
  } else if (!d.size_gb) {
    set(
      "unknown",
      "No servable weights found",
      "An adapter or projector-only repo, or the weights are named unusually — check the model card.",
    );
  } else if (d.fits_budget === undefined) {
    set(
      "unknown",
      `${d.size_gb.toFixed(1)} GB`,
      `No budget configured on this machine, so there is nothing to compare it against${of}.`,
    );
  } else if (!d.budget_gb) {
    set(
      "no",
      `Nothing fits — ${d.size_gb.toFixed(1)} GB`,
      `This machine has no room for model weights: ${d.budget_reason}.`,
    );
  } else if (d.fits_budget) {
    // When a repo holds several quants, the headline should be the best one
    // that fits rather than the smallest -- the smallest is the safe answer
    // to "could anything here run" and the worst thing to actually serve.
    if (d.best_variant) {
      set(
        "yes",
        `Fits — ${d.best_variant.size_gb.toFixed(1)} GB at ${d.best_variant.display || d.best_variant.label}`,
        `The largest of ${d.variants} quants that fits this machine's ~${Math.round(d.budget_gb)} GB budget. The smallest is ${d.size_gb.toFixed(1)} GB.`,
      );
    } else {
      set(
        "yes",
        `Fits — ${d.size_gb.toFixed(1)} GB`,
        `Under this machine's ~${Math.round(d.budget_gb)} GB budget${of}.`,
      );
    }
  } else {
    set(
      "no",
      `Too big — ${d.size_gb.toFixed(1)} GB`,
      `Over this machine's ~${Math.round(d.budget_gb)} GB budget${of}. You can still pull it, but expect it not to load.`,
    );
  }
}

// The quants a repo holds, as a control rather than a readout: a repo with a
// 2-bit and an 8-bit copy should not have that decided silently. The best fit
// is pre-selected so the common case is still one click.
//
// Options vLLM could not load are shown disabled rather than hidden, because
// "the 65 GB one is right there and greyed out" answers a question that its
// absence would only raise.
function renderQuants(d) {
  el.quantSelect.innerHTML = "";
  if (!d.all_variants || d.all_variants.length < 2) {
    el.quants.hidden = true;
    return;
  }
  el.quants.hidden = false;
  el.quantsCount.textContent = `(${d.all_variants.length})`;

  for (const v of d.all_variants) {
    const opt = document.createElement("option");
    // The value stays the full label: it is what vLLM is given.
    opt.value = v.label;
    opt.title = v.blocked ? `${v.label} — ${v.blocked}` : v.label;

    const name = (v.display || v.label).padEnd(26);
    const size = `${v.size_gb.toFixed(1)} GB`.padStart(8);
    // Only the head of a blocked reason: the full sentence is repeated on
    // every option otherwise, and the verdict above already gives it once.
    const why = v.blocked ? v.blocked.split(/, | -- /)[0] : "";
    if (v.blocked) {
      opt.disabled = true;
      opt.textContent = `${name} ${size}   can't load: ${why}`;
    } else if (v.gguf && !d.gguf_ready) {
      opt.disabled = true;
      opt.textContent = `${name} ${size}   needs the GGUF plugin`;
    } else if (d.budget_known && v.size_gb > d.budget_gb) {
      opt.textContent = `${name} ${size}   too big for this machine`;
    } else {
      opt.textContent = `${name} ${size}`;
    }
    if (d.best_variant && v.name === d.best_variant.name) {
      opt.selected = true;
    }
    el.quantSelect.append(opt);
  }

  el.quantNote.textContent = d.best_variant
    ? `${d.best_variant.display || d.best_variant.label} is the biggest this machine can serve.`
    : "None of these can be served on this machine.";
}

// showGGUFOffer fills in the install panel from the server, so the command
// and the caveats are the machine's own rather than a copy that could drift
// from what the CLI says.
async function showGGUFOffer() {
  el.gguf.hidden = false;
  el.ggufStatus.textContent = "";
  el.ggufLog.hidden = true;
  try {
    const g = await api("/api/gguf");
    el.ggufCmd.textContent = g.command || g.reason || "";
    el.ggufCaveats.innerHTML = "";
    for (const c of g.caveats || []) {
      const li = document.createElement("li");
      li.textContent = c;
      el.ggufCaveats.append(li);
    }
    // Nowhere to install into is not something a button can fix.
    el.ggufInstall.hidden = !g.command;

    // A build started before this page loaded is still the machine's build.
    // Rejoin it rather than offering to start a second one.
    if (g.installing) {
      el.ggufInstall.disabled = true;
      el.ggufInstall.innerHTML = '<span class="spinner"></span>Installing…';
      watchInstall();
    } else if (g.install_error) {
      el.ggufStatus.textContent = g.install_error;
      if (g.install_log) {
        el.ggufLog.textContent = g.install_log;
        el.ggufLog.hidden = false;
      }
    }
  } catch (err) {
    el.ggufStatus.textContent = err.message;
  }
}

// slowAfter is when a wait stops being a pause and starts being a worry. The
// first inspect on a machine can ask vLLM what it supports, which imports
// torch and takes seconds; saying so is better than a spinner that just keeps
// spinning.
const SLOW_AFTER_MS = 4000;

// INSTALL_POLL_MS is how often the page asks whether a build has finished.
// The work is minutes, so this is about keeping the elapsed counter honest
// rather than catching the moment it lands.
const INSTALL_POLL_MS = 3000;

let installPoll = null;

// watchInstall follows a build that is already running on the machine, which
// is what makes this survive a reload: the install belongs to the server, not
// to the page that started it, so a refreshed tab picks it back up.
function watchInstall() {
  clearInterval(installPoll);
  installPoll = setInterval(async () => {
    let g;
    try {
      g = await api("/api/gguf");
    } catch {
      return; // a blip mid-build is not worth abandoning the watch for
    }
    if (g.installing) {
      const mins = Math.floor((g.elapsed_seconds || 0) / 60);
      const secs = (g.elapsed_seconds || 0) % 60;
      el.ggufStatus.textContent = `Building from source… ${mins}m ${String(secs).padStart(2, "0")}s elapsed.`;
      return;
    }

    clearInterval(installPoll);
    installPoll = null;
    if (g.install_error) {
      el.ggufStatus.textContent = g.install_error;
      if (g.install_log) {
        el.ggufLog.textContent = g.install_log;
        el.ggufLog.hidden = false;
      }
      el.ggufInstall.disabled = false;
      el.ggufInstall.textContent = "Install the plugin";
      return;
    }
    // Re-measure the model that is on screen, so the quants it just became
    // possible to serve stop being listed as unavailable.
    el.ggufStatus.textContent = "Installed. Re-checking this model…";
    if (selected) inspect(selected);
  }, INSTALL_POLL_MS);
}

el.ggufInstall.addEventListener("click", async () => {
  if (
    !confirm(
      "Install the GGUF plugin?\n\nIt builds from source, so this can take several minutes. Nothing currently running is affected — the plugin is read the next time a model starts.",
    )
  ) {
    return;
  }
  el.ggufInstall.disabled = true;
  el.ggufInstall.innerHTML = '<span class="spinner"></span>Installing…';
  el.ggufLog.hidden = true;
  el.ggufStatus.textContent = "Starting…";
  try {
    const res = await api("/api/gguf/install", { method: "POST" });
    if (res.already) {
      // Nothing to wait for; just show the truth.
      if (selected) inspect(selected);
      return;
    }
    el.ggufStatus.textContent = res.message || "Building from source…";
    watchInstall();
  } catch (err) {
    el.ggufStatus.textContent = err.message;
    el.ggufInstall.disabled = false;
    el.ggufInstall.textContent = "Install the plugin";
  }
});

// A thinking model reasons before it answers, which reads as a slow reply
// unless it is said up front. The switch is only offered where a new profile
// can set it.
function renderThinking(d) {
  el.thinkingSelect.innerHTML = "";
  el.thinking.hidden = !d.thinking;
  if (!d.thinking) return;
  const def = d.thinking.default ? "on" : "off";
  el.thinkingNote.textContent = d.thinking.default
    ? `Thinking model: replies begin after a reasoning pass. ${d.thinking.why}`
    : `This model can reason before it answers, but does not unless asked. ${d.thinking.why}`;
  const offer = d.thinking_switch && !d.existing_profile;
  el.thinkingSelect.hidden = !offer;
  if (!offer) return;
  for (const [value, label] of [
    ["", `Model default (${def})`],
    ["on", "On"],
    ["off", "Off"],
  ]) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = label;
    el.thinkingSelect.append(opt);
  }
}

async function inspect(repo) {
  selected = repo;
  clearInterval(installPoll);
  installPoll = null;
  el.empty.hidden = true;
  el.detail.hidden = false;
  el.title.textContent = repo;
  el.link.href = `https://huggingface.co/${repo}`;
  el.loading.hidden = false;
  el.loadingText.textContent = "Measuring against this machine…";
  const slow = setTimeout(() => {
    if (selected === repo) {
      el.loadingText.textContent =
        "Still measuring. Reading the repo, and asking vLLM what it supports when it is the server.";
    }
  }, SLOW_AFTER_MS);
  el.body.hidden = true;
  el.detailError.hidden = true;
  el.pull.disabled = true;
  el.pull.hidden = false;
  el.pull.textContent = "Pull and switch";
  el.pullNote.hidden = true;
  el.gguf.hidden = true;
  el.thinking.hidden = true;
  el.ggufInstall.disabled = false;
  el.ggufInstall.textContent = "Install the plugin";

  try {
    const d = await api(`/api/inspect?repo=${encodeURIComponent(repo)}`);
    clearTimeout(slow);
    if (selected !== repo) return; // a newer pick won the race

    renderVerdict(d);

    el.facts.innerHTML = "";
    fact("license", d.license);
    fact("downloads", fmtCount(d.downloads));
    fact(
      "quant",
      d.quant_tags?.length ? d.quant_tags.join(", ") : "none in tags",
    );
    if (d.detected_max_model_len) {
      fact("context", Number(d.detected_max_model_len).toLocaleString());
    }

    renderQuants(d);

    // A repo that is entirely GGUF on a machine without the plugin cannot be
    // served at all, whichever quant is picked -- so offer to fix that instead
    // of a pull button that could only fail.
    // An architecture the plugin cannot map is not fixable by installing
    // anything, so offering the install here would be misleading.
    if (d.needs_gguf && d.arch_blocked) {
      el.dropped.hidden = false;
      el.dropped.textContent = `This model can't be served as GGUF — it is ${d.arch_blocked}`;
      el.pull.disabled = true;
      el.loading.hidden = true;
      el.body.hidden = false;
      return;
    }
    if (d.needs_gguf && !d.gguf_ready) {
      await showGGUFOffer();
      el.pull.hidden = true;
      el.loading.hidden = true;
      el.body.hidden = false;
      return;
    }
    el.gguf.hidden = true;
    el.pull.hidden = false;

    el.flags.hidden = !d.detected_flags;
    if (d.detected_flags) {
      el.flags.textContent = [
        `Serving flags found on the model card and accepted by your vLLM: ${d.detected_flags}`,
        ...(d.flag_notes ?? []),
      ].join("\n");
    }
    renderThinking(d);
    el.dropped.hidden = !d.dropped_flags?.length;
    if (d.dropped_flags?.length) {
      el.dropped.textContent = d.dropped_flags.join("\n");
    }

    // An existing profile means this is a switch, not a new download. Saying
    // so stops the button implying work that isn't about to happen.
    if (d.existing_profile) {
      el.pull.textContent = `Switch to saved “${d.existing_profile}”`;
      el.pullNote.textContent =
        "You already have a profile for this model, so its saved settings are used rather than fresh ones.";
      el.pullNote.hidden = false;
    }
    // Pulling ends in a restart, so it needs the same permission the machine
    // page's buttons do. Offering it without that produces a 403 on click;
    // saying why up front is the behaviour the rest of the panel promises.
    el.pull.disabled = !serviceControl.ok;
    if (!serviceControl.ok) {
      el.pullNote.textContent = serviceControl.reason;
      el.pullNote.hidden = false;
    }
    el.loading.hidden = true;
    el.body.hidden = false;
  } catch (err) {
    clearTimeout(slow);
    if (selected !== repo) return;
    el.loading.hidden = true;
    el.detailError.textContent = err.message;
    el.detailError.hidden = false;
  }
}

el.pull.addEventListener("click", async () => {
  const repo = selected;
  if (!repo) return;
  if (
    !confirm(
      `Pull ${repo} and switch to it?\n\nThis restarts the server and unloads the current model. The server downloads the weights as it starts, so a large model can take a long time before it serves.`,
    )
  ) {
    return;
  }
  el.pull.disabled = true;
  el.pull.innerHTML = '<span class="spinner"></span>Saving…';
  try {
    let path = `/api/pull?repo=${encodeURIComponent(repo)}`;
    if (!el.quants.hidden && el.quantSelect.value) {
      path += `&quant=${encodeURIComponent(el.quantSelect.value)}`;
    }
    if (!el.thinkingSelect.hidden && el.thinkingSelect.value) {
      path += `&thinking=${encodeURIComponent(el.thinkingSelect.value)}`;
    }
    const res = await api(path, { method: "POST" });
    // The interesting thing now is the machine coming up, which is the other
    // page's whole job -- so hand over to it rather than reporting from here.
    sessionStorage.setItem(
      "vllm-helper-notes",
      JSON.stringify({
        notes: res.notes,
        warnings: res.warnings,
        model: repo,
      }),
    );
    location.href = "/";
  } catch (err) {
    el.detailError.textContent = err.message;
    el.detailError.hidden = false;
    el.pull.disabled = false;
    el.pull.textContent = "Pull and switch";
  }
});

fillChrome().then((s) => {
  if (s) {
    serviceControl = s.cluster
      ? { ok: false, reason: s.control_reason || "" }
      : { ok: !!s.privileged, reason: s.priv_reason || "" };
  }
});
fillBudget();
el.input.focus();
