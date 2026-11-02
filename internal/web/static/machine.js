// The machine page: what this box is serving right now, what that costs it,
// and the saved models it could serve instead.
//
// Polls rather than streams: the events worth seeing here are minutes apart,
// and a poll survives the server restarting underneath it without any
// reconnect logic.
"use strict";

// True while a switch/restart/stop is in flight, so the buttons stay disabled
// until the poll after it lands rather than inviting a second click.
let busy = false;

const el = {
  host: $("[data-host]"),
  build: $("[data-build]"),
  phase: $("[data-phase-label]"),
  model: $("[data-model]"),
  profile: $("[data-profile]"),
  detail: $("[data-detail]"),
  failure: $("[data-failure]"),
  privilege: $("[data-privilege]"),
  error: $("[data-error]"),
  rail: $("[data-rail]"),
  railUsed: $("[data-rail-used]"),
  railBudget: $("[data-rail-budget]"),
  used: $("[data-used]"),
  budget: $("[data-budget]"),
  total: $("[data-total]"),
  totalLabel: $("[data-total-label]"),
  gpu: $("[data-gpu]"),
  models: $("[data-models]"),
  modelCount: $("[data-model-count]"),
  log: $("[data-log]"),
  follow: $("[data-follow]"),
  restart: $("[data-restart]"),
  stop: $("[data-stop]"),
  notes: $("[data-notes]"),
  notesTitle: $("[data-notes-title]"),
  notesList: $("[data-notes-list]"),
  notesDismiss: $("[data-notes-dismiss]"),
};

const PHASES = {
  serving: "Serving",
  loading: "Loading model",
  failing: "Failing to start",
  stopped: "Stopped",
  idle: "No model selected",
};

function showError(message) {
  el.error.textContent = message;
  el.error.hidden = !message;
}

// showNotes reports what a switch decided rather than what went wrong.
//
// Separate from showError because these are not failures -- "serving the Q8_0
// quant", "patched the chat template" -- and colouring them red taught people
// to distrust a screen that was working correctly. It stays until dismissed,
// since it is the only record of choices made on the user's behalf, and a
// poll every thirty seconds must not wipe it out from under a reader.
function showNotes(items, title) {
  el.notesList.innerHTML = "";
  const lines = (items || []).filter((s) => s && s.trim());
  el.notes.hidden = lines.length === 0;
  if (!lines.length) return;
  el.notesTitle.textContent = title || "What this switch did";
  for (const line of lines) {
    const li = document.createElement("li");
    li.textContent = line;
    el.notesList.append(li);
  }
}

el.notesDismiss.addEventListener("click", () => {
  el.notes.hidden = true;
});

function renderStatus(s) {
  el.host.textContent = s.host || "";
  el.build.textContent = s.build || "";

  el.phase.textContent = PHASES[s.phase] || s.phase;
  el.phase.dataset.state = s.phase;

  el.model.textContent = s.model && s.model !== "unknown" ? s.model : "—";
  el.profile.textContent = s.profile ? `profile ${s.profile}` : "";

  const bits = [`port ${s.port}`];
  if (s.phase === "serving") bits.push(`http ${s.http_status}`);
  if (s.restarts > 0) bits.push(`${s.restarts} restarts`);
  // Only while it is failing: the field keeps the previous run's verdict, so
  // a unit that crashed once and is now serving would read "http 200 ·
  // exit-code". Same rule as the log below.
  if (s.fail_reason && s.phase === "failing") bits.push(s.fail_reason);
  el.detail.textContent = bits.join(" · ");

  // Only shown when it explains something. A stale "exit-code" on a unit
  // that is running again is noise.
  const failing = s.phase === "failing" && s.last_log;
  el.failure.hidden = !failing;
  if (failing) el.failure.textContent = s.last_log;

  // A recipe cluster is reported but not driven from here, so its reason
  // takes the same slot as a missing privilege. Removing a profile other than
  // the active one is still fine: it does not touch whatever is serving.
  const canControl = s.privileged && !s.cluster;
  el.privilege.hidden = canControl;
  if (!canControl) {
    el.privilege.textContent = s.cluster ? s.control_reason : s.priv_reason;
  }

  const canAct = canControl && !busy;
  el.restart.disabled = !canAct;
  el.stop.disabled = !canAct || s.phase === "stopped" || s.phase === "idle";
  document
    .querySelectorAll("[data-switch]")
    .forEach((b) => (b.disabled = !canAct || b.dataset.switch === s.profile));
  document
    .querySelectorAll("[data-remove]")
    .forEach(
      (b) =>
        (b.disabled = !s.privileged || busy || b.dataset.remove === s.profile),
    );
}

function renderMachine(m) {
  // One rail, one pool. The budget is measured against whichever memory the
  // weights land in, so the axis has to be that same memory: drawing a VRAM
  // budget over installed RAM puts the marker somewhere that means nothing.
  const pool = m.pool_bytes || 0;
  if (!pool) {
    el.rail.hidden = true;
    return;
  }
  el.rail.hidden = false;

  const pct = (b) => `${Math.min(100, Math.max(0, (b / pool) * 100))}%`;
  const decimalGB = (b) => `${Math.round(b / 1e9)} GB`;

  // Usage is a host-RAM reading, so it belongs on the rail only when the rail
  // is host RAM. Nothing here measures what a discrete card or the Metal
  // working set is holding, and showing the host's figure against either
  // would just be a wrong number.
  const measured = m.pool === "ram";
  const used = measured
    ? Math.max(0, pool - m.available_mem_mb * 1024 * 1024)
    : 0;
  el.railUsed.style.width = measured ? pct(used) : "0%";
  el.railBudget.style.left = pct(m.budget_bytes);

  el.used.textContent = measured ? decimalGB(used) : "not measured";
  el.budget.textContent = decimalGB(m.budget_bytes);
  el.total.textContent = decimalGB(pool);
  el.totalLabel.textContent = measured ? "total" : m.pool;
  el.gpu.textContent = m.gpu_name || "none detected";
  el.gpu.title = m.budget_reason || "";
}

function renderProfiles(list, activeName) {
  el.modelCount.textContent = list.length ? `(${list.length})` : "";
  if (!list.length) {
    el.models.innerHTML =
      '<li class="empty">No saved models yet. Find one under <strong>Add model</strong>.</li>';
    return;
  }
  el.models.innerHTML = "";
  for (const p of list) {
    const li = document.createElement("li");
    li.dataset.active = String(p.active);

    const main = document.createElement("div");
    main.className = "m-main";
    const name = document.createElement("div");
    name.className = "m-name";
    name.textContent = p.model || p.name;
    const meta = document.createElement("div");
    meta.className = "m-meta";
    meta.textContent = [
      p.name,
      p.max_model_len
        ? `${Number(p.max_model_len).toLocaleString()} ctx`
        : null,
      p.gpu_memory_utilization ? `mem ${p.gpu_memory_utilization}` : null,
    ]
      .filter(Boolean)
      .join(" · ");
    main.append(name, meta);
    li.append(main);

    if (p.active) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = "active";
      li.append(tag);
    } else {
      const btn = document.createElement("button");
      btn.className = "btn";
      btn.textContent = "Switch";
      btn.dataset.switch = p.name;
      btn.addEventListener("click", () => switchTo(p.name, p.model || p.name));
      li.append(btn);

      // Only inactive profiles get this. Removing the running one is
      // refused by the server anyway, and offering a button that cannot
      // work is worse than not offering it.
      const rm = document.createElement("button");
      rm.className = "m-remove";
      rm.textContent = "Remove";
      rm.dataset.remove = p.name;
      rm.addEventListener("click", () => removeProfile(p.name));
      li.append(rm);
    }
    el.models.append(li);
  }
}

async function switchTo(name, label) {
  if (
    !confirm(
      `Switch to ${label}?\n\nThis restarts the server and unloads the current model. Loading the new one can take several minutes.`,
    )
  ) {
    return;
  }
  busy = true;
  showError("");
  try {
    const res = await api(`/api/switch/${encodeURIComponent(name)}`, {
      method: "POST",
    });
    // Plugin output, not a failure -- the switch succeeded or we would be in
    // the catch below.
    showNotes(res.warnings, `Switched to ${label}`);
  } catch (e) {
    showError(e.message);
  } finally {
    busy = false;
    refresh();
  }
}

async function act(path, question) {
  if (question && !confirm(question)) return;
  busy = true;
  showError("");
  try {
    await api(path, { method: "POST" });
  } catch (e) {
    showError(e.message);
  } finally {
    busy = false;
    refresh();
  }
}

el.restart.addEventListener("click", () =>
  act(
    "/api/restart",
    "Restart the server? The model reloads, which takes a few minutes.",
  ),
);
el.stop.addEventListener("click", () =>
  act(
    "/api/stop",
    "Stop the server? The endpoint goes down until you start it again.",
  ),
);

async function refreshLog() {
  try {
    const { logs, hint } = await api("/api/logs?n=300");
    const atBottom =
      el.log.scrollTop + el.log.clientHeight >= el.log.scrollHeight - 40;
    el.log.textContent = logs || hint || "No log output yet.";
    if (el.follow.checked && atBottom) el.log.scrollTop = el.log.scrollHeight;
  } catch (e) {
    el.log.textContent = e.message;
  }
}

// Polling cadence. Checking the status reaches the vLLM server, which writes
// a line to its own log for every request it answers, so a tab left open on a
// working machine must not poll at conversational speed -- it drowns the log
// someone is reading to debug something else. Poll quickly only while
// something is actually changing.
const FAST_MS = 2000; // a load or a crash loop: the screen changes second to second
const IDLE_MS = 30000; // steady state: nothing moves here unless someone asks it to

function intervalFor(phase) {
  return phase === "loading" || phase === "failing" ? FAST_MS : IDLE_MS;
}

let timer = null;
let activeProfile = null;

// Chained timeouts rather than an interval, so a slow poll delays the next
// one instead of stacking up behind it.
function schedule(ms) {
  clearTimeout(timer);
  // A hidden tab polls not at all; visibilitychange starts it again.
  if (document.hidden) return;
  timer = setTimeout(refresh, ms);
}

async function refresh() {
  // Assume steady state, so a failure to reach our own server backs off
  // rather than retrying hard against something that is down.
  let phase = "stopped";
  try {
    const [status, machine] = await Promise.all([
      api("/api/status"),
      api("/api/machine"),
    ]);
    phase = status.phase;
    // Rebuild the list only when the active model changed. Doing it on
    // every poll would fight the pointer mid-click.
    if (status.profile !== activeProfile) {
      // Fetch first, record second. Assigning before the request settles
      // means one failed fetch leaves the old list on screen and every later
      // poll skipping the rebuild, so the page never recovers on its own.
      const profiles = await api("/api/profiles");
      activeProfile = status.profile;
      renderProfiles(profiles, status.profile);
    }
    renderStatus(status);
    renderMachine(machine);
  } catch (e) {
    showError(e.message);
  }
  schedule(intervalFor(phase));
}

// The log is on its own clock. Reading it costs a journalctl call and
// nothing else -- it never touches the vLLM server -- so it can stay live
// while someone is following it without adding a line to the log they are
// reading.
const LOG_FOLLOW_MS = 2000;
const LOG_IDLE_MS = 15000;

let logTimer = null;

function scheduleLog() {
  clearTimeout(logTimer);
  if (document.hidden) return;
  logTimer = setTimeout(
    logLoop,
    el.follow.checked ? LOG_FOLLOW_MS : LOG_IDLE_MS,
  );
}

async function logLoop() {
  await refreshLog();
  scheduleLog();
}

// Ticking the box should show the tail now, not in fifteen seconds.
el.follow.addEventListener("change", () => {
  if (el.follow.checked) logLoop();
});

// Coming back to the tab should show the truth immediately, not whatever was
// on screen when it was hidden.
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    clearTimeout(timer);
    clearTimeout(logTimer);
  } else {
    refresh();
    logLoop();
  }
});

async function removeProfile(name) {
  if (
    !confirm(
      `Delete the saved profile '${name}'?

This removes its settings and its Coder model entry. The downloaded weights stay in your HuggingFace cache.`,
    )
  ) {
    return;
  }
  try {
    const res = await api(`/api/profiles/${encodeURIComponent(name)}/remove`, {
      method: "POST",
    });
    // A plugin that could not clean up leaves something behind, so say so:
    // the profile is gone either way, but a Coder entry pointing at it is
    // not, and nothing else would ever mention it.
    showNotes(res.warnings, `Removed '${name}'`);
    activeProfile = null; // force the list to rebuild without it
    refresh();
  } catch (err) {
    showError(err.message);
  }
}

// A pull on the add-model page navigates here, since what matters next is
// this machine coming up. Carry its notes across so they are not lost in the
// navigation.
const handover = sessionStorage.getItem("vllm-helper-notes");
if (handover !== null) {
  sessionStorage.removeItem("vllm-helper-notes");
  try {
    const { notes, warnings, model } = JSON.parse(handover);
    showNotes(
      [...(notes || []), ...(warnings || [])],
      model ? `Pulled ${model}` : "What this pull did",
    );
  } catch {
    // An older tab may have left a plain string behind.
    if (handover) showNotes([handover]);
  }
}

refresh();
logLoop();
