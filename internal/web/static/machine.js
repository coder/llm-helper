// The machine page: the models this box is serving right now, what that costs
// it, and the saved models it could serve as well.
//
// Polls rather than streams: the events worth seeing here are minutes apart,
// and a poll survives the server restarting underneath it without any
// reconnect logic.
"use strict";

// True while an action is in flight, so the buttons stay disabled until the
// poll after it lands rather than inviting a second click.
let busy = false;

const el = {
  host: $("[data-host]"),
  build: $("[data-build]"),
  phase: $("[data-phase-label]"),
  summary: $("[data-summary]"),
  instances: $("[data-instances]"),
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
  logPick: $("[data-log-instance]"),
  follow: $("[data-follow]"),
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
  idle: "Nothing running",
};

// The page-level phase is the one most worth attention.
const PHASE_ORDER = ["failing", "loading", "stopped", "serving"];

function showError(message) {
  el.error.textContent = message;
  el.error.hidden = !message;
}

// showNotes reports what an action decided rather than what went wrong.
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
  el.notesTitle.textContent = title || "What this did";
  for (const line of lines) {
    const li = document.createElement("li");
    li.textContent = line;
    el.notesList.append(li);
  }
}

el.notesDismiss.addEventListener("click", () => {
  el.notes.hidden = true;
});

function overallPhase(list) {
  if (!list.length) return "idle";
  for (const p of PHASE_ORDER) {
    if (list.some((i) => i.phase === p)) return p;
  }
  return list[0].phase;
}

function button(label, onClick, quiet) {
  const b = document.createElement("button");
  b.className = quiet ? "btn btn-quiet" : "btn";
  b.textContent = label;
  b.addEventListener("click", onClick);
  return b;
}

function instanceRow(i, privileged) {
  const li = document.createElement("li");

  const head = document.createElement("div");
  head.className = "i-head";
  const phase = document.createElement("span");
  phase.className = "phase";
  phase.dataset.state = i.phase;
  phase.textContent = PHASES[i.phase] || i.phase;
  const name = document.createElement("span");
  name.className = "m-name";
  name.textContent = i.model && i.model !== "unknown" ? i.model : i.profile;
  head.append(phase, name);
  if (i.default) {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = "default";
    head.append(tag);
  }
  li.append(head);

  const bits = [`profile ${i.profile}`, `port ${i.port}`];
  if (i.phase === "serving") bits.push(`http ${i.http_status}`);
  if (i.restarts > 0) bits.push(`${i.restarts} restarts`);
  // The field keeps the previous run's verdict, so it only explains a unit
  // that is failing now.
  if (i.fail_reason && i.phase === "failing") bits.push(i.fail_reason);
  const meta = document.createElement("div");
  meta.className = "m-meta";
  meta.textContent = bits.join(" · ");
  li.append(meta);

  if (i.phase === "failing" && i.last_log) {
    const pre = document.createElement("pre");
    pre.className = "failure";
    pre.textContent = i.last_log;
    li.append(pre);
  }

  if (i.cluster) {
    const note = document.createElement("div");
    note.className = "m-meta";
    note.textContent = i.control_reason || "";
    li.append(note);
  }

  const canAct = privileged && !i.cluster && !busy;
  const label = i.model || i.profile;
  const actions = document.createElement("div");
  actions.className = "i-actions";
  const restart = button("Restart", () =>
    act(
      i.profile,
      "restart",
      `Restart ${label}? The model reloads, which takes a few minutes.`,
    ),
  );
  const stop = button(
    "Stop",
    () =>
      act(
        i.profile,
        "down",
        `Stop ${label}? Its endpoint goes down until you run it again.`,
      ),
    true,
  );
  restart.disabled = !canAct;
  stop.disabled = !canAct;
  actions.append(restart, stop);
  if (!i.default) {
    const def = button(
      "Make default",
      () => act(i.profile, "default", null),
      true,
    );
    def.disabled = !privileged || busy;
    actions.append(def);
  }
  li.append(actions);
  return li;
}

let instancesKey = null;
let running = [];

function renderStatus(s) {
  el.host.textContent = s.host || "";
  el.build.textContent = s.build || "";

  const list = s.instances || [];
  running = list.map((i) => i.profile);
  const phase = overallPhase(list);
  el.phase.textContent = PHASES[phase] || phase;
  el.phase.dataset.state = phase;

  if (list.length === 1) {
    const m = list[0].model;
    el.summary.textContent = m && m !== "unknown" ? m : list[0].profile;
  } else {
    el.summary.textContent = list.length
      ? `${list.length} models running`
      : "—";
  }

  el.privilege.hidden = s.privileged;
  if (!s.privileged) el.privilege.textContent = s.priv_reason;

  // Rebuilding on every poll would fight the pointer mid-click.
  const key = JSON.stringify([list, s.privileged, busy]);
  if (key !== instancesKey) {
    instancesKey = key;
    el.instances.innerHTML = "";
    for (const i of list) el.instances.append(instanceRow(i, s.privileged));
  }

  renderLogPick(list);

  document
    .querySelectorAll("[data-run]")
    .forEach(
      (b) =>
        (b.disabled = !s.privileged || busy || running.includes(b.dataset.run)),
    );
  document
    .querySelectorAll("[data-remove]")
    .forEach(
      (b) =>
        (b.disabled =
          !s.privileged || busy || running.includes(b.dataset.remove)),
    );
}

function renderLogPick(list) {
  const names = list.map((i) => i.profile);
  const current = el.logPick.value;
  const have = Array.from(el.logPick.options).map((o) => o.value);
  if (JSON.stringify(have) !== JSON.stringify(names)) {
    el.logPick.innerHTML = "";
    for (const n of names) {
      const opt = document.createElement("option");
      opt.value = n;
      opt.textContent = n;
      el.logPick.append(opt);
    }
    if (names.includes(current)) el.logPick.value = current;
  }
  el.logPick.hidden = names.length < 2;
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

function renderProfiles(list) {
  el.modelCount.textContent = list.length ? `(${list.length})` : "";
  if (!list.length) {
    el.models.innerHTML =
      '<li class="empty">No saved models yet. Find one under <strong>Add model</strong>.</li>';
    return;
  }
  el.models.innerHTML = "";
  for (const p of list) {
    const li = document.createElement("li");
    li.dataset.active = String(p.running);

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

    if (p.running) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = "running";
      li.append(tag);
    } else {
      const btn = document.createElement("button");
      btn.className = "btn";
      btn.textContent = "Run";
      btn.dataset.run = p.name;
      btn.addEventListener("click", () =>
        runProfile(p.name, p.model || p.name),
      );
      li.append(btn);

      // Removing a running profile is refused by the server anyway, and
      // offering a button that cannot work is worse than not offering it.
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

function instancePath(name, verb) {
  return `/api/instances/${encodeURIComponent(name)}/${verb}`;
}

async function runProfile(name, label) {
  if (
    !confirm(
      `Run ${label}?\n\nIt starts alongside any model already running, if it fits in memory. Loading can take several minutes.`,
    )
  ) {
    return;
  }
  busy = true;
  showError("");
  try {
    const res = await api(instancePath(name, "up"), { method: "POST" });
    showNotes(res.warnings, `Started ${label}`);
  } catch (e) {
    showError(e.message);
  } finally {
    busy = false;
    profilesKey = null;
    refresh();
  }
}

async function act(name, verb, question) {
  if (question && !confirm(question)) return;
  busy = true;
  showError("");
  try {
    const res = await api(instancePath(name, verb), { method: "POST" });
    if (res.warnings?.length) showNotes(res.warnings, `${name}: ${verb}`);
  } catch (e) {
    showError(e.message);
  } finally {
    busy = false;
    profilesKey = null;
    refresh();
  }
}

async function refreshLog() {
  if (!running.length) {
    el.log.textContent = "No model is running.";
    return;
  }
  const name = el.logPick.value || running[0];
  try {
    const { logs, hint } = await api(
      `/api/logs?n=300&instance=${encodeURIComponent(name)}`,
    );
    const atBottom =
      el.log.scrollTop + el.log.clientHeight >= el.log.scrollHeight - 40;
    el.log.textContent = logs || hint || "No log output yet.";
    if (el.follow.checked && atBottom) el.log.scrollTop = el.log.scrollHeight;
  } catch (e) {
    el.log.textContent = e.message;
  }
}

el.logPick.addEventListener("change", () => logLoop());

// Polling cadence. Checking the status reaches each model server, which writes
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
let profilesKey = null;

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
    phase = overallPhase(status.instances || []);
    // Rebuild the list only when what runs changed. Fetch first, record
    // second, so one failed fetch does not leave the page stuck.
    const key = (status.instances || []).map((i) => i.profile).join(",");
    if (key !== profilesKey) {
      const profiles = await api("/api/profiles");
      profilesKey = key;
      renderProfiles(profiles);
    }
    renderStatus(status);
    renderMachine(machine);
  } catch (e) {
    showError(e.message);
  }
  schedule(intervalFor(phase));
}

// The log is on its own clock. Reading it costs a journalctl call and
// nothing else -- it never touches a model server -- so it can stay live
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
    profilesKey = null; // force the list to rebuild without it
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

refresh().then(logLoop);
