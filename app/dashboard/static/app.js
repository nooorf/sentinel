"use strict";
/* Sentinel console. Every figure on the page comes from the API it is served by. */

const FEATURES = [...Array.from({ length: 28 }, (_, i) => `V${i + 1}`), "Amount"];
const VIEWS = ["overview", "score", "monitoring", "models"];
const TERMINAL = new Set(["promoted", "rejected", "failed", "skipped"]);
const S = { eval: null, info: null, examples: [], selected: 0, session: [] };

const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const nf = new Intl.NumberFormat("en-GB");
const eur = new Intl.NumberFormat("en-GB", { style: "currency", currency: "EUR", maximumFractionDigits: 0 });
const eur2 = new Intl.NumberFormat("en-GB", { style: "currency", currency: "EUR", minimumFractionDigits: 2 });
const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
const fx = (x, d = 3) => (x == null ? "–" : Number(x).toFixed(d));
const shortV = (v) => (v ? v.split("-").pop() : "–");
const when = (iso) => (iso ? new Date(iso).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "–");
const ordinal = (n) => n + (["th", "st", "nd", "rd"][(n % 100 >= 11 && n % 100 <= 13) || n % 10 > 3 ? 0 : n % 10]);
const MODEL_NAMES = { xgboost: "XGBoost", ensemble: "XGBoost + Isolation Forest" };

async function api(path, opts = {}) {
  const init = { ...opts, headers: { "content-type": "application/json", ...(opts.headers || {}) } };
  if (init.body && typeof init.body !== "string") init.body = JSON.stringify(init.body);
  let r;
  try { r = await fetch(path, init); } catch { throw new Error("The Sentinel API is unreachable. Check that the service is running."); }
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : body.error || `Request failed (${r.status})`);
  return body;
}

function banner(msg) { const b = $("#banner"); b.textContent = msg || ""; b.hidden = !msg; }
function busy(btn, on) { btn.disabled = on; btn.setAttribute("aria-busy", on ? "true" : "false"); }

/* ------------------------------------------------------------------ routing */
function route() {
  const v = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "overview";
  VIEWS.forEach((name) => { $(`#view-${name}`).hidden = name !== v; });
  document.querySelectorAll(".tabs a").forEach((a) => a.toggleAttribute("aria-current", a.dataset.view === v) && a.setAttribute("aria-current", "page"));
  document.title = `Sentinel · ${v[0].toUpperCase()}${v.slice(1)}`;
  if (v === "models") loadModels();
  if (v === "monitoring") loadMonitoring();
}

/* ------------------------------------------------------------------ health */
async function health() {
  const el = $("#status");
  try {
    const h = await api("/health");
    el.className = `status ${h.model_loaded ? "is-up" : "is-down"}`;
    const lat = h.model_loaded ? (await api("/metrics").catch(() => null))?.predict_latency_ms : null;
    $(".status__text", el).textContent = h.model_loaded ? `Live · model ${shortV(h.model_version)}${lat ? ` · median ${lat.p50.toFixed(1)} ms` : ""}` : "No model loaded";
    if (!h.model_loaded) banner("No model is loaded yet. Train one with python scripts/train.py, then restart the API.");
    return h;
  } catch (e) {
    el.className = "status is-down";
    $(".status__text", el).textContent = "API unreachable";
    banner(e.message);
    return null;
  }
}

/* ------------------------------------------------------------------ overview */
function pairRow(label, base, sent, fmt) {
  const max = Math.max(base, sent) || 1;
  const wb = (base / max) * 82, ws = (sent / max) * 82;
  const delta = base ? sent / base - 1 : 0;
  return `<div class="pair">
    <div class="pair__label">${esc(label)}</div>
    <div class="pair__bars">
      <div class="bar" style="--w:${wb}%"><span class="bar__fill"></span><span class="bar__value">${fmt(base)}</span></div>
      <div class="bar bar--sentinel" style="--w:${wb}%" data-w="${ws}%"><span class="bar__fill"></span><span class="bar__value">${fmt(sent)}</span></div>
    </div>
    <div class="pair__delta">${delta <= 0 ? "−" : "+"}${Math.abs(Math.round(delta * 100))}%<small>vs baseline</small></div>
  </div>`;
}

function figure(term, value, note) {
  return `<div><dt>${esc(term)}</dt><dd>${value}${note ? `<small>${esc(note)}</small>` : ""}</dd></div>`;
}

function renderOverview(metrics) {
  const ev = S.eval, dm = ev.decision_model;
  const B = ev.models.baseline.test, T = ev.models[dm].test;
  const BV = ev.models.baseline.test_value, TV = ev.models[dm].test_value;
  const fpCut = B.fp ? 1 - T.fp / B.fp : 0;

  $("#ov-title").textContent = fpCut > 0 ? `Sentinel cuts false alarms by ${pct(fpCut)}` : "Sentinel fraud screening results";
  $("#ov-standfirst").textContent = `Held-out test window of ${nf.format(T.n)} card transactions, ${T.n_fraud} of them fraud. ` +
    `Sentinel's ${MODEL_NAMES[dm] || dm} model against a logistic-regression baseline, both tuned on validation data to an ${pct(ev.target_recall)} recall target.`;

  const cuts = [T.fp < B.fp, T.tp + T.fp < B.tp + B.fp, !BV || TV.legit_value_flagged < BV.legit_value_flagged];
  $("#ov-chart-title").textContent = cuts.every(Boolean) ? "Fewer false alarms, fewer reviews, less customer spend held"
    : "Sentinel vs. the baseline model, held-out test window";
  const rows = [pairRow("False alarms on legitimate transactions", B.fp, T.fp, nf.format),
                pairRow("Transactions sent to manual review", B.tp + B.fp, T.tp + T.fp, nf.format)];
  if (BV && TV) rows.push(pairRow("Legitimate customer spend held for review", BV.legit_value_flagged, TV.legit_value_flagged, eur.format));
  const pairs = $("#ov-pairs");
  pairs.classList.add("is-intro");
  pairs.innerHTML = rows.join("");
  requestAnimationFrame(() => requestAnimationFrame(() => {   // Sentinel bars contract from the baseline length to their own
    pairs.classList.remove("is-intro");
    pairs.querySelectorAll(".bar--sentinel").forEach((b) => b.style.setProperty("--w", b.dataset.w));
  }));
  $("#ov-source").textContent = `Source: Sentinel evaluation of model ${S.info.version}, test split never used for training or tuning.`;

  const lat = ev.model_only_latency_ms_single;
  const live = metrics && metrics.predict_latency_ms;
  const figs = [
    figure("Precision of review queue", fx(T.precision), `baseline ${fx(B.precision)}`),
    figure("PR-AUC", fx(T.pr_auc), `baseline ${fx(B.pr_auc)}`),
  ];
  if (TV) {
    figs.push(figure("Fraud value caught", pct(TV.fraud_value_caught_share), `${eur.format(TV.fraud_value_caught)} of ${eur.format(TV.fraud_value_total)}`));
    figs.push(figure("Fraud cost vs. no screening", `−${pct(1 - TV.total_cost / TV.fraud_value_total)}`, `missed fraud + ${eur.format(TV.review_cost)} per review`));
  }
  if (lat) figs.push(figure("Model latency, p95", `${lat.p95.toFixed(1)} ms`, "per transaction"));
  if (live) figs.push(figure("Live API latency, median", `${live.p50.toFixed(1)} ms`, `last ${nf.format(live.n)} predictions`));
  $("#ov-figures").innerHTML = figs.join("");

  renderPR(ev, dm);
  renderQueue(B, T);
}

function renderQueue(B, T) {
  const qb = B.tp + B.fp, qs = T.tp + T.fp, max = Math.max(qb, qs) || 1;
  const row = (name, sub, m, q) => `<div class="qrow"><div class="qrow__name">${esc(name)}<small>${esc(sub)}</small></div><div>
      <div class="qbar" style="--w:${(q / max) * 100}%" role="img" aria-label="${q} reviews: ${m.tp} fraud, ${m.fp} legitimate">
        <i class="f" style="flex:${m.tp}">${m.tp} fraud</i><i class="l" style="flex:${m.fp}">${m.fp} legitimate</i></div>
      <div class="qnote">${nf.format(q)} reviews · ${pct(m.precision)} fraud</div></div></div>`;
  $("#ov-queue").innerHTML = row("Baseline", "logistic regression", B, qb) + row("Sentinel", MODEL_NAMES[S.eval.decision_model] || "", T, qs);
  $("#ov-out-title").textContent = qs < qb && T.precision > 0.5
    ? `Analysts review ${pct(1 - qs / qb)} fewer transactions, and most of them are fraud` : "The analysts' review queue";
}

function renderPR(ev, dm) {
  const W = 620, H = 380, m = { l: 66, r: 16, t: 14, b: 46 };
  const x = (r) => m.l + r * (W - m.l - m.r), y = (p) => m.t + (1 - p) * (H - m.t - m.b);
  const path = (c) => c.recall.map((r, i) => `${i ? "L" : "M"}${x(r).toFixed(1)},${y(c.precision[i]).toFixed(1)}`).join("");
  const B = ev.models.baseline, T = ev.models[dm];
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const grid = ticks.map((t) => `<line x1="${m.l}" x2="${W - m.r}" y1="${y(t)}" y2="${y(t)}"/>`).join("");
  const xl = ticks.map((t) => `<text x="${x(t)}" y="${H - m.b + 18}" text-anchor="middle">${t * 100}%</text>`).join("");
  const yl = ticks.map((t) => `<text x="${m.l - 8}" y="${y(t) + 4}" text-anchor="end">${t * 100}%</text>`).join("");
  const op = (mm, cls) => ({ cx: x(mm.test.recall), cy: y(mm.test.precision), cls });
  const s = op(T, "sentinel"), b = op(B, "base");
  const sx = Math.max(m.l + 150, s.cx - 190), sy = Math.max(m.t + 34, s.cy - 70);
  const bx = Math.max(m.l + 150, b.cx - 170), by = Math.min(H - m.b - 20, b.cy + 62);
  $("#ov-pr").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Precision-recall curves for Sentinel and the baseline">
    <g class="grid">${grid}</g><g class="axis">${xl}${yl}
      <text x="${(W + m.l) / 2}" y="${H - 6}" text-anchor="middle">Recall: share of fraud caught</text>
      <text transform="translate(12 ${(H - m.b) / 2}) rotate(-90)" text-anchor="middle">Precision: share of alerts that are fraud</text></g>
    <path class="curve curve--base" d="${path(B.pr_curve_test)}"/>
    <path class="curve curve--sentinel" d="${path(T.pr_curve_test)}"/>
    <line class="lead" x1="${s.cx}" y1="${s.cy}" x2="${sx + 4}" y2="${sy + 6}"/>
    <text class="note" x="${sx}" y="${sy - 10}" text-anchor="end">Sentinel operating point</text>
    <text class="note note--sub" x="${sx}" y="${sy + 6}" text-anchor="end">${pct(T.test.precision)} precision at ${pct(T.test.recall)} recall</text>
    <line class="lead" x1="${b.cx}" y1="${b.cy}" x2="${bx + 4}" y2="${by - 14}"/>
    <text class="note" x="${bx}" y="${by - 4}" text-anchor="end">Baseline</text>
    <text class="note note--sub" x="${bx}" y="${by + 12}" text-anchor="end">${pct(B.test.precision)} precision at ${pct(B.test.recall)} recall</text>
    <circle class="op op--base" cx="${b.cx}" cy="${b.cy}" r="6"/><circle class="op op--sentinel" cx="${s.cx}" cy="${s.cy}" r="7"/>
  </svg>`;
  $("#ov-pr-sub").textContent = `Area under the curve: ${fx(T.test.pr_auc)} for Sentinel, ${fx(B.test.pr_auc)} for the baseline.`;
  const c = T.pr_curve_test;   // highest recall reached with every alert still being fraud
  const perfect = Math.max(0, ...c.recall.filter((r, i) => c.precision[i] >= 0.999));
  $("#ov-pr-title").textContent = perfect >= 0.25 ? `Every alert is fraud until Sentinel has caught ${pct(perfect)} of all fraud`
    : "Precision and recall on unseen transactions";
}

/* ------------------------------------------------------------------ scoring */
const exName = (i) => `Test transaction ${String.fromCharCode(65 + i)}`;
const exLabel = (e) => (e.label ? "Fraud" : "Legitimate");

function renderExamples() {
  const opts = S.examples.map((e, i) => `<option value="${i}">${exName(i)} · ${eur2.format(e.features.Amount)} · ${exLabel(e).toLowerCase()}</option>`).join("");
  $("#ov-example").innerHTML = opts;
  const firstFraud = Math.max(0, S.examples.findIndex((e) => e.label === 1));
  $("#ov-example").value = String(firstFraud);
  S.selected = firstFraud;
  $("#sc-table tbody").innerHTML = S.examples.map((e, i) => `<tr data-i="${i}">
      <td><input type="radio" name="ex" value="${i}" aria-label="${exName(i)}"></td><td>${exName(i)}</td>
      <td class="num">${eur2.format(e.features.Amount)}</td><td><span class="tag tag--${e.label ? "fraud" : "legit"}">${exLabel(e)}</span></td></tr>`).join("");
  selectExample(firstFraud);
}

function selectExample(i) {
  S.selected = i;
  document.querySelectorAll("#sc-table tbody tr").forEach((tr) => {
    const on = Number(tr.dataset.i) === i;
    tr.classList.toggle("is-selected", on);
    $("input", tr).checked = on;
  });
  const f = S.examples[i].features;
  $("#sc-features .features__grid").innerHTML = FEATURES.map((k) => `<div><span>${k}</span><b>${k === "Amount" ? eur2.format(f[k]) : f[k].toFixed(3)}</b></div>`).join("");
}

async function score(i) {
  const ex = S.examples[i];
  const t0 = performance.now();
  const res = await api("/predict", { method: "POST", body: ex.features });
  const ms = performance.now() - t0;
  S.session.unshift({ at: new Date(), i, ex, res, ms });
  renderSession();
  return { res, ms, ex };
}

function verdictHTML({ res, ms, ex }) {
  const thr = S.info.threshold;
  const review = res.decision === "REVIEW";
  return `<div class="decision decision--${review ? "review" : "approve"}">
      <span class="decision__word">${review ? "Send to review" : "Approve"}</span>
      <span class="decision__meta">${esc(res.risk_category)} risk<br>${ms.toFixed(0)} ms round trip</span></div>
    <div class="scale" style="--t:${(thr * 100).toFixed(2)}%;--p:${(res.decision_score * 100).toFixed(2)}%" aria-hidden="true">
      <div class="scale__track"></div><div class="scale__thr"><span>Review threshold ${thr.toFixed(2)}</span></div>
      <div class="scale__marker${res.decision_score > 0.85 ? " is-high" : res.decision_score < 0.15 ? " is-low" : ""}"><span>${res.decision_score.toFixed(3)}</span></div>
      <div class="scale__ends"><span>0</span><span>1</span></div></div>
    <dl class="figures">
      ${figure("Fraud probability", res.fraud_probability.toFixed(4), "XGBoost")}
      ${figure("Anomaly rank", ordinal(Math.min(99, Math.floor(res.anomaly_score * 100))), "percentile vs. training traffic")}
      ${figure("Actual outcome", exLabel(ex), "label in the dataset")}
      ${figure("Model", `<span class="mono">${esc(shortV(res.model_version))}</span>`, MODEL_NAMES[res.decision_model] || res.decision_model)}
    </dl>`;
}

function renderSession() {
  const tb = $("#sc-session tbody");
  if (!S.session.length) return;
  tb.innerHTML = S.session.slice(0, 12).map((s) => `<tr><td>${s.at.toLocaleTimeString("en-GB")}</td><td>${exName(s.i)}</td>
      <td class="num">${eur2.format(s.ex.features.Amount)}</td><td>${exLabel(s.ex)}</td><td class="num">${s.res.fraud_probability.toFixed(4)}</td>
      <td><span class="tag tag--${s.res.decision === "REVIEW" ? "review" : "approve"}">${s.res.decision === "REVIEW" ? "Review" : "Approve"}</span></td>
      <td class="num">${s.ms.toFixed(0)} ms</td></tr>`).join("");
}

/* ------------------------------------------------------------------ monitoring */
function driftRow(name, sub, rep) {
  const fd = rep.feature_drift;
  const cells = FEATURES.map((f) => {
    const c = fd.columns[f] || {};
    return `<i class="${c.drift_detected ? "on" : ""}" title="${f}: ${c.drift_detected ? "drift detected" : "stable"}"></i>`;
  }).join("");
  return `<div class="drift__row"><div class="drift__name">${esc(name)}<small>${esc(sub)}</small></div>
    <div class="drift__cells" role="img" aria-label="${fd.n_drifted} of ${fd.n_features} features drifted">${cells}</div>
    <div class="drift__count">${fd.n_drifted}/${fd.n_features}<small>${rep.drift_detected ? '<span class="alert">alert raised</span>' : "no alert"}</small></div></div>`;
}
const driftAxis = () => `<div class="drift__axis" aria-hidden="true"><span></span><div class="drift__ticks">${FEATURES.map((f) => `<span>${f === "Amount" ? "Amt" : f}</span>`).join("")}</div><span></span></div>`;

async function loadMonitoring() {
  if (S.simulated) return;
  try {
    const st = await api("/monitor/status");
    const r = st.latest_report;
    if (r && r.status === "ok") {
      $("#mo-drift").innerHTML = driftRow("Latest check", when(r.created_at), r) + driftAxis() + driftFinding(r, null);
      $("#mo-source").textContent = `Source: Evidently DataDriftPreset against ${nf.format(r.reference.rows)} reference transactions; ${nf.format(r.current.rows)} transactions in the window.`;
    }
  } catch (e) { banner(e.message); }
}

function driftFinding(after, sim) {
  const fd = after.feature_drift;
  const thr = Math.ceil(after.config.share_threshold * fd.n_features);
  const head = after.drift_detected ? `Drift alert raised: ${fd.n_drifted} of ${fd.n_features} features moved` : `Traffic is stable: ${fd.n_drifted} of ${fd.n_features} features moved, below the alert level`;
  const lines = [];
  if (after.prediction_drift?.drift_detected) lines.push(`<span class="alert">Model scores shifted as well.</span>`);
  if (sim?.retraining?.triggered) lines.push(`Retraining started automatically: run <span class="mono">${esc(sim.retraining.run_id)}</span>. <a href="#models">Follow it under Models</a>.`);
  else if (sim?.retraining && sim.retraining.reason) lines.push(esc(sim.retraining.reason));
  lines.unshift(`An alert is raised when ${thr} or more of the ${fd.n_features} features drift.`);
  $("#mo-chart-title").textContent = head;
  return `${lines.length ? `<p>${lines.join(" ")}</p>` : ""}
    <div class="drift__links"><a href="/monitor/reports/${encodeURIComponent(after.id)}/html?kind=features" target="_blank" rel="noopener">Open the Evidently feature report</a>
    <a href="/monitor/reports/${encodeURIComponent(after.id)}/html?kind=prediction" target="_blank" rel="noopener">Open the prediction report</a></div>`;
}

async function runSimulation(ev) {
  ev.preventDefault();
  const f = ev.target, btn = $("#mo-run");
  const mag = Number(f.magnitude.value), n = Number(f.n.value);
  busy(btn, true);
  try {
    const sim = await api("/monitor/simulate", { method: "POST", body: { n, magnitude: mag, amount_factor: 3, auto_trigger: f.auto.checked } });
    S.simulated = true;
    const b = sim.before, a = sim.after;
    if (b.status !== "ok" || a.status !== "ok") throw new Error(a.message || b.message || "The drift check did not complete.");
    $("#mo-drift").innerHTML = driftRow("Unchanged batch", `${nf.format(n)} held-out transactions`, b) +
      driftRow("Shifted batch", `V1–V10 moved ${mag}σ, amounts ×3`, a) + driftAxis() + driftFinding(a, sim);
    $("#mo-source").textContent = `Source: Evidently DataDriftPreset against ${nf.format(a.reference.rows)} reference transactions from the serving model's validation data. Batches are simulated from held-out traffic.`;
  } catch (e) { banner(e.message); }
  finally { busy(btn, false); }
}

/* ------------------------------------------------------------------ models */
async function loadModels() {
  try {
    const [reg, rs] = await Promise.all([api("/model/versions"), api("/retraining/status?limit=20")]);
    const vs = Object.entries(reg.versions).sort((a, b) => (a[1].registered_at < b[1].registered_at ? 1 : -1));
    $("#md-versions tbody").innerHTML = vs.length ? vs.map(([v, r]) => {
      const st = v === reg.active ? "active" : r.status;
      const note = (r.history || []).slice(-1)[0]?.reason || "";
      return `<tr><td class="mono">${esc(v)}</td><td><span class="tag tag--${esc(st)}">${esc(st.replace("_", " "))}</span></td>
        <td>${when(r.registered_at)}</td><td>${esc(note)}</td></tr>`;
    }).join("") : `<tr class="table__empty"><td colspan="4">No versions registered.</td></tr>`;
    $("#md-runs tbody").innerHTML = rs.runs.length ? rs.runs.map((r) => `<tr><td>${when(r.requested_at)}</td><td>${esc(r.trigger)}</td>
        <td><span class="tag tag--${esc(r.status)}">${esc(r.status)}</span></td><td>${esc(r.decision || r.error || r.reason || "")}</td></tr>`).join("")
      : `<tr class="table__empty"><td colspan="4">No retraining runs yet.</td></tr>`;

    const info = S.info, hm = info?.headline_metrics?.test;
    $("#md-active").innerHTML = info ? [
      figure("Version", `<span class="mono">${esc(shortV(info.version))}</span>`, when(info.created_at)),
      figure("Decision model", esc(MODEL_NAMES[info.decision_model] || info.decision_model), `threshold ${info.threshold.toFixed(3)}`),
      hm ? figure("Test PR-AUC", fx(hm.pr_auc), `precision ${fx(hm.precision)} · recall ${fx(hm.recall)}`) : "",
    ].join("") : "";

    const stack = reg.approved_stack || [];
    const target = stack.length >= 2 && stack[stack.length - 1] === reg.active ? stack[stack.length - 2] : null;
    $("#md-rollback").hidden = !target;
    if (target) {
      $("#md-rollback-note").textContent = `Switch serving back to the previous approved version, ${shortV(target)}.`;
      $("#md-rollback-target").textContent = shortV(target);
    }
    $("#md-retrain").disabled = rs.active_run;
    if (rs.active_run) $("#md-retrain-msg").textContent = "A retraining run is in progress.";
  } catch (e) { banner(e.message); }
}

async function retrain() {
  const btn = $("#md-retrain"), msg = $("#md-retrain-msg");
  busy(btn, true);
  msg.textContent = "Training a candidate model…";
  try {
    const run = await api("/retraining/trigger", { method: "POST", body: { reason: "Retrained from the Sentinel console" } });
    let r = run;
    while (!TERMINAL.has(r.status)) {
      await new Promise((ok) => setTimeout(ok, 2000));
      r = (await api("/retraining/status?limit=20")).runs.find((x) => x.id === run.id) || r;
      msg.textContent = `Run ${run.id}: ${r.status}…`;
    }
    msg.textContent = r.status === "promoted" ? `Promoted ${shortV(r.candidate_version)}; it is serving now.` : `Run finished: ${r.decision || r.error || r.status}.`;
    await refreshModel();
    loadModels();
  } catch (e) { msg.textContent = e.message; }
  finally { busy(btn, false); }
}

async function rollback() {
  const msg = $("#md-rollback-msg");
  try {
    const r = await api("/retraining/rollback", { method: "POST", body: { reason: "Rolled back from the Sentinel console" } });
    msg.textContent = `Serving ${shortV(r.active_model)}.`;
    $("#md-rollback-confirm").hidden = true;
    await refreshModel();
    loadModels();
  } catch (e) { msg.textContent = e.message; }
}

/* ------------------------------------------------------------------ boot */
async function refreshModel() {
  const [info, ev, ex, metrics] = await Promise.all([api("/model/info"), api("/model/evaluation"), api("/model/examples"), api("/metrics").catch(() => null)]);
  Object.assign(S, { info, eval: ev, examples: ex });
  renderOverview(metrics);
  renderExamples();
  health();
}

function wire() {
  window.addEventListener("hashchange", route);
  $("#ov-example").addEventListener("change", (e) => { S.selected = Number(e.target.value); });
  $("#ov-score").addEventListener("click", async (e) => {
    const btn = e.currentTarget; busy(btn, true);
    try {
      const { res, ms } = await score(Number($("#ov-example").value));
      const review = res.decision === "REVIEW";
      $("#ov-result").innerHTML = `<div class="mini"><b class="${review ? "review" : "approve"}">${review ? "Send to review" : "Approve"}</b>
        <span>fraud probability ${res.fraud_probability.toFixed(4)} · ${ms.toFixed(0)} ms · <a href="#score">details</a></span></div>`;
    } catch (err) { $("#ov-result").textContent = err.message; }
    finally { busy(btn, false); }
  });
  $("#sc-table tbody").addEventListener("click", (e) => { const tr = e.target.closest("tr"); if (tr) selectExample(Number(tr.dataset.i)); });
  $("#sc-score").addEventListener("click", async (e) => {
    const btn = e.currentTarget; busy(btn, true);
    try { $("#sc-verdict").innerHTML = verdictHTML(await score(S.selected)); }
    catch (err) { $("#sc-verdict").innerHTML = `<p class="empty">${esc(err.message)}</p>`; }
    finally { busy(btn, false); }
  });
  $("#mo-form").addEventListener("submit", runSimulation);
  $("#md-retrain").addEventListener("click", retrain);
  $("#md-rollback-btn").addEventListener("click", () => { $("#md-rollback-confirm").hidden = false; });
  $("#md-rollback-no").addEventListener("click", () => { $("#md-rollback-confirm").hidden = true; });
  $("#md-rollback-yes").addEventListener("click", rollback);
}

(async function boot() {
  wire();
  route();
  const h = await health();
  if (!h || !h.model_loaded) return;
  try { await refreshModel(); } catch (e) { banner(e.message); }
  setInterval(health, 15000);
})();
