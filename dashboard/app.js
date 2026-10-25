/* tradesim results dashboard: reads data.json (scripts/build_dashboard.py) and renders SVG charts. */
"use strict";

const NS = "http://www.w3.org/2000/svg";
const SERIES_VARS = ["--series-1", "--series-2", "--series-3"];
const state = { data: null, agent: 0, episode: 0, symbol: null };

// ---------------------------------------------------------------- helpers
const $ = (id) => document.getElementById(id);
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const agentColor = (i) => cssVar(SERIES_VARS[i % SERIES_VARS.length]);

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    if (k === "class") node.className = v;
    else if (k === "style") node.style.cssText = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));  // text is always textContent
  }
  return node;
}

function s(tag, attrs = {}, text) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

const isNum = (v) => typeof v === "number" && Number.isFinite(v);
function fmtNum(v, digits = 2) {
  if (v === "inf") return "∞";
  if (v === "-inf") return "−∞";
  if (!isNum(v)) return "n/a";
  return v.toLocaleString("es-ES", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
function fmtPct(v, signed = true, digits = 2) {
  if (!isNum(v)) return fmtNum(v);
  const sign = signed && v > 0 ? "+" : v < 0 ? "−" : "";
  return `${sign}${fmtNum(Math.abs(v), digits)} %`;
}
function fmtMoney(v, digits = 2) {
  if (!isNum(v)) return fmtNum(v);
  return `${v < 0 ? "−" : ""}$${fmtNum(Math.abs(v), digits)}`;
}
function fmtPrice(v) {
  if (!isNum(v)) return "n/a";
  const a = Math.abs(v);
  const digits = a >= 1000 ? 2 : a >= 1 ? 2 : a >= 0.01 ? 4 : 6;
  return fmtNum(v, digits);
}
const shortDate = (t) => (t || "").replace("T00:00", "").replace("T", " ");

function niceTicks(min, max, count = 5) {
  if (min === max) { min -= 1; max += 1; }
  const span = max - min;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / count)));
  const err = (span / count) / step0;
  const step = step0 * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
  const ticks = [];
  const start = Math.floor(min / step + 1e-9) * step, end = Math.ceil(max / step - 1e-9) * step;
  for (let v = start; v <= end + step * 1e-9; v += step) ticks.push(+v.toPrecision(12));
  return ticks;  // first <= min and last >= max: every mark stays inside the plot
}
/** Compact number (1,2K / 3M). With ``step`` (tick spacing) the decimals are just enough to tell ticks apart. */
function compact(v, step) {
  const a = Math.abs(v);
  const [div, suf] = a >= 1e6 ? [1e6, "M"] : a >= 1e3 ? [1e3, "K"] : [1, ""];
  let digits;
  if (isNum(step) && step > 0) digits = Math.max(0, Math.min(6, -Math.floor(Math.log10(step / div) + 1e-9)));
  else digits = div > 1 ? (a / div >= 10 ? 0 : 1) : a >= 1 ? (Number.isInteger(v) ? 0 : 1) : a >= 0.01 ? 3 : 5;
  return `${fmtNum(v / div, digits)}${suf}`;
}

// ---------------------------------------------------------------- tooltip
const tip = {
  el: null,
  show(rows, head, x, y) {
    this.el = this.el || $("tooltip");
    this.el.replaceChildren();
    if (head) this.el.append(h("div", { class: "tt-head" }, head));
    for (const r of rows) {
      this.el.append(h("div", { class: "tt-row" },
        r.color ? h("span", { class: "key", style: `background:${r.color}` }) : null,
        h("strong", {}, r.value), h("span", { class: "name" }, r.name)));
    }
    this.el.hidden = false;
    const w = this.el.offsetWidth, hgt = this.el.offsetHeight;
    let left = x + 14, top = y + 14;
    if (left + w > window.innerWidth - 8) left = x - w - 14;
    if (top + hgt > window.innerHeight - 8) top = y - hgt - 14;
    this.el.style.left = `${Math.max(8, left)}px`;
    this.el.style.top = `${Math.max(8, top)}px`;
  },
  hide() { if (this.el) this.el.hidden = true; },
};

// ---------------------------------------------------------------- line chart
/**
 * Responsive SVG line chart with a crosshair tooltip.
 * opts: { x: [labels], series: [{name, color, values, area?, endLabel?}], yFormat, tipFormat, height,
 *         markers: [{i, y, color, up, text}], zeroLine, minY0 }
 */
function lineChart(root, opts) {
  const draw = () => {
    root.replaceChildren();
    const W = Math.max(root.clientWidth, 280);
    const H = opts.height || 260;
    const hasEnd = opts.series.some((sr) => sr.endLabel);
    const m = { l: 52, r: hasEnd ? (W < 560 ? 84 : 104) : 14, t: 10, b: 26 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const n = opts.x.length;
    let lo = Infinity, hi = -Infinity;
    for (const sr of opts.series) for (const v of sr.values) if (isNum(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    for (const mk of opts.markers || []) if (isNum(mk.y)) { lo = Math.min(lo, mk.y); hi = Math.max(hi, mk.y); }
    if (opts.zeroLine || opts.minY0 !== undefined) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    const pad = (hi - lo) * 0.06 || 1;
    const ticks = niceTicks(lo - (opts.minY0 === 0 && lo >= 0 ? 0 : pad), hi + (hi === 0 ? 0 : pad), H < 200 ? 3 : 5);
    const tickStep = ticks.length > 1 ? ticks[1] - ticks[0] : undefined;
    const y0 = ticks[0], y1 = ticks[ticks.length - 1];
    const X = (i) => m.l + (n <= 1 ? 0 : (i / (n - 1)) * pw);
    const Y = (v) => m.t + ph - ((v - y0) / (y1 - y0)) * ph;

    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", tabindex: 0,
      "aria-label": opts.label || "Gráfico de líneas" });
    // grid + y axis
    for (const t of ticks) {
      svg.append(s("line", { x1: m.l, x2: m.l + pw, y1: Y(t), y2: Y(t), stroke: t === 0 && opts.zeroLine ? cssVar("--axis") : cssVar("--grid"), "stroke-width": 1 }));
      svg.append(s("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end", class: "axis-text" }, (opts.yFormat || compact)(t, tickStep)));
    }
    // x ticks
    const xt = Math.max(2, Math.min(6, n, Math.floor(pw / 110)));
    for (let k = 0; k < xt; k++) {
      const i = Math.round((k / Math.max(xt - 1, 1)) * (n - 1));
      svg.append(s("text", { x: X(i), y: H - 6, "text-anchor": k === 0 ? "start" : k === xt - 1 ? "end" : "middle", class: "axis-text" },
        opts.xFormat ? opts.xFormat(opts.x[i], i) : shortDate(opts.x[i])));
    }
    svg.append(s("line", { x1: m.l, x2: m.l + pw, y1: m.t + ph, y2: m.t + ph, stroke: cssVar("--axis") }));
    // series
    for (const sr of opts.series) {
      let d = "", started = false;
      sr.values.forEach((v, i) => {
        if (!isNum(v)) { started = false; return; }
        d += `${started ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`;
        started = true;
      });
      if (sr.area) {
        const base = Y(Math.max(y0, Math.min(0, y1)));
        const first = sr.values.findIndex(isNum), last = sr.values.length - 1;
        svg.append(s("path", { d: `${d}L${X(last)},${base}L${X(first)},${base}Z`, fill: sr.color, "fill-opacity": 0.18, stroke: "none" }));
      }
      svg.append(s("path", { d, fill: "none", stroke: sr.color, "stroke-width": sr.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    }
    // markers (buy / sell triangles with a surface ring)
    for (const mk of opts.markers || []) {
      const cx = X(mk.i), cy = Y(mk.y), r = 6;
      const pts = mk.up ? `${cx},${cy - r} ${cx - r},${cy + r * 0.8} ${cx + r},${cy + r * 0.8}`
        : `${cx},${cy + r} ${cx - r},${cy - r * 0.8} ${cx + r},${cy - r * 0.8}`;
      svg.append(s("polygon", { points: pts, fill: mk.color, stroke: cssVar("--surface"), "stroke-width": 2, "stroke-linejoin": "round" }));
    }
    // end labels (pushed apart when they collide)
    const ends = opts.series.filter((sr) => sr.endLabel).map((sr) => {
      let i = sr.values.length - 1;
      while (i > 0 && !isNum(sr.values[i])) i--;
      return { sr, y: Y(sr.values[i]), v: sr.values[i] };
    }).sort((a, b) => a.y - b.y);
    for (let k = 1; k < ends.length; k++) if (ends[k].y - ends[k - 1].y < 28) ends[k].y = ends[k - 1].y + 28;
    for (const e of ends) {
      const g = s("text", { x: m.l + pw + 8, y: e.y - 2, class: "end-label" });
      g.append(s("tspan", {}, (opts.yFormat || compact)(e.v)));
      g.append(s("tspan", { x: m.l + pw + 8, dy: 13, class: "sub" }, e.sr.endLabel));
      svg.append(s("line", { x1: m.l + pw + 1, x2: m.l + pw + 5, y1: e.y - 6, y2: e.y - 6, stroke: e.sr.color, "stroke-width": 2 }));
      svg.append(g);
    }
    // crosshair
    const cross = s("line", { y1: m.t, y2: m.t + ph, stroke: cssVar("--axis"), "stroke-width": 1, visibility: "hidden" });
    const dots = opts.series.map((sr) => s("circle", { r: 4, fill: sr.color, stroke: cssVar("--surface"), "stroke-width": 2, visibility: "hidden" }));
    svg.append(cross, ...dots);
    let cur = -1;
    const at = (i, cx, cy) => {
      cur = Math.max(0, Math.min(n - 1, i));
      cross.setAttribute("x1", X(cur)); cross.setAttribute("x2", X(cur)); cross.setAttribute("visibility", "visible");
      const rows = [];
      opts.series.forEach((sr, k) => {
        const v = sr.values[cur];
        if (isNum(v)) {
          dots[k].setAttribute("cx", X(cur)); dots[k].setAttribute("cy", Y(v)); dots[k].setAttribute("visibility", "visible");
          rows.push({ color: sr.color, value: (opts.tipFormat || opts.yFormat || compact)(v), name: sr.name });
        } else dots[k].setAttribute("visibility", "hidden");
      });
      for (const mk of (opts.markers || []).filter((q) => q.i === cur)) rows.push({ color: mk.color, value: mk.up ? "▲" : "▼", name: mk.text });
      const box = svg.getBoundingClientRect();
      const px = cx ?? box.left + (X(cur) / W) * box.width, py = cy ?? box.top + 20;
      tip.show(rows, opts.xFormat ? opts.xFormat(opts.x[cur], cur) : shortDate(opts.x[cur]), px, py);
    };
    const hit = s("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent" });
    hit.addEventListener("pointermove", (ev) => {
      const box = svg.getBoundingClientRect();
      const xv = ((ev.clientX - box.left) / box.width) * W;
      at(Math.round(((xv - m.l) / pw) * (n - 1)), ev.clientX, ev.clientY);
    });
    const leave = () => { cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); tip.hide(); };
    hit.addEventListener("pointerleave", leave);
    svg.addEventListener("blur", leave);
    svg.addEventListener("keydown", (ev) => {
      if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
        ev.preventDefault();
        at((cur < 0 ? n - 1 : cur) + (ev.key === "ArrowRight" ? 1 : -1));
      }
    });
    svg.append(hit);
    root.append(svg);
  };
  draw();
  observe(root, draw);
}

// ---------------------------------------------------------------- strip plot
/** One row per group, one dot per value. rows: [{name, color, points: [{v, text, onClick}], mean}] */
function stripPlot(root, rows, opts = {}) {
  const draw = () => {
    root.replaceChildren();
    const W = Math.max(root.clientWidth, 280);
    const rowH = 44;
    const m = { l: W < 560 ? 92 : 120, r: 16, t: 8, b: 26 };
    const H = m.t + m.b + rows.length * rowH;
    const pw = W - m.l - m.r;
    const all = rows.flatMap((r) => r.points.map((p) => p.v)).filter(isNum);
    const ticks = niceTicks(Math.min(0, ...all), Math.max(0, ...all));
    const x0 = ticks[0], x1 = ticks[ticks.length - 1];
    const inset = 14;  // keeps edge dots off the row keys and the card border
    const X = (v) => m.l + inset + ((v - x0) / (x1 - x0)) * (pw - 2 * inset);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": opts.label || "Gráfico de puntos" });
    for (const t of ticks) {
      svg.append(s("line", { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, stroke: t === 0 ? cssVar("--axis") : cssVar("--grid") }));
      svg.append(s("text", { x: X(t), y: H - 6, "text-anchor": "middle", class: "axis-text" }, `${t > 0 ? "+" : t < 0 ? "−" : ""}${Math.abs(t)}%`));
    }
    rows.forEach((row, r) => {
      const cy = m.t + r * rowH + rowH / 2;
      const label = s("text", { x: m.l - 12, y: cy + 4, "text-anchor": "end", class: "row-label" }, row.name);
      svg.append(s("line", { x1: m.l - 8, x2: m.l - 2, y1: cy, y2: cy, stroke: row.color, "stroke-width": 3, "stroke-linecap": "round" }), label);
      if (isNum(row.mean)) {
        svg.append(s("line", { x1: X(row.mean), x2: X(row.mean), y1: cy - 15, y2: cy + 15, stroke: cssVar("--text"), "stroke-width": 2 }));
      }
      row.points.forEach((p, k) => {
        if (!isNum(p.v)) return;
        const cyj = cy + ((k % 5) - 2) * 4;  // deterministic vertical spread so dots don't hide each other
        const g = s("g", { tabindex: 0, role: "button", "aria-label": p.text });
        g.append(s("circle", { cx: X(p.v), cy: cyj, r: 12, fill: "transparent" }));
        const dot = s("circle", { cx: X(p.v), cy: cyj, r: 5, fill: row.color, stroke: cssVar("--surface"), "stroke-width": 2 });
        g.append(dot);
        const show = (ev) => {
          dot.setAttribute("r", 6.5);
          const b = g.getBoundingClientRect();
          tip.show([{ color: row.color, value: fmtPct(p.v), name: row.name }], p.text, ev.clientX ?? b.right, ev.clientY ?? b.top);
        };
        const hide = () => { dot.setAttribute("r", 5); tip.hide(); };
        g.addEventListener("pointermove", show);
        g.addEventListener("pointerleave", hide);
        g.addEventListener("focus", show);
        g.addEventListener("blur", hide);
        if (p.onClick) {
          g.style.cursor = "pointer";
          g.addEventListener("click", p.onClick);
          g.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); p.onClick(); } });
        }
        svg.append(g);
      });
    });
    root.append(svg);
  };
  draw();
  observe(root, draw);
}

const observers = new WeakMap();
function observe(root, draw) {
  observers.get(root)?.disconnect();
  let w = root.clientWidth;
  const ro = new ResizeObserver(() => { if (Math.abs(root.clientWidth - w) > 2) { w = root.clientWidth; draw(); } });
  ro.observe(root);
  observers.set(root, ro);
}

// ---------------------------------------------------------------- tables & tiles
function table(el, columns, rows, opts = {}) {
  el.replaceChildren();
  const thead = h("thead", {}, h("tr", {}, columns.map((c) => h("th", { scope: "col" }, c.label))));
  const tbody = h("tbody");
  rows.forEach((row, i) => {
    const tr = h("tr", { class: opts.selected === i ? "selected" : null, tabindex: opts.onRow ? 0 : null },
      columns.map((c) => {
        const val = c.get(row);
        return h("td", { class: c.cls ? c.cls(row) : null }, val instanceof Node ? val : String(val));
      }));
    if (opts.onRow) {
      tr.addEventListener("click", () => opts.onRow(i));
      tr.addEventListener("keydown", (ev) => { if (ev.key === "Enter") opts.onRow(i); });
    }
    tbody.append(tr);
  });
  el.append(thead, tbody);
}

function tiles(el, items) {
  el.replaceChildren(...items.map((t) => h("div", { class: "tile" },
    h("div", { class: "label" }, t.label), h("div", { class: "value" }, t.value), t.sub ? h("div", { class: "sub" }, t.sub) : null)));
}

const keyed = (name, color) => h("span", {}, h("span", { class: "key", style: `background:${color}` }), name);
const mean = (summary, k) => summary[k]?.mean;

// ---------------------------------------------------------------- sections
function renderHeader() {
  const md = state.data.meta;
  $("subtitle").textContent = `${md.n_episodes} episodios de ${md.episode_length} barras (${md.timeframe}) entre ${md.window_start} y ${md.window_end} · ` +
    `${md.symbols.length} activos · capital inicial ${fmtMoney(md.initial_cash, 0)} · comisiones "${md.fees}" · ` +
    `slippage ${md.execution.slippage_model}, participación máx. ${Math.round(md.execution.participation_rate * 100)} %`;
  if (md.synthetic) {
    $("notice").replaceChildren(h("div", { class: "notice", role: "note" },
      h("span", { class: "icon", "aria-hidden": "true" }, "⚠"),
      h("span", {}, h("strong", {}, "Datos sintéticos. "),
        "Estos resultados sirven para probar el sistema, no para juzgar una estrategia. Regenera con datos reales: ",
        h("code", {}, "python scripts/build_dashboard.py --data yfinance"))));
  }
  $("footer").replaceChildren(
    `Generado ${md.generated_at.replace("T", " ").replace("Z", " UTC")} · tradesim ${md.tradesim_version}` +
    (md.commit ? ` · commit ${md.commit}` : "") + ` · semillas ${md.seed}–${md.seed + md.n_episodes - 1} · activos: ${md.symbols.join(", ")} · `,
    h("code", {}, "scripts/build_dashboard.py"));
}

function renderComparison() {
  const { agents } = state.data;
  const bench = agents[0].summary.benchmark_return_pct;
  $("compare-hint").textContent = `Todos los agentes se evaluaron sobre los mismos ${state.data.meta.n_episodes} episodios. ` +
    `Buy & hold equiponderado (sin costes): retorno medio ${fmtPct(bench?.mean)}, mediana ${fmtPct(bench?.median)}.`;
  const cols = [
    { label: "Agente", get: (a) => keyed(a.name, a.color) },
    { label: "Retorno medio", get: (a) => fmtPct(mean(a.summary, "total_return_pct")) },
    { label: "Mediana", get: (a) => fmtPct(a.summary.total_return_pct?.median) },
    { label: "vs buy & hold", get: (a) => fmtPct(mean(a.summary, "excess_return_pct")) },
    { label: "Sharpe medio", get: (a) => fmtNum(mean(a.summary, "sharpe")) },
    { label: "Sortino medio", get: (a) => fmtNum(mean(a.summary, "sortino")) },
    { label: "Max DD medio", get: (a) => fmtPct(-mean(a.summary, "max_drawdown_pct")) },
    { label: "Win rate", get: (a) => fmtPct(mean(a.summary, "win_rate_pct"), false, 1) },
    { label: "Trades / ep.", get: (a) => fmtNum(mean(a.summary, "num_trades"), 1) },
    { label: "Comisiones / ep.", get: (a) => fmtMoney(mean(a.summary, "total_fees")) },
    { label: "Episodios +", get: (a) => fmtPct(a.summary.pct_positive_episodes, false, 0) },
    { label: "Baten B&H", get: (a) => fmtPct(a.summary.pct_beat_benchmark, false, 0) },
  ];
  table($("compare-table"), cols, agents);
}

function renderPathChart() {
  const { agents, benchmark_path: bp } = state.data;
  const n = Math.min(...agents.map((a) => a.path.mean.length));
  const series = agents.map((a) => ({ name: a.name, color: a.color, values: a.path.mean.slice(0, n), endLabel: a.short }));
  if (bp) series.push({ name: "Buy & hold (sin costes)", color: cssVar("--bench"), values: bp.mean.slice(0, n), endLabel: "Buy & hold", width: 1.5 });
  const x = Array.from({ length: n }, (_, i) => i);
  $("path-table").replaceChildren();
  lineChart($("chart-path"), {
    x, series, height: 280, zeroLine: true, label: "Retorno medio por barra del episodio",
    yFormat: (v) => fmtPct(v, true, Math.abs(v) < 10 ? 1 : 0), tipFormat: (v) => fmtPct(v),
    xFormat: (v) => `barra ${v}`,
  });
  const step = Math.max(1, Math.round(n / 6));
  const idx = Array.from({ length: Math.ceil(n / step) }, (_, k) => Math.min(k * step, n - 1));
  if (idx[idx.length - 1] !== n - 1) idx.push(n - 1);
  table($("path-table").appendChild(document.createElement("table")), [
    { label: "Barra", get: (i) => i },
    ...series.map((sr) => ({ label: sr.name, get: (i) => fmtPct(sr.values[i]) })),
  ], idx);
  $("path-table").firstChild.className = "data-table";
}

function renderStrip() {
  const { agents } = state.data;
  const rows = agents.map((a, ai) => ({
    name: a.short, color: a.color, mean: mean(a.summary, "total_return_pct"),
    points: a.episodes.map((e, ei) => ({
      v: e.metrics.total_return_pct,
      text: `Episodio ${e.episode} · ${shortDate(e.start)} → ${shortDate(e.end)} · B&H ${fmtPct(e.metrics.benchmark_return_pct)}`,
      onClick: () => { selectAgent(ai, ei); $("agent-section").scrollIntoView({ behavior: "smooth", block: "start" }); },
    })),
  }));
  rows.push({
    name: "Buy & hold", color: cssVar("--bench"), mean: mean(agents[0].summary, "benchmark_return_pct"),
    points: agents[0].episodes.map((e) => ({ v: e.metrics.benchmark_return_pct, text: `Episodio ${e.episode} · ${shortDate(e.start)} → ${shortDate(e.end)}` })),
  });
  stripPlot($("chart-strip"), rows, { label: "Retorno por episodio y agente" });
}

function renderAgent() {
  const a = state.data.agents[state.agent];
  const sm = a.summary;
  tiles($("agent-tiles"), [
    { label: "Retorno medio", value: fmtPct(mean(sm, "total_return_pct")), sub: `mediana ${fmtPct(sm.total_return_pct?.median)}` },
    { label: "Mejor / peor episodio", value: fmtPct(sm.best_return_pct), sub: `peor ${fmtPct(sm.worst_return_pct)}` },
    { label: "Sharpe medio", value: fmtNum(mean(sm, "sharpe")), sub: `Sortino ${fmtNum(mean(sm, "sortino"))}` },
    { label: "Max drawdown medio", value: fmtPct(-mean(sm, "max_drawdown_pct")), sub: `peor ${fmtPct(-sm.max_drawdown_pct?.max)}` },
    { label: "Baten buy & hold", value: fmtPct(sm.pct_beat_benchmark, false, 0), sub: `episodios positivos ${fmtPct(sm.pct_positive_episodes, false, 0)}` },
    { label: "Capital final medio", value: fmtMoney(mean(sm, "final_equity")), sub: `trades / ep. ${fmtNum(mean(sm, "num_trades"), 1)}` },
  ]);
  table($("episodes-table"), [
    { label: "#", get: (e) => e.episode },
    { label: "Periodo", get: (e) => `${shortDate(e.start)} → ${shortDate(e.end)}` },
    { label: "Capital final", get: (e) => fmtMoney(e.metrics.final_equity) },
    { label: "Retorno", get: (e) => fmtPct(e.metrics.total_return_pct) },
    { label: "Buy & hold", get: (e) => fmtPct(e.metrics.benchmark_return_pct) },
    { label: "Sharpe", get: (e) => fmtNum(e.metrics.sharpe) },
    { label: "Sortino", get: (e) => fmtNum(e.metrics.sortino) },
    { label: "Max DD", get: (e) => fmtPct(-e.metrics.max_drawdown_pct) },
    { label: "Trades", get: (e) => e.metrics.num_trades },
    { label: "Win rate", get: (e) => fmtPct(e.metrics.win_rate_pct, false, 1) },
    { label: "Profit factor", get: (e) => fmtNum(e.metrics.profit_factor) },
    { label: "PnL medio", get: (e) => fmtMoney(e.metrics.avg_trade_pnl) },
  ], a.episodes, { selected: state.episode, onRow: (i) => selectAgent(state.agent, i) });
  renderEpisode();
}

function renderEpisode() {
  const a = state.data.agents[state.agent];
  const e = a.episodes[state.episode];
  const mt = e.metrics;
  $("episode-title").textContent = `Episodio ${e.episode} · ${shortDate(e.start)} → ${shortDate(e.end)} · ${a.name}`;
  tiles($("episode-tiles"), [
    { label: "Capital final", value: fmtMoney(mt.final_equity), sub: `retorno ${fmtPct(mt.total_return_pct)}` },
    { label: "Buy & hold", value: fmtPct(mt.benchmark_return_pct), sub: `diferencia ${fmtPct(mt.excess_return_pct)}` },
    { label: "Sharpe / Sortino", value: fmtNum(mt.sharpe), sub: `Sortino ${fmtNum(mt.sortino)}` },
    { label: "Max drawdown", value: fmtPct(-mt.max_drawdown_pct), sub: `volatilidad ${fmtPct(mt.volatility_annual_pct, false)}` },
    { label: "Trades cerrados", value: String(mt.num_trades ?? 0), sub: `win rate ${fmtPct(mt.win_rate_pct, false, 1)} · PF ${fmtNum(mt.profit_factor)}` },
    { label: "Mejor / peor trade", value: fmtMoney(mt.best_trade), sub: `peor ${fmtMoney(mt.worst_trade)}` },
  ]);

  const idxOf = new Map(e.t.map((t, i) => [t, i]));
  const fillMarkers = e.fills.filter((f) => idxOf.has(f.t)).map((f) => ({
    i: idxOf.get(f.t), y: e.equity[idxOf.get(f.t)], up: f.side === "buy", color: cssVar(f.side === "buy" ? "--buy" : "--sell"),
    text: `${f.side === "buy" ? "Compra" : "Venta"} ${f.quantity} ${f.symbol} @ ${fmtPrice(f.price)}`,
  }));
  const eqSeries = [{ name: "Agente", color: a.color, values: e.equity, endLabel: "Agente" }];
  if (e.benchmark) eqSeries.push({ name: "Buy & hold", color: cssVar("--bench"), values: e.benchmark, endLabel: "Buy & hold", width: 1.5 });
  lineChart($("chart-equity"), { x: e.t, series: eqSeries, markers: fillMarkers, height: 280, yFormat: (v, st) => `$${compact(v, st)}`,
    tipFormat: (v) => fmtMoney(v), label: "Equity del agente frente a buy and hold" });

  let peak = -Infinity;
  const dd = e.equity.map((v) => { peak = Math.max(peak, v); return (v / peak - 1) * 100; });
  lineChart($("chart-dd"), { x: e.t, series: [{ name: "Drawdown", color: a.color, values: dd, area: true }], height: 150,
    yFormat: (v) => fmtPct(v, true, Math.abs(v) < 10 ? 1 : 0), tipFormat: (v) => fmtPct(v), label: "Drawdown del episodio" });

  const syms = Object.keys(e.prices);
  const sel = $("sel-symbol");
  sel.replaceChildren(...syms.map((sy) => h("option", { value: sy }, sy)));
  if (!syms.includes(state.symbol)) state.symbol = syms[0] || null;
  sel.value = state.symbol || "";
  sel.disabled = !syms.length;
  renderPrice(e);

  table($("trades-table"), [
    { label: "Símbolo", get: (t) => t.symbol },
    { label: "Entrada", get: (t) => shortDate(t.opened) },
    { label: "Salida", get: (t) => shortDate(t.closed) },
    { label: "Cantidad", get: (t) => fmtNum(t.quantity, 6) },
    { label: "Precio entrada", get: (t) => fmtPrice(t.entry_price) },
    { label: "Precio salida", get: (t) => fmtPrice(t.exit_price) },
    { label: "PnL neto", get: (t) => fmtMoney(t.pnl) },
    { label: "Retorno", get: (t) => fmtPct(t.return_pct) },
  ], e.trades.length ? e.trades : []);
  if (!e.trades.length) $("trades-table").append(h("caption", {}, "Sin trades cerrados en este episodio (las posiciones abiertas cuentan en el equity)."));
  table($("fills-table"), [
    { label: "Fecha", get: (f) => shortDate(f.t) },
    { label: "Lado", get: (f) => (f.side === "buy" ? "compra" : "venta"), cls: (f) => `side-${f.side}` },
    { label: "Símbolo", get: (f) => f.symbol },
    { label: "Cantidad", get: (f) => f.quantity },
    { label: "Precio", get: (f) => fmtPrice(f.price) },
    { label: "Comisión", get: (f) => fmtMoney(f.fee, 4) },
    { label: "Liquidez", get: (f) => f.liquidity },
  ], e.fills);
}

function renderPrice(e) {
  const root = $("chart-price");
  const p = state.symbol ? e.prices[state.symbol] : null;
  if (!p) { root.replaceChildren(h("p", { class: "hint" }, "Sin operaciones en este episodio.")); return; }
  const idxOf = new Map(p.t.map((t, i) => [t, i]));
  const markers = e.fills.filter((f) => f.symbol === state.symbol && idxOf.has(f.t)).map((f) => ({
    i: idxOf.get(f.t), y: f.price, up: f.side === "buy", color: cssVar(f.side === "buy" ? "--buy" : "--sell"),
    text: `${f.side === "buy" ? "Compra" : "Venta"} ${f.quantity} @ ${fmtPrice(f.price)}`,
  }));
  root.replaceChildren();
  root.append(h("div", { class: "legend" },
    h("span", {}, h("span", { class: "key", style: `background:${cssVar("--text-2")}` }), `${state.symbol} cierre`),
    h("span", {}, h("span", { class: "tri", style: `color:${cssVar("--buy")}` }, "▲"), "compra (precio de ejecución)"),
    h("span", {}, h("span", { class: "tri", style: `color:${cssVar("--sell")}` }, "▼"), "venta")));
  const holder = h("div");
  root.append(holder);
  lineChart(holder, { x: p.t, series: [{ name: `${state.symbol} cierre`, color: cssVar("--text-2"), values: p.close, width: 1.5 }],
    markers, height: 240, yFormat: compact, tipFormat: fmtPrice, label: `Precio de ${state.symbol} con compras y ventas` });
}

// ---------------------------------------------------------------- selection
function fillSelectors() {
  const { agents } = state.data;
  $("sel-agent").replaceChildren(...agents.map((a, i) => h("option", { value: i }, a.name)));
  $("sel-agent").value = state.agent;
  $("sel-episode").replaceChildren(...agents[state.agent].episodes.map((e, i) =>
    h("option", { value: i }, `#${e.episode} · ${shortDate(e.start)} → ${shortDate(e.end)} · ${fmtPct(e.metrics.total_return_pct)}`)));
  $("sel-episode").value = state.episode;
}

function selectAgent(ai, ei = 0) {
  state.agent = ai;
  state.episode = ei;
  fillSelectors();
  renderAgent();
  writeHash();
}

function writeHash() { history.replaceState(null, "", `#agent=${state.agent}&episode=${state.episode}`); }
function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  const a = +q.get("agent"), e = +q.get("episode");
  if (Number.isInteger(a) && state.data.agents[a]) state.agent = a;
  if (Number.isInteger(e) && state.data.agents[state.agent].episodes[e]) state.episode = e;
}

function renderAll() {
  const SHORT = { SimpleRSIAgent: "RSI", MovingAverageCrossAgent: "Cruce medias", RandomAgent: "Aleatorio",
                  BuyAndHoldAgent: "Buy & hold (agente)" };
  state.data.agents.forEach((a, i) => {
    a.color = agentColor(i);
    a.short = SHORT[a.class] || a.name.replace(/\(.*\)/, "").replace(/Agent$/, "").slice(0, 16);
  });
  renderHeader();
  renderComparison();
  renderPathChart();
  renderStrip();
  fillSelectors();
  renderAgent();
}

// ---------------------------------------------------------------- boot
function initTheme() {
  let stored = null;
  try { stored = localStorage.getItem("tradesim-theme"); } catch { /* storage unavailable */ }
  if (stored) document.documentElement.dataset.theme = stored;
  $("theme-toggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("tradesim-theme", next); } catch { /* ignore */ }
    if (state.data) renderAll();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (state.data && !document.documentElement.dataset.theme) renderAll();
  });
}

async function boot() {
  initTheme();
  try {
    const res = await fetch("data.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.data = await res.json();
  } catch (err) {
    $("app").replaceChildren(h("div", { class: "error" },
      `No se pudo cargar data.json (${err.message}). Genera los datos con `, h("code", {}, "python scripts/build_dashboard.py"),
      " y sirve la carpeta con un servidor HTTP (no abriendo el archivo directamente)."));
    $("subtitle").textContent = "";
    return;
  }
  readHash();
  $("sel-agent").addEventListener("change", (ev) => selectAgent(+ev.target.value, 0));
  $("sel-episode").addEventListener("change", (ev) => selectAgent(state.agent, +ev.target.value));
  $("sel-symbol").addEventListener("change", (ev) => {
    state.symbol = ev.target.value;
    renderPrice(state.data.agents[state.agent].episodes[state.episode]);
  });
  renderAll();
  $("app").setAttribute("aria-busy", "false");
}

boot();
