/* Live operations page: reads live.json (written by scripts/live_run.py). Needs charts.js. */
"use strict";

const live = { data: null, source: null, account: 0, prices: {}, pricesAt: null, timer: null, rejected: {}, journalAll: false };
const LIVE_REFRESH_MS = 30_000;
const MAX_LIVE_DEVIATION = 0.5;  // a live price >50% away from the last valuation is treated as bad data

// ---------------------------------------------------------------- formatting
const fmtWhen = (iso) => (iso ? new Date(iso).toLocaleString("es-ES", { dateStyle: "medium", timeStyle: "short" }) : "n/a");
const fmtDay = (iso) => (iso ? new Date(iso).toLocaleDateString("es-ES", { day: "2-digit", month: "short", year: "numeric" }) : "n/a");
function fmtAgo(iso) {
  if (!iso) return "";
  const min = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (min < 1) return "ahora mismo";
  if (min < 60) return `hace ${min} min`;
  const hours = Math.round(min / 60);
  if (hours < 48) return `hace ${hours} h`;
  return `hace ${Math.round(hours / 24)} días`;
}
const STATUS = {
  ok: ["Decidió", "st-ok"], synced: ["Valorada", "st-synced"], skipped: ["Sin decisión", "st-skipped"],
  halted: ["Detenida por riesgo", "st-halted"], rejected: ["Rechazada", "st-rejected"], error: ["Error", "st-error"],
  "not started": ["Sin iniciar", "st-skipped"],
};
function statusChip(status) {
  const [label, cls] = STATUS[status] || [status || "n/a", "st-skipped"];
  return h("span", { class: `chip ${cls}` }, h("span", { class: "dot", "aria-hidden": "true" }), label);
}
const signed = (v, f = fmtMoney) => (isNum(v) && v > 0 ? `+${f(v)}` : f(v));

// ---------------------------------------------------------------- colours
function assignColors() {
  let slot = 0;
  for (const a of live.data.accounts) {
    a.color = a.agent === "BuyAndHoldAgent" ? cssVar("--bench") : cssVar(SERIES_VARS[slot++ % SERIES_VARS.length]);
  }
}

// ---------------------------------------------------------------- schedule
/** Next run times for "M H * * D" crons (H = hour or "*", D = "*" or "a-b"), in UTC. */
function nextRuns(schedule, n = 3) {
  const now = new Date();
  const out = [];
  for (const job of schedule || []) {
    const [m, hr, , , dow] = job.cron.split(/\s+/);
    const days = dow === "*" ? null : (() => { const [a, b] = dow.split("-").map(Number); return [a, b ?? a]; })();
    const hours = hr === "*" ? [...Array(24).keys()] : [+hr];
    let found = null;
    for (let k = 0; k < 8 && !found; k++) {
      for (const hh of hours) {
        const t = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + k, hh, +m));
        const wd = t.getUTCDay();
        if (t > now && (!days || (wd >= days[0] && wd <= days[1]))) { found = t; break; }
      }
    }
    if (found) out.push({ t: found, mode: job.mode });
  }
  return out.sort((a, b) => a.t - b.t).slice(0, n);
}
const MODE_ES = { decide: "decisión", mark: "valoración", research: "investigación" };

// ---------------------------------------------------------------- live prices
const cryptoSymbols = () => live.data.meta.symbols.filter((s) => live.data.meta.asset_class?.[s] === "crypto");

async function refreshLivePrices() {
  if (document.hidden || !live.data) return;
  const results = await Promise.allSettled(cryptoSymbols().map(async (sym) => {
    const res = await fetch(`https://api.exchange.coinbase.com/products/${sym.split("-")[0]}-USD/ticker`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const q = await res.json();
    return [sym, { last: +q.price, bid: +q.bid, ask: +q.ask, time: q.time }];
  }));
  let got = 0;
  for (const r of results) {
    if (r.status !== "fulfilled" || !isNum(r.value[1].last)) continue;
    const [sym, q] = r.value;
    const ref = live.data.prices?.[sym]?.last;
    if (isNum(ref) && Math.abs(q.last / ref - 1) > MAX_LIVE_DEVIATION) {  // sanity guard: never show nonsense values
      live.rejected[sym] = q.last;
      delete live.prices[sym];
      continue;
    }
    delete live.rejected[sym];
    live.prices[sym] = q;
    got++;
  }
  if (got) live.pricesAt = new Date().toISOString();
  renderPrices();
  renderAccountLive();
}

const livePrice = (sym) => live.prices[sym]?.last;

// ---------------------------------------------------------------- header
function renderStatus() {
  const d = live.data;
  const bar = $("statusbar");
  bar.replaceChildren();
  const last = d.last_run;
  if (last) {
    const errs = Object.keys(last.data_errors || {}).length +
      Object.values(last.accounts || {}).filter((a) => a.status === "error").length;
    bar.append(h("span", {}, "Última ejecución: ", h("strong", {}, fmtWhen(last.started_at)), ` (${fmtAgo(last.started_at)}, ${MODE_ES[last.mode] || last.mode}, ${fmtNum(last.duration_s, 1)} s)`));
    bar.append(h("span", {}, errs ? `⚠ ${errs} incidencia${errs > 1 ? "s" : ""} en la última ejecución` : "✓ sin incidencias"));
    if (live.source) bar.append(h("span", {}, `datos: ${live.source}`));
  }
  const rs = d.research?.status;
  if (rs?.started_at) {
    bar.append(h("span", {}, h("a", { href: "research.html" }, "Investigación"), ` actualizada ${fmtAgo(rs.started_at)}`,
      rs.llm === "enabled" ? "" : " (motor local gratuito)"));
  }
  const next = nextRuns(d.meta.schedule);
  if (next.length) {
    bar.append(h("span", {}, "Próximas: ", next.map((r) => `${fmtWhen(r.t.toISOString())} (${MODE_ES[r.mode] || r.mode})`).join(" · ")));
  }
  const started = d.accounts.some((a) => a.summary);
  $("notice").replaceChildren();
  if (!started) {
    $("notice").append(h("div", { class: "notice", role: "note" }, h("span", { class: "icon", "aria-hidden": "true" }, "⏳"),
      h("span", {}, h("strong", {}, "Todavía no hay operaciones. "),
        "Las cuentas se abren en la primera decisión diaria, que solo ocurre con la bolsa de Nueva York abierta (lunes a viernes, sin festivos).")));
  }
  $("footer").replaceChildren(`Datos generados ${fmtWhen(d.meta.generated_at)} · activos: ${d.meta.symbols.join(", ")} · `,
    `límites de riesgo: máx. ${d.meta.risk.max_position_pct} % por posición, parada con drawdown ≥ ${d.meta.risk.halt_drawdown_pct} % · `,
    "cuentas de práctica con ejecución simulada sobre precios reales (spread, impacto y comisiones de Alpaca).");
}

// ---------------------------------------------------------------- accounts overview
function renderAccounts() {
  const rows = live.data.accounts;
  table($("accounts-table"), [
    { label: "Cuenta", get: (a) => keyed(a.name, a.color) },
    { label: "Equity", get: (a) => fmtMoney(a.summary?.equity) },
    { label: "P&L total", get: (a) => (a.summary ? `${signed(a.summary.pnl)} (${fmtPct(a.summary.pnl_pct)})` : "—") },
    { label: "Hoy", get: (a) => (a.summary ? `${signed(a.summary.day_pnl)} (${fmtPct(a.summary.day_pnl_pct)})` : "—") },
    { label: "Invertido", get: (a) => fmtPct(a.summary?.invested_pct, false, 0) },
    { label: "Abiertas", get: (a) => a.summary?.open_positions ?? "—" },
    { label: "Cerradas", get: (a) => a.summary?.num_trades ?? "—" },
    { label: "Win rate", get: (a) => fmtPct(a.summary?.win_rate_pct, false, 0) },
    { label: "Max DD", get: (a) => (a.summary ? fmtPct(-a.summary.max_drawdown_pct) : "—") },
    { label: "Sharpe", get: (a) => fmtNum(a.summary?.sharpe) },
    { label: "Comisiones", get: (a) => fmtMoney(a.summary?.fees) },
    { label: "Último run", get: (a) => statusChip(a.status) },
  ], rows, { selected: live.account, onRow: (i) => selectAccount(i, true) });
}

function renderEdge() {
  const days = (v) => (v === null || v === undefined ? "no se alcanza al ritmo actual" : v === 0 ? "ya significativo" : `${v} días más`);
  table($("edge-table"), [
    { label: "Cuenta", get: (a) => keyed(a.name, a.color) },
    { label: "Días", get: (a) => a.significance?.days ?? "—" },
    { label: "Sharpe [IC 95 %]", get: (a) => (isNum(a.significance?.sharpe)
      ? `${fmtNum(a.significance.sharpe)} [${fmtNum(a.significance.sharpe_ci95[0])}, ${fmtNum(a.significance.sharpe_ci95[1])}]` : "pocos días") },
    { label: "P(bate a B&H)", get: (a) => (isNum(a.significance?.p_beats_buy_hold) ? fmtPct(a.significance.p_beats_buy_hold * 100, false, 0) : "—") },
    { label: "IC (t)", get: (a) => (a.significance?.rank_ic ? `${fmtNum(a.significance.rank_ic.mean, 3)} (t ${fmtNum(a.significance.rank_ic.t)})` : "—") },
    { label: "Para Sharpe significativo", get: (a) => (isNum(a.significance?.sharpe) ? days(a.significance.days_to_significant_sharpe) : "—") },
    { label: "Para batir a B&H con significancia", get: (a) => ("p_beats_buy_hold" in (a.significance || {}) ? days(a.significance.days_to_significant_vs_buy_hold) : "—") },
  ], live.data.accounts);
}

function renderPerformance() {
  const accs = live.data.accounts.filter((a) => a.equity_curve?.t?.length);
  const root = $("chart-perf");
  if (!accs.length) { root.replaceChildren(h("p", { class: "empty" }, "Aparecerá con la primera valoración.")); return; }
  const base = accs.reduce((a, b) => (b.equity_curve.t.length > a.equity_curve.t.length ? b : a));
  const x = base.equity_curve.t;
  const series = accs.map((a) => {
    const byT = new Map(a.equity_curve.t.map((t, i) => [t, a.equity_curve.equity[i]]));
    return { name: a.name, color: a.color, endLabel: a.name.length > 14 ? a.name.slice(0, 13) + "…" : a.name,
      values: x.map((t) => (byT.has(t) ? (byT.get(t) / a.initial_cash - 1) * 100 : null)),
      width: a.agent === "BuyAndHoldAgent" ? 1.5 : 2 };
  });
  lineChart(root, { x, series, height: 270, zeroLine: true, label: "Retorno acumulado por cuenta",
    xFormat: (t) => fmtWhen(t), yFormat: (v, st) => fmtPct(v, true, st !== undefined && st < 1 ? 1 : Math.abs(v) < 10 ? 1 : 0),
    tipFormat: (v) => fmtPct(v) });
}

function renderPrices() {
  const d = live.data;
  const live_n = Object.keys(live.prices).length;
  const bad = Object.keys(live.rejected);
  $("prices-hint").replaceChildren(
    live_n ? h("span", {}, h("span", { class: "live-dot", "aria-hidden": "true" }, "●"),
      `Cripto en tiempo real (Coinbase, ${fmtAgo(live.pricesAt)}); acciones a la última valoración.`)
      : "Precios de la última valoración.",
    bad.length ? h("span", {}, ` Precio en vivo descartado para ${bad.join(", ")}: difiere más de un 50 % de la última valoración.`) : "");
  const rows = d.meta.symbols.map((s) => ({ sym: s, run: d.prices?.[s] || {}, now: live.prices[s] }));
  table($("prices-table"), [
    { label: "Activo", get: (r) => r.sym },
    { label: "Precio", get: (r) => {
      const v = r.now?.last ?? r.run.last;
      return r.now ? h("span", { class: "pos" }, h("span", { class: "live-dot", "aria-hidden": "true" }, "●"), fmtPrice(v)) : fmtPrice(v);
    } },
    { label: "vs. valoración", get: (r) => (r.now && isNum(r.run.last) ? fmtPct((r.now.last / r.run.last - 1) * 100) : "—") },
    { label: "Spread", get: (r) => {
      const q = r.now || r.run;
      const bps = isNum(q.bid) && isNum(q.ask) && q.ask > 0 ? ((q.ask - q.bid) / ((q.ask + q.bid) / 2)) * 1e4 : r.run.spread_bps;
      return isNum(bps) ? `${fmtNum(bps, bps < 1 ? 2 : 1)} bps${r.run.modelled_spread && !r.now ? " (modelado)" : ""}` : "—";
    } },
    { label: "Mercado", get: (r) => {
      if (r.run.error) return "sin datos";
      if (live.data.meta.asset_class?.[r.sym] === "crypto") return "abierto 24/7";
      return r.run.market_open === undefined ? "—" : r.run.market_open ? "abierto" : "cerrado";
    } },
    { label: "Fuente", get: (r) => (r.now ? "coinbase (vivo)" : (r.run.source || "").replace("ccxt:", "")) },
  ], rows);
}

// ---------------------------------------------------------------- account detail
function liveEquity(a) {
  if (!a.summary) return null;
  let eq = a.summary.cash;
  for (const p of a.positions) eq += parseFloat(p.quantity) * (livePrice(p.symbol) ?? p.last_price);
  return eq;
}

function renderAccount() {
  const a = live.data.accounts[live.account];
  $("account-desc").textContent = a.description || "";
  const started = Boolean(a.summary);
  $("account-body").hidden = !started;
  $("account-empty").hidden = started;
  if (!started) {
    $("account-tiles").replaceChildren();
    $("account-empty").textContent = a.status === "error"
      ? `Error: ${a.error || "ver el registro de ejecuciones"}`
      : "Esta cuenta aún no ha operado: se abrirá en la próxima decisión diaria con la bolsa de Nueva York abierta.";
    renderJournal(a);
    return;
  }
  renderAccountLive();
  const c = a.equity_curve;
  lineChart($("chart-equity"), { x: c.t, series: [{ name: a.name, color: a.color, values: c.equity }], height: 220,
    label: `Equity de ${a.name}`, xFormat: (t) => fmtWhen(t), yFormat: (v, st) => `$${compact(v, st)}`, tipFormat: (v) => fmtMoney(v) });

  table($("trades-table"), [
    { label: "Activo", get: (t) => t.symbol },
    { label: "Apertura", get: (t) => fmtDay(t.opened_at) },
    { label: "Cierre", get: (t) => fmtDay(t.closed_at) },
    { label: "Días", get: (t) => fmtNum(t.days_held, 1) },
    { label: "Asignado", get: (t) => fmtMoney(t.allocated) },
    { label: "% cuenta", get: (t) => fmtPct(t.allocated_pct_at_entry, false, 1) },
    { label: "Entrada", get: (t) => fmtPrice(t.entry_price) },
    { label: "Salida", get: (t) => fmtPrice(t.exit_price) },
    { label: "Resultado", get: (t) => signed(t.pnl) },
    { label: "Rendimiento", get: (t) => fmtPct(t.return_pct) },
    { label: "Comisiones", get: (t) => fmtMoney(t.fees) },
    { label: "Motivo entrada", get: (t) => reasonCell(t.entry_reason), cls: () => "reason" },
    { label: "Salida por", get: (t) => ({ stop_loss: "stop-loss", take_profit: "objetivo" }[t.exit_kind] || "decisión") },
    { label: "Motivo salida", get: (t) => reasonCell(t.exit_reason), cls: () => "reason" },
  ], a.closed_trades);
  if (!a.closed_trades.length) $("trades-table").append(h("caption", { class: "empty" }, "Aún no hay operaciones cerradas."));

  table($("orders-table"), [
    { label: "Creada", get: (o) => fmtWhen(o.created_at) },
    { label: "Lado", get: (o) => (o.side === "buy" ? "compra" : "venta"), cls: (o) => `side-${o.side}` },
    { label: "Activo", get: (o) => o.symbol },
    { label: "Cantidad", get: (o) => o.quantity },
    { label: "Tipo", get: (o) => ({ stop: "stop-loss", limit: "límite / objetivo", market: "mercado", stop_limit: "stop límite" }[o.order_type] || o.order_type) },
    { label: "Precio", get: (o) => { const v = parseFloat(o.stop_price || o.limit_price); return isNum(v) ? fmtPrice(v) : "—"; } },
    { label: "Motivo", get: (o) => reasonCell(o.tag), cls: () => "reason" },
  ], a.open_orders);
  if (!a.open_orders.length) $("orders-table").append(h("caption", { class: "empty" }, "Sin órdenes pendientes."));
  renderLearning(a);
  renderJournal(a);
}

/** Long rationales: two lines in the table, the full text on hover. */
const reasonCell = (text) => (text ? h("span", { class: "clamp", title: text }, text) : "—");

/** Protective level with its distance from the current price ("—" when the position has none). */
function exitCell(p, level) {
  if (!isNum(level)) return "—";
  const px = livePrice(p.symbol) ?? p.last_price;
  return isNum(px) && px > 0 ? `${fmtPrice(level)} (${fmtPct((level / px - 1) * 100, true, 1)})` : fmtPrice(level);
}

/** Tiles and open positions: the parts that move with live crypto prices. */
function renderAccountLive() {
  const a = live.data?.accounts[live.account];
  if (!a?.summary) return;
  const s = a.summary;
  const leq = liveEquity(a);
  const hasLive = a.positions.some((p) => livePrice(p.symbol) !== undefined);
  tiles($("account-tiles"), [
    { label: "Equity (última valoración)", value: fmtMoney(s.equity), sub: `inicial ${fmtMoney(a.initial_cash, 0)}` },
    hasLive ? { label: "Equity estimada ahora", value: fmtMoney(leq), sub: `cripto en vivo · ${signed((leq / a.initial_cash - 1) * 100, (v) => fmtPct(v, false))}` } : null,
    { label: "Resultado total", value: signed(s.pnl), sub: `${fmtPct(s.pnl_pct)} · hoy ${signed(s.day_pnl)}` },
    { label: "Cash / invertido", value: fmtMoney(s.cash), sub: `invertido ${fmtPct(s.invested_pct, false, 0)}` },
    { label: "No realizado / realizado", value: signed(s.unrealized_pnl), sub: `realizado ${signed(s.realized_pnl)}` },
    { label: "Max drawdown", value: fmtPct(-s.max_drawdown_pct), sub: `actual ${fmtPct(-s.current_drawdown_pct)}` },
    { label: "Trades cerrados", value: String(s.num_trades), sub: `win rate ${fmtPct(s.win_rate_pct, false, 0)} · PF ${fmtNum(s.profit_factor)}` },
    { label: "Sharpe", value: fmtNum(s.sharpe), sub: s.days_live < 20 ? `necesita 20 días (lleva ${s.days_live})` : `Sortino ${fmtNum(s.sortino)}` },
    { label: "Costes", value: fmtMoney(s.fees), sub: `slippage ${fmtMoney(s.slippage_cost)}` },
  ].filter(Boolean));

  table($("positions-table"), [
    { label: "Activo", get: (p) => p.symbol },
    { label: "Abierta", get: (p) => fmtDay(p.opened_at) },
    { label: "Días", get: (p) => fmtNum(p.days_held, 1) },
    { label: "Cantidad", get: (p) => p.quantity },
    { label: "Asignado", get: (p) => fmtMoney(p.allocated) },
    { label: "% cuenta", get: (p) => fmtPct(p.allocated_pct_at_entry, false, 1) },
    { label: "Precio medio", get: (p) => fmtPrice(p.avg_price) },
    { label: "Precio", get: (p) => {
      const lp = livePrice(p.symbol);
      return lp !== undefined ? h("span", { class: "pos" }, h("span", { class: "live-dot", "aria-hidden": "true" }, "●"), fmtPrice(lp)) : fmtPrice(p.last_price);
    } },
    { label: "Valor", get: (p) => fmtMoney(parseFloat(p.quantity) * (livePrice(p.symbol) ?? p.last_price)) },
    { label: "Resultado", get: (p) => signed(parseFloat(p.quantity) * (livePrice(p.symbol) ?? p.last_price) - p.allocated) },
    { label: "Rendimiento", get: (p) => fmtPct((parseFloat(p.quantity) * (livePrice(p.symbol) ?? p.last_price) / p.allocated - 1) * 100) },
    { label: "Peso", get: (p) => fmtPct(p.weight_pct, false, 1) },
    { label: "Stop-loss", get: (p) => exitCell(p, p.stop_loss) },
    { label: "Objetivo", get: (p) => exitCell(p, p.take_profit) },
    { label: "Motivo entrada", get: (p) => reasonCell(p.entry_reason), cls: () => "reason" },
  ], a.positions);
  if (!a.positions.length) $("positions-table").append(h("caption", { class: "empty" }, "Sin posiciones abiertas: todo en cash."));
}

function renderLearning(a) {
  const root = $("learning");
  root.replaceChildren();
  const L = a.learning;
  if (!L || !Object.keys(L).length) { root.append(h("p", { class: "empty" }, "Este agente no aprende: sus reglas son fijas.")); return; }
  if (L.error) { root.append(h("p", { class: "empty" }, `No se pudo leer el estado: ${L.error}`)); return; }
  if ("open_theses" in L) { renderAnalystMemory(root, L); return; }
  const kv = h("dl", { class: "kv" });
  const add = (k, v) => kv.append(h("dt", {}, k), h("dd", {}, v));
  if (L.method) add("Método", L.method);
  if ("samples_learned" in L) {
    add("Muestras aprendidas", fmtNum(L.samples_learned, 0) + (L.pretrained_samples ? ` (${fmtNum(L.pretrained_samples, 0)} de pre-entrenamiento)` : ""));
    add("Opera", L.trading_enabled ? "sí" : (L.ic_gate ? "no: sin ventaja estadística (en efectivo)" : "todavía no (aprendiendo)"));
    if (L.horizon_bars) add("Horizonte de predicción", `${L.horizon_bars} día${L.horizon_bars > 1 ? "s" : ""}`);
    add("Correlación por ranking (IC)", isNum(L.rank_ic_mean)
      ? `${fmtNum(L.rank_ic_mean, 3)} · t = ${fmtNum(L.rank_ic_t, 2)} (${L.rank_ic_days} días; t ≥ 2 = ventaja significativa)` : "n/a");
    add("Aciertos de dirección", fmtPct(L.direction_hit_rate_pct, false, 1));
  }
  if (L.params) {
    add("Parámetros actuales", `RSI ${L.params.period} · compra < ${L.params.oversold} · venta > ${L.params.overbought}`);
    add("Último reajuste", L.last_tune ? `${fmtWhen(L.last_tune)} (cada ${L.retune_every_days} días)` : "pendiente");
  }
  root.append(kv);
  if (L.weights) {
    root.append(h("div", { class: "chart-title" }, "Pesos aprendidos (qué mira el modelo)"));
    const max = Math.max(...Object.values(L.weights).map(Math.abs), 1e-9);
    for (const [k, w] of Object.entries(L.weights)) {
      const pct = (Math.abs(w) / max) * 50;
      const color = w >= 0 ? cssVar("--buy") : cssVar("--sell");
      root.append(h("div", { class: "wbar" }, h("span", {}, k),
        h("span", { class: "track" }, h("span", { class: "mid" }),
          h("span", { class: "fill", style: `background:${color};${w >= 0 ? `left:50%;width:${pct}%` : `right:50%;width:${pct}%`}` })),
        h("span", { class: "num" }, fmtNum(w, 3))));
    }
  }
  if (L.recent_tunes?.length) {
    root.append(h("div", { class: "chart-title" }, "Reajustes recientes"));
    const tbl = h("table", { class: "data-table" });
    table(tbl, [
      { label: "Fecha", get: (t) => fmtDay(t.date) },
      { label: "Antes", get: (t) => t.previous.join(" / ") },
      { label: "Mejor", get: (t) => t.best.join(" / ") },
      { label: "Sharpe actual → mejor", get: (t) => `${fmtNum(t.current_sharpe)} → ${fmtNum(t.best_sharpe)}` },
      { label: "¿Cambió?", get: (t) => (t.switched ? "sí" : "no") },
    ], [...L.recent_tunes].reverse());
    root.append(h("div", { class: "table-scroll" }, tbl));
  }
}

const REGIME = { risk_on: "apetito por el riesgo", neutral: "neutral", risk_off: "aversión al riesgo" };
const CONVICTION = { low: "baja", medium: "media", high: "alta" };

/** Memory of the discretionary agent: last view, open theses, lessons, engine, cost. */
function renderAnalystMemory(root, L) {
  const kv = h("dl", { class: "kv" });
  const add = (k, v) => kv.append(h("dt", {}, k), h("dd", {}, v));
  add("Método", "Análisis discrecional: patrones técnicos + macro + noticias + calendario, con memoria");
  add("Motores", (L.engines || []).map((e) => ({ gemini: "Gemini (gratis)", groq: "Groq (gratis)", anthropic: "Claude (de pago)" }[e] || e)).join(" → ") + " → motor local (gratis)");
  if (L.last_engine) add("Última decisión con", L.last_engine);
  add("Decisiones tomadas", String(L.decisions ?? 0));
  if (L.last_regime) add("Lectura del mercado", REGIME[L.last_regime] || L.last_regime);
  add("Coste de IA de pago (este agente)", fmtMoney(L.llm_cost_usd_total || 0));
  root.append(kv);
  if (L.last_view) root.append(h("div", { class: "chart-title" }, "Última visión del mercado"), h("p", { class: "view" }, L.last_view));
  const theses = Object.entries(L.theses || {});
  if (theses.length) {
    root.append(h("div", { class: "chart-title" }, "Tesis abiertas"));
    const tbl = h("table", { class: "data-table wrap theses" });
    table(tbl, [
      { label: "Activo", get: ([s]) => s },
      { label: "Peso", get: ([, t]) => fmtPct(t.target_weight * 100, false, 1) },
      { label: "Convicción", get: ([, t]) => CONVICTION[t.conviction] || t.conviction || "—" },
      { label: "Tesis", get: ([, t]) => t.thesis || "—" },
      { label: "Invalidación", get: ([, t]) => t.invalidation || "—" },
      { label: "Catalizadores", get: ([, t]) => (t.catalysts || []).join(" · ") || "—" },
    ], theses);
    root.append(h("div", { class: "table-scroll" }, tbl));
  }
  if (L.lessons?.length) {
    root.append(h("div", { class: "chart-title" }, "Lecciones que ha anotado"), h("ul", { class: "lessons" }, L.lessons.map((x) => h("li", {}, x))));
  }
  if (!L.decisions) root.append(h("p", { class: "empty" }, "Todavía no ha decidido."));
}

/** Runner and agent messages are written in English in the code; show them in Spanish. */
const TRANSLATIONS = [
  [/^market closed for (\d+) symbols \(e\.g\. ([^)]+)\): no decisions today$/, "Mercado cerrado para $1 activos (p. ej. $2): hoy no se decide"],
  [/^no trades$/, "sin operaciones"], [/^already decided today$/, "ya decidió hoy"], [/^hold$/, "mantener"],
  [/^no prices available$/, "sin precios disponibles"],
  [/^learning: not trading yet$/, "aprendiendo: todavía no opera"], [/^open orders pending: waiting$/, "órdenes pendientes: espera"],
  [/^drawdown (.+) >= halt limit (.+)$/, "drawdown $1 ≥ límite de parada $2"],
  [/^agent wanted (\d+) orders > limit (\d+); nothing sent$/, "el agente pidió $1 órdenes (límite $2): no se envió nada"],
  [/^buy /, "compra "], [/^sell /, "venta "], [/^trim /, "reduce "], [/ to (\d+%)/, " hasta $1"], [/prediction /, "predicción "],
];
const tr = (msg) => TRANSLATIONS.reduce((m, [re, rep]) => m.replace(re, rep), msg || "");
const quiet = (e) => !e.orders?.length && !e.explain?.retuned && !e.explain?.market_view && e.status !== "error" && e.status !== "halted" && e.status !== "rejected";

function renderJournal(a) {
  const ul = $("journal");
  ul.replaceChildren();
  const all = a.journal || [];
  if (!all.length) { ul.append(h("li", { class: "empty" }, "Sin entradas todavía.")); return; }
  const hidden = all.filter(quiet).length;
  const entries = live.journalAll ? all : all.filter((e) => !quiet(e));
  ul.append(h("li", { class: "msg" }, live.journalAll
    ? `Mostrando los ${all.length} días. `
    : `Mostrando los días con actividad (${entries.length}); ${hidden} días sin operaciones ocultos. `,
    h("button", { class: "theme-toggle", type: "button", onclick: () => { live.journalAll = !live.journalAll; renderJournal(a); } },
      live.journalAll ? "Ocultar días sin operaciones" : "Ver todos los días")));
  if (!entries.length) ul.append(h("li", { class: "empty" }, "Todavía no ha operado."));
  for (const e of entries.slice(0, 60)) {
    const li = h("li", {}, h("div", { class: "head" }, h("span", { class: "when" }, fmtWhen(e.time)), statusChip(e.status),
      e.message ? h("span", { class: "msg" }, tr(e.message)) : null,
      isNum(e.equity_after) ? h("span", { class: "msg" }, `equity ${fmtMoney(e.equity_after)}`) : null));
    const ex = e.explain || {};
    if (ex.market_view) {
      li.append(h("p", { class: "view" }, h("strong", {}, `Visión (${REGIME[ex.regime] || ex.regime || "?"}${ex.engine ? ` · ${ex.engine}` : ""}): `), ex.market_view));
      const bits = [];
      if (ex.changes?.length) bits.push(...ex.changes.map((c) => `Cambio: ${c}`));
      if (ex.key_risks?.length) bits.push(`Riesgos: ${ex.key_risks.join(" · ")}`);
      if (ex.watchlist?.length) bits.push(`Vigilando: ${ex.watchlist.join(" · ")}`);
      if (ex.lessons_new?.length) bits.push(...ex.lessons_new.map((x) => `Lección: ${x}`));
      if (ex.sources?.length) bits.push(`Consultó en la web: ${ex.sources.slice(0, 5).map((s) => s.title || s.url).join(" · ")}`);
      if (isNum(ex.usage?.cost_usd_estimate)) bits.push(`Coste de esta decisión: ${fmtMoney(ex.usage.cost_usd_estimate)}`);
      if (bits.length) li.append(h("ul", { class: "signals" }, bits.map((b) => h("li", {}, b))));
    } else if (ex.status === "inactive" || (ex.status === "error" && ex.message)) {
      li.append(h("ul", { class: "signals" }, h("li", {}, ex.message)));
    }
    const sig = e.explain?.signals;
    if (sig?.length) li.append(h("ul", { class: "signals" }, sig.slice(0, 8).map((s) => h("li", {}, tr(s)))));
    if (e.explain?.retuned) {
      const r = e.explain.retuned;
      li.append(h("ul", { class: "signals" }, h("li", {}, `Reajuste: mejor ${r.best.join("/")} (Sharpe ${fmtNum(r.best_sharpe)}) vs actual ${fmtNum(r.current_sharpe)} → ${r.switched ? "cambia" : "se mantiene"}`)));
    }
    if (e.orders?.length) {
      const tbl = h("table", { class: "data-table orders" });
      table(tbl, [
        { label: "Orden", get: (o) => (o.side === "buy" ? "compra" : "venta"), cls: (o) => `side-${o.side}` },
        { label: "Activo", get: (o) => o.symbol },
        { label: "Cantidad", get: (o) => o.quantity },
        { label: "Precio medio", get: (o) => (parseFloat(o.avg_fill_price) ? fmtPrice(parseFloat(o.avg_fill_price)) : "—") },
        { label: "Estado", get: (o) => o.status + (o.reject_reason ? `: ${o.reject_reason}` : "") },
        { label: "Motivo", get: (o) => reasonCell(o.tag), cls: () => "reason" },
      ], e.orders);
      li.append(h("div", { class: "table-scroll" }, tbl));
    }
    if (e.dropped_symbols) li.append(h("div", { class: "msg" }, `Sin datos: ${Object.keys(e.dropped_symbols).join(", ")}`));
    ul.append(li);
  }
}

// ---------------------------------------------------------------- selection & boot
function selectAccount(i, scroll = false) {
  live.account = i;
  $("sel-account").value = i;
  history.replaceState(null, "", `#account=${encodeURIComponent(live.data.accounts[i].id)}`);
  renderAccounts();
  renderAccount();
  if (scroll) $("detail").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderAll() {
  assignColors();
  renderStatus();
  renderAccounts();
  renderEdge();
  renderPerformance();
  renderPrices();
  $("sel-account").replaceChildren(...live.data.accounts.map((a, i) => h("option", { value: i }, a.name)));
  $("sel-account").value = live.account;
  renderAccount();
}

async function boot() {
  initTheme(() => { if (live.data) renderAll(); });
  try {
    [live.source, live.data] = await loadLiveData();
  } catch (err) {
    $("statusbar").textContent = "";
    $("app").replaceChildren(h("div", { class: "error" },
      `No se pudo cargar live.json (${err.message}). Se genera con `, h("code", {}, "python scripts/live_run.py"),
      "; la primera ejecución programada lo crea."));
    return;
  }
  const want = new URLSearchParams(location.hash.slice(1)).get("account");
  const idx = live.data.accounts.findIndex((a) => a.id === want);
  if (idx >= 0) live.account = idx;
  $("sel-account").addEventListener("change", (ev) => selectAccount(+ev.target.value));
  renderAll();
  $("app").setAttribute("aria-busy", "false");
  refreshLivePrices();
  live.timer = setInterval(refreshLivePrices, LIVE_REFRESH_MS);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshLivePrices(); });
  setInterval(renderStatus, 60_000);  // keep "hace X min" fresh
}

boot();
