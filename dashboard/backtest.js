/* Backtest page: reads data.json (scripts/build_dashboard.py). Needs charts.js. */
"use strict";

const state = { data: null, agent: 0, episode: 0, symbol: null };
const agentColor = (i) => cssVar(SERIES_VARS[i % SERIES_VARS.length]);

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
async function boot() {
  initTheme(() => { if (state.data) renderAll(); });
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
