/* Research page: calendar, news analyses, macro and technical patterns from live.json ("research"). */
"use strict";

const R = { data: null, source: null };

// ---------------------------------------------------------------- vocabulary
const PATTERNS = {
  doji: "Doji", hammer: "Martillo", hanging_man: "Hombre colgado", shooting_star: "Estrella fugaz",
  inverted_hammer: "Martillo invertido", bullish_engulfing: "Envolvente alcista", bearish_engulfing: "Envolvente bajista",
  bullish_harami: "Harami alcista", bearish_harami: "Harami bajista", piercing_line: "Línea penetrante",
  dark_cloud_cover: "Nube oscura", morning_star: "Estrella de la mañana", evening_star: "Estrella del atardecer",
  three_white_soldiers: "Tres soldados blancos", three_black_crows: "Tres cuervos negros", inside_bar: "Barra interior",
  resistance_breakout: "Ruptura de resistencia", support_breakdown: "Pérdida de soporte", double_top: "Doble techo",
  double_bottom: "Doble suelo", head_and_shoulders: "Hombro-cabeza-hombro", inverse_head_and_shoulders: "HCH invertido",
  ascending_triangle: "Triángulo ascendente", descending_triangle: "Triángulo descendente",
  symmetrical_triangle: "Triángulo simétrico", rising_wedge: "Cuña ascendente", falling_wedge: "Cuña descendente",
  golden_cross: "Cruce dorado (50/200)", death_cross: "Cruce de la muerte (50/200)", new_52w_high: "Máximo de 52 semanas",
  new_52w_low: "Mínimo de 52 semanas", bullish_rsi_divergence: "Divergencia alcista RSI",
  bearish_rsi_divergence: "Divergencia bajista RSI", rsi_oversold: "RSI en sobreventa", rsi_overbought: "RSI en sobrecompra",
  macd_bullish_cross: "Cruce alcista MACD", macd_bearish_cross: "Cruce bajista MACD", bollinger_squeeze: "Compresión de Bollinger",
  gap_up: "Hueco alcista", gap_down: "Hueco bajista", volume_spike: "Pico de volumen",
};
const LEVELS = { neckline: "línea clavicular", target: "objetivo", invalidation: "invalidación", level: "nivel",
  upper: "directriz superior", lower: "directriz inferior", gap_fill: "cierre del hueco" };
const CATEGORY = { central_bank: "Bancos centrales", inflation: "Inflación", employment: "Empleo", growth: "Crecimiento",
  consumer: "Consumo", earnings: "Resultados", dividend: "Dividendos", options_expiry: "Vencimientos", holiday: "Festivos",
  politics: "Política", crypto: "Cripto", commodities: "Materias primas", other: "Otros" };
const NEWS_CATEGORY = { earnings: "resultados", guidance: "previsiones", macro: "macro", central_bank: "banco central",
  policy: "política", regulation: "regulación", geopolitics: "geopolítica", corporate: "empresa", product: "producto",
  analyst_rating: "analistas", markets: "mercados", crypto: "cripto", commodities: "materias primas", opinion: "opinión", other: "otros" };
const IMPORTANCE = { high: ["alta", "●●●"], medium: ["media", "●●○"], low: ["baja", "●○○"] };
const IMPACT = { high: "alto", medium: "medio", low: "bajo" };
const DIRECTION = { up: ["▲", "sube", "up"], down: ["▼", "baja", "down"], mixed: ["◆", "mixto", "flat"], unclear: ["●", "incierto", "flat"] };
const PRICED = { likely: "probablemente sí", partially: "en parte", unlikely: "probablemente no", unknown: "no se sabe" };
const HORIZON = { intraday: "intradía", days: "días", weeks: "semanas", months: "meses" };
const MAGNITUDE = { low: "leve", medium: "moderado", high: "fuerte" };
const PROFILE = { equity_broad: "Índices de acciones (SPY, QQQ, IWM, DIA)", "Information Technology": "Tecnología",
  "Communication Services": "Comunicaciones", "Consumer Discretionary": "Consumo discrecional",
  "Consumer Staples": "Consumo básico", Financials: "Financieras", Energy: "Energía", "Health Care": "Salud",
  Industrials: "Industriales", bonds: "Bonos (TLT)", precious_metals: "Metales preciosos (GLD, SLV)",
  emerging: "Emergentes (EEM)", crypto: "Cripto" };
const MACRO_GROUPS = [
  ["Política monetaria y tipos", ["fed_funds", "ust_3m_live", "ust_2y", "ust_10y", "ust_10y_live", "curve_10y_2y", "real_10y", "hy_spread"]],
  ["Inflación", ["cpi_yoy", "core_cpi_yoy", "core_pce_yoy"]],
  ["Empleo y actividad", ["unemployment", "payrolls_change", "jobless_claims", "gdp_growth", "retail_sales_yoy", "consumer_sentiment", "m2_yoy"]],
  ["Mercados", ["vix", "dxy", "wti", "gold", "copper"]],
];
const TITLE_ES = [
  [/^FOMC rate decision( \+ economic projections \(SEP\))? and Powell press conference$/, (m, sep) => `Fed: decisión de tipos${sep ? " + proyecciones" : ""} y rueda de prensa de Powell`],
  [/^FOMC minutes$/, "Fed: actas del FOMC"],
  [/^Employment Situation \(nonfarm payrolls, unemployment rate\)$/, "Informe de empleo (nóminas no agrícolas y paro)"],
  [/^CPI \(consumer prices\)$/, "IPC (precios al consumo)"], [/^PPI \(producer prices\)$/, "IPP (precios de producción)"],
  [/^Retail sales$/, "Ventas minoristas"], [/^PCE inflation \/ personal income and outlays$/, "Inflación PCE / ingresos y gastos personales"],
  [/^GDP \((advance|second|third) estimate\)$/, (m, k) => `PIB (${{ advance: "avance", second: "segunda estimación", third: "tercera estimación" }[k]})`],
  [/^ISM manufacturing PMI$/, "PMI manufacturero ISM"], [/^ISM services PMI$/, "PMI de servicios ISM"],
  [/^UMich consumer sentiment \(preliminary\)$/, "Confianza del consumidor UMich (preliminar)"],
  [/^Initial jobless claims$/, "Peticiones semanales de paro"],
  [/^Monthly options expiry$/, "Vencimiento mensual de opciones"],
  [/^Quarterly options\/futures expiry \(quad witching\)$/, "Vencimiento trimestral (cuádruple hora bruja)"],
  [/^US stock market holiday \(NYSE\/Nasdaq closed\)$/, "Festivo: bolsa de EE. UU. cerrada"],
  [/^(\S+) quarterly earnings$/, "$1: resultados trimestrales"], [/^(\S+) ex-dividend date$/, "$1: fecha ex-dividendo"],
];
const titleEs = (t) => TITLE_ES.reduce((s, [re, rep]) => s.replace(re, rep), t || "");

// ---------------------------------------------------------------- helpers
const fmtWhen = (iso) => (iso ? new Date(iso).toLocaleString("es-ES", { dateStyle: "medium", timeStyle: "short" }) : "n/a");
function fmtAgo(iso) {
  if (!iso) return "";
  const min = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (min < 1) return "ahora mismo";
  if (min < 60) return `hace ${min} min`;
  const hours = Math.round(min / 60);
  return hours < 48 ? `hace ${hours} h` : `hace ${Math.round(hours / 24)} días`;
}
const dirMark = (d) => {
  const [icon, label, cls] = d === "bullish" ? ["▲", "alcista", "up"] : d === "bearish" ? ["▼", "bajista", "down"] : ["●", "neutral", "flat"];
  return h("span", { class: `tri ${cls}`, title: label, "aria-label": label }, icon);
};
const assetsLabel = (syms) => (syms || []).map((s) => ({ "*all": "todos", "*equities": "acciones", "*crypto": "cripto" }[s] || s)).join(", ");
const nyDate = (d) => new Date(`${d}T12:00:00Z`);
function dayLabel(d) {
  const today = new Date(new Date().toLocaleDateString("en-CA", { timeZone: "America/New_York" }) + "T12:00:00Z");
  const diff = Math.round((nyDate(d) - today) / 86400000);
  const name = nyDate(d).toLocaleDateString("es-ES", { weekday: "long", day: "numeric", month: "long", timeZone: "UTC" });
  const rel = diff === 0 ? "hoy" : diff === 1 ? "mañana" : diff > 1 ? `en ${diff} días` : diff === -1 ? "ayer" : `hace ${-diff} días`;
  return [name.charAt(0).toUpperCase() + name.slice(1), rel];
}
function eventTime(e) {
  if (!e.when) return "";
  const t = new Date(e.when);
  const local = t.toLocaleTimeString("es-ES", { hour: "2-digit", minute: "2-digit" });
  const ny = t.toLocaleTimeString("es-ES", { hour: "2-digit", minute: "2-digit", timeZone: "America/New_York" });
  return local === ny ? `${ny} NY` : `${local} (${ny} NY)`;
}
function optionList(sel, values, labelOf, allLabel = "todos") {
  const cur = sel.value;
  sel.replaceChildren(h("option", { value: "" }, allLabel), ...values.map((v) => h("option", { value: v }, labelOf ? labelOf(v) : v)));
  if (values.includes(cur)) sel.value = cur;
}

// ---------------------------------------------------------------- status
function renderStatus() {
  const r = R.data.research;
  const bar = $("statusbar");
  bar.replaceChildren();
  const st = r.status || {};
  bar.append(h("span", {}, "Investigación actualizada: ", h("strong", {}, fmtWhen(st.started_at)), ` (${fmtAgo(st.started_at)}, ${fmtNum(st.duration_s, 0)} s)`));
  const llmOn = st.llm === "enabled";
  bar.append(h("span", {}, "Motor: ", h("strong", {}, (st.engines || ["local (gratis, sin IA)"]).join(" → "))));
  if (isNum(r.llm_cost_usd_total) && r.llm_cost_usd_total > 0) bar.append(h("span", {}, `coste estimado de IA de pago: ${fmtMoney(r.llm_cost_usd_total)} en total`));
  else bar.append(h("span", {}, "coste: 0 $"));
  const nErr = Object.keys(r.errors || {}).length;
  bar.append(h("span", { title: Object.entries(r.errors || {}).map(([k, v]) => `${k}: ${v}`).join("\n") },
    nErr ? `⚠ ${nErr} fuente${nErr > 1 ? "s" : ""} con incidencias` : "✓ todas las fuentes respondieron"));
  if (R.source) bar.append(h("span", {}, `datos: ${R.source}`));
  $("notice").replaceChildren();
  if (!llmOn) {
    $("notice").append(h("div", { class: "notice", role: "note" }, h("span", { class: "icon", "aria-hidden": "true" }, "i"),
      h("span", {}, h("strong", {}, "Modo gratuito sin IA. "),
        "Todo se analiza con el motor local (diccionario financiero y reglas). Para análisis con IA gratis añade una clave gratuita de Google AI Studio (",
        h("code", {}, "GEMINI_API_KEY"), ") o de Groq (", h("code", {}, "GROQ_API_KEY"), ") en GitHub → Settings → Secrets and variables → Actions.")));
  }
  $("footer").replaceChildren(`Datos generados ${fmtWhen(R.data.meta.generated_at)} · modelos: noticias ${r.models?.triage_model || "?"} (clasificación) y `,
    `${r.models?.deep_model || "?"} (análisis), calendario ${r.models?.calendar_model || "?"} · fuentes: Yahoo Finance, Google News, CoinDesk, `,
    "Reserva Federal, FRED, Forex Factory. Información para los agentes de práctica, no es asesoramiento financiero.");
}

// ---------------------------------------------------------------- calendar
function calendarEvents() {
  return R.data.research.calendar?.events || [];
}
function eventMatches(e, minImp, cat, asset) {
  const rank = { high: 0, medium: 1, low: 2 };
  if (rank[e.importance] > rank[minImp]) return false;
  if (cat && e.category !== cat) return false;
  if (asset) {
    const cls = R.data.meta.asset_class?.[asset];
    const syms = e.symbols || [];
    if (!(syms.includes(asset) || syms.includes("*all") || (syms.includes("*equities") && cls !== "crypto") ||
      (syms.includes("*crypto") && cls === "crypto"))) return false;
  }
  return true;
}
function eventRow(e) {
  const [impLabel, dots] = IMPORTANCE[e.importance] || ["", ""];
  const es = (v) => String(v).replace(/\brevenue\b/g, "ingresos").replace(/\bEPS\b/g, "BPA");
  const nums = [["previsión", e.forecast], ["anterior", e.previous], ["dato", e.actual]].filter(([, v]) => v)
    .map(([k, v]) => h("span", { class: k === "dato" ? "actual" : "" }, `${k} ${es(v)}`));
  return h("li", { class: `ev imp-${e.importance}` },
    h("span", { class: "imp", title: `importancia ${impLabel}`, "aria-label": `importancia ${impLabel}` }, dots),
    h("span", { class: "time" }, eventTime(e) || (e.category === "earnings" ? "hora por confirmar" : "todo el día")),
    h("span", { class: "country" }, e.country),
    h("span", { class: "title" }, titleEs(e.title),
      e.confirmed === false ? h("span", { class: "tag", title: "fecha calculada por la regla habitual; se confirma cuando la fuente oficial la publica" }, "fecha estimada") : null,
      e.source === "claude-web" ? h("span", { class: "tag", title: e.notes || "" }, "añadido por la IA") : null),
    h("span", { class: "nums" }, ...nums),
    h("span", { class: "meta" }, `${CATEGORY[e.category] || e.category} · ${assetsLabel(e.symbols)}`));
}
function renderCalendar() {
  const cal = R.data.research.calendar || {};
  const events = calendarEvents();
  const minImp = $("cal-importance").value, cat = $("cal-category").value, asset = $("cal-asset").value;
  const todayNY = new Date().toLocaleDateString("en-CA", { timeZone: "America/New_York" });
  const future = events.filter((e) => e.date >= todayNY && eventMatches(e, minImp, cat, asset));
  const past = events.filter((e) => e.date < todayNY && eventMatches(e, minImp, cat, asset)).reverse();
  const nHigh = events.filter((e) => e.date >= todayNY && e.importance === "high").length;
  $("cal-hint").replaceChildren(`${events.filter((e) => e.date >= todayNY).length} eventos hasta el ${cal.window ? fmtDayShort(cal.window[1]) : "?"} (${nHigh} de importancia alta). `,
    cal.researched_at ? `Verificado con búsqueda web ${fmtAgo(cal.researched_at)}. ` : "Sin verificar con búsqueda web todavía. ",
    "Horas en tu zona horaria y en Nueva York.");
  const sum = $("cal-summary");
  sum.hidden = !cal.summary;
  if (cal.summary) sum.replaceChildren(h("strong", {}, "Resumen del mes (IA): "), cal.summary);
  $("cal-list").replaceChildren(...groupByDay(future));
  if (!future.length) $("cal-list").append(h("p", { class: "empty" }, "Ningún evento con estos filtros."));
  $("cal-past").replaceChildren(...groupByDay(past));
  if (!past.length) $("cal-past").append(h("p", { class: "empty" }, "Nada en los últimos 7 días con estos filtros."));
}
const fmtDayShort = (d) => nyDate(d).toLocaleDateString("es-ES", { day: "numeric", month: "long", timeZone: "UTC" });
function groupByDay(events) {
  const out = [];
  let cur = null, ul = null;
  for (const e of events) {
    if (e.date !== cur) {
      cur = e.date;
      const [name, rel] = dayLabel(e.date);
      ul = h("ul", { class: "events" });
      out.push(h("div", { class: "day" }, h("div", { class: "day-head" }, h("strong", {}, name), h("span", {}, rel)), ul));
    }
    ul.append(eventRow(e));
  }
  return out;
}

// ---------------------------------------------------------------- news
function eventTitle(id) {
  const e = calendarEvents().find((x) => x.id === id);
  return e ? `${titleEs(e.title)} (${fmtDayShort(e.date)})` : null;
}
function analysisCard(a, open) {
  const an = a.analysis;
  const chips = (an.affected || []).map((x) => {
    const [icon, label, cls] = DIRECTION[x.direction] || DIRECTION.unclear;
    return h("span", { class: "chip", title: `${label}, ${MAGNITUDE[x.magnitude] || x.magnitude}, ${HORIZON[x.horizon] || x.horizon}` },
      h("span", { class: `tri ${cls}`, "aria-hidden": "true" }, icon), x.symbol);
  });
  const related = (an.related_events || []).map(eventTitle).filter(Boolean);
  const body = h("div", { class: "an-body" },
    h("p", {}, h("strong", {}, "Qué pasó. "), an.what_happened || ""),
    h("p", {}, h("strong", {}, "Por qué importa. "), an.why_it_matters || ""),
    an.affected?.length ? h("div", { class: "table-scroll" }, (() => {
      const t = h("table", { class: "data-table wrap" });
      table(t, [
        { label: "Activo", get: (x) => x.symbol },
        { label: "Efecto", get: (x) => { const [i, l, c] = DIRECTION[x.direction] || DIRECTION.unclear; return h("span", {}, h("span", { class: `tri ${c}` }, i), ` ${l}`); } },
        { label: "Intensidad", get: (x) => MAGNITUDE[x.magnitude] || x.magnitude },
        { label: "Plazo", get: (x) => HORIZON[x.horizon] || x.horizon },
        { label: "Por qué", get: (x) => x.rationale },
      ], an.affected);
      return t;
    })()) : null,
    an.macro_link ? h("p", {}, h("strong", {}, "Contexto macro. "), an.macro_link) : null,
    h("p", { class: "facts" }, `¿Descontado por el mercado? ${PRICED[an.priced_in] || an.priced_in} · confianza ${fmtPct((an.confidence ?? 0) * 100, false, 0)}`
      + (an.read_full_article ? " · leyó el artículo completo" : "")
      + ` · ${String(an.model || "").startsWith("local") ? "análisis automático del motor local (sin IA)" : `IA: ${String(an.model || "").split(":")[0]}`}`),
    an.what_to_watch ? h("p", {}, h("strong", {}, "Qué vigilar. "), an.what_to_watch) : null,
    related.length ? h("p", {}, h("strong", {}, "Eventos relacionados. "), related.join(" · ")) : null,
    h("p", { class: "facts" }, h("a", { href: a.url, target: "_blank", rel: "noopener" }, `Fuente: ${a.publisher || "original"} ↗`),
      ` · titular original: ${a.title}`));
  return h("details", { class: "analysis", open: open || null },
    h("summary", {}, h("span", { class: "when" }, fmtAgo(a.published)), h("span", { class: "headline" }, an.headline || a.title),
      h("span", { class: "chips" }, ...chips)), body);
}
function renderNews() {
  const n = R.data.research.news || {};
  const asset = $("news-asset").value;
  const touches = (syms) => !asset || (syms || []).includes(asset);
  const analyses = (n.analyses || []).filter((a) => touches((a.analysis.affected || []).map((x) => x.symbol).concat(a.triage?.symbols || [])));
  const eng = Object.entries(R.data.research.status?.news_engines || {}).map(([k, v]) => `${k === "local" ? "motor local" : k.split(":")[0]} ${v}`).join(", ");
  $("news-hint").textContent = `${n.stored ?? 0} noticias de los últimos 7 días. Todas se clasifican (relevancia, sentimiento e impacto) y las importantes se analizan en detalle`
    + (eng ? ` (última actualización: ${eng})` : "")
    + (n.untriaged ? `; ${n.untriaged} pendientes de clasificar.` : ".");
  $("analyses").replaceChildren(...analyses.slice(0, 40).map((a, i) => analysisCard(a, i < 2)));
  if (!analyses.length) $("analyses").append(h("p", { class: "empty" }, R.data.research.status?.llm === "enabled"
    ? "Todavía no hay análisis detallados para este filtro." : "Todavía no hay análisis detallados para este filtro."));
  const feed = (n.feed || []).filter((f) => touches(f.symbols));
  table($("feed-table"), [
    { label: "Hora", get: (f) => fmtAgo(f.published) },
    { label: "Relevancia", get: (f) => (isNum(f.relevance) ? relevanceBar(f.relevance) : "—") },
    { label: "Sentimiento", get: (f) => (isNum(f.sentiment) ? h("span", {}, dirMark(f.sentiment > 0.15 ? "bullish" : f.sentiment < -0.15 ? "bearish" : "neutral"), ` ${fmtNum(f.sentiment, 1)}`) : "—") },
    { label: "Impacto", get: (f) => IMPACT[f.impact] || "—" },
    { label: "Activos", get: (f) => (f.symbols || []).join(", ") || "macro" },
    { label: "Noticia", get: (f) => h("a", { href: f.url, target: "_blank", rel: "noopener", title: f.title },
      f.summary || f.title, f.category ? h("span", { class: "tag" }, NEWS_CATEGORY[f.category] || f.category) : null) },
  ], feed.slice(0, 150));
  if (!feed.length) $("feed-table").append(h("caption", { class: "empty" }, "Sin noticias para este filtro."));
}
function relevanceBar(v) {
  return h("span", { class: "rbar", title: fmtNum(v, 2), "aria-label": `relevancia ${fmtNum(v, 2)}` },
    h("span", { class: "rtrack" }, h("span", { class: "rfill", style: `width:${Math.round(v * 100)}%` })), fmtNum(v, 1));
}

// ---------------------------------------------------------------- macro
function renderMacro() {
  const m = R.data.research.macro || {};
  const ind = m.indicators || {};
  $("macro-hint").textContent = `Datos oficiales (FRED) y de mercado (Yahoo), actualizados ${fmtAgo(m.generated_at)}.`;
  const rows = [];
  for (const [group, ids] of MACRO_GROUPS) {
    const present = ids.filter((id) => ind[id]);
    if (!present.length) continue;
    rows.push({ group });
    for (const id of present) rows.push(ind[id]);
  }
  const fmtVal = (r, v) => {
    if (!isNum(v)) return "—";
    if (r.unit === "miles") return `${fmtNum(v, 0)} mil`;
    return `${fmtNum(v, Math.abs(v) >= 1000 ? 0 : 2)}${r.unit === "%" ? " %" : r.unit === "pp" ? " pp" : ""}`;
  };
  // Macro moves are not good or bad per se (falling yields help bonds, hurt banks): neutral arrows.
  const trendMark = (t) => h("span", { class: "tri flat", "aria-hidden": "true" }, t === "up" ? "▲" : t === "down" ? "▼" : "●");
  table($("macro-table"), [
    { label: "Indicador", get: (r) => (r.group ? h("strong", {}, r.group) : h("span", { class: "indent" }, r.name)) },
    { label: "Valor", get: (r) => (r.group ? "" : fmtVal(r, r.value)) },
    { label: "Anterior", get: (r) => (r.group ? "" : fmtVal(r, r.previous)) },
    { label: "Tendencia", get: (r) => (r.group || !r.trend ? "" : h("span", {}, trendMark(r.trend),
      r.trend === "up" ? " sube" : r.trend === "down" ? " baja" : " estable")) },
    { label: "1 mes", get: (r) => (r.group || !isNum(r.change_1m_pct) ? "" : fmtPct(r.change_1m_pct, true, 1)) },
    { label: "Fecha", get: (r) => (r.group ? "" : r.as_of + (r.stale ? " (sin actualizar)" : "")) },
  ], rows);
  const names = Object.fromEntries(Object.entries(ind).map(([k, v]) => [k, v.name]));
  const prof = m.profiles || {};
  const rel = Object.entries(m.relevance || {}).map(([k, ids]) => ({ k, ids, assets: Object.keys(prof).filter((s) => prof[s] === k) }))
    .filter((r) => r.assets.length);
  table($("relevance-table"), [
    { label: "Tipo de activo", get: (r) => h("span", {}, h("strong", {}, PROFILE[r.k] || r.k), h("br"), h("span", { class: "muted" }, r.assets.join(", "))) },
    { label: "Datos que lo mueven", get: (r) => r.ids.map((id) => names[id] || id).join(" · ") },
  ], rel);
}

// ---------------------------------------------------------------- technicals
function scoreBar(v) {
  const pct = Math.min(Math.abs(v), 1) * 50;
  const color = v >= 0 ? cssVar("--buy") : cssVar("--sell");
  return h("span", { class: "sbar", title: fmtNum(v, 2) },
    h("span", { class: "track" }, h("span", { class: "mid" }),
      h("span", { class: "fill", style: `background:${color};${v >= 0 ? `left:50%;width:${pct}%` : `right:50%;width:${pct}%`}` })),
    h("span", { class: "num" }, (v > 0 ? "+" : "") + fmtNum(v, 2)));
}
function patternChip(p) {
  const lv = Object.entries(p.levels || {}).map(([k, v]) => `${LEVELS[k] || k} ${fmtPrice(v)}`).join(", ");
  const when = p.bars_ago ? ` · hace ${p.bars_ago} sesión${p.bars_ago > 1 ? "es" : ""}` : "";
  const status = p.status === "forming" ? " (en formación)" : "";
  return h("span", { class: "chip pat", title: `fuerza ${fmtNum(p.strength, 2)}${when}${lv ? " · " + lv : ""}` },
    dirMark(p.direction), `${PATTERNS[p.name] || p.name}${status}`);
}
function renderTechnicals() {
  const t = R.data.research.technicals || {};
  const cls = $("tech-class").value, sort = $("tech-sort").value;
  let rows = Object.entries(t.symbols || {}).map(([sym, r]) => ({ sym, ...r }))
    .filter((r) => !cls || R.data.meta.asset_class?.[r.sym] === cls);
  rows.sort(sort === "symbol" ? (a, b) => a.sym.localeCompare(b.sym) : sort === "-score" ? (a, b) => a.score - b.score : (a, b) => b.score - a.score);
  $("tech-hint").textContent = `Velas, figuras de precio, soportes y resistencias, cruces y divergencias en barras diarias completas. Actualizado ${fmtAgo(t.generated_at)}. Pasa el cursor por un patrón para ver sus niveles.`;
  table($("tech-table"), [
    { label: "Activo", get: (r) => h("span", { title: R.data.meta.names?.[r.sym] || "" }, r.sym) },
    { label: "Precio", get: (r) => fmtPrice(r.close) },
    { label: "Puntuación", get: (r) => scoreBar(r.score ?? 0) },
    { label: "Tendencia 60 d", get: (r) => h("span", { title: r.trend?.ma_alignment || "" }, dirMark(r.trend?.direction), ` ${fmtPct(r.trend?.change_pct, true, 1)}`) },
    { label: "RSI", get: (r) => fmtNum(r.rsi, 0) },
    { label: "Soporte", get: (r) => (r.support ? `${fmtPrice(r.support.price)} (${r.support.touches}×)` : "—") },
    { label: "Resistencia", get: (r) => (r.resistance ? `${fmtPrice(r.resistance.price)} (${r.resistance.touches}×)` : "—") },
    { label: "Patrones activos", get: (r) => {
      const seen = new Set();
      const pats = (r.patterns || []).filter((p) => !seen.has(p.name) && seen.add(p.name));  // strongest first
      return pats.length ? h("span", { class: "chips" }, ...pats.slice(0, 5).map(patternChip)) : "ninguno";
    } },
  ], rows);
}

// ---------------------------------------------------------------- boot
function renderAll() {
  renderStatus();
  renderCalendar();
  renderNews();
  renderMacro();
  renderTechnicals();
}

async function boot() {
  initTheme(() => { if (R.data) renderAll(); });
  try {
    [R.source, R.data] = await loadLiveData();
  } catch (err) {
    $("statusbar").textContent = "";
    $("app").replaceChildren(h("div", { class: "error" }, `No se pudo cargar live.json (${err.message}).`));
    return;
  }
  if (!R.data.research || !R.data.research.status) {
    $("statusbar").textContent = "";
    $("app").replaceChildren(h("div", { class: "notice" }, h("span", { class: "icon" }, "⏳"),
      h("span", {}, "La investigación aún no se ha ejecutado: aparecerá tras la próxima ejecución horaria.")));
    return;
  }
  const syms = R.data.meta.symbols;
  optionList($("cal-asset"), syms);
  optionList($("news-asset"), syms);
  const cats = [...new Set(calendarEvents().map((e) => e.category))].sort((a, b) => (CATEGORY[a] || a).localeCompare(CATEGORY[b] || b));
  optionList($("cal-category"), cats, (c) => CATEGORY[c] || c);
  for (const id of ["cal-importance", "cal-category", "cal-asset"]) $(id).addEventListener("change", renderCalendar);
  $("news-asset").addEventListener("change", renderNews);
  for (const id of ["tech-class", "tech-sort"]) $(id).addEventListener("change", renderTechnicals);
  renderAll();
  $("app").setAttribute("aria-busy", "false");
  setInterval(renderStatus, 60_000);
}

boot();
