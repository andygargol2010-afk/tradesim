/* Chronicle page: the daily journal written by the AI chronicler (diario.json). */
"use strict";

const D = { data: null, source: null, shown: 14 };

async function loadDiario() {
  const remote = document.querySelector('meta[name="tradesim-data-url"]')?.content;
  const sources = [["repositorio", remote], ["despliegue", "diario.json"]].filter(([, url]) => url);
  const got = await Promise.allSettled(sources.map(async ([name, url]) => {
    const res = await fetch(url, { cache: "no-cache" });
    if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
    return [name, await res.json()];
  }));
  const ok = got.filter((r) => r.status === "fulfilled").map((r) => r.value);
  if (!ok.length) throw new Error(got.map((r) => r.reason?.message).join("; "));
  ok.sort((a, b) => (b[1].updated_at || "").localeCompare(a[1].updated_at || ""));
  return ok[0];
}

const pctCell = (v) => (isNum(v) ? h("span", { class: `tri ${v > 0 ? "up" : v < 0 ? "down" : "flat"}` }, fmtPct(v)) : "—");

function renderStats() {
  const rows = [];
  for (const s of Object.values(D.data.stats || {})) {
    for (const [tag, v] of Object.entries(s.by_situation)) rows.push({ name: s.name, tag, ...v });
  }
  table($("stats-table"), [
    { label: "Agente", get: (r) => r.name },
    { label: "Situación", get: (r) => r.tag },
    { label: "Días", get: (r) => r.days },
    { label: "Días con operaciones", get: (r) => `${r.days_traded} (${Math.round((r.days_traded / r.days) * 100)} %)` },
    { label: "Compras", get: (r) => r.buys },
    { label: "Ventas", get: (r) => r.sells },
    { label: "Resultado medio del día", get: (r) => pctCell(r.avg_day_pnl_pct) },
  ], rows);
}

function list(title, items) {
  if (!items?.length) return null;
  return h("div", {}, h("h4", {}, title), h("ul", { class: "signals" }, items.map((x) => h("li", {}, x))));
}

function entryCard(e) {
  const t = e.text || {};
  const names = Object.fromEntries((e.facts?.agents || []).map((a) => [a.id, a]));
  const agents = (t.agentes || []).map((a) => {
    const f = names[a.id] || {};
    return h("div", { class: "card agent-day" },
      h("h4", {}, f.name || a.id, " ", pctCell(f.day_pnl_pct)),
      h("dl", {},
        h("dt", {}, "Qué hizo"), h("dd", {}, a.que_hizo),
        h("dt", {}, "Por qué"), h("dd", {}, a.por_que),
        h("dt", {}, "Según la situación"), h("dd", {}, a.segun_situacion),
        h("dt", {}, "Resultado"), h("dd", {}, a.resultado)));
  });
  const pats = (t.patrones_mercado || []).map((p) => `${p.patron} (${(p.activos || []).join(", ")}): ${p.lectura}`);
  return h("section", { class: "card", id: `d-${e.date}` },
    h("h2", {}, `${e.date} · ${t.titulo || ""}`),
    h("p", {}, h("span", { class: "chip" }, e.situation_tag), " ", t.situacion || ""),
    list("Patrones del mercado", pats),
    h("h3", {}, "Qué hizo cada agente"),
    h("div", { class: "agent-days" }, agents),
    t.comparativa ? h("p", {}, h("strong", {}, "Comparativa. "), t.comparativa) : null,
    list("Patrones recurrentes", t.patrones_recurrentes),
    list("A vigilar", t.a_vigilar),
    h("p", { class: "hint" }, `Escrito con ${e.engine} el ${shortDate(e.written_at).slice(0, 16)} UTC.`));
}

function renderEntries() {
  const entries = D.data.entries || [];
  const more = entries.length > D.shown
    ? h("button", { class: "theme-toggle", type: "button", onclick: () => { D.shown += 14; renderEntries(); } }, "Ver días anteriores")
    : null;
  $("entries").replaceChildren(...entries.slice(0, D.shown).map(entryCard), ...(more ? [more] : []));
}

async function boot() {
  initTheme(() => {});
  try {
    [D.source, D.data] = await loadDiario();
  } catch (err) {
    $("statusbar").textContent = "";
    $("app").replaceChildren(h("div", { class: "notice" }, h("span", { class: "icon" }, "⏳"),
      h("span", {}, "El diario aún no tiene entradas: la primera se escribe tras el primer día completo de simulación.")));
    return;
  }
  $("statusbar").textContent = `${D.data.entries.length} días · actualizado ${shortDate(D.data.updated_at).slice(0, 16)} UTC · fuente: ${D.source}`;
  renderStats();
  renderEntries();
  $("app").setAttribute("aria-busy", "false");
}

boot();
