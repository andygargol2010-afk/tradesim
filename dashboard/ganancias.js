/* Daily gain of every agent, read from ganancias/ganancias.csv in the repository (fresh after every live run). */
"use strict";

function parseCsv(text) {
  const [head, ...lines] = text.trim().split("\n");
  const cols = head.split(",");
  return lines.map((line) => {
    const cells = line.match(/("[^"]*"|[^,]+)/g).map((c) => c.replace(/^"|"$/g, ""));
    const row = Object.fromEntries(cols.map((c, i) => [c, cells[i]]));
    for (const k of ["equity_usd", "day_pnl_usd", "day_pnl_pct", "total_pnl_usd"]) row[k] = parseFloat(row[k]);
    return row;
  });
}

const money = (v) => h("span", { class: `tri ${v > 0 ? "up" : v < 0 ? "down" : "flat"}` },
  `${v > 0 ? "+" : ""}${fmtNum(v, 2)} $`);

async function boot() {
  initTheme(() => {});
  const remote = document.querySelector('meta[name="tradesim-data-url"]')?.content;
  $("csv-link").href = remote;
  let rows;
  try {
    const res = await fetch(remote, { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    rows = parseCsv(await res.text());
  } catch (err) {
    $("statusbar").textContent = "";
    $("app").replaceChildren(h("div", { class: "error" }, `No se pudieron cargar las ganancias (${err.message}).`));
    return;
  }
  const names = [...new Set(rows.map((r) => r.name))];
  const days = [...new Set(rows.map((r) => r.date))].sort().reverse();
  const by = new Map(rows.map((r) => [`${r.date}|${r.name}`, r]));
  const last = names.map((n) => days.map((d) => by.get(`${d}|${n}`)).find(Boolean));
  const ranked = last.filter(Boolean).sort((a, b) => b.total_pnl_usd - a.total_pnl_usd);
  table($("total-table"), [
    { label: "Agente", get: (r) => r.name },
    { label: "Valor", get: (r) => fmtMoney(r.equity_usd) },
    { label: "Ganancia total", get: (r) => money(r.total_pnl_usd) },
    { label: "Hoy", get: (r) => (r.date === days[0] ? money(r.day_pnl_usd) : "—") },
  ], ranked);
  table($("days-table"), [
    { label: "Día", get: (d) => d },
    ...names.map((n) => ({ label: n, get: (d) => {
      const r = by.get(`${d}|${n}`);
      return r ? h("span", {}, money(r.day_pnl_usd), h("br"), h("small", { class: "muted" }, `${r.day_pnl_pct > 0 ? "+" : ""}${fmtNum(r.day_pnl_pct, 2)} %`)) : "—";
    } })),
  ], days);
  $("statusbar").textContent = `${names.length} agentes · ${days.length} días · último dato ${days[0]}`;
  $("app").setAttribute("aria-busy", "false");
}

boot();
