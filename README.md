# tradesim

Simulador de trading de alta fidelidad para **entrenar agentes de IA** (RL y LLM) que después operarán en mercados reales.
El objetivo de diseño es que lo aprendido aquí se transfiera a real con la menor fricción posible: por eso el simulador
es deliberadamente **conservador** (nunca regala precios, liquidez ni información del futuro).

- **2.028 activos** incluidos: S&P 500, Nasdaq-100, S&P MidCap 400, S&P SmallCap 600, 169 ETFs, top 300 cripto por
  capitalización, 28 pares forex y 13 índices (solo datos). Regenerable desde fuentes vivas.
- **Datos reales** (yfinance, ccxt/Binance/Coinbase/Kraken, Alpaca) con caché en disco, o **sintéticos** reproducibles.
- **Backtesting histórico** y **paper trading en tiempo real** con exactamente las mismas reglas de ejecución.
- **Fracciones** con aritmética `Decimal` exacta (0.003 BTC, 1.75 AAPL...).
- Órdenes **market, limit, stop, stop-limit** · TIF **GTC, DAY, IOC, FOK** · spread, slippage, impacto de mercado,
  liquidez limitada, fills parciales, comisiones por bróker, márgenes, cortos y financiación.
- Interfaz **Gymnasium** (`PaperTradingEnv`) + API "humana" + **tools JSON para LLMs** (Claude, OpenAI-style).
- Persistencia, logs JSONL + SQLite, seeds, escenarios de estrés y métricas de rendimiento.

---

## Instalación

```bash
cd tradesim
pip install -e ".[all,dev]"        # todo: gymnasium, yfinance, ccxt, alpaca-py, pyarrow...
# o lo mínimo (solo numpy + pandas; datos sintéticos):
pip install -e .
```

Python 3.11+. Tests: `pytest -q` (47 tests, < 3 s, sin red).

## Inicio rápido

```python
from tradesim import TradingSimulator, SimulatorConfig, CachedProvider
from tradesim.data.yfinance_provider import YFinanceProvider

sim = TradingSimulator(
    provider=CachedProvider(YFinanceProvider()),          # o SyntheticProvider(seed=1) sin red
    config=SimulatorConfig(initial_cash=10_000, timeframe="1d", fees="alpaca", seed=42),
)
sim.reset(symbols=["AAPL", "BTC-USD", "SPY"], start="2023-01-01", end="2024-01-01")

sim.get_price("BTC-USD")                     # {'bid':..., 'ask':..., 'mid':..., 'spread_bps':..., ...}
sim.place_order("BTC-USD", "buy", 0.003)     # fracción de BTC
sim.place_order("AAPL", "buy", 1.75, order_type="limit", limit_price=180.0)
sim.place_order("SPY", "buy", notional=2_000)                 # por importe en USD
sim.place_order("SPY", "sell", 5, order_type="stop", stop_price=400)
sim.step()                                   # avanza una barra: las órdenes se ejecutan aquí
sim.get_portfolio(); sim.get_trades(); sim.get_metrics()
sim.get_historical_data("AAPL", timeframe="1w", limit=20, indicators=True)
sim.save_state("runs/state.json")
```

Scripts de ejemplo (`examples/`):

| Script | Qué demuestra |
|---|---|
| `demo.py` | Carga 2.000+ activos, compras fraccionadas, todos los tipos de orden, rechazos, liquidez limitada con fills parciales, estrategia, portfolio, métricas y persistencia. `--data synthetic|yfinance|mixed` |
| `rl_agent.py` | Entorno Gymnasium: baselines (random, buy&hold, momentum) en episodios fuera de muestra; PPO opcional con `stable-baselines3` |
| `llm_agent.py` | Agente Claude con function calling sobre el simulador (`--dry-run` para un agente guionizado sin API) |
| `live_paper.py` | Paper trading en vivo con cotizaciones y libro L2 reales vía ccxt, con estado reanudable |

---

## Arquitectura

```
                 ┌─────────────────────── agentes ───────────────────────┐
                 │  RL (Gymnasium)        LLM (tools JSON)      scripts  │
                 │  env.PaperTradingEnv   agents/llm_tools      API Python│
                 └───────────┬───────────────────┬──────────────┬────────┘
                             ▼                   ▼              ▼
              ┌──────────────────────────────────────────────────────────┐
              │ simulator.BaseSimulator  (bróker: validación pre-trade,   │
              │   reservas de cash, API humana, métricas, persistencia)  │
              ├───────────────────────────┬──────────────────────────────┤
              │ TradingSimulator (reloj    │ live.LivePaperTrader (reloj  │
              │ histórico, barra a barra)  │ de pared, cotizaciones+L2)   │
              └─────────────┬─────────────┴───────────────┬──────────────┘
                            ▼                             ▼
   ┌──────────────────────────────────┐   ┌───────────────────────────────────┐
   │ execution.engine.ExecutionEngine │──▶│ portfolio.Portfolio (Decimal)       │
   │  camino intrabarra, stops/gaps,  │   │  posiciones, PnL, comisiones,       │
   │  maker/taker, capacidad, TIF     │   │  margen, financiación, equity curve │
   │ market.liquidity  spread + libro │   │ portfolio.metrics  Sharpe, DD...    │
   │ execution.fees    perfiles bróker│   └───────────────────────────────────┘
   └──────────────┬───────────────────┘
                  ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ data.store.MarketData  (barras + estimaciones sin look-ahead + escenarios)│
   │ data.base.DataProvider ← YFinance | CCXT | Alpaca | Synthetic | Routing   │
   │                          └─ CachedProvider (parquet en ~/.tradesim/cache)│
   │ assets.universe.AssetUniverse (reglas por activo: tick, lote, mínimos)  │
   └──────────────────────────────────────────────────────────────────────────┘
```

| Módulo | Responsabilidad |
|---|---|
| `types.py` | `Order`, `Fill`, `ClosedTrade`, `Quote`, enums; conversión `D()` a `Decimal` sin artefactos binarios |
| `assets/universe.py` | `Asset` (clase, tier de liquidez, tick, paso de lote, mínimo nocional, sesión) y `AssetUniverse` (CSV, filtros) |
| `data/` | Proveedores, caché, timeframes/resampling, `MarketData` con volatilidad y volumen típico calculados solo con barras pasadas |
| `market/liquidity.py` | Perfiles por clase/tier, modelo de spread, libro sintético por ley de la raíz cuadrada, barrido de libro L2 real |
| `market/conditions.py` | Escenarios: volatilidad ×N, tendencia, gaps, flash crash, sequía de liquidez |
| `execution/engine.py` | Motor de fills único para histórico y vivo |
| `execution/fees.py` | `FeeSchedule` (maker/taker %, fijo, por acción, mín./máx., tasas SEC/FINRA) y presets por bróker |
| `portfolio/` | Contabilidad exacta y métricas |
| `simulator.py` / `live.py` | Relojes histórico y en vivo sobre el mismo núcleo de bróker |
| `env.py` | `PaperTradingEnv` (Gymnasium) |
| `agents/llm_tools.py` | Definiciones de tools + `dispatch()` para agentes LLM |
| `persistence.py` | Log de eventos JSONL, SQLite de órdenes/fills/equity, snapshots JSON atómicos |

### Modelo temporal (clave para no hacer trampas)

El simulador está situado en el **cierre de la barra k** (`sim.now`). El agente ve todo hasta la barra k (incluida),
envía órdenes y llama a `step()`. Las órdenes se ejecutan **en la barra k+1** (apertura y luego camino intrabarra), se
valora la cartera al cierre de k+1 y avanza el reloj. Es exactamente la latencia mínima que tendrías en real. La
volatilidad y el volumen típico que usan el spread y el impacto se calculan solo con barras anteriores a la ejecutada.
`get_historical_data` nunca devuelve barras no cerradas, tampoco al agregar a timeframes mayores.

---

## Realismo de ejecución

| Aspecto | Cómo se simula |
|---|---|
| **Bid/ask** | Real si el feed lo da (ccxt, Alpaca, modo vivo). Si no: `spread = base(clase, tier) × (σ/σ_típica)^0.5 × multiplicador de escenario`, con mínimo de 1 tick. Ej.: AAPL ~1,5 bps, small cap ~25 bps, token long-tail ~100 bps |
| **Market** | Apertura de la siguiente barra ± medio spread + impacto + slippage fijo/ruido opcional |
| **Impacto** | Ley de la raíz cuadrada: el libro sintético cumple `Δp = Y·σ·√(Q/V)` (Y=`impact_coefficient`, por defecto 1.0). Con libro L2 real (ccxt en vivo) se barre nivel a nivel |
| **Liquidez** | Máximo `participation_rate` (10 %) del volumen de la barra por barra, compartido entre todas las órdenes del símbolo. El resto sigue trabajando (GTC/DAY) o se cancela (IOC). FOK: todo o nada |
| **Limit** | Marketable al llegar → taker hasta su límite (con mejora de precio). Si no, descansa y solo se ejecuta **al límite** (maker) si el precio **lo atraviesa**; si solo lo toca, con probabilidad `limit_touch_fill_prob` (incertidumbre de cola) |
| **Stop / stop-limit** | Se disparan al tocar el stop en el camino O→L→H→C (barra alcista) u O→H→L→C (bajista). **Gaps**: si la apertura salta el stop, se ejecuta a la apertura (te comes el gap) |
| **Ticks y lotes** | Reg NMS (0,01 $ ≥ 1 $, 0,0001 $ por debajo); cripto ~6 cifras significativas. Precios redondeados en contra del trader; cantidades hacia abajo al paso del lote (1e-9 acciones, 1e-8 cripto). Mínimo nocional 1 $ |
| **Comisiones** | Por clase de activo y tipo de liquidez (maker/taker), con mínimos/máximos por orden y tasas regulatorias en ventas de acciones |
| **Cash / margen** | Cuenta de efectivo por defecto (sin apalancamiento ni cortos). Reservas por órdenes abiertas (no se gasta dos veces el cash). Opcional: `max_leverage`, cortos con coste de préstamo, interés de margen, margin call con liquidación |
| **Horarios** | Acciones solo en barras de sesión (las órdenes esperan en fin de semana); cripto 24/7; forex 24/5. Órdenes DAY expiran al cambiar la fecha de sesión (Nueva York) |

Configuración (todo en `SimulatorConfig` / `ExecutionConfig`):

```python
from decimal import Decimal
from tradesim import SimulatorConfig, ExecutionConfig, MarginConfig, SpreadModel

config = SimulatorConfig(
    initial_cash=50_000, timeframe="15m", fees="ibkr_tiered", seed=7, scenario="high_volatility",
    execution=ExecutionConfig(slippage_model="sqrt_impact", impact_coefficient=0.8, participation_rate=0.05,
                              fixed_slippage_bps=1.0, slippage_noise_bps=2.0, limit_touch_fill_prob=0.3,
                              spread=SpreadModel(spread_multiplier=1.5)),
    margin=MarginConfig(max_leverage=Decimal(2), allow_short=True, maintenance_margin=Decimal("0.25")),
    max_position_pct=25, log_dir="runs/exp1",
)
```

`slippage_model="none"` desactiva spread/impacto/capacidad (solo para ablaciones: medir cuánto rendimiento se "come"
la microestructura).

### Comisiones (presets `fees=`)

| Preset | Acciones/ETFs | Cripto | Forex |
|---|---|---|---|
| `alpaca` (defecto) | 0 $ + SEC/FINRA en ventas | 0,15 % maker / 0,25 % taker | 0 |
| `ibkr_fixed` | 0,005 $/acción, mín. 1 $, máx. 1 % | 0,18 %, mín. 1,75 $ | 0,2 bps, mín. 2 $ |
| `ibkr_tiered` | 0,004 $/acción (incl. aprox. tasas de exchange), mín. 0,35 $ | igual | igual |
| `binance` | (Alpaca) | 0,10 % / 0,10 % | - |
| `coinbase` | (Alpaca) | 0,60 % / 1,20 % | - |
| `kraken` | (Alpaca) | 0,25 % / 0,40 % | - |
| `pct_plus_fixed` | 0,10 % + 1 $ por orden | igual | igual |
| `zero` | 0 | 0 | 0 |

Son tarifas del nivel de entrada (2025) para particulares. **Verifica las tuyas**: cualquier campo se sobrescribe con
`FeeModel.preset("alpaca").with_override(AssetClass.CRYPTO, FeeSchedule(...))`.

---

## Datos y API keys

| Proveedor | Activos | Key | Notas |
|---|---|---|---|
| `SyntheticProvider(seed)` | todos | no | GARCH + colas t + saltos + factor común + gaps nocturnos; estable por prefijo y coherente entre timeframes. Para desarrollo, CI y episodios ilimitados; **no** para validar una estrategia |
| `YFinanceProvider()` | acciones, ETFs, cripto, forex, índices | no | No oficial y con rate limit (429): úsalo siempre con `CachedProvider`. Intradía corto: 1m ≈ 30 días, 5m–30m 60 días, 1h 730 días. Precios ajustados por splits y dividendos |
| `CCXTProvider("binance" \| "coinbase" \| "kraken" \| ...)` | cripto | no (endpoints públicos) | OHLCV + **bid/ask y libro L2 reales**. Binance bloquea algunas regiones (HTTP 451); Kraken solo sirve las últimas 720 velas |
| `AlpacaProvider()` | acciones, ETFs, cripto | sí (gratis) | `export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...`. Plan gratuito = feed IEX: el volumen se reescala ×40 (`iex_volume_scale`) para aproximar el consolidado; con SIP usa `feed="sip"` |

Combinar fuentes y cachear:

```python
from tradesim import RoutingProvider, CachedProvider
from tradesim.data.ccxt_provider import CCXTProvider
from tradesim.data.yfinance_provider import YFinanceProvider

provider = CachedProvider(RoutingProvider([CCXTProvider("coinbase"), YFinanceProvider()]), "~/.tradesim/cache")
```

Detrás de un proxy corporativo exporta `HTTPS_PROXY` y `REQUESTS_CA_BUNDLE`; `CCXTProvider` las respeta
(`trust_env=True`, ccxt las ignora por defecto).

Para añadir otra fuente (Polygon, tu base de datos, ficheros...) implementa `DataProvider.supports()` y `get_bars()`
devolviendo un DataFrame con índice UTC (apertura de barra) y columnas `open, high, low, close, volume`; opcionalmente
`spread_bps` (spread medio observado, que el motor usará en lugar del modelo), `get_quote()` y `get_order_book()`.

## Activos: cómo añadir más

- **Regenerar desde fuentes vivas**: `python scripts/build_universe.py [--crypto 300]` (Wikipedia para índices,
  CoinGecko para cripto, excluyendo stablecoins y tokens envueltos/stakeados).
- **CSV propio**: `AssetUniverse.from_csv("mis_activos.csv")`, columnas mínimas `symbol,asset_class` (+ opcionales
  `name, liquidity_tier, tick_size, qty_step, min_notional, shortable, session...`). Combinar: `AssetUniverse.default().merge(otro)`.
- **En código**: `universe.add(Asset("XYZ", AssetClass.EQUITY, liquidity_tier=4, fractionable=False))`.

El `liquidity_tier` (1 = SPY/AAPL/BTC … 5 = micro cap / token long-tail) determina spread base, volumen típico de
respaldo y (en sintético) la volatilidad.

---

## Entrenar agentes

### RL (Gymnasium)

```python
from tradesim.env import PaperTradingEnv
env = PaperTradingEnv(["SPY", "QQQ", "TLT", "GLD", "BTC-USD"], "2018-01-01", "2022-12-31", provider=provider,
                      episode_length=252, window=30, action_mode="weights", reward="log_return",
                      drawdown_penalty=0.5)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

- **Acciones**: `"weights"` = pesos objetivo por activo (`Box[0,1]`, o `[-1,1]` con cortos), ejecutados con órdenes
  de mercado y una banda de no-operación; `"discrete"` = `MultiDiscrete` hold/buy/sell una fracción del equity. En
  cualquier modo se acepta `{"orders": [{...}, ...]}` con los argumentos de `place_order`.
- **Observación**: ventana de retornos normalizados, rangos H/L, volumen z-score; RSI, régimen de volatilidad,
  distancia a SMA20, peso actual y spread por activo; cash, apalancamiento, drawdown, PnL no realizado, progreso.
- **Recompensa**: `log_return` · `pnl` · `sharpe` (Sharpe diferencial) − `drawdown_penalty` × aumento del drawdown. Los
  costes no se añaden aparte: ya están en el equity (spread, impacto, comisiones), como en real.
- **Episodios** con inicio aleatorio y seeds: `env.reset(seed=s)` reproduce exactamente el mismo episodio.
- `stable-baselines3`: `PPO("MlpPolicy", env).learn(...)` (ver `examples/rl_agent.py`).

Buenas prácticas para transferir a real: separa train/validación/test **por tiempo**; entrena también con
`scenario="high_volatility" | "crash" | "gaps" | "illiquid"` y `slippage_noise_bps > 0` para robustez; compara siempre
contra buy & hold después de costes; valida en `LivePaperTrader` antes de arriesgar dinero.

### Agentes LLM

`tradesim.agents.llm_tools.TOOLS` define 10 herramientas (`get_available_assets`, `get_price`, `get_portfolio`,
`get_historical_data`, `place_order`, `cancel_order`, `get_open_orders`, `close_position`, `get_trades`,
`get_metrics`) en formato de la Messages API de Anthropic (`openai_tools()` las convierte al formato
`{"type": "function"}`). `dispatch(sim, name, args)` las ejecuta y devuelve JSON; los errores vuelven como
`{"error": ...}` para que el modelo pueda corregirse. Funcionan igual sobre `TradingSimulator` y `LivePaperTrader`, así
que el mismo prompt pasa de backtest a paper trading en vivo sin cambios. Ver `examples/llm_agent.py`
(`ANTHROPIC_API_KEY` o `ant auth login`).

### Paper trading en vivo

```python
from tradesim.live import LivePaperTrader
trader = LivePaperTrader(CCXTProvider("kraken"), config=SimulatorConfig(initial_cash=10_000, fees="kraken"))
trader.place_order("BTC-USD", "buy", notional=500)
trader.sync()            # ejecuta contra la cotización y el libro L2 actuales, valora y registra equity
trader.run(agent_fn, interval_seconds=60)
trader.save_state("runs/live/state.json")   # y LivePaperTrader.load_state(...) para continuar
```

---

## Escenarios de mercado

`SimulatorConfig(scenario=...)` con presets `normal`, `high_volatility`, `crash`, `bull`, `gaps`, `flash_crash`,
`illiquid`, o un `Scenario` propio:

```python
from tradesim.market.conditions import Scenario, VolatilityScale, GapInjection, LiquidityDrought, FlashCrash, Trend
scenario = Scenario("stress", [VolatilityScale(2.5), GapInjection(prob=0.03, size=0.07), LiquidityDrought(0.4, 2.0)])
```

Se aplican a datos reales o sintéticos, solo dentro de la ventana del episodio, conservando la coherencia OHLC, y
ensanchan spreads / reducen profundidad donde corresponde. Son deterministas dada la seed.

## Persistencia, logs y reproducibilidad

- `sim.save_state(path)` / `TradingSimulator.load_state(path, provider)`: cartera, órdenes, contadores de IDs, estado
  del RNG y posición del reloj; continuar tras cargar da resultados idénticos (verificado en tests).
- `log_dir=...`: `events.jsonl` (cada orden enviada/rechazada/cancelada/expirada, fill, margin call, reset) y
  `trades.sqlite` (tablas `orders`, `fills`, `closed_trades`, `equity`).
- Una seed fija los datos sintéticos, los escenarios, el ruido de ejecución y el muestreo de episodios.

## Métricas

`sim.get_metrics()`: retorno total, CAGR, volatilidad, Sharpe, Sortino, Calmar, max drawdown y su duración, VaR/CVaR
95 %, skew, número de trades, win rate, profit factor, payoff, expectancy, mayor ganancia/pérdida, tiempo medio en
posición, tiempo en mercado, turnover anual, comisiones, coste de slippage y financiación. Se anualiza según el
timeframe (252 días × 6,5 h para acciones, 365 × 24 h para cripto).

---

## Limitaciones conocidas (léelas antes de pasar a real)

1. **Sesgo de supervivencia**: el universo es la composición *actual* de los índices; aplicada al pasado sobrestima
   rendimientos (faltan empresas que quebraron o salieron del índice).
2. **Camino intrabarra supuesto** (O→L→H→C / O→H→L→C): con barras diarias, el orden real de máximos y mínimos es
   desconocido. Para estrategias con stops ajustados usa timeframes más finos.
3. **Tu impacto no mueve el histórico**: el impacto se paga en el precio de ejecución, pero las barras siguientes no
   reaccionan a tus órdenes (sin impacto permanente). Con tamaños grandes, sé más conservador (`impact_coefficient`).
4. **Sin festivos** en el calendario sintético; los datos reales sí los reflejan. Sin subastas de apertura/cierre,
   pre/post-market, halts ni corporate actions más allá de precios ajustados.
5. **Spreads modelados** cuando no hay datos de cotización: calibrados a medianas típicas, no a cada instante.
6. Los **datos sintéticos** sirven para depurar y aumentar datos, no para validar una estrategia.
7. Las comisiones y tasas regulatorias cambian: verifica las de tu bróker.

## Tests

```bash
pytest -q
```

Cubren contabilidad (precio medio, PnL realizado, giros largo→corto, precisión fraccionaria), cada tipo de orden,
gaps, capacidad/fills parciales, IOC/FOK, impacto creciente con el tamaño, barrido de libro L2, comisiones, ausencia de
look-ahead, rechazos, reservas de cash, expiración DAY, margen y cortos, reproducibilidad, guardar/cargar,
escenarios, caché y el entorno Gymnasium (`check_env`).
