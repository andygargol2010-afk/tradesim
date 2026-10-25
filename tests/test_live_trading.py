import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from tradesim import SyntheticProvider
from tradesim.assets.universe import Asset
from tradesim.data.replay import ReplayLiveProvider
from tradesim.live import is_market_open, nyse_holidays
from tradesim.trading import AccountSpec, LiveConfig, LiveRunner, RiskLimits
from tradesim.trading.export import export_dashboard
from tradesim.types import AssetClass

SYMBOLS = ["SPY", "AAPL", "MSFT", "BTC-USD", "ETH-USD"]


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def make_runner(tmp_path, universe, accounts=None, clock=None, risk=None):
    clock = clock or Clock(datetime(2024, 3, 4, 14, 45, tzinfo=timezone.utc))  # a Monday, NYSE open
    cfg = LiveConfig(symbols=SYMBOLS, accounts=accounts or [
        AccountSpec("rsi", "RSI", "SimpleRSIAgent", {"oversold": 45, "overbought": 55}),
        AccountSpec("ma", "Medias", "MovingAverageCrossAgent", {"max_weight": 0.25}),
        AccountSpec("bh", "Buy & hold", "BuyAndHoldAgent", {}),
    ], state_dir=str(tmp_path / "accounts"), runs_log=str(tmp_path / "runs.jsonl"),
        dashboard_file=str(tmp_path / "live.json"), risk=risk or RiskLimits())
    provider = ReplayLiveProvider(SyntheticProvider(seed=3), clock)
    return LiveRunner(cfg, provider=provider, universe=universe, clock=clock), clock


def test_nyse_calendar():
    assert datetime(2026, 7, 3).date() in nyse_holidays(2026)  # July 4th on a Saturday -> Friday
    assert datetime(2027, 12, 24).date() in nyse_holidays(2027)
    from tradesim.assets.universe import AssetUniverse

    spy = AssetUniverse.default().get("SPY")
    assert not is_market_open(spy, datetime(2026, 11, 26, 15, 0, tzinfo=timezone.utc))  # Thanksgiving
    assert is_market_open(spy, datetime(2026, 11, 27, 15, 0, tzinfo=timezone.utc))
    assert not is_market_open(spy, datetime(2026, 11, 27, 22, 0, tzinfo=timezone.utc))  # after the close


def test_daily_operation_end_to_end(tmp_path, universe):
    runner, clock = make_runner(tmp_path, universe)
    statuses = []
    for d in range(10):
        day = datetime(2024, 3, 4, tzinfo=timezone.utc) + timedelta(days=d)
        clock.t = day.replace(hour=14, minute=45)
        statuses.append(runner.run("decide", export=False)["accounts"]["bh"]["status"])
        clock.t = day.replace(hour=19, minute=45)
        runner.run("mark", export=False)
    # weekdays decide, weekends are skipped (stocks closed)
    assert statuses[:5] == ["ok"] * 5 and statuses[5:7] == ["skipped", "skipped"]
    # a second decide on the same day does nothing
    clock.t = clock.t.replace(hour=15)
    assert runner.run("decide", export=False)["accounts"]["rsi"]["status"] == "skipped"

    data = export_dashboard(runner.config, universe, now=clock.t)
    accounts = {a["id"]: a for a in data["accounts"]}
    bh = accounts["bh"]
    assert bh["summary"]["open_positions"] == len(SYMBOLS)
    pos = bh["positions"][0]
    for key in ("allocated", "allocated_pct_at_entry", "unrealized_pnl", "unrealized_pct", "days_held", "entry_reason"):
        assert key in pos
    assert sum(p["allocated_pct_at_entry"] for p in bh["positions"]) == pytest.approx(95, abs=2)
    assert len(bh["equity_curve"]["t"]) >= 20
    rsi = accounts["rsi"]
    assert rsi["summary"]["num_fills"] > 0 and rsi["journal"]
    if rsi["closed_trades"]:
        t = rsi["closed_trades"][0]
        assert t["entry_reason"].startswith("RSI") and t["allocated"] > 0 and t["exit_reason"]
    assert data["prices"]["BTC-USD"]["last"] > 0 and data["last_run"]["mode"] == "decide"  # the duplicate check above
    json.dumps(data)  # dashboard file is valid JSON


def test_state_survives_new_runner(tmp_path, universe):
    runner, clock = make_runner(tmp_path, universe)
    runner.run("decide", export=False)
    eq = runner.run("mark", export=False)["accounts"]["bh"]["equity"]
    runner2, _ = make_runner(tmp_path, universe, clock=clock)
    assert runner2.run("mark", export=False)["accounts"]["bh"]["equity"] == pytest.approx(eq)


def test_risk_limits(tmp_path, universe):
    runner, clock = make_runner(tmp_path, universe, risk=RiskLimits(max_orders_per_run=2),
                                accounts=[AccountSpec("bh", "BH", "BuyAndHoldAgent", {})])
    rec = runner.run("decide", export=False)
    assert rec["accounts"]["bh"]["status"] == "rejected"  # wanted 5 orders > 2
    runner2, _ = make_runner(tmp_path / "b", universe, risk=RiskLimits(max_position_pct=10),
                             accounts=[AccountSpec("ma", "MA", "MovingAverageCrossAgent", {"max_weight": 0.5})],
                             clock=clock)
    runner2.run("decide", export=False)
    journal = [json.loads(x) for x in (tmp_path / "b" / "accounts" / "ma" / "journal.jsonl").read_text().splitlines()]
    rejected = [o for e in journal for o in e.get("orders", []) if o["status"] == "rejected"]
    assert all("max_position_pct" in o["reject_reason"] for o in rejected)


def test_failing_account_does_not_stop_others(tmp_path, universe):
    runner, _ = make_runner(tmp_path, universe, accounts=[
        AccountSpec("broken", "Broken", "NoSuchAgent", {}), AccountSpec("bh", "BH", "BuyAndHoldAgent", {})])
    rec = runner.run("decide", export=True)
    assert rec["accounts"]["broken"]["status"] == "error" and rec["accounts"]["bh"]["status"] == "ok"
    data = json.loads((tmp_path / "live.json").read_text())
    assert {a["id"] for a in data["accounts"]} == {"broken", "bh"}


# ------------------------------------------------------------------ Alpaca adapter
class FakeAlpaca:
    """Minimal stand-in for alpaca.trading.client.TradingClient."""

    def __init__(self, cash=10_000.0, price=100.0):
        self.cash, self.price = cash, price
        self.positions: dict[str, float] = {}
        self.orders: dict[str, SimpleNamespace] = {}
        self.fee_drift = 0.0

    def get_account(self):
        eq = self.cash + sum(q * self.price for q in self.positions.values())
        return SimpleNamespace(cash=str(self.cash - self.fee_drift), equity=str(eq), buying_power=str(self.cash))

    def get_all_positions(self):
        return [SimpleNamespace(symbol=s, qty=str(q), avg_entry_price=str(self.price), current_price=str(self.price))
                for s, q in self.positions.items() if q]

    def submit_order(self, req):
        oid = f"b{len(self.orders) + 1}"
        self.orders[oid] = SimpleNamespace(id=oid, symbol=req.symbol, side=req.side.value, qty=req.qty,
                                           filled_qty="0", filled_avg_price=None, filled_at=None, status="accepted",
                                           tif=req.time_in_force.value)
        return self.orders[oid]

    def fill(self, oid, qty=None, price=None):
        o = self.orders[oid]
        q = float(qty if qty is not None else o.qty)
        p = price or self.price
        sign = 1 if o.side == "buy" else -1
        self.cash -= sign * q * p
        sym = o.symbol.replace("/", "")
        self.positions[sym] = self.positions.get(sym, 0.0) + sign * q
        o.filled_qty, o.filled_avg_price = str(q), str(p)
        o.status = "filled" if q >= float(o.qty) else "partially_filled"

    def get_order_by_id(self, oid):
        return self.orders[oid]

    def cancel_order_by_id(self, oid):
        self.orders[oid].status = "canceled"


def test_alpaca_trader_books_broker_fills_and_reconciles(universe):
    from tradesim.trading.alpaca import AlpacaTrader

    fake = FakeAlpaca()
    clock = Clock(datetime(2024, 3, 4, 15, 0, tzinfo=timezone.utc))
    data = ReplayLiveProvider(SyntheticProvider(seed=3), clock)
    t = AlpacaTrader(paper=True, client=fake, universe=universe, data_provider=data, clock=clock, symbols=["AAPL"])
    assert t.portfolio.cash == Decimal("10000")
    res = t.place_order("AAPL", "buy", "1.5")
    assert res["status"] == "new" and fake.orders["b1"].tif == "day"  # fractional equity -> DAY
    assert t.place_order("AAPL", "sell", 5)["status"] == "rejected"  # local check: no shorting
    fake.fill("b1", qty=1.0, price=100.0)
    fake.orders["b1"].status = "partially_filled"
    t.sync()
    assert t.portfolio.quantity("AAPL") == Decimal("1.0") and t.orders[res["id"]].status.value == "partially_filled"
    fake.orders["b1"].filled_qty, fake.orders["b1"].filled_avg_price = "1.5", str((100 + 0.5 * 106) / 1.5)
    fake.orders["b1"].status = "filled"
    fake.positions["AAPL"] = 1.5
    fake.cash = 10_000 - 153
    t.sync()
    assert t.portfolio.quantity("AAPL") == Decimal("1.5")
    assert float(t.portfolio.fills[-1].price) == pytest.approx(106.0)  # price of the second slice
    assert t.orders[res["id"]].status.value == "filled"
    fake.fee_drift = 0.37  # broker charged something the model did not
    t.sync()
    assert float(t.portfolio.cash) == pytest.approx(10_000 - 153 - 0.37)
    assert any(r["kind"] == "cash" for r in t.reconciliation)
    fake.positions["AAPL"] = 2.0  # position changed outside tradesim
    t.sync()
    assert any(r["kind"] == "position" and r["symbol"] == "AAPL" for r in t.reconciliation)


def test_real_money_requires_explicit_opt_in(universe, monkeypatch):
    from tradesim.trading.alpaca import AlpacaTrader

    monkeypatch.delenv("TRADESIM_ALLOW_LIVE", raising=False)
    with pytest.raises(PermissionError):
        AlpacaTrader(paper=False, client=FakeAlpaca(), universe=universe, data_provider=SyntheticProvider())


from tradesim.agents.base import Agent, ClosePosition, OrderIntent  # noqa: E402


class SwapAgent(Agent):
    """Day 1: put 90% in SPY. Day 2: sell it all and put 90% in AAPL (needs the sale's cash)."""

    def act(self, obs):
        if obs.position("SPY") == 0 and obs.position("AAPL") == 0:
            return OrderIntent.buy("SPY", percent_of_equity=90, tag="day 1")
        if obs.position("SPY") > 0:
            return [OrderIntent.buy("AAPL", percent_of_equity=90, tag="swap"), ClosePosition("SPY")]
        return []


def test_sells_fund_buys_in_the_same_decision(tmp_path, universe):
    runner, clock = make_runner(tmp_path, universe, risk=RiskLimits(max_position_pct=100),
                                accounts=[AccountSpec("swap", "Swap", "test_live_trading:SwapAgent", {})])
    runner.run("decide", export=False)
    clock.t = clock.t + timedelta(days=1)
    runner.run("decide", export=False)
    journal = [json.loads(x) for x in (tmp_path / "accounts" / "swap" / "journal.jsonl").read_text().splitlines()]
    day2 = journal[-1]
    by_symbol = {o["symbol"]: o for o in day2["orders"]}
    assert by_symbol["SPY"]["status"] == "filled" and by_symbol["AAPL"]["status"] == "filled"
    assert float(by_symbol["AAPL"]["avg_fill_price"]) > 0  # journal shows the executed price, not "new"
    data = export_dashboard(runner.config, universe, now=clock.t)
    acc = data["accounts"][0]
    assert [p["symbol"] for p in acc["positions"]] == ["AAPL"]
    assert acc["closed_trades"][0]["entry_reason"] == "day 1" and acc["closed_trades"][0]["exit_reason"] == "close_position"


def test_yfinance_ignores_stale_out_of_session_quotes(monkeypatch):
    yf = pytest.importorskip("yfinance")
    from tradesim.data.yfinance_provider import YFinanceProvider

    class FakeTicker:
        info_by_symbol = {
            "AAA": {"bid": 99.98, "ask": 100.02, "marketState": "REGULAR"},  # good, in session
            "BBB": {"bid": 92.0, "ask": 108.0, "marketState": "REGULAR"},   # absurd spread
            "CCC": {"bid": 99.98, "ask": 100.02, "marketState": "CLOSED"},  # after hours
            "DDD": {"bid": 95.0, "ask": 95.1, "marketState": "REGULAR"},    # does not bracket the last trade
        }

        def __init__(self, symbol):
            self.fast_info = {"last_price": 100.0}
            self.info = self.info_by_symbol[symbol]

    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    provider = YFinanceProvider()
    quotes = {s: provider.get_quote(Asset(s, AssetClass.EQUITY)) for s in FakeTicker.info_by_symbol}
    assert not quotes["AAA"].modelled and quotes["AAA"].bid == 99.98
    for s in ("BBB", "CCC", "DDD"):
        assert quotes[s].modelled and quotes[s].bid == quotes[s].ask == 100.0


def test_weekend_crypto_partial_session(tmp_path, universe):
    sat = Clock(datetime(2024, 3, 9, 14, 45, tzinfo=timezone.utc))
    runner, clock = make_runner(tmp_path, universe, clock=sat,
                                accounts=[AccountSpec("bh", "BH", "BuyAndHoldAgent", {}, auto_stop=False)])
    assert runner.run("decide", export=False)["accounts"]["bh"]["status"] == "skipped"  # opt-in
    runner.config.weekend_crypto = True
    clock.t = clock.t.replace(day=10)
    rec = runner.run("decide", export=False)["accounts"]["bh"]
    assert rec["status"] == "ok"
    journal = [json.loads(x) for x in (tmp_path / "accounts" / "bh" / "journal.jsonl").read_text().splitlines()]
    last = [e for e in journal if e.get("session") == "partial"][-1]
    filled = {o["symbol"] for o in last["orders"] if o["status"] != "rejected"}
    assert filled == {"BTC-USD", "ETH-USD"}  # stocks wait for Monday
    assert set(last["deferred_to_next_session"]) == {"SPY", "AAPL", "MSFT"}
    clock.t = clock.t.replace(day=11)  # Monday: the stock part is executed
    runner.run("decide", export=False)
    data = export_dashboard(runner.config, universe, now=clock.t)
    assert data["accounts"][0]["summary"]["open_positions"] == len(SYMBOLS)


def test_partial_session_keeps_only_open_markets_and_free_cash(tmp_path, universe):
    from tradesim.agents.base import ClosePosition, OrderIntent, TargetWeights
    from tradesim.trading.runner import _fund_from_cash

    runner, _ = make_runner(tmp_path, universe)
    trader = SimpleNamespace(portfolio=SimpleNamespace(equity=Decimal(100), cash=Decimal(50),
                                                       quantity=lambda s: Decimal(0), open_positions=lambda: []),
                             get_price=lambda s: {"mid": 10.0, "ask": 10.0})
    entry: dict = {}
    # the agent prefers stocks: crypto is not bought just because it is the only open market
    out = runner._restrict([TargetWeights({"SPY": 0.5, "AAPL": 0.5})], {"BTC-USD", "ETH-USD"}, trader, entry)
    assert out == [] and entry["deferred_to_next_session"] == ["AAPL", "SPY"]
    out = runner._restrict([TargetWeights({"SPY": 0.4, "BTC-USD": 0.4}), ClosePosition("MSFT")],
                           {"BTC-USD"}, trader, entry)
    assert [(a.symbol, a.side, a.notional) for a in out] == [("BTC-USD", "buy", 40.0)]
    big = _fund_from_cash([OrderIntent.buy("BTC-USD", notional=80), OrderIntent.buy("ETH-USD", notional=20)],
                          trader, entry)
    assert sum(a.notional for a in big) == pytest.approx(49.5) and "funding" in entry


def test_traditional_investor_holds_and_rebalances(tmp_path, universe):
    from tradesim.agents import TraditionalInvestorAgent

    inv = TraditionalInvestorAgent()
    t = inv.targets({"SPY", "AAPL", "BTC-USD"})  # missing funds: their weight goes to the rest of the policy
    assert set(t) == {"SPY", "BTC-USD"} and t["SPY"] == 0.25  # capped: the excess stays in cash
    runner, clock = make_runner(tmp_path, universe, accounts=[
        AccountSpec("inv", "Inversor", "TraditionalInvestorAgent", {"lookback": 40}, auto_stop=False)])
    runner.config.weekend_crypto = True
    fills_per_day = []
    for d in range(100):
        clock.t = datetime(2024, 3, 4, 14, 45, tzinfo=timezone.utc) + timedelta(days=d)
        runner.run("decide", export=False)
        journal = (tmp_path / "accounts" / "inv" / "journal.jsonl").read_text().splitlines()
        fills_per_day.append(sum(len(json.loads(x).get("orders", [])) for x in journal))
    trading_days = sum(1 for a, b in zip([0] + fills_per_day, fills_per_day) if b > a)
    assert fills_per_day[0] >= 3 and trading_days <= 4  # initial investment + quarterly / band rebalances only
    acc = {a["id"]: a for a in export_dashboard(runner.config, universe, now=clock.t)["accounts"]}["inv"]
    assert acc["summary"]["open_positions"] >= 3


def test_daily_chronicle(tmp_path, universe, monkeypatch):
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from tradesim.trading.chronicle import write_entry

    runner, clock = make_runner(tmp_path, universe)
    runner.config.chronicle = True
    days = []
    for d in range(4):
        day = datetime(2024, 3, 4, tzinfo=timezone.utc) + timedelta(days=d)
        for hh in (14, 19):
            clock.t = day.replace(hour=hh, minute=45)
            rec = runner.run("decide" if hh == 14 else "mark", export=True)
            days.append(rec.get("chronicle"))
    store = json.loads((tmp_path / "diario.json").read_text())
    assert [e["date"] for e in store["entries"]] == ["2024-03-06", "2024-03-05", "2024-03-04"]  # yesterday, once
    e = store["entries"][-1]
    assert e["engine"] == "agentes" and {a["id"] for a in e["text"]["agentes"]} == {"rsi", "ma", "bh"}
    bh = next(a for a in e["facts"]["agents"] if a["id"] == "bh")
    assert bh["buys"] == sorted(SYMBOLS) and bh["day_pnl_pct"] is not None
    assert store["stats"]["bh"]["by_situation"]
    md = (tmp_path / "diario.md").read_text()
    assert "## Patrones acumulados" in md and "### Buy & hold (escrito por el agente)" in md and "Compré" in md
    write_entry(runner.config, datetime(2024, 3, 6).date())  # rewriting a day replaces it
    assert len(json.loads((tmp_path / "diario.json").read_text())["entries"]) == 3


def test_agents_write_their_own_diary(tmp_path, universe):
    from tradesim.trading.chronicle import write_entry

    runner, clock = make_runner(tmp_path, universe, accounts=[
        AccountSpec("rsi", "RSI", "WalkForwardRSIAgent", {}), AccountSpec("ma", "Medias", "MovingAverageCrossAgent", {}),
        AccountSpec("bh", "Buy & hold", "BuyAndHoldAgent", {}), AccountSpec("inv", "Inversor", "TraditionalInvestorAgent", {}),
        AccountSpec("ap", "Aprendiz", "OnlineLinearAgent", {"pretrain_bars": 300}),
        AccountSpec("an", "Analista", "DiscretionaryAgent", {"engines": []})])
    runner.run("decide", export=True)
    for aid in ("rsi", "ma", "bh", "inv", "ap", "an"):
        line = (tmp_path / "accounts" / aid / "journal.jsonl").read_text().splitlines()[-1]
        note = json.loads(line)["diary"]
        assert note["que_hice"] and note["por_que"] and "tag" in note["situation"], aid
    bh = json.loads((tmp_path / "accounts" / "bh" / "journal.jsonl").read_text().splitlines()[-1])["diary"]
    assert bh["que_hice"].startswith("Compré") and "referencia pasiva" in bh["segun_situacion"]
    e = write_entry(runner.config, clock.t.date())
    by_id = {a["id"]: a for a in e["text"]["agentes"]}
    assert by_id["bh"]["autor"] == "agente" and by_id["bh"]["que_hizo"].startswith("Compré")
    assert "(escrito por el agente)" in (tmp_path / "diario.md").read_text()


def test_protective_stops_do_not_block_decisions(tmp_path, universe):
    runner, clock = make_runner(tmp_path, universe, accounts=[AccountSpec("ma", "MA", "MovingAverageCrossAgent", {})])
    runner.run("decide", export=False)
    trader = runner.load_trader(runner.config.accounts[0], runner.base_provider)
    assert trader.open_order_ids and all(trader.orders[i].reduce_only for i in trader.open_order_ids)
    assert trader.observe().open_orders == 0  # automatic stops rest for the life of a position
