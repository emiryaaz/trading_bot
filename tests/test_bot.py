"""İnternet gerektirmeyen testler: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot import indicators as ind
from bot.backtest import run_backtest
from bot.data import synthetic_ohlcv, synthetic_portfolio
from bot.exchange import FuturesClient, FuturesExchange, PaperFutures, Position
from bot.live import Trader
from bot.risk import LONG, SHORT, RiskManager, liquidation_price
from bot.strategy import SmaCrossRsi

RISK = {"margin_per_trade_pct": 0.20, "max_open_positions": 3, "max_total_margin_pct": 0.60, "stop_mode": "roe", "stop_loss_roe": 0.05, "take_profit_roe": 0.05,
        "stop_atr_mult": 2.0, "take_profit_atr_mult": 4.0, "max_drawdown_pct": 0.10,
        "max_daily_loss_pct": 0.0, "kill_switch_file": "STOP_TEST_NONE"}
PARAMS = {"fast": 20, "slow": 50, "rsi_period": 14, "rsi_max_entry": 70, "rsi_min_entry": 30, "atr_period": 14}
LEV = 2


def bt(df, risk=None, **kw):
    return run_backtest(df, SmaCrossRsi(**PARAMS), RiskManager(risk or RISK, LEV), 1000, 0.0005, 0.0003, **kw)


# ------------------------------------------------------------------ göstergeler
def test_rsi_bounds_and_trend():
    up = pd.Series(np.linspace(100, 200, 100))
    assert (ind.rsi(up, 14).dropna() > 99).all()
    rnd = pd.Series(100 + np.random.default_rng(0).normal(0, 1, 500).cumsum())
    assert ind.rsi(rnd).dropna().between(0, 100).all()


# ------------------------------------------------------------------ risk
def test_roe_protection_prices():
    rm = RiskManager(RISK, LEV)
    assert rm.protection_prices(100, LONG) == pytest.approx((97.5, 102.5))    # %5 ROE / 2x = %2.5 fiyat
    assert rm.protection_prices(100, SHORT) == pytest.approx((102.5, 97.5))


def test_plan_sizing():
    rm = RiskManager(RISK, LEV)
    p = rm.plan_entry(equity=1000, available=1000, price=100, side=LONG)
    assert p.margin == pytest.approx(200)            # bakiyenin %20'si teminat
    assert p.qty == pytest.approx(4)                 # 200 x 2 / 100
    loss_at_stop = (100 - p.stop) * p.qty
    assert loss_at_stop == pytest.approx(10)         # = teminatın %5'i = bakiyenin %1'i


def test_stop_before_liquidation():
    for side in (LONG, SHORT):
        liq = liquidation_price(100, side, LEV)
        stop, _ = RiskManager(RISK, LEV).protection_prices(100, side)
        assert (stop > liq) if side == LONG else (stop < liq)
    assert liquidation_price(100, LONG, 2) == pytest.approx(50.2, abs=0.1)


def test_drawdown_breaker():
    rm = RiskManager(RISK, LEV)
    assert rm.drawdown_hit(1000, 900)
    assert not rm.drawdown_hit(1000, 901)


# ------------------------------------------------------------------ backtest
def test_backtest_consistent_and_two_sided():
    res = bt(synthetic_ohlcv(n=3000))
    assert {t.side for t in res.trades} == {LONG, SHORT}
    assert res.equity.iloc[-1] == pytest.approx(1000 + sum(t.pnl for t in res.trades), abs=1e-6)
    for t in res.trades:
        assert t.pnl >= -t.margin - 1e-9                       # isolated: teminattan fazla kayıp yok
        if t.reason == "stop-loss":
            assert t.roe == pytest.approx(-0.05, abs=0.03)    # ~%5 + ücret / boşluk payı


def test_long_only_flag():
    res = bt(synthetic_ohlcv(n=3000), allow_short=False)
    assert res.trades and all(t.side == LONG for t in res.trades)


def test_funding_direction():
    df = synthetic_ohlcv(n=3000)
    res = bt(df, funding_fallback=0.001)
    for t in res.trades:
        if abs(t.funding) > 0:
            assert np.sign(t.funding) == t.side             # pozitif oranda long öder, short alır


def test_no_lookahead():
    df = synthetic_ohlcv(n=2000)
    cut = 1500
    a = bt(df.iloc[:cut])
    df2 = df.copy()
    df2.iloc[cut:, :4] *= 1.5
    b = bt(df2)
    ca = [t for t in a.trades if t.reason not in ("test sonu",)]
    assert [(t.entry_time, round(t.pnl, 6)) for t in ca] == [(t.entry_time, round(t.pnl, 6)) for t in b.trades[:len(ca)]]


def test_backtest_breaker_triggers():
    risk = {**RISK, "margin_per_trade_pct": 1.0, "take_profit_roe": 0}   # agresif: her stop = bakiyenin %5'i
    res = bt(synthetic_ohlcv(n=5000, seed=3), risk=risk)
    assert len(res.breaker_events) >= 1
    ev = res.breaker_events[0]
    assert ev["equity"] <= ev["baseline"] * 0.9 + 1e-6


# ------------------------------------------------------------------ portföy (çoklu coin) backtest
COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "LINK/USDT"]


def _overlap_stats(trades):
    events = sorted([(t.entry_time, 1, t.margin) for t in trades] + [(t.exit_time, -1, t.margin) for t in trades],
                    key=lambda e: (e[0], e[1]))       # aynı anda kapanış önce
    n = m = max_n = max_m = 0
    for _, k, mg in events:
        n += k
        m += k * mg
        max_n, max_m = max(max_n, n), max(max_m, m)
    return max_n, max_m


def test_portfolio_limits_respected():
    data = synthetic_portfolio(COINS, n=3000)
    res = bt(data)
    assert {t.symbol for t in res.trades} == set(COINS)
    max_n, max_m = _overlap_stats(res.trades)
    assert max_n <= 3                                           # en fazla 3 açık pozisyon
    assert max_m <= res.equity.max() * 0.60 + 1e-6              # toplam teminat sınırı
    assert res.equity.iloc[-1] == pytest.approx(1000 + sum(t.pnl for t in res.trades), abs=1e-6)
    for t in res.trades:                                        # aynı coinde üst üste binen pozisyon yok
        same = [u for u in res.trades if u.symbol == t.symbol and u is not t]
        assert all(u.exit_time <= t.entry_time or u.entry_time >= t.exit_time for u in same)


def test_portfolio_single_position_limit():
    risk = {**RISK, "max_open_positions": 1}
    res = bt(synthetic_portfolio(COINS, n=2000), risk=risk)
    assert _overlap_stats(res.trades)[0] == 1


def test_portfolio_different_lengths():
    """Sonradan listelenen coin (ör. HYPE) geç başlasa da test çalışmalı."""
    data = synthetic_portfolio(COINS[:3], n=3000)
    data["NEW/USDT"] = synthetic_ohlcv(n=1200, seed=99).set_axis(data["BTC/USDT"].index[-1200:])
    res = bt(data)
    new = [t for t in res.trades if t.symbol == "NEW/USDT"]
    assert new and min(t.entry_time for t in new) >= data["NEW/USDT"].index[0]
    assert "NEW/USDT" in res.per_symbol().index


# ------------------------------------------------------------------ canlı döngü (sahte borsa)
class FakeSource(FuturesClient):
    """Sentetik veriyi mum mum 'oynatan' sahte fiyat kaynağı (çoklu coin)."""

    def __init__(self, data: dict):
        self.data = {k if ":" in k else f"{k}:USDT": v for k, v in data.items()}
        self.i = 200

    def fetch_ohlcv(self, symbol, timeframe, limit=500):
        return self.data[symbol].iloc[max(0, self.i - limit + 1): self.i + 1]

    def fetch_price(self, symbol):
        return float(self.data[symbol]["close"].iloc[self.i])

    def setup(self, *a): pass
    def available_balance(self): return 0
    def get_position(self, s): return None
    def open_position(self, *a): raise NotImplementedError
    def close_position(self, *a): raise NotImplementedError
    def place_protection(self, *a): pass
    def cancel_protection(self, *a): pass
    def closed_result(self, *a): return None


def _cfg(tmp_path, capital=1000, coins=COINS):
    return {"exchange": {"symbols": coins, "timeframe": "1h"}, "mode": "paper",
            "futures": {"leverage": LEV, "margin_mode": "isolated", "allow_short": True},
            "risk": RISK, "strategy": {"name": "sma_cross_rsi", "params": PARAMS},
            "live": {"state_file": str(tmp_path / "state.json"), "poll_seconds": 0, "capital": capital}}


def _trader(tmp_path, src, capital=1000, risk=None, paper=None, coins=COINS):
    paper = paper or PaperFutures(src, capital, 0.0005, 0.0003, LEV)
    cfg = _cfg(tmp_path, capital, coins)
    cfg["risk"] = risk or RISK
    return Trader(cfg, paper, SmaCrossRsi(**PARAMS), RiskManager(risk or RISK, LEV),
                  state_path=tmp_path / "state.json"), paper


def test_trader_multi_coin_loop_and_restore(tmp_path):
    src = FakeSource(synthetic_portfolio(COINS, n=1500))
    t, paper = _trader(tmp_path, src)
    max_open, peak = 0, 1000.0
    for i in range(200, 1499):
        src.i = i
        if not t.step():
            break
        max_open = max(max_open, len(t.positions))
        peak = max(peak, t.equity())
        assert len(paper.pos) == len(t.positions)              # bot ve hesap senkron
        assert t.used_margin() <= peak * 0.60 + 1e-6            # sınır, açılış anındaki bakiyeye göre
    trades = t.state["trades"]
    assert max_open == 3 and len(trades) > 10
    assert {x["side"] for x in trades} == {"LONG", "SHORT"}
    assert len({x["symbol"] for x in trades}) >= 4
    assert "stop-loss" in {x["reason"] for x in trades}
    if not t.positions:
        assert t.equity() == pytest.approx(paper.balance, rel=1e-9)
    t2, p2 = _trader(tmp_path, src)
    assert len(t2.state["trades"]) == len(trades) and p2.balance == pytest.approx(paper.balance)
    assert set(p2.pos) == set(t2.positions)


def test_trader_breaker_halts_and_reports(tmp_path):
    src = FakeSource(synthetic_portfolio(COINS, n=3000))
    # agresif: tek pozisyon, bakiyenin %90'ı teminat → her stop ≈ bakiyenin %4.5'i
    risk = {**RISK, "margin_per_trade_pct": 0.9, "max_total_margin_pct": 1.0, "max_open_positions": 1,
            "take_profit_roe": 0}
    t, paper = _trader(tmp_path, src, risk=risk)
    stopped = False
    for i in range(200, 2999):
        src.i = i
        if not t.step():
            stopped = True
            break
    assert stopped and t.state["halted"]
    assert not t.positions and not paper.pos                   # tüm pozisyonlar kapatıldı
    assert t.equity() <= 1000 * 0.9 + 1e-6
    rep = Path(t.state["last_report"]).read_text(encoding="utf-8")
    assert "kayıp sınırına" in rep and "Coin bazında" in rep
    assert t.step() is False
    eq = t.resume()
    assert not t.state["halted"] and t.state["session"]["baseline"] == pytest.approx(eq)
    src.i += 1
    assert t.step() is True


def test_kill_switch_closes_all(tmp_path, monkeypatch):
    src = FakeSource(synthetic_portfolio(COINS, n=600))
    t, paper = _trader(tmp_path, src)
    src.i = 300
    for sym, side in (("BTC/USDT:USDT", SHORT), ("ETH/USDT:USDT", LONG)):
        f = paper.open_position(sym, side, 0.01)
        t.positions[sym] = {"side": side, "qty": 0.01, "entry": f.price, "entry_fee": f.fee, "margin": 1,
                            "stop": 1e9 if side == SHORT else 0, "take_profit": 0, "opened": "x", "opened_ms": 0}
    monkeypatch.setattr(t.risk, "kill_switch", lambda: True)
    assert t.step() is False
    assert not t.positions and not paper.pos and len(t.state["trades"]) == 2


def test_foreign_position_untouched(tmp_path):
    src = FakeSource(synthetic_portfolio(COINS, n=1500))
    t, paper = _trader(tmp_path, src)
    paper.pos["BTC/USDT:USDT"] = Position(LONG, 0.01, 100)   # botun açmadığı pozisyon
    for i in range(200, 1400):
        src.i = i
        t.step()
    assert paper.pos["BTC/USDT:USDT"].qty == 0.01
    assert all(x["symbol"] != "BTC/USDT" for x in t.state["trades"])
    assert t.state["trades"]                                  # diğer coinlerde işlem devam etti


def test_protection_failure_closes_position(tmp_path):
    src = FakeSource(synthetic_portfolio(COINS, n=1500))

    class BadPaper(PaperFutures):
        def place_protection(self, *a):
            raise RuntimeError("borsa hatası")

    paper = BadPaper(src, 1000, 0.0005, 0.0003, LEV)
    t, _ = _trader(tmp_path, src, paper=paper)
    for i in range(200, 1499):
        src.i = i
        t.step()
        assert not paper.pos                                  # korumasız pozisyon asla kalmamalı
    assert any(x["reason"].startswith("koruma emri hatası") for x in t.state["trades"])


def test_live_mode_locked():
    from bot.exchange import build_exchange
    cfg = {"mode": "live", "exchange": {"name": "binanceusdm", "symbols": ["BTC/USDT"]},
           "futures": {"leverage": 2}, "live": {"capital": 100},
           "secrets": {"api_key": "x", "api_secret": "y", "allow_live": False}}
    with pytest.raises(RuntimeError):
        build_exchange(cfg)


def test_config_validation():
    from bot.config import validate
    base = {"mode": "paper", "futures": {"leverage": 2, "margin_mode": "isolated"}, "risk": dict(RISK)}
    validate(base)
    for bad in ({"futures": {"leverage": 20, "margin_mode": "isolated"}},
                {"futures": {"leverage": 2, "margin_mode": "cross"}},
                {"risk": {**RISK, "stop_loss_roe": 0}},
                {"risk": {**RISK, "max_open_positions": 0}},
                {"exchange": {"symbols": ["BTC/USDT", "BTC/USDT"]}}):
        with pytest.raises(ValueError):
            validate({**base, **bad})


# ------------------------------------------------------------------ borsa sonucu okuma
class _FakeCcxt:
    def fetch_my_trades(self, symbol, since=None):
        return [  # açılış (short sat) + iki parça kapanış (al)
            {"side": "sell", "amount": 0.01, "price": 50000, "fee": {"cost": 0.25, "currency": "USDT"},
             "info": {"realizedPnl": "0"}},
            {"side": "buy", "amount": 0.004, "price": 48750, "fee": {"cost": 0.0975, "currency": "USDT"},
             "info": {"realizedPnl": "5"}},
            {"side": "buy", "amount": 0.006, "price": 48750, "fee": {"cost": 0.14625, "currency": "USDT"},
             "info": {"realizedPnl": "7.5"}},
        ]

    def fetch_funding_history(self, symbol, since=None):
        return [{"amount": 0.05}]                    # short olarak funding aldık


def test_closed_result_from_exchange():
    c = FuturesExchange.__new__(FuturesExchange)
    c.ex = _FakeCcxt()
    r = c.closed_result("BTC/USDT:USDT", SHORT, 0, 0.25)
    assert r.exit_price == pytest.approx(48750)
    assert r.fees == pytest.approx(0.49375)
    assert r.funding == pytest.approx(-0.05)
    assert r.pnl == pytest.approx(12.5 - 0.49375 + 0.05)


# ------------------------------------------------------------------ çok katmanlı strateji (v0.5)
from bot.strategy import MultiFactor


def test_multifactor_no_lookahead():
    df = synthetic_ohlcv(n=1500, seed=5)
    btc = synthetic_ohlcv(n=1500, seed=6)
    s = MultiFactor()
    cut = 1100
    a = s.generate(df.iloc[:cut], btc.iloc[:cut])
    df2, btc2 = df.copy(), btc.copy()
    df2.iloc[cut:, :4] *= 1.7
    btc2.iloc[cut:, :4] *= 0.5
    b = s.generate(df2, btc2).iloc[:cut]
    for col in ("long_entry", "short_entry", "long_exit", "short_exit", "dir"):
        assert (a[col].values == b[col].values).all(), col


def test_multifactor_direction_rules():
    df = synthetic_ohlcv(n=4000, seed=7)
    out = MultiFactor(use_btc_filter=False).generate(df)
    assert out["long_entry"].sum() > 0 and out["short_entry"].sum() > 0
    assert (out.loc[out["long_entry"], "dir"] == 1).all()        # LONG yalnızca yükselen trendde
    assert (out.loc[out["short_entry"], "dir"] == -1).all()      # SHORT yalnızca düşen trendde
    assert (out.loc[out["long_entry"] | out["short_entry"], "adx"] >= 20).all()
    assert not (out["long_entry"] & out["short_entry"]).any()


def test_multifactor_btc_filter():
    df = synthetic_ohlcv(n=4000, seed=7)
    btc = synthetic_ohlcv(n=4000, seed=8)
    out = MultiFactor().generate(df, btc)
    assert (out.loc[out["long_entry"], "btc_dir"] != -1).all()   # BTC düşerken altcoinde long yok
    assert (out.loc[out["short_entry"], "btc_dir"] != 1).all()
    free = MultiFactor(use_btc_filter=False).generate(df)
    assert (out["long_entry"] | out["short_entry"]).sum() <= (free["long_entry"] | free["short_entry"]).sum()


def test_layers_reduce_trades():
    data = synthetic_portfolio(COINS, n=3000)
    off = dict(use_regime=False, use_htf=False, use_btc_filter=False, trigger="cross",
               use_macd=False, use_fib=False, use_volume=False)
    s_off = run_backtest(data, MultiFactor(**off), RiskManager(RISK, LEV), 1000, 0.0005, 0.0003)
    s_all = run_backtest(data, MultiFactor(), RiskManager(RISK, LEV), 1000, 0.0005, 0.0003)
    assert 0 < len(s_all.trades) < len(s_off.trades)


def test_atr_risk_sizing():
    risk = {**RISK, "stop_mode": "atr", "stop_atr_mult": 2.0, "take_profit_atr_mult": 4.0, "risk_per_trade_pct": 0.01}
    rm = RiskManager(risk, LEV)
    p = rm.plan_entry(equity=1000, available=1000, price=100, side=LONG, atr=2.0)   # stop 96
    assert p.stop == pytest.approx(96) and p.take_profit == pytest.approx(108)
    assert (100 - p.stop) * p.qty == pytest.approx(10)          # kayıp = bakiyenin %1'i (teminat 125)
    wide = rm.plan_entry(equity=1000, available=1000, price=100, side=LONG, atr=0.1)  # çok dar stop
    assert wide.margin <= 1000 * 0.20 + 1e-9                     # teminat tavanı aşılmaz


def test_subset_uses_warm_indicators():
    data = synthetic_portfolio(COINS[:3], n=3000)
    res = run_backtest(data, MultiFactor(), RiskManager(RISK, LEV), 1000, 0.0005, 0.0003)
    cut = res.equity.index[2100]
    tr, te = res.subset(None, cut), res.subset(cut)
    assert len(tr.trades) + len(te.trades) == len(res.trades)
    assert te.initial == pytest.approx(float(res.equity.loc[:cut].iloc[-2]))
    assert set(te.buy_hold) == set(COINS[:3])


def test_trader_multifactor_with_btc_context(tmp_path):
    coins = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    src = FakeSource(synthetic_portfolio(coins, n=2600))
    paper = PaperFutures(src, 1000, 0.0005, 0.0003, LEV)
    cfg = _cfg(tmp_path, coins=coins)
    t = Trader(cfg, paper, MultiFactor(adx_min=15), RiskManager(RISK, LEV), state_path=tmp_path / "state.json")
    src.i = 1700
    for i in range(src.i, 2599):
        src.i = i
        assert t.step()
    assert len(t.state["trades"]) > 0
    assert len(paper.pos) == len(t.positions)


# ------------------------------------------------------------------ v0.6: limit emir, ücretler, yeni veriler
from bot.strategy import Strategy as _BaseStrategy


class _OneShot(_BaseStrategy):
    """Belirli mumda LONG sinyali veren deneme stratejisi."""

    def __init__(self, at):
        super().__init__()
        self.at = at

    @property
    def warmup(self):
        return 1

    def generate(self, df, btc=None):
        out = df.copy()
        out["atr"] = 1.0
        for c in ("long_entry", "short_entry", "long_exit", "short_exit"):
            out[c] = False
        out.iloc[self.at, out.columns.get_loc("long_entry")] = True
        return out


def _bars(rows):
    idx = pd.date_range("2025-01-01", periods=len(rows), freq="1h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)


RISK_NO_TP = {**RISK, "take_profit_roe": 0}


def test_limit_entry_fills_only_when_price_crosses():
    # sinyal mumu 1 (kapanış 100). Mum 2'nin dibi 99.5 < 100 -> dolar, maker ücreti, kaymasız.
    df = _bars([(100, 101, 99, 100), (100, 101, 99, 100), (100.5, 101, 99.5, 100.8), (100.8, 101, 100, 100.5)])
    res = run_backtest(df, _OneShot(1), RiskManager(RISK_NO_TP, LEV), 1000, 0.0005, 0.0003,
                       maker_fee=0.0002, entry_order="limit")
    t = res.trades[0]
    assert t.entry_price == pytest.approx(100.0)
    assert t.entry_fee == pytest.approx(t.qty * 100 * 0.0002)
    # dibi tam 100'e DOKUNUP geçmeyen mum -> dolmaz (kuyruk önceliği temkini)
    df2 = _bars([(100, 101, 99, 100), (100, 101, 99, 100), (100.5, 101, 100.0, 100.8), (100.8, 101, 100.2, 100.5)])
    res2 = run_backtest(df2, _OneShot(1), RiskManager(RISK_NO_TP, LEV), 1000, 0.0005, 0.0003,
                        maker_fee=0.0002, entry_order="limit")
    assert res2.trades == [] and res2.limit_stats == {"limit_placed": 1, "limit_filled": 0}
    # aynı senaryo piyasa emriyle: mum 2 açılışından + kayma, taker ücreti
    res3 = run_backtest(df2, _OneShot(1), RiskManager(RISK_NO_TP, LEV), 1000, 0.0005, 0.0003,
                        maker_fee=0.0002, entry_order="market")
    t3 = res3.trades[0]
    assert t3.entry_price == pytest.approx(100.5 * 1.0003)
    assert t3.entry_fee == pytest.approx(t3.qty * t3.entry_price * 0.0005)


def test_limit_gap_fills_at_better_open():
    df = _bars([(100, 101, 99, 100), (100, 101, 99, 100), (99.0, 99.5, 98.8, 99.2), (99.2, 99.5, 99, 99.3)])
    res = run_backtest(df, _OneShot(1), RiskManager(RISK_NO_TP, LEV), 1000, 0.0005, 0.0003,
                       maker_fee=0.0002, entry_order="limit")
    assert res.trades[0].entry_price == pytest.approx(99.0)        # açılış limitten iyiyse açılıştan


def test_take_profit_limit_is_maker():
    risk = {**RISK, "take_profit_roe": 0.05}                         # 2x -> fiyatın %2.5'i
    df = _bars([(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99.5, 100.5), (100.5, 103.5, 100, 103)])
    res = run_backtest(df, _OneShot(1), RiskManager(risk, LEV), 1000, 0.0005, 0.0003,
                       maker_fee=0.0002, entry_order="limit", tp_order="limit")
    t = res.trades[0]
    assert t.reason == "kâr-al" and t.exit_price == pytest.approx(102.5)
    assert t.exit_fee == pytest.approx(t.qty * 102.5 * 0.0002)


def test_limit_vs_market_on_portfolio():
    data = synthetic_portfolio(COINS, n=3000)
    s = MultiFactor(use_btc_filter=True)
    mk = run_backtest(data, s, RiskManager(RISK, LEV), 1000, 0.0005, 0.0003, maker_fee=0.0002)
    lm = run_backtest(data, s, RiskManager(RISK, LEV), 1000, 0.0005, 0.0003, maker_fee=0.0002,
                      entry_order="limit", tp_order="limit")
    st = lm.limit_stats
    assert 0 < st["limit_filled"] <= st["limit_placed"]
    assert len(lm.trades) <= len(mk.trades)
    fee_mk = sum(t.entry_fee for t in mk.trades) / max(1, len(mk.trades))
    fee_lm = sum(t.entry_fee for t in lm.trades) / max(1, len(lm.trades))
    assert fee_lm < fee_mk


def test_funding_layer_blocks_crowded_side():
    df = synthetic_ohlcv(n=4000, seed=7)
    f = pd.Series(0.0001, index=df.index)
    f.iloc[2000:2600] = 0.002                                         # aşırı pozitif funding: kalabalık long
    df["funding_rate"] = f
    on = MultiFactor(use_btc_filter=False, use_funding=True).generate(df)
    off = MultiFactor(use_btc_filter=False).generate(df)
    win = slice(2000, 2600)
    assert on["long_entry"].iloc[win].sum() == 0
    assert off["long_entry"].sum() >= on["long_entry"].sum()
    assert on["short_entry"].iloc[win].sum() == off["short_entry"].iloc[win].sum()   # short engellenmez


def test_taker_flow_layer():
    df = synthetic_ohlcv(n=4000, seed=7)
    df["taker_buy_volume"] = df["volume"] * 0.3                       # sürekli satıcı baskısı
    out = MultiFactor(use_btc_filter=False, use_taker_flow=True).generate(df)
    assert out["long_entry"].sum() == 0
    df["taker_buy_volume"] = df["volume"] * 0.7                       # sürekli alıcı baskısı
    out = MultiFactor(use_btc_filter=False, use_taker_flow=True).generate(df)
    assert out["short_entry"].sum() == 0


def test_attach_funding_no_lookahead():
    from bot.backtest import attach_funding
    df = synthetic_ohlcv(n=48)
    f = pd.Series([0.001, 0.002], index=pd.DatetimeIndex(
        ["2024-01-01 08:00:00.005", "2024-01-01 16:00:00.003"], tz="UTC"))
    out = attach_funding(df, f)
    assert pd.isna(out.loc["2024-01-01 07:00", "funding_rate"])
    assert out.loc["2024-01-01 08:00", "funding_rate"] == 0.001
    assert out.loc["2024-01-01 15:00", "funding_rate"] == 0.001
    assert out.loc["2024-01-01 16:00", "funding_rate"] == 0.002


def test_trader_limit_entries_paper(tmp_path):
    src = FakeSource(synthetic_portfolio(COINS, n=1500))
    paper = PaperFutures(src, 1000, 0.0005, 0.0003, LEV, maker_fee=0.0002, tp_limit=True)
    cfg = _cfg(tmp_path)
    cfg["execution"] = {"entry_order": "limit", "limit_offset_atr": 0.0, "limit_ttl_bars": 1, "tp_order": "limit"}
    t = Trader(cfg, paper, SmaCrossRsi(**PARAMS), RiskManager(RISK, LEV), state_path=tmp_path / "state.json")
    placed = 0
    for i in range(200, 1499):
        src.i = i
        before = len(t.pending)
        alive = t.step()
        placed += max(0, len(t.pending) - before)
        assert t.slots_used() <= 3
        assert len(t.pending) <= 3
        assert set(paper.pos) == set(t.positions)                     # bot ve hesap senkron
        if not alive:                                                 # devre kesici durdurduysa
            assert not t.pending and not t.positions
            break
    trades = t.state["trades"]
    assert trades and placed > len(trades)                            # bazı limit emirler dolmadı
    for x in trades:                                                  # giriş ücreti maker oranında
        assert x["fees"] > 0
    # yeniden başlatınca bekleyen emirler geri yüklenir
    t2 = Trader(cfg, PaperFutures(src, 1000, 0.0005, 0.0003, LEV, maker_fee=0.0002, tp_limit=True),
                SmaCrossRsi(**PARAMS), RiskManager(RISK, LEV), state_path=tmp_path / "state.json")
    assert set(t2.pending) == set(t.pending) and set(t2.ex.orders) == set(paper.orders)


def test_breaker_cancels_pending(tmp_path):
    src = FakeSource(synthetic_portfolio(COINS, n=800))
    paper = PaperFutures(src, 1000, 0.0005, 0.0003, LEV, maker_fee=0.0002)
    cfg = _cfg(tmp_path)
    cfg["execution"] = {"entry_order": "limit", "limit_ttl_bars": 3}
    t = Trader(cfg, paper, SmaCrossRsi(**PARAMS), RiskManager(RISK, LEV), state_path=tmp_path / "s.json")
    src.i = 300
    oid = paper.place_limit_entry("ETH/USDT:USDT", LONG, 0.01, 1.0)   # dolmayacak kadar düşük fiyat
    t.pending["ETH/USDT:USDT"] = {"id": oid, "side": LONG, "qty": 0.01, "price": 1.0, "atr": 1.0,
                                  "margin": 0.005, "placed": "x", "placed_ms": 0, "candles": 0}
    t._close_all("devre kesici")
    assert not t.pending and not paper.orders


# ------------------------------------------------------------------ veri toplayıcı
class _FakeMetricsEx:
    def __init__(self):
        self.calls = 0
        now = int(pd.Timestamp.now(tz="UTC").floor("h").timestamp() * 1000)
        self.ts = [now - k * 3600_000 for k in range(1200, 0, -1)]   # 50 gün: API penceresinden uzun

    def market(self, s):
        return {"id": s.split(":")[0].replace("/", "")}

    def _rows(self, params, fields):
        # Binance davranışı: endTime'dan önceki en son `limit` kayıt
        self.calls += 1
        rows = [{"timestamp": t, **{k: str(1 + j * 0.1) for j, k in enumerate(fields)}}
                for t in self.ts if t <= params.get("endTime", 10**15)]
        return rows[-params["limit"]:]

    def fapiDataGetOpenInterestHist(self, p): return self._rows(p, ["sumOpenInterest", "sumOpenInterestValue"])
    def fapiDataGetTopLongShortPositionRatio(self, p): return self._rows(p, ["longShortRatio", "longAccount", "shortAccount"])
    def fapiDataGetTopLongShortAccountRatio(self, p): return self._rows(p, ["longShortRatio", "longAccount", "shortAccount"])
    def fapiDataGetGlobalLongShortAccountRatio(self, p): return self._rows(p, ["longShortRatio", "longAccount", "shortAccount"])
    def fapiDataGetTakerlongshortRatio(self, p): return self._rows(p, ["buySellRatio", "buyVol", "sellVol"])


def test_collector_merges_without_duplicates(tmp_path, monkeypatch):
    from bot import collector
    monkeypatch.setattr(collector, "METRICS_DIR", tmp_path)
    cfg = {"exchange": {"symbols": ["BTC/USDT", "ETH/USDT"]}}
    ex = _FakeMetricsEx()
    s1 = collector.collect_recent(cfg, ex)
    n = s1[("BTC/USDT", "open_interest")]
    assert 29 * 24 - 2 <= n <= 29 * 24 + 1                            # 500'lük sayfalarla tüm 29 gün
    s2 = collector.collect_recent(cfg, ex)                            # tekrar: yeni kayıt yok
    assert all(v == 0 for v in s2.values())
    m = collector.load_metrics("BTC/USDT:USDT")
    assert len(m) == n and "open_interest_oi" in m and "taker_ratio_ratio" in m


def test_archive_parser():
    import io, zipfile
    from bot.collector import parse_archive_csv
    csv = ("create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
           "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
           "2025-01-01 00:05:00,BTCUSDT,100,1000,1.1,1.2,0.9,1.05\n2025-01-01 00:10:00,BTCUSDT,101,1010,1.1,1.2,0.9,0.95\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("BTCUSDT-metrics-2025-01-01.csv", csv)
    df = parse_archive_csv(buf.getvalue())
    assert len(df) == 2 and df["sum_open_interest"].iloc[1] == 101 and "symbol" not in df


def test_klines_parser_keeps_taker_volume():
    from bot.exchange import klines_to_df
    rows = [[1735689600000, "100", "101", "99", "100.5", "10", 1735693199999, "1000", 50, "6", "600", "0"]]
    df = klines_to_df(rows)
    assert df["taker_buy_volume"].iloc[0] == 6 and df["close"].iloc[0] == 100.5


# ------------------------------------------------------------------ v0.7: tek / ikili indikatör stratejileri
from bot.signals import COMMON_PAIRS, CONFIRMS, INDICATORS, SignalStrategy


@pytest.mark.parametrize("name", list(INDICATORS))
def test_indicator_no_lookahead(name):
    df = synthetic_ohlcv(n=1400, seed=3)
    cut = 1000
    s = SignalStrategy(name, None)
    a = s.generate(df.iloc[:cut])
    b = df.copy()
    b.iloc[cut:, :5] *= 1.5
    bb = s.generate(b).iloc[:cut]
    for col in ("long_entry", "short_entry", "long_exit", "short_exit"):
        assert (a[col].values == bb[col].values).all(), (name, col)


def test_confirm_filters_and_respects_direction():
    df = synthetic_ohlcv(n=3000, seed=4)
    single = SignalStrategy("macd", None).generate(df)
    paired = SignalStrategy("macd", "ema200").generate(df)
    assert paired["long_entry"].sum() < single["long_entry"].sum()
    assert (paired["long_entry"] <= single["long_entry"]).all()       # onay sadece eler, yeni sinyal üretmez
    ema = paired["close"].ewm(span=200, adjust=False, min_periods=200).mean()
    assert (paired.loc[paired["long_entry"], "close"] > ema[paired["long_entry"]]).all()
    assert (paired.loc[paired["short_entry"], "close"] < ema[paired["short_entry"]]).all()


def test_all_common_pairs_valid():
    for p, cs in COMMON_PAIRS.items():
        assert p in INDICATORS
        for c in cs:
            assert c in CONFIRMS
    df = synthetic_ohlcv(n=600, seed=9)
    for c in CONFIRMS:                                                 # her onay fonksiyonu çalışmalı
        SignalStrategy("fibonacci", c).generate(df)


def test_fib_lookback_param_changes_signals():
    df = synthetic_ohlcv(n=3000, seed=4)
    a = SignalStrategy("fibonacci", None, fib_lookback=48).generate(df)["long_entry"]
    b = SignalStrategy("fibonacci", None, fib_lookback=120).generate(df)["long_entry"]
    assert (a != b).any()


def test_default_config_builds_signal_strategy():
    from bot.config import load_config
    from bot.strategy import build_strategy
    s = build_strategy(load_config())
    assert isinstance(s, SignalStrategy) and s.primary == "fibonacci" and s.confirm == "volume"


# ------------------------------------------------------------------ v0.7.1: kademeli izleyen stop
TRAIL = {"enabled": True, "step_r": 1.0, "gap_r": 1.0, "move_tp": True}


def test_trail_update_long_and_short():
    rm = RiskManager({**RISK, "trailing": TRAIL}, LEV)
    # long: giriş 80k, stop 78k, TP 83k
    assert rm.trail_update(1, 80000, 78000, 83000, 0, 81900) == (0, None, None)
    assert rm.trail_update(1, 80000, 78000, 83000, 0, 82000) == (1, 80000, 85000)
    assert rm.trail_update(1, 80000, 78000, 83000, 1, 84100) == (2, 82000, 87000)
    assert rm.trail_update(1, 80000, 78000, 83000, 2, 83000) == (2, None, None)     # geri gevşemez
    # short: giriş 100, stop 102.5 (R=2.5), TP 0 (yok)
    k, st, tp = rm.trail_update(-1, 100, 102.5, 0, 0, 95.0)
    assert (k, st, tp) == (2, pytest.approx(97.5), 0)
    assert RiskManager(RISK, LEV).trail_update(1, 80000, 78000, 83000, 0, 90000) == (0, None, None)  # kapalı


def test_backtest_trailing_user_example():
    risk = {**RISK, "take_profit_roe": 0.075, "trailing": TRAIL}      # 2x: stop 78k, TP 83k
    df = _bars([(80000, 80100, 79900, 80000), (80000, 80100, 79900, 80000),
                (80050, 80200, 79950, 80100),                         # limit 80k dolar
                (80100, 82100, 80050, 82000),                         # 82k -> stop 80k, TP 85k
                (82000, 82500, 79900, 80000)])                        # 80k'den başa baş
    r = run_backtest(df, _OneShot(1), RiskManager(risk, LEV), 1000, 0.0005, 0.0, maker_fee=0.0002,
                     entry_order="limit", tp_order="limit")
    t = r.trades[0]
    assert (t.init_stop, t.init_tp) == (pytest.approx(78000), pytest.approx(83000))
    assert (t.stop, t.take_profit, t.steps) == (pytest.approx(80000), pytest.approx(85000), 1)
    assert t.exit_price == pytest.approx(80000) and t.reason == "başa baş"


def test_backtest_trailing_conservative_same_bar():
    """Aynı mumda hem yeni tepe hem de eski stop'un altı varsa: eski stop'tan çıkılır (temkinli)."""
    risk = {**RISK, "take_profit_roe": 0, "trailing": TRAIL}
    df = _bars([(80000, 80100, 79900, 80000), (80000, 80100, 79900, 80000),
                (80050, 80200, 79950, 80100), (80100, 82500, 77900, 80000)])
    r = run_backtest(df, _OneShot(1), RiskManager(risk, LEV), 1000, 0.0005, 0.0, maker_fee=0.0002,
                     entry_order="limit")
    assert r.trades[0].reason == "stop-loss" and r.trades[0].exit_price == pytest.approx(78000)


def test_trader_trailing_paper(tmp_path):
    src = FakeSource(synthetic_portfolio(COINS, n=1500))
    risk = {**RISK, "take_profit_roe": 0, "trailing": {**TRAIL, "step_r": 0.5}}
    paper = PaperFutures(src, 1000, 0.0005, 0.0003, LEV)
    cfg = _cfg(tmp_path)
    cfg["risk"] = risk
    t = Trader(cfg, paper, SmaCrossRsi(**PARAMS), RiskManager(risk, LEV), state_path=tmp_path / "s.json")
    last_stop, raised = {}, 0
    for i in range(200, 1499):
        src.i = i
        if not t.step():
            break
        for sym, p in t.positions.items():
            key = (sym, p["entry"], p["side"], p["init_stop"])     # pozisyonu benzersiz tanımlar
            prev = last_stop.get(key)
            if prev is not None:
                assert (p["stop"] >= prev) if p["side"] == LONG else (p["stop"] <= prev)   # stop geri gitmez
                raised += p["stop"] != prev
            last_stop[key] = p["stop"]
            assert paper.protect[sym][0] == pytest.approx(p["stop"])                     # borsadaki stop = durum
    reasons = {x["reason"] for x in t.state["trades"]}
    assert raised > 0 and ({"izleyen stop", "başa baş"} & reasons)


class _RecCcxt:
    def __init__(self):
        self.log, self.n = [], 0

    def create_order(self, symbol, type_, side, qty, price=None, params=None):
        self.n += 1
        kind = "sl" if params and "stopLossPrice" in params else "tp"
        self.log.append(("create", kind, (params or {}).get("stopLossPrice", price)))
        return {"id": f"o{self.n}"}

    def cancel_order(self, oid, symbol, params=None):
        self.log.append(("cancel", oid, bool(params and params.get("trigger"))))

    def amount_to_precision(self, s, q): return q
    def price_to_precision(self, s, p): return p


def test_update_protection_places_new_before_cancel():
    c = FuturesExchange.__new__(FuturesExchange)
    c.ex, c.tp_limit = _RecCcxt(), True
    pos = Position(LONG, 0.01, 80000)
    ids = c.place_protection("BTC/USDT:USDT", pos, 78000, 83000)
    c.ex.log.clear()
    new = c.update_protection("BTC/USDT:USDT", pos, ids, 80000, 85000)
    kinds = [(a, b) for a, b, _ in c.ex.log]
    assert kinds == [("create", "sl"), ("cancel", ids["sl"]), ("create", "tp"), ("cancel", ids["tp"])]
    assert c.ex.log[1][2] is True and c.ex.log[3][2] is False          # stop = koşullu emir, TP = limit
    assert new["sl"] != ids["sl"] and new["tp_price"] == 85000


# ------------------------------------------------------------------ v0.7.2: kısmi kâr-al
PARTIAL = {"enabled": True, "at_r": 1.0, "fraction": 0.5, "lock_r": 0.5, "step_r": 0.5, "gap_r": 1.0}


def test_runner_stop_rules():
    rm = RiskManager({**RISK, "partial": PARTIAL}, LEV)
    assert rm.partial_target(1, 80000, 78000) == pytest.approx(82000)
    assert rm.runner_stop(1, 80000, 78000, 78000, 82000) == pytest.approx(81000)   # taban: giriş + 0.5R
    assert rm.runner_stop(1, 80000, 78000, 81000, 84100) == pytest.approx(82000)
    assert rm.runner_stop(1, 80000, 78000, 82000, 83000) is None                   # geri gelmez
    assert rm.runner_stop(-1, 100, 102, 102, 96) == pytest.approx(98)              # short: 96 + 1R


def test_backtest_partial_user_example():
    risk = {**RISK, "partial": PARTIAL}
    df = _bars([(80000, 80100, 79900, 80000), (80000, 80100, 79900, 80000), (80050, 80200, 79950, 80100),
                (80100, 82100, 81100, 82000), (82000, 84100, 81500, 84000), (84000, 84200, 81900, 82000)])
    r = run_backtest(df, _OneShot(1), RiskManager(risk, LEV), 1000, 0.0005, 0.0, maker_fee=0.0002,
                     entry_order="limit", tp_order="limit")
    a, b = r.trades
    assert a.reason == "kısmi kâr-al" and a.exit_price == pytest.approx(82000) and a.part == "kısmi"
    assert b.reason == "izleyen stop" and b.exit_price == pytest.approx(82000)
    assert a.qty == pytest.approx(b.qty)
    assert r.equity.iloc[-1] == pytest.approx(1000 + a.pnl + b.pnl)


def test_backtest_partial_same_bar_is_conservative():
    risk = {**RISK, "partial": PARTIAL}
    df = _bars([(80000, 80100, 79900, 80000), (80000, 80100, 79900, 80000), (80050, 80200, 79950, 80100),
                (80100, 82100, 80050, 82000), (82000, 84100, 81500, 84000)])
    r = run_backtest(df, _OneShot(1), RiskManager(risk, LEV), 1000, 0.0005, 0.0, maker_fee=0.0002,
                     entry_order="limit", tp_order="limit")
    assert [t.reason for t in r.trades] == ["kısmi kâr-al", "izleyen stop"]
    assert r.trades[1].exit_price == pytest.approx(81000)          # aynı mumda yeni stop'un altı görüldü


def test_live_refuses_partial(tmp_path):
    src = FakeSource(synthetic_portfolio(COINS, n=400))
    with pytest.raises(RuntimeError):
        Trader(_cfg(tmp_path), PaperFutures(src, 1000, 0.0005, 0.0003, LEV), SmaCrossRsi(**PARAMS),
               RiskManager({**RISK, "partial": PARTIAL}, LEV), state_path=tmp_path / "s.json")


# ------------------------------------------------------------------ v0.8: sinyal stratejisine trend filtreleri
def test_signal_trend_filters_respect_direction():
    from bot.signals import htf_trend
    df = synthetic_ohlcv(n=4000, seed=7)
    s = SignalStrategy("fibonacci", None, use_htf=True, htf_mode="strict")
    out = s.generate(df)
    base = SignalStrategy("fibonacci", None).generate(df)
    assert 0 < out["long_entry"].sum() + out["short_entry"].sum() < base["long_entry"].sum() + base["short_entry"].sum()
    assert (out.loc[out["long_entry"], "htf_dir"] == 1).all()
    assert (out.loc[out["short_entry"], "htf_dir"] == -1).all()
    loose = htf_trend(df, {**s.params, "htf_mode": "loose"})["dir"]
    strict = htf_trend(df, s.params)["dir"]
    assert ((strict != 0) <= (loose != 0)).all()                    # katı tanım gevşeğin alt kümesi


def test_signal_btc_filter_and_flags():
    df, btc = synthetic_ohlcv(n=3000, seed=7), synthetic_ohlcv(n=3000, seed=8)
    s = SignalStrategy("fibonacci", None, use_btc_filter=True)
    assert s.uses_btc and not SignalStrategy().uses_btc
    out = s.generate(df, btc)
    assert (out.loc[out["long_entry"], "btc_dir"] != -1).all()
    assert (out.loc[out["short_entry"], "btc_dir"] != 1).all()


def test_signal_filters_no_lookahead():
    df, btc = synthetic_ohlcv(n=2000, seed=3), synthetic_ohlcv(n=2000, seed=4)
    s = SignalStrategy(use_regime=True, use_htf=True, use_btc_filter=True)
    cut = 1500
    a = s.generate(df.iloc[:cut], btc.iloc[:cut])
    b, bb = df.copy(), btc.copy()
    b.iloc[cut:, :5] *= 1.5
    bb.iloc[cut:, :5] *= 0.6
    x = s.generate(b, bb).iloc[:cut]
    for c in ("long_entry", "short_entry", "long_exit", "short_exit"):
        assert (a[c].values == x[c].values).all()


def test_default_config_has_strict_htf():
    from bot.config import load_config
    from bot.strategy import build_strategy
    s = build_strategy(load_config())
    assert s.params["use_htf"] and s.params["htf_mode"] == "strict" and not s.params["use_btc_filter"]
