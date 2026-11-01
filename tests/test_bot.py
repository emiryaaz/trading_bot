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
