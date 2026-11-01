"""Çoklu coin, vadeli işlem (USDⓈ-M, isolated) PORTFÖY backtest motoru.

Tüm coinler aynı zaman çizelgesinde, ORTAK bir bakiye üzerinden simüle edilir:
  * Sinyal mum KAPANIŞINDA oluşur, işlem bir SONRAKİ mumun açılışında yapılır (geleceği görme yok).
  * Her açılış/kapanışta pozisyon büyüklüğü üzerinden işlem ücreti ve aleyhte kayma düşülür.
  * 8 saatte bir (00:00, 08:00, 16:00 UTC) funding ödenir/alınır. Gerçek funding verisi varsa o kullanılır.
  * Stop-loss / kâr al mum içi en düşük/en yüksek fiyata göre kontrol edilir; aynı mumda ikisi de
    tetiklenirse kötümser varsayımla önce stop sayılır. Boşlukla stop'un ötesinde açılırsa açılıştan kapanır.
  * Isolated marjin: bir işlemde kayıp asla o işlemin teminatını aşamaz.
  * Portföy sınırları: aynı anda en fazla `max_open_positions` pozisyon, toplam teminat en fazla
    bakiyenin `max_total_margin_pct`'i. Aynı mumda birden çok sinyal gelirse config'deki coin sırası önceliklidir.
  * DEVRE KESİCİ: toplam bakiye son başlatmaya göre `max_drawdown_pct` düşerse TÜM pozisyonlar kapatılır.
    Simülasyon "kullanıcı inceleyip yeniden başlattı" varsayımıyla devam eder ki kaç kez tetiklendiği görülsün.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .risk import LONG, RiskManager
from .strategy import Strategy


@dataclass
class Trade:
    symbol: str
    side: int
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    margin: float
    stop: float
    take_profit: float
    entry_fee: float
    exit_time: pd.Timestamp | None = None
    exit_price: float = 0.0
    exit_fee: float = 0.0
    funding: float = 0.0      # ödenen (+) / alınan (-) funding toplamı
    reason: str = ""

    @property
    def gross(self) -> float:
        return self.side * (self.exit_price - self.entry_price) * self.qty

    @property
    def pnl(self) -> float:
        return max(self.gross - self.entry_fee - self.exit_fee - self.funding, -self.margin)

    @property
    def roe(self) -> float:
        return self.pnl / self.margin if self.margin else 0.0

    def as_dict(self) -> dict:
        return {"coin": self.symbol, "yön": "LONG" if self.side == LONG else "SHORT",
                "giriş_zamanı": self.entry_time, "giriş": self.entry_price, "çıkış_zamanı": self.exit_time,
                "çıkış": self.exit_price, "miktar": self.qty, "teminat": round(self.margin, 2),
                "ücret": round(self.entry_fee + self.exit_fee, 4), "funding": round(self.funding, 4),
                "kz": round(self.pnl, 4), "roe_%": round(self.roe * 100, 2), "neden": self.reason}


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade] = field(default_factory=list)
    breaker_events: list[dict] = field(default_factory=list)
    initial: float = 0.0
    buy_hold: dict = field(default_factory=dict)      # coin -> al-ve-tut getirisi
    bars_per_year: float = 8760

    @property
    def buy_hold_return(self) -> float:
        return float(np.mean(list(self.buy_hold.values()))) if self.buy_hold else 0.0

    def metrics(self) -> dict:
        eq = self.equity
        rets = eq.pct_change().dropna()
        dd = (eq / eq.cummax() - 1).min()
        tr = self.trades
        wins = [t for t in tr if t.pnl > 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = -sum(t.pnl for t in tr if t.pnl <= 0)
        longs = [t for t in tr if t.side == LONG]
        shorts = [t for t in tr if t.side != LONG]
        sharpe = (rets.mean() / rets.std() * np.sqrt(self.bars_per_year)) if rets.std() > 0 else 0.0
        reasons = pd.Series([t.reason for t in tr]).value_counts().to_dict() if tr else {}
        first = self.breaker_events[0]["time"] if self.breaker_events else "-"
        return {
            "Başlangıç bakiyesi": round(self.initial, 2),
            "Bitiş bakiyesi": round(float(eq.iloc[-1]), 2),
            "Toplam getiri %": round((eq.iloc[-1] / self.initial - 1) * 100, 2),
            "Al-ve-tut (coin ortalaması) %": round(self.buy_hold_return * 100, 2),
            "Maks. düşüş (drawdown) %": round(dd * 100, 2),
            "İşlem sayısı (long/short)": f"{len(tr)} ({len(longs)}/{len(shorts)})",
            "Kazanma oranı %": round(len(wins) / len(tr) * 100, 1) if tr else 0.0,
            "Long K/Z": round(sum(t.pnl for t in longs), 2),
            "Short K/Z": round(sum(t.pnl for t in shorts), 2),
            "Kâr faktörü": round(gross_win / gross_loss, 2) if gross_loss > 0 else (float("inf") if gross_win else 0.0),
            "Ödenen ücret": round(sum(t.entry_fee + t.exit_fee for t in tr), 2),
            "Net funding (ödenen)": round(sum(t.funding for t in tr), 2),
            "Kapanış nedenleri": reasons,
            "Devre kesici tetiklenme": f"{len(self.breaker_events)} kez, ilk: {first}",
            "Sharpe (yıllık)": round(float(sharpe), 2),
        }

    def per_symbol(self) -> pd.DataFrame:
        rows = []
        for sym in sorted({t.symbol for t in self.trades} | set(self.buy_hold), key=str):
            tr = [t for t in self.trades if t.symbol == sym]
            w = [t for t in tr if t.pnl > 0]
            gw, gl = sum(t.pnl for t in w), -sum(t.pnl for t in tr if t.pnl <= 0)
            rows.append({"Coin": sym, "İşlem": len(tr),
                         "Kazanma %": round(len(w) / len(tr) * 100, 1) if tr else 0.0,
                         "K/Z": round(sum(t.pnl for t in tr), 2),
                         "Long K/Z": round(sum(t.pnl for t in tr if t.side == LONG), 2),
                         "Short K/Z": round(sum(t.pnl for t in tr if t.side != LONG), 2),
                         "Kâr faktörü": round(gw / gl, 2) if gl > 0 else (float("inf") if gw else 0.0),
                         "Stop %": round(sum(t.reason == "stop-loss" for t in tr) / len(tr) * 100) if tr else 0,
                         "Al-ve-tut %": round(self.buy_hold.get(sym, 0) * 100, 1)})
        return pd.DataFrame(rows).set_index("Coin").sort_values("K/Z", ascending=False)


def _funding_rates(index: pd.DatetimeIndex, funding: pd.Series | None, fallback: float) -> np.ndarray:
    """Her mum için o mumun açılışında uygulanacak funding oranı (funding saati değilse 0)."""
    is_funding_time = (index.hour % 8 == 0) & (index.minute == 0)
    rates = np.where(is_funding_time, fallback, 0.0)
    if funding is not None and len(funding):
        f = funding.copy()
        f.index = f.index.floor("h")
        f = f[~f.index.duplicated()]
        aligned = f.reindex(index.floor("h"))
        mask = is_funding_time & aligned.notna().to_numpy()
        rates[mask] = aligned.to_numpy()[mask]
    return rates


class _Sym:
    """Tek bir coinin ortak zaman çizelgesine hizalanmış dizileri."""

    def __init__(self, name, df, strategy, index, funding, fallback):
        sig = strategy.generate(df).reindex(index)
        self.name = name
        self.o, self.h, self.l, self.c = (sig[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        self.le, self.se, self.lx, self.sx = (sig[k].fillna(False).to_numpy(dtype=bool)
                                              for k in ("long_entry", "short_entry", "long_exit", "short_exit"))
        self.atr = sig["atr"].to_numpy(dtype=float)
        self.fund = _funding_rates(index, funding, fallback)
        valid = np.where(~np.isnan(self.c))[0]
        self.first = valid[0] if len(valid) else len(index)
        self.ready_from = self.first + strategy.warmup
        self.last_close = np.nan


def run_backtest(data: pd.DataFrame | dict[str, pd.DataFrame], strategy: Strategy, risk: RiskManager,
                 initial: float, fee_rate: float, slippage: float,
                 allow_short: bool = True, funding: pd.Series | dict | None = None,
                 funding_fallback: float = 0.0001, bars_per_year: float = 8760) -> BacktestResult:
    if isinstance(data, pd.DataFrame):
        data = {"COIN": data}
    if funding is None or isinstance(funding, pd.Series):
        funding = {s: funding for s in data}
    index = sorted(set().union(*[d.index for d in data.values()]))
    index = pd.DatetimeIndex(index)
    syms = [_Sym(s, d, strategy, index, funding.get(s), funding_fallback) for s, d in data.items()]

    max_pos = risk.c.get("max_open_positions", 1)
    max_margin = risk.c.get("max_total_margin_pct", 1.0)

    cash = float(initial)
    pos: dict[str, Trade] = {}
    trades: list[Trade] = []
    breaker: list[dict] = []
    equity = np.empty(len(index))
    baseline = initial
    day, day_start_eq = None, initial
    pend_exit: set[str] = set()
    pend_entry: dict[str, int] = {}

    def px_of(s: _Sym, i: int, arr) -> float:
        v = arr[i]
        return s.last_close if np.isnan(v) else v

    def unreal(i: int, kind: str = "c") -> float:
        tot = 0.0
        for s in syms:
            t = pos.get(s.name)
            if t:
                p = px_of(s, i, getattr(s, kind))
                tot += t.side * (p - t.entry_price) * t.qty
        return tot

    def close_pos(s: _Sym, i: int, raw: float, reason: str) -> None:
        nonlocal cash
        t = pos.pop(s.name)
        px = raw * (1 - t.side * slippage)
        t.exit_fee = px * t.qty * fee_rate
        t.exit_time, t.exit_price, t.reason = index[i], px, reason
        cash += t.pnl + t.entry_fee + t.funding     # giriş ücreti ve funding zaten düşülmüştü
        trades.append(t)

    def open_pos(s: _Sym, i: int, side: int) -> None:
        nonlocal cash
        used = sum(t.margin for t in pos.values())
        eq_now = cash + unreal(i, "o")
        room = max(0.0, eq_now * max_margin - used)
        available = min(cash - used, room)
        plan = risk.plan_entry(eq_now, available, s.o[i], side, s.atr[i - 1])
        if not plan:
            return
        px = s.o[i] * (1 + side * slippage)
        stop, tp = risk.protection_prices(px, side, s.atr[i - 1])
        fee = px * plan.qty * fee_rate
        cash -= fee
        pos[s.name] = Trade(s.name, side, index[i], px, plan.qty, plan.margin, stop, tp, fee)

    for i in range(len(index)):
        d = index[i].date()
        if d != day:
            day, day_start_eq = d, cash + unreal(i, "o")

        for s in syms:
            if np.isnan(s.o[i]):
                continue
            t = pos.get(s.name)
            # 1) funding
            if t and s.fund[i]:
                pay = t.side * t.qty * s.o[i] * s.fund[i]
                t.funding += pay
                cash -= pay
            # 2) önceki mumun çıkış kararı
            if t and s.name in pend_exit:
                close_pos(s, i, s.o[i], "sinyal")

        # 2b) girişler (öncelik sırası = config sırası)
        if pend_entry and not risk.daily_limit_hit(day_start_eq, cash + unreal(i, "o")):
            for s in syms:
                side = pend_entry.get(s.name)
                if side and s.name not in pos and len(pos) < max_pos and not np.isnan(s.o[i]):
                    open_pos(s, i, side)
        pend_exit.clear()
        pend_entry.clear()

        # 3) mum içi stop-loss / kâr al
        for s in syms:
            t = pos.get(s.name)
            if not t or np.isnan(s.o[i]):
                continue
            o, h, l = s.o[i], s.h[i], s.l[i]
            if t.side == LONG:
                if l <= t.stop:
                    close_pos(s, i, min(o, t.stop), "stop-loss")
                elif t.take_profit and h >= t.take_profit:
                    close_pos(s, i, max(o, t.take_profit), "kâr-al")
            else:
                if h >= t.stop:
                    close_pos(s, i, max(o, t.stop), "stop-loss")
                elif t.take_profit and l <= t.take_profit:
                    close_pos(s, i, min(o, t.take_profit), "kâr-al")

        for s in syms:
            if not np.isnan(s.c[i]):
                s.last_close = s.c[i]

        # 4) devre kesici (toplam bakiye)
        eq_close = cash + unreal(i)
        if risk.drawdown_hit(baseline, eq_close):
            for s in syms:
                if s.name in pos:
                    close_pos(s, i, s.last_close, "devre kesici")
            breaker.append({"time": index[i], "baseline": round(baseline, 2), "equity": round(cash, 2)})
            baseline = cash
            eq_close = cash

        # 5) mum kapanışında yeni sinyaller
        for s in syms:
            if i < s.ready_from or np.isnan(s.c[i]):
                continue
            t = pos.get(s.name)
            exiting = bool(t and ((t.side == LONG and s.lx[i]) or (t.side != LONG and s.sx[i])))
            if exiting:
                pend_exit.add(s.name)
            if t and not exiting:
                continue
            if s.le[i]:
                pend_entry[s.name] = LONG
            elif s.se[i] and allow_short:
                pend_entry[s.name] = -1

        equity[i] = eq_close

    for s in syms:
        if s.name in pos:
            close_pos(s, len(index) - 1, s.last_close, "test sonu")
    equity[-1] = cash

    bh = {}
    for s in syms:
        w = min(s.ready_from, len(index) - 1)
        valid = s.c[~np.isnan(s.c)]
        if len(valid) and not np.isnan(s.o[w]):
            bh[s.name] = valid[-1] / s.o[w] - 1
    return BacktestResult(pd.Series(equity, index=index), trades, breaker, initial, bh, bars_per_year)
