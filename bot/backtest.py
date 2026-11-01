"""Vadeli işlem (USDⓈ-M, isolated) backtest motoru.

Gerçekçilik kuralları:
  * Sinyal mum KAPANIŞINDA oluşur, işlem bir SONRAKİ mumun açılışında yapılır (geleceği görme yok).
  * Her açılış/kapanışta pozisyon büyüklüğü üzerinden işlem ücreti ve aleyhte kayma düşülür.
  * 8 saatte bir (00:00, 08:00, 16:00 UTC) funding ödenir/alınır. Gerçek funding verisi varsa o kullanılır.
  * Stop-loss / kâr al mum içi en düşük/en yüksek fiyata göre kontrol edilir; aynı mumda ikisi de
    tetiklenirse kötümser varsayımla önce stop sayılır. Fiyat boşlukla stop'un ötesinde açılırsa
    açılış fiyatından kapanır (gerçekte de böyle olur).
  * Isolated marjin: bir işlemde kayıp asla o işlemin teminatını aşamaz (likidasyon).
  * DEVRE KESİCİ: özsermaye son başlatmaya göre max_drawdown_pct kadar düşerse pozisyon kapatılır,
    olay kaydedilir. Simülasyon, "kullanıcı botu yeniden başlattı" varsayımıyla devam eder ki
    tüm dönemde kaç kez tetikleneceği görülebilsin.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .risk import LONG, RiskManager
from .strategy import Strategy


@dataclass
class Trade:
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
        return {"yön": "LONG" if self.side == LONG else "SHORT", "giriş_zamanı": self.entry_time,
                "giriş": round(self.entry_price, 2), "çıkış_zamanı": self.exit_time,
                "çıkış": round(self.exit_price, 2), "miktar": self.qty, "teminat": round(self.margin, 2),
                "ücret": round(self.entry_fee + self.exit_fee, 4), "funding": round(self.funding, 4),
                "kz": round(self.pnl, 4), "roe_%": round(self.roe * 100, 2), "neden": self.reason}


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade] = field(default_factory=list)
    breaker_events: list[dict] = field(default_factory=list)
    initial: float = 0.0
    buy_hold_return: float = 0.0
    bars_per_year: float = 8760

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
            "Al-ve-tut getirisi %": round(self.buy_hold_return * 100, 2),
            "Maks. düşüş (drawdown) %": round(dd * 100, 2),
            "İşlem sayısı (long/short)": f"{len(tr)} ({len(longs)}/{len(shorts)})",
            "Kazanma oranı %": round(len(wins) / len(tr) * 100, 1) if tr else 0.0,
            "Long K/Z": round(sum(t.pnl for t in longs), 2),
            "Short K/Z": round(sum(t.pnl for t in shorts), 2),
            "Kâr faktörü": round(gross_win / gross_loss, 2) if gross_loss > 0 else (float("inf") if gross_win else 0.0),
            "Ödenen ücret": round(sum(t.entry_fee + t.exit_fee for t in tr), 2),
            "Net funding (ödenen)": round(sum(t.funding for t in tr), 2),
            "Kapanış nedenleri": reasons,
            "Devre kesici (%10) tetiklenme": f"{len(self.breaker_events)} kez, ilk: {first}",
            "Sharpe (yıllık)": round(float(sharpe), 2),
        }


def _funding_series(index: pd.DatetimeIndex, funding: pd.Series | None, fallback: float) -> np.ndarray:
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


def run_backtest(df: pd.DataFrame, strategy: Strategy, risk: RiskManager,
                 initial: float, fee_rate: float, slippage: float,
                 allow_short: bool = True, funding: pd.Series | None = None,
                 funding_fallback: float = 0.0001, bars_per_year: float = 8760) -> BacktestResult:
    data = strategy.generate(df)
    o, h, l, c = (data[k].to_numpy() for k in ("open", "high", "low", "close"))
    le, se = data["long_entry"].to_numpy(), data["short_entry"].to_numpy()
    lx, sx = data["long_exit"].to_numpy(), data["short_exit"].to_numpy()
    atr = data["atr"].to_numpy()
    idx = data.index
    fund = _funding_series(idx, funding, funding_fallback)

    cash = float(initial)          # cüzdan bakiyesi (açık pozisyonun teminatı dahil)
    pos: Trade | None = None
    trades: list[Trade] = []
    breaker: list[dict] = []
    equity = np.empty(len(data))
    baseline = initial             # devre kesici referansı
    day, day_start_eq = None, initial
    pend_exit, pend_entry = False, 0

    def unreal(px: float) -> float:
        return pos.side * (px - pos.entry_price) * pos.qty if pos else 0.0

    def close_pos(i: int, raw: float, reason: str) -> None:
        nonlocal cash, pos
        px = raw * (1 - pos.side * slippage)       # aleyhte kayma
        pos.exit_fee = px * pos.qty * fee_rate
        pos.exit_time, pos.exit_price, pos.reason = idx[i], px, reason
        # giriş ücreti ve funding zaten anında düşülmüştü; toplam değişim = pos.pnl olsun
        cash += pos.pnl + pos.entry_fee + pos.funding
        trades.append(pos)
        pos = None

    def open_pos(i: int, side: int) -> None:
        nonlocal cash, pos
        eq_now = cash
        plan = risk.plan_entry(eq_now, cash, o[i], side, atr[i - 1])
        if not plan:
            return
        px = o[i] * (1 + side * slippage)
        stop, tp = risk.protection_prices(px, side, atr[i - 1])
        fee = px * plan.qty * fee_rate
        cash -= fee
        pos = Trade(side, idx[i], px, plan.qty, plan.margin, stop, tp, fee)

    for i in range(len(data)):
        d = idx[i].date()
        if d != day:
            day, day_start_eq = d, cash + unreal(o[i])

        # 1) funding (mum açılışında)
        if pos and fund[i]:
            pay = pos.side * pos.qty * o[i] * fund[i]   # pozitif oranda long öder, short alır
            pos.funding += pay
            cash -= pay

        # 2) önceki mumun kararlarını açılışta uygula
        if pend_exit and pos:
            close_pos(i, o[i], "sinyal")
        if pend_entry and pos is None and not np.isnan(atr[i - 1]):
            if not risk.daily_limit_hit(day_start_eq, cash):
                open_pos(i, pend_entry)
        pend_exit, pend_entry = False, 0

        # 3) mum içi stop-loss / kâr al
        if pos:
            if pos.side == LONG:
                if l[i] <= pos.stop:
                    close_pos(i, min(o[i], pos.stop), "stop-loss")
                elif pos.take_profit and h[i] >= pos.take_profit:
                    close_pos(i, max(o[i], pos.take_profit), "kâr-al")
            else:
                if h[i] >= pos.stop:
                    close_pos(i, max(o[i], pos.stop), "stop-loss")
                elif pos.take_profit and l[i] <= pos.take_profit:
                    close_pos(i, min(o[i], pos.take_profit), "kâr-al")

        # 4) devre kesici (mum kapanışındaki özsermayeye göre)
        eq_close = cash + unreal(c[i])
        if risk.drawdown_hit(baseline, eq_close):
            if pos:
                close_pos(i, c[i], "devre kesici")
            breaker.append({"time": idx[i], "baseline": round(baseline, 2), "equity": round(cash, 2)})
            baseline = cash                        # kullanıcı inceleyip yeniden başlattı varsayımı
            eq_close = cash

        # 5) mum kapanışında yeni sinyaller
        if i >= strategy.warmup:
            if pos and ((pos.side == LONG and lx[i]) or (pos.side != LONG and sx[i])):
                pend_exit = True
            if le[i]:
                pend_entry = LONG
            elif se[i] and allow_short:
                pend_entry = -1
            if pos and not pend_exit:
                pend_entry = 0                     # zaten pozisyondaysak yeni giriş yok

        equity[i] = eq_close

    if pos:
        close_pos(len(data) - 1, c[-1], "test sonu")
        equity[-1] = cash

    w = strategy.warmup
    bh = c[-1] / o[w] - 1
    return BacktestResult(pd.Series(equity, index=idx), trades, breaker, initial, bh, bars_per_year)
