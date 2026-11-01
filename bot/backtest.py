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
    init_stop: float = 0.0    # izleyen stop için ilk seviyeler
    init_tp: float = 0.0
    steps: int = 0            # ulaşılan kademe sayısı
    partial_done: bool = False  # kısmi kâr-al gerçekleşti mi (kalan yarı "koşuyor")
    part: str = ""            # "kısmi" = pozisyonun kısmi kâr-al ile kapanan parçası

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
    closes: pd.DataFrame | None = None                # coin kapanış fiyatları (dönem al-ve-tut için)

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

    def subset(self, start=None, end=None) -> "BacktestResult":
        """Aynı simülasyonun bir dönemi: yalnızca o dönemde AÇILAN işlemler ve o dönemin bakiye eğrisi.
        Göstergeler önceki veriyle ısınmış olduğu için eğitim/test ayrımında doğru yöntem budur."""
        eq = self.equity
        before = eq.loc[:start].iloc[:-1] if start is not None else eq.iloc[:0]
        initial = float(before.iloc[-1]) if len(before) else self.initial
        part_eq = eq.loc[start:end]
        tr = [t for t in self.trades if (start is None or t.entry_time >= start) and (end is None or t.entry_time < end)]
        ev = [e for e in self.breaker_events if (start is None or e["time"] >= start) and (end is None or e["time"] < end)]
        bh = self.buy_hold
        if self.closes is not None:
            c = self.closes.loc[start:end]
            bh = {k: float(c[k].dropna().iloc[-1] / c[k].dropna().iloc[0] - 1) for k in c if c[k].notna().sum() > 1}
        return BacktestResult(part_eq, tr, ev, initial, bh, self.bars_per_year, self.closes)

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


def attach_funding(df: pd.DataFrame, funding: pd.Series | None) -> pd.DataFrame:
    """Son ödenmiş funding oranını her muma 'funding_rate' kolonu olarak ekler (geleceği görmeden:
    T anındaki ödeme oranı, T'de açılan mumdan itibaren bilinir)."""
    if funding is None or not len(funding) or "funding_rate" in df:
        return df
    f = funding.copy()
    f.index = f.index.floor("h")
    f = f[~f.index.duplicated(keep="last")].sort_index()
    out = df.copy()
    out["funding_rate"] = f.reindex(out.index, method="ffill")
    return out


class _Sym:
    """Tek bir coinin ortak zaman çizelgesine hizalanmış dizileri."""

    def __init__(self, name, df, strategy, index, funding, fallback, btc=None):
        df = attach_funding(df, funding)
        sig = strategy.generate(df, btc).reindex(index)
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
                 funding_fallback: float = 0.0001, bars_per_year: float = 8760,
                 maker_fee: float | None = None, entry_order: str = "market",
                 limit_offset_atr: float = 0.0, limit_ttl_bars: int = 1,
                 tp_order: str = "market", same_bar_runner_check: bool = True) -> BacktestResult:
    """fee_rate = taker (piyasa emri) ücreti, maker_fee = limit emir ücreti.

    entry_order="limit": sinyal mumunun kapanışında (kapanış -/+ limit_offset_atr x ATR) fiyatına deftere
    limit emir konur. Sonraki `limit_ttl_bars` mum içinde fiyat bu seviyeyi GEÇERSE (sadece dokunması yetmez)
    emir maker olarak dolar; geçmezse iptal edilir. Fiyat seviyenin ötesinde açılırsa açılış fiyatından dolar.
    tp_order="limit": kâr-al deftere yazılı limit emirdir (maker, kaymasız). Stop-loss her zaman piyasa emridir.
    """
    taker_fee = fee_rate
    maker_fee = taker_fee if maker_fee is None else maker_fee
    if isinstance(data, pd.DataFrame):
        data = {"COIN": data}
    if funding is None or isinstance(funding, pd.Series):
        funding = {s: funding for s in data}
    index = sorted(set().union(*[d.index for d in data.values()]))
    index = pd.DatetimeIndex(index)
    btc_key = next((k for k in data if k.upper().startswith("BTC/")), None)
    btc_df = data[btc_key] if (btc_key and getattr(strategy, "uses_btc", False)) else None
    syms = [_Sym(s, d, strategy, index, funding.get(s), funding_fallback,
                 None if s == btc_key else btc_df) for s, d in data.items()]

    max_pos = risk.c.get("max_open_positions", 1)
    max_margin = risk.c.get("max_total_margin_pct", 1.0)
    use_limit = entry_order == "limit"

    cash = float(initial)
    pos: dict[str, Trade] = {}
    trades: list[Trade] = []
    breaker: list[dict] = []
    equity = np.empty(len(index))
    baseline = initial
    day, day_start_eq = None, initial
    pend_exit: set[str] = set()
    pend_entry: dict[str, dict] = {}     # coin -> {side, limit, expires, atr}
    stats = {"limit_placed": 0, "limit_filled": 0}

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

    def close_pos(s: _Sym, i: int, raw: float, reason: str, maker: bool = False) -> None:
        nonlocal cash
        t = pos.pop(s.name)
        px = raw if maker else raw * (1 - t.side * slippage)
        t.exit_fee = px * t.qty * (maker_fee if maker else taker_fee)
        t.exit_time, t.exit_price, t.reason = index[i], px, reason
        cash += t.pnl + t.entry_fee + t.funding     # giriş ücreti ve funding zaten düşülmüştü
        trades.append(t)

    def realize_part(s: _Sym, i: int, t: Trade, frac: float, px: float, maker: bool) -> None:
        """Pozisyonun `frac` kadarını px'ten kapatır (kısmi kâr-al); kalan pozisyon devam eder."""
        nonlocal cash
        q = t.qty * frac
        part = Trade(t.symbol, t.side, t.entry_time, t.entry_price, q, t.margin * frac, t.stop, px,
                     t.entry_fee * frac, funding=t.funding * frac, init_stop=t.init_stop, init_tp=t.init_tp,
                     part="kısmi")
        fill = px if maker else px * (1 - t.side * slippage)
        part.exit_fee = fill * q * (maker_fee if maker else taker_fee)
        part.exit_time, part.exit_price, part.reason = index[i], fill, "kısmi kâr-al"
        cash += part.pnl + part.entry_fee + part.funding
        trades.append(part)
        keep = 1 - frac
        t.qty, t.margin, t.entry_fee, t.funding = t.qty * keep, t.margin * keep, t.entry_fee * keep, t.funding * keep

    def open_pos(s: _Sym, i: int, side: int, px: float, atr: float, maker: bool) -> None:
        nonlocal cash
        used = sum(t.margin for t in pos.values())
        eq_now = cash + unreal(i, "o")
        room = max(0.0, eq_now * max_margin - used)
        available = min(cash - used, room)
        plan = risk.plan_entry(eq_now, available, px, side, atr)
        if not plan:
            return
        if not maker:
            px = px * (1 + side * slippage)
        stop, tp = risk.protection_prices(px, side, atr)
        if risk.partial:
            tp = risk.partial_target(side, px, stop)     # ilk kâr-al = kısmi kâr-al seviyesi
        fee = px * plan.qty * (maker_fee if maker else taker_fee)
        cash -= fee
        pos[s.name] = Trade(s.name, side, index[i], px, plan.qty, plan.margin, stop, tp, fee,
                            init_stop=stop, init_tp=tp)

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
        pend_exit.clear()

        # 2b) girişler (öncelik sırası = config sırası)
        if pend_entry:
            blocked = risk.daily_limit_hit(day_start_eq, cash + unreal(i, "o"))
            for s in syms:
                pe = pend_entry.get(s.name)
                if not pe or np.isnan(s.o[i]):
                    continue
                if s.name in pos or blocked or len(pos) >= max_pos:
                    if not use_limit or s.name in pos or i >= pe["expires"]:
                        pend_entry.pop(s.name, None)
                    continue
                side = pe["side"]
                if not use_limit:
                    open_pos(s, i, side, s.o[i], pe["atr"], maker=False)
                    pend_entry.pop(s.name, None)
                    continue
                lim = pe["limit"]
                crossed = s.l[i] < lim if side == LONG else s.h[i] > lim
                if crossed:
                    fill = min(s.o[i], lim) if side == LONG else max(s.o[i], lim)
                    open_pos(s, i, side, fill, pe["atr"], maker=True)
                    if s.name in pos:
                        stats["limit_filled"] += 1
                    pend_entry.pop(s.name, None)
                elif i >= pe["expires"]:
                    pend_entry.pop(s.name, None)    # dolmadı -> iptal

        # 3) mum içi stop-loss / kâr al, ardından izleyen stop güncellemesi
        #    Temkinli sıra: önce MEVCUT stop/kâr-al kontrol edilir, stop ancak sonra yükseltilir
        #    (mum içinde tepe mi dip mi önce geldi bilinemez).
        tp_maker = tp_order == "limit"
        for s in syms:
            t = pos.get(s.name)
            if not t or np.isnan(s.o[i]):
                continue
            o, h, l = s.o[i], s.h[i], s.l[i]
            locked = (t.stop - t.entry_price) * t.side          # >0: stop kâr bölgesinde
            sl_reason = "izleyen stop" if locked > 1e-12 else ("başa baş" if t.steps else "stop-loss")
            pc = risk.partial
            if pc and not t.partial_done:
                # --- kısmi kâr-al aşaması: önce stop, sonra hedef (temkinli sıra)
                hit_stop = l <= t.stop if t.side == LONG else h >= t.stop
                hit_tgt = h >= t.take_profit if t.side == LONG else l <= t.take_profit
                if hit_stop:
                    close_pos(s, i, min(o, t.stop) if t.side == LONG else max(o, t.stop), sl_reason)
                    continue
                if hit_tgt:
                    fill = max(o, t.take_profit) if t.side == LONG else min(o, t.take_profit)
                    realize_part(s, i, t, float(pc.get("fraction", 0.5)), fill, tp_maker)
                    t.partial_done, t.take_profit = True, 0.0
                    t.stop = t.entry_price + t.side * float(pc.get("lock_r", 0.5)) * abs(t.entry_price - t.init_stop)
                    # aynı mumda yeni stop'un altına inildiyse kalan yarı da kapanır (temkinli varsayım)
                    if same_bar_runner_check and ((l <= t.stop) if t.side == LONG else (h >= t.stop)):
                        close_pos(s, i, t.stop, "izleyen stop")
                continue
            if t.partial_done:
                # --- koşan yarı: sadece (yükselen) stop
                if (l <= t.stop) if t.side == LONG else (h >= t.stop):
                    close_pos(s, i, min(o, t.stop) if t.side == LONG else max(o, t.stop), "izleyen stop")
                else:
                    ns = risk.runner_stop(t.side, t.entry_price, t.init_stop, t.stop, h if t.side == LONG else l)
                    if ns is not None:
                        t.stop = ns
                continue
            if t.side == LONG:
                if l <= t.stop:
                    close_pos(s, i, min(o, t.stop), sl_reason)
                elif t.take_profit and h >= t.take_profit:
                    close_pos(s, i, max(o, t.take_profit), "kâr-al", maker=tp_maker)
            else:
                if h >= t.stop:
                    close_pos(s, i, max(o, t.stop), sl_reason)
                elif t.take_profit and l <= t.take_profit:
                    close_pos(s, i, min(o, t.take_profit), "kâr-al", maker=tp_maker)
            if s.name in pos and risk.trailing:
                peak = h if t.side == LONG else l
                k, ns, ntp = risk.trail_update(t.side, t.entry_price, t.init_stop, t.init_tp, t.steps, peak)
                if ns is not None:
                    t.steps, t.stop, t.take_profit = k, ns, ntp

        for s in syms:
            if not np.isnan(s.c[i]):
                s.last_close = s.c[i]

        # 4) devre kesici (toplam bakiye)
        eq_close = cash + unreal(i)
        if risk.drawdown_hit(baseline, eq_close):
            for s in syms:
                if s.name in pos:
                    close_pos(s, i, s.last_close, "devre kesici")
            pend_entry.clear()
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
            side = LONG if s.le[i] else (-1 if (s.se[i] and allow_short) else 0)
            if side:
                limit = s.c[i] - side * limit_offset_atr * s.atr[i]
                pend_entry[s.name] = {"side": side, "limit": limit, "atr": s.atr[i],
                                      "expires": i + max(1, int(limit_ttl_bars))}
                stats["limit_placed"] += 1 if use_limit else 0

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
    closes = pd.DataFrame({s.name: s.c for s in syms}, index=index)
    res = BacktestResult(pd.Series(equity, index=index), trades, breaker, initial, bh, bars_per_year, closes)
    res.limit_stats = stats
    return res
