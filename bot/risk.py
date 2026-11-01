"""Risk yönetimi (vadeli işlemler, isolated marjin).

Terimler:
  teminat (margin)  : işleme kilitlenen para. Isolated modda en fazla bu kadar kaybedilebilir.
  pozisyon (notional): teminat x kaldıraç. Kâr/zarar bunun üzerinden oluşur.
  ROE               : kâr/zararın teminata oranı. 2x'te fiyat %2.5 hareket ederse ROE %5 olur.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import ROOT

LONG, SHORT = 1, -1


@dataclass
class TradePlan:
    side: int           # 1 = long, -1 = short
    qty: float          # kontrat miktarı (ör. BTC)
    margin: float       # kilitlenecek teminat (USDT)
    stop: float         # stop-loss fiyatı
    take_profit: float  # kâr al fiyatı (0 = yok)
    liquidation: float  # yaklaşık likidasyon fiyatı (bilgi amaçlı)


def liquidation_price(entry: float, side: int, leverage: float, mmr: float = 0.004) -> float:
    """Isolated pozisyon için yaklaşık likidasyon fiyatı (bakım teminatı oranı mmr ile)."""
    if side == LONG:
        return entry * (1 - 1 / leverage) / (1 - mmr)
    return entry * (1 + 1 / leverage) / (1 + mmr)


class RiskManager:
    def __init__(self, risk_cfg: dict, leverage: float):
        self.c = risk_cfg
        self.lev = float(leverage)

    def protection_prices(self, entry: float, side: int, atr: float = 0.0) -> tuple[float, float]:
        """Giriş fiyatına göre stop-loss ve kâr al fiyatlarını döndürür."""
        if self.c["stop_mode"] == "roe":
            sl_move = self.c["stop_loss_roe"] / self.lev           # %5 ROE / 2x = fiyatın %2.5'i
            tp_roe = self.c.get("take_profit_roe", 0) or 0
            stop = entry * (1 - side * sl_move)
            tp = entry * (1 + side * tp_roe / self.lev) if tp_roe > 0 else 0.0
        else:
            stop = entry - side * self.c["stop_atr_mult"] * atr
            tp_mult = self.c.get("take_profit_atr_mult", 0) or 0
            tp = entry + side * tp_mult * atr if tp_mult > 0 else 0.0
        return stop, tp

    def plan_entry(self, equity: float, available: float, price: float, side: int,
                   atr: float = 0.0) -> TradePlan | None:
        if price <= 0 or equity <= 0:
            return None
        if self.c["stop_mode"] == "atr" and atr <= 0:
            return None
        stop, tp = self.protection_prices(price, side, atr)
        margin_cap = min(equity * self.c["margin_per_trade_pct"], available * 0.98)
        risk_pct = self.c.get("risk_per_trade_pct", 0) or 0
        if self.c["stop_mode"] == "atr" and risk_pct > 0:
            # Oynaklığa göre boyut: stop'a takılınca kayıp = bakiye x risk_pct olacak şekilde
            stop_dist = abs(price - stop)
            if stop_dist <= 0:
                return None
            margin = min(equity * risk_pct / stop_dist * price / self.lev, margin_cap)
        else:
            margin = margin_cap
        if margin <= 0:
            return None
        qty = margin * self.lev / price
        liq = liquidation_price(price, side, self.lev)
        # stop, likidasyon fiyatından önce gelmeli
        if (side == LONG and stop <= liq) or (side == SHORT and stop >= liq) or stop <= 0:
            return None
        return TradePlan(side, qty, margin, stop, tp, liq)

    # ------------------------------------------------------------ kademeli izleyen stop
    @property
    def trailing(self) -> dict | None:
        t = self.c.get("trailing") or {}
        return t if t.get("enabled") else None

    def trail_update(self, side: int, entry: float, init_stop: float, init_tp: float,
                     steps_done: int, peak: float) -> tuple[int, float | None, float | None]:
        """Kademeli izleyen stop.

        R = ilk stop mesafesi. Fiyat girişten her `step_r` x R lehe gittiğinde (kademe k):
            stop = giriş + (k x step_r - gap_r) x R          (asla geri gevşemez)
            kâr-al = ilk kâr-al + k x step_r x R             (move_tp açıksa; kâr-al yoksa yok)
        Örnek (long, giriş 80k, stop 78k -> R = 2k, step_r = gap_r = 1):
            82k'ye çıkınca (k=1) stop 80k (başa baş), 84k'de (k=2) stop 82k, ...
        Dönüş: (yeni kademe, yeni stop, yeni kâr-al) — kademe değişmediyse (steps_done, None, None).
        """
        t = self.trailing
        if not t:
            return steps_done, None, None
        R = abs(entry - init_stop)
        move = side * (peak - entry)
        if R <= 0 or move <= 0:
            return steps_done, None, None
        step, gap = float(t.get("step_r", 1.0)), float(t.get("gap_r", 1.0))
        k = int(math.floor(move / (step * R) + 1e-12))
        if k <= steps_done:
            return steps_done, None, None
        stop = entry + side * (k * step - gap) * R
        stop = max(stop, init_stop) if side == LONG else min(stop, init_stop)
        tp = init_tp + side * k * step * R if (init_tp and t.get("move_tp", True)) else init_tp
        return k, stop, tp

    # ------------------------------------------------------------ kısmi kâr-al + koşan yarı
    @property
    def partial(self) -> dict | None:
        p = self.c.get("partial") or {}
        return p if p.get("enabled") else None

    def partial_target(self, side: int, entry: float, init_stop: float) -> float:
        """Kısmi kâr-al fiyatı: girişten at_r x R lehe."""
        R = abs(entry - init_stop)
        return entry + side * float(self.partial.get("at_r", 1.0)) * R

    def runner_stop(self, side: int, entry: float, init_stop: float, current_stop: float,
                    peak: float) -> float | None:
        """Kısmi kâr-al'dan sonra kalan yarının stop'u.

        Taban: giriş + lock_r x R (girişin ÜSTÜNDE -> kalan yarı da zararla kapanamaz).
        Fiyat her step_r x R ilerledikçe stop, ulaşılan seviyenin gap_r x R gerisine çekilir.
        Örnek (long, giriş 80k, R=2k, lock 0.5, step 0.5, gap 1): kısmi 82k'de -> stop 81k;
        84k'de stop 82k; 86k'de 84k ...  Stop asla geri gelmez. Değişmediyse None döner.
        """
        pc = self.partial
        R = abs(entry - init_stop)
        if R <= 0:
            return None
        lock = entry + side * float(pc.get("lock_r", 0.5)) * R
        step, gap = float(pc.get("step_r", 0.5)), float(pc.get("gap_r", 1.0))
        move = side * (peak - entry)
        k = int(math.floor(move / (step * R) + 1e-12)) if move > 0 else 0
        trail = entry + side * (k * step - gap) * R
        new = max(lock, trail) if side == LONG else min(lock, trail)
        better = new > current_stop + 1e-12 if side == LONG else new < current_stop - 1e-12
        return new if better else None

    def drawdown_hit(self, baseline: float, equity: float) -> bool:
        """DEVRE KESİCİ: son başlatmadaki bakiyeye göre kayıp max_drawdown_pct'e ulaştı mı?"""
        if not baseline or baseline <= 0:
            return False
        return (baseline - equity) / baseline >= self.c["max_drawdown_pct"] - 1e-12

    def daily_limit_hit(self, day_start_equity: float, equity: float) -> bool:
        lim = self.c.get("max_daily_loss_pct", 0) or 0
        if lim <= 0 or not day_start_equity or day_start_equity <= 0:
            return False
        return (day_start_equity - equity) / day_start_equity >= lim

    def kill_switch(self) -> bool:
        return (ROOT / self.c.get("kill_switch_file", "STOP")).exists()
