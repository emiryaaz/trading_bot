"""Risk yönetimi (vadeli işlemler, isolated marjin).

Terimler:
  teminat (margin)  : işleme kilitlenen para. Isolated modda en fazla bu kadar kaybedilebilir.
  pozisyon (notional): teminat x kaldıraç. Kâr/zarar bunun üzerinden oluşur.
  ROE               : kâr/zararın teminata oranı. 2x'te fiyat %2.5 hareket ederse ROE %5 olur.
"""
from __future__ import annotations

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
        margin = min(equity * self.c["margin_per_trade_pct"], available * 0.98)
        if margin <= 0:
            return None
        qty = margin * self.lev / price
        stop, tp = self.protection_prices(price, side, atr)
        liq = liquidation_price(price, side, self.lev)
        # stop, likidasyon fiyatından önce gelmeli
        if (side == LONG and stop <= liq) or (side == SHORT and stop >= liq) or stop <= 0:
            return None
        return TradePlan(side, qty, margin, stop, tp, liq)

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
