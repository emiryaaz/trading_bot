"""Strateji katmanı.

Her strateji mum verisini alır ve her mum için dört karar sütunu üretir:
    long_entry   -> long (yükseliş) pozisyonu aç
    short_entry  -> short (düşüş) pozisyonu aç
    long_exit    -> açık long'u kapat
    short_exit   -> açık short'u kapat
Ayrıca 'atr' sütunu (oynaklık) risk katmanı tarafından kullanılabilir.

Strateji yalnızca SİNYAL üretir; ne kadar açılacağına ve stop'a risk katmanı karar verir.
Yeni strateji eklemek için Strategy sınıfından türetip STRATEGIES sözlüğüne ekleyin.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from . import indicators as ind

SIGNAL_COLUMNS = ("long_entry", "short_entry", "long_exit", "short_exit")


class Strategy(ABC):
    def __init__(self, **params):
        self.params = params

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Göstergelerin oturması için gereken minimum mum sayısı."""

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """df'ye SIGNAL_COLUMNS ve 'atr' sütunlarını ekleyip döndürür."""


class SmaCrossRsi(Strategy):
    """Hızlı SMA yavaş SMA'yı yukarı keserse LONG (RSI aşırı alımda değilse),
    aşağı keserse SHORT (RSI aşırı satımda değilse). Ters kesişimde açık pozisyon kapanır.
    Basit bir başlangıç stratejisi — kârlı olduğu varsayılmamalı."""

    @property
    def warmup(self) -> int:
        p = self.params
        return max(p["slow"], p["rsi_period"], p["atr_period"]) + 2

    def generate(self, df):
        p = self.params
        out = df.copy()
        out["fast"] = ind.sma(out["close"], p["fast"])
        out["slow"] = ind.sma(out["close"], p["slow"])
        out["rsi"] = ind.rsi(out["close"], p["rsi_period"])
        out["atr"] = ind.atr(out, p["atr_period"])

        ready = out["slow"].notna() & out["atr"].notna()
        above = out["fast"] > out["slow"]
        cross_up = above & ~above.shift(1, fill_value=False) & ready
        cross_down = ~above & above.shift(1, fill_value=False) & ready

        out["long_entry"] = cross_up & (out["rsi"] < p.get("rsi_max_entry", 100))
        out["short_entry"] = cross_down & (out["rsi"] > p.get("rsi_min_entry", 0))
        out["long_exit"] = cross_down
        out["short_exit"] = cross_up
        return out


STRATEGIES = {
    "sma_cross_rsi": SmaCrossRsi,
}


def build_strategy(cfg: dict) -> Strategy:
    s = cfg["strategy"]
    if s["name"] not in STRATEGIES:
        raise ValueError(f"Bilinmeyen strateji: {s['name']}. Seçenekler: {list(STRATEGIES)}")
    return STRATEGIES[s["name"]](**s["params"])
