"""Strateji katmanı.

Her strateji mum verisini alır ve her mum için dört karar sütunu üretir:
    long_entry   -> long (yükseliş) pozisyonu aç
    short_entry  -> short (düşüş) pozisyonu aç
    long_exit    -> açık long'u kapat
    short_exit   -> açık short'u kapat
Ayrıca 'atr' sütunu (oynaklık) risk katmanı tarafından stop / kâr-al için kullanılır.

`generate(df, btc=None)`: `btc`, altcoinler için BTC'nin aynı zaman dilimindeki mum verisidir
(BTC filtresi için). BTC'nin kendisi için None verilir.

Strateji yalnızca SİNYAL üretir; ne kadar açılacağına ve stop'a risk katmanı karar verir.
Yeni strateji eklemek için Strategy sınıfından türetip STRATEGIES sözlüğüne ekleyin.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from . import indicators as ind

SIGNAL_COLUMNS = ("long_entry", "short_entry", "long_exit", "short_exit")


class Strategy(ABC):
    uses_btc = False

    def __init__(self, **params):
        self.params = params

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Göstergelerin oturması için gereken minimum mum sayısı."""

    @abstractmethod
    def generate(self, df: pd.DataFrame, btc: pd.DataFrame | None = None) -> pd.DataFrame:
        """df'ye SIGNAL_COLUMNS ve 'atr' sütunlarını ekleyip döndürür."""


# =====================================================================================
class SmaCrossRsi(Strategy):
    """(v0.1-v0.4) Hızlı SMA yavaş SMA'yı yukarı keserse LONG, aşağı keserse SHORT; RSI aşırı bölge filtresi.
    Karşılaştırma için tutuluyor."""

    @property
    def warmup(self) -> int:
        p = self.params
        return max(p["slow"], p["rsi_period"], p["atr_period"]) + 2

    def generate(self, df, btc=None):
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


# =====================================================================================
MULTI_DEFAULTS = {
    # --- 1) piyasa rejimi: trend yeterince güçlü mü? (ADX, üst zaman diliminde)
    "use_regime": True, "adx_period": 14, "adx_min": 20,
    # --- 2) yön: üst zaman dilimi trendi + BTC'nin yönü
    "use_htf": True, "htf": "4h", "htf_fast": 21, "htf_slow": 55,
    "dir_fast": 50, "dir_slow": 200,            # use_htf kapalıyken 1s'te yön için
    "use_btc_filter": True,
    # --- 3) giriş zamanlaması
    "trigger": "pullback",                       # pullback = trend yönünde geri çekilme | cross = SMA kesişimi
    "fast": 20, "slow": 50,                      # trigger: cross için
    "rsi_period": 14, "pull_window": 8,
    "rsi_pull_long": 40, "rsi_trigger_long": 45,     # long: RSI 40'ın altına inmiş, şimdi 45'i yukarı kesti
    "rsi_pull_short": 60, "rsi_trigger_short": 55,   # short: RSI 60'ın üstüne çıkmış, şimdi 55'i aşağı kesti
    "use_macd": True,                            # MACD histogramı giriş yönüne dönmüş olmalı
    "use_fib": True, "fib_lookback": 72, "fib_min": 0.382, "fib_max": 0.786,
    # --- 4) onay
    "use_volume": True, "volume_period": 20, "volume_min": 1.0,
    # --- 5) vadeli piyasa verileri (fiyattan bağımsız bilgi)
    "use_funding": False,                        # funding aşırıysa kalabalık tarafa katılma
    "funding_window_days": 30, "funding_z_max": 2.0, "funding_abs_max": 0.0005,
    "use_taker_flow": False,                     # agresif alıcı / satıcı baskısı giriş yönünde olmalı
    "flow_period": 6, "flow_min": 0.5,
    # --- diğer
    "atr_period": 14,
}


class MultiFactor(Strategy):
    """Çok katmanlı strateji. Bir işlem açılması için açık olan TÜM katmanların onay vermesi gerekir.

      1. Piyasa rejimi  : 4s ADX >= adx_min  (yatay piyasada işlem yok)
      2. Yön            : 4s'te fiyat > EMA21 > EMA55 ise sadece LONG, tersi ise sadece SHORT.
                          Altcoinlerde BTC'nin 4s yönü ters olmamalı.
      3. Giriş          : trend yönünde geri çekilme bitti mi? (RSI geri çekilip toparlandı)
                          + MACD histogramı giriş yönüne dönmüş + geri çekilme Fibonacci 0.382-0.786 bölgesinde
      4. Onay           : hacim ortalamanın üstünde
      5. Vadeli veriler : funding aşırı pozitifken LONG, aşırı negatifken SHORT açma (kalabalığa katılma);
                          son mumlarda agresif alıcıların payı (taker buy / hacim) giriş yönünü desteklemeli
    Çıkış: stop-loss / kâr-al (borsada) veya 4s'te fiyat EMA55'in ters tarafına geçerse (trend bozuldu).

    Her katman `use_*` parametresiyle kapatılabilir; böylece her birinin katkısı ayrı ayrı test edilir.
    """

    def __init__(self, **params):
        super().__init__(**{**MULTI_DEFAULTS, **params})
        self.uses_btc = bool(self.params["use_btc_filter"])

    @property
    def warmup(self) -> int:
        p = self.params
        hours = pd.Timedelta(p["htf"]) / pd.Timedelta("1h")
        htf_need = int((max(p["htf_slow"], p["adx_period"] * 3) + 2) * hours) if (p["use_htf"] or p["use_regime"]) else 0
        return max(htf_need, p["dir_slow"] + 2, p["slow"] + 2, p["fib_lookback"] + 2, 60)

    # -------------------------------------------------------------- yön (trend) bilgisi
    def _trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Her mum için: dir (+1 / -1 / 0), above_slow (fiyat yavaş EMA'nın üstünde mi), adx."""
        p = self.params
        if p["use_htf"] or p["use_regime"]:
            def fn(h):
                c = h["close"]
                f, s = ind.ema(c, p["htf_fast"]), ind.ema(c, p["htf_slow"])
                d = np.where((c > f) & (f > s), 1, np.where((c < f) & (f < s), -1, 0))
                return pd.DataFrame({"dir_htf": d, "above_htf": (c > s).astype(float),
                                     "adx_htf": ind.adx(h, p["adx_period"]), "slow_htf": s}, index=h.index)
            t = ind.higher_timeframe(df, p["htf"], fn)
        else:
            t = pd.DataFrame(index=df.index)
        if p["use_htf"]:
            t["dir"] = t["dir_htf"].where(t["slow_htf"].notna(), 0)
            t["above"] = t["above_htf"]
        else:
            c = df["close"]
            f, s = ind.ema(c, p["dir_fast"]), ind.ema(c, p["dir_slow"])
            t["dir"] = np.where((c > f) & (f > s), 1, np.where((c < f) & (f < s), -1, 0))
            t["dir"] = t["dir"].where(s.notna(), 0)
            t["above"] = (c > s).astype(float).where(s.notna())
        t["adx"] = t["adx_htf"] if "adx_htf" in t else np.nan
        return t

    def generate(self, df, btc=None):
        p = self.params
        out = df.copy()
        out["atr"] = ind.atr(out, p["atr_period"])
        tr = self._trend(out)
        out["dir"], out["adx"] = tr["dir"].fillna(0), tr["adx"]
        above = tr["above"]

        long_ok = pd.Series(True, index=out.index)
        short_ok = pd.Series(True, index=out.index)

        # 1) rejim
        if p["use_regime"]:
            strong = out["adx"] >= p["adx_min"]
            long_ok &= strong
            short_ok &= strong

        # 2) yön
        if p["use_htf"] or p["trigger"] == "pullback":
            long_ok &= out["dir"] == 1
            short_ok &= out["dir"] == -1
        if p["use_btc_filter"] and btc is not None:
            btc_dir = self._trend(btc.reindex(out.index, method="ffill"))["dir"].fillna(0)
            out["btc_dir"] = btc_dir
            long_ok &= btc_dir != -1
            short_ok &= btc_dir != 1

        # 3) giriş tetikleyicisi
        out["rsi"] = ind.rsi(out["close"], p["rsi_period"])
        if p["trigger"] == "cross":
            f, s = ind.sma(out["close"], p["fast"]), ind.sma(out["close"], p["slow"])
            ab = f > s
            trig_long = ab & ~ab.shift(1, fill_value=False) & s.notna()
            trig_short = ~ab & ab.shift(1, fill_value=False) & s.notna()
            exit_long, exit_short = trig_short, trig_long
        else:
            r, w = out["rsi"], p["pull_window"]
            dipped = r.rolling(w, min_periods=1).min() <= p["rsi_pull_long"]
            spiked = r.rolling(w, min_periods=1).max() >= p["rsi_pull_short"]
            trig_long = dipped & (r.shift(1) < p["rsi_trigger_long"]) & (r >= p["rsi_trigger_long"])
            trig_short = spiked & (r.shift(1) > p["rsi_trigger_short"]) & (r <= p["rsi_trigger_short"])
            # çıkış: fiyat üst zaman diliminin yavaş EMA'sının ters tarafına geçti -> trend bozuldu
            exit_long = above == 0
            exit_short = above == 1

        if p["use_macd"]:
            h = ind.macd(out["close"])["hist"]
            long_ok &= h > h.shift(1)
            short_ok &= h < h.shift(1)

        if p["use_fib"]:
            fib = ind.fib_retracement(out, p["fib_lookback"], p["pull_window"])
            out["fib_long"], out["fib_short"] = fib["long_level"], fib["short_level"]
            long_ok &= fib["long_level"].between(p["fib_min"], p["fib_max"])
            short_ok &= fib["short_level"].between(p["fib_min"], p["fib_max"])

        # 5) vadeli işlem piyasası verileri
        if p["use_funding"]:
            if "funding_rate" in out and out["funding_rate"].notna().any():
                f = out["funding_rate"].ffill()
                step = out.index.to_series().diff().median()
                win = max(10, int(pd.Timedelta(days=p["funding_window_days"]) / step))
                mu = f.rolling(win, min_periods=win // 3).mean()
                sd = f.rolling(win, min_periods=win // 3).std().replace(0, np.nan)
                z = ((f - mu) / sd).fillna(0)
                out["funding_z"] = z
                crowded_long = (z > p["funding_z_max"]) | (f > p["funding_abs_max"])
                crowded_short = (z < -p["funding_z_max"]) | (f < -p["funding_abs_max"])
                long_ok &= ~crowded_long
                short_ok &= ~crowded_short
            else:
                warnings.warn("use_funding açık ama funding verisi yok; funding katmanı uygulanmadı.")
        if p["use_taker_flow"]:
            if "taker_buy_volume" in out:
                ratio = (out["taker_buy_volume"] / out["volume"].replace(0, np.nan))
                flow = ratio.rolling(p["flow_period"], min_periods=p["flow_period"]).mean()
                out["taker_flow"] = flow
                long_ok &= flow > p["flow_min"]
                short_ok &= flow < 1 - p["flow_min"]
            else:
                warnings.warn("use_taker_flow açık ama mum verisinde taker hacmi yok; katman uygulanmadı.")

        # 4) onay
        if p["use_volume"]:
            vr = ind.volume_ratio(out["volume"], p["volume_period"])
            out["vol_ratio"] = vr
            long_ok &= vr >= p["volume_min"]
            short_ok &= vr >= p["volume_min"]

        ready = out["atr"].notna()
        out["long_entry"] = (trig_long & long_ok & ready).fillna(False).astype(bool)
        out["short_entry"] = (trig_short & short_ok & ready).fillna(False).astype(bool)
        out["long_exit"] = exit_long.fillna(False).astype(bool)
        out["short_exit"] = exit_short.fillna(False).astype(bool)
        return out


STRATEGIES = {
    "sma_cross_rsi": SmaCrossRsi,
    "multi_factor": MultiFactor,
}


def build_strategy(cfg: dict) -> Strategy:
    s = cfg["strategy"]
    if s["name"] not in STRATEGIES:
        raise ValueError(f"Bilinmeyen strateji: {s['name']}. Seçenekler: {list(STRATEGIES)}")
    return STRATEGIES[s["name"]](**(s.get("params") or {}))
