"""Tek tek indikatör kuralları ve ikili kombinasyonlar.

Her indikatörün iki rolü vardır:
  * GİRİŞ (entry): indikatörün klasik "şimdi al / şimdi sat" anı. Tekli testte işlemi bu açar.
  * ONAY (confirm): ikili testte başka bir indikatörün sinyalini süzen filtre. Sinyal anında
    onay indikatörü aynı yönü destekliyorsa işlem açılır.

Çıkış: borsadaki stop-loss / kâr-al (risk ayarları) veya giriş indikatörünün TERS sinyali.
Tüm hesaplar yalnızca geçmiş veriyi kullanır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .strategy import Strategy


def _cross_up(a: pd.Series, b) -> pd.Series:
    b_prev = b.shift(1) if isinstance(b, pd.Series) else b
    return (a > b) & (a.shift(1) <= b_prev)


def _cross_down(a: pd.Series, b) -> pd.Series:
    b_prev = b.shift(1) if isinstance(b, pd.Series) else b
    return (a < b) & (a.shift(1) >= b_prev)


def _state(cond_up: pd.Series, cond_down: pd.Series) -> pd.Series:
    return pd.Series(np.where(cond_up, 1, np.where(cond_down, -1, 0)), index=cond_up.index)


# ------------------------------------------------------------------ indikatör tanımları
# Her fonksiyon: df -> DataFrame(long, short, state)  [state: +1 yükseliş, -1 düşüş, 0 nötr]

def _ema_cross(df):
    f, s = ind.ema(df["close"], 9), ind.ema(df["close"], 21)
    return pd.DataFrame({"long": _cross_up(f, s), "short": _cross_down(f, s), "state": _state(f > s, f < s)})


def _ema200(df):
    c, e = df["close"], ind.ema(df["close"], 200)
    return pd.DataFrame({"long": _cross_up(c, e), "short": _cross_down(c, e), "state": _state(c > e, c < e)})


def _macd(df):
    m = ind.macd(df["close"])
    return pd.DataFrame({"long": _cross_up(m["macd"], m["signal"]), "short": _cross_down(m["macd"], m["signal"]),
                         "state": _state(m["macd"] > m["signal"], m["macd"] < m["signal"])})


def _rsi(df):
    r = ind.rsi(df["close"], 14)
    return pd.DataFrame({"long": _cross_up(r, 30), "short": _cross_down(r, 70), "state": _state(r > 50, r < 50)})


def _stoch(df):
    s = ind.stochastic(df)
    k, d = s["k"], s["d"]
    return pd.DataFrame({"long": _cross_up(k, d) & (k.shift(1) < 20), "short": _cross_down(k, d) & (k.shift(1) > 80),
                         "state": _state(k > d, k < d)})


def _bollinger(df):
    b, c = ind.bollinger(df["close"]), df["close"]
    return pd.DataFrame({"long": _cross_up(c, b["lower"]), "short": _cross_down(c, b["upper"]),
                         "state": _state(c > b["mid"], c < b["mid"]),
                         # onay rolü (ör. RSI + Bollinger): son 3 mumda banda değmiş olmak
                         "touch_low": (df["low"] <= b["lower"]).rolling(3, min_periods=1).max().astype(bool),
                         "touch_high": (df["high"] >= b["upper"]).rolling(3, min_periods=1).max().astype(bool)})


def _bollinger_break(df):
    b, c = ind.bollinger(df["close"]), df["close"]
    return pd.DataFrame({"long": _cross_up(c, b["upper"]), "short": _cross_down(c, b["lower"]),
                         "state": _state(c > b["mid"], c < b["mid"])})


def _dmi(df):
    d = ind.dmi(df)
    strong = d["adx"] > 20
    return pd.DataFrame({"long": _cross_up(d["plus"], d["minus"]) & strong,
                         "short": _cross_down(d["plus"], d["minus"]) & strong,
                         "state": _state(d["plus"] > d["minus"], d["plus"] < d["minus"]), "adx": d["adx"]})


def _supertrend(df):
    st = ind.supertrend(df)
    return pd.DataFrame({"long": (st == 1) & (st.shift(1) == -1), "short": (st == -1) & (st.shift(1) == 1),
                         "state": st.astype(int)})


def _psar(df):
    p = ind.parabolic_sar(df)
    return pd.DataFrame({"long": (p == 1) & (p.shift(1) == -1), "short": (p == -1) & (p.shift(1) == 1),
                         "state": p.astype(int)})


def _ichimoku(df):
    ic, c = ind.ichimoku(df), df["close"]
    above, below = c > ic["cloud_top"], c < ic["cloud_bot"]
    return pd.DataFrame({"long": _cross_up(ic["tenkan"], ic["kijun"]) & above,
                         "short": _cross_down(ic["tenkan"], ic["kijun"]) & below,
                         "state": _state(above, below)})


def _donchian(df):
    d, c = ind.donchian(df, 20), df["close"]
    mid = (d["upper"] + d["lower"]) / 2
    return pd.DataFrame({"long": (c > d["upper"]) & (c.shift(1) <= d["upper"].shift(1)),
                         "short": (c < d["lower"]) & (c.shift(1) >= d["lower"].shift(1)),
                         "state": _state(c > mid, c < mid)})


def _cci(df):
    x = ind.cci(df)
    return pd.DataFrame({"long": _cross_up(x, -100), "short": _cross_down(x, 100), "state": _state(x > 0, x < 0)})


def _mfi(df):
    x = ind.mfi(df)
    return pd.DataFrame({"long": _cross_up(x, 20), "short": _cross_down(x, 80), "state": _state(x > 50, x < 50)})


def _obv(df):
    o = ind.obv(df)
    e = ind.ema(o, 20)
    return pd.DataFrame({"long": _cross_up(o, e), "short": _cross_down(o, e), "state": _state(o > e, o < e)})


def _vwap(df):
    v, c = ind.rolling_vwap(df, 24), df["close"]
    return pd.DataFrame({"long": _cross_up(c, v), "short": _cross_down(c, v), "state": _state(c > v, c < v)})


def _fibonacci(df, lookback: int = 72):
    """Son 72 mumun salınımında fiyat 0.5 - 0.786 'altın bölgeye' geri çekilip 0.5 seviyesinin üstüne
    (yükselişte) / altına (düşüşte) geri dönerse işlem."""
    h, l, c = df["high"], df["low"], df["close"]
    hi = h.rolling(lookback, min_periods=lookback).max()
    lo = l.rolling(lookback, min_periods=lookback).min()
    hi_pos = h.rolling(lookback, min_periods=lookback).apply(np.argmax, raw=True)
    lo_pos = l.rolling(lookback, min_periods=lookback).apply(np.argmin, raw=True)
    upswing, downswing = hi_pos > lo_pos, hi_pos < lo_pos        # tepe dipten sonra mı geldi?
    rng = hi - lo
    up50, up786 = hi - 0.5 * rng, hi - 0.786 * rng
    dn50, dn786 = lo + 0.5 * rng, lo + 0.786 * rng
    rec_lo = l.rolling(3, min_periods=1).min()
    rec_hi = h.rolling(3, min_periods=1).max()
    long = upswing & (rec_lo <= up50) & (rec_lo >= up786) & _cross_up(c, up50)
    short = downswing & (rec_hi >= dn50) & (rec_hi <= dn786) & _cross_down(c, dn50)
    up618, dn618 = hi - 0.618 * rng, lo + 0.618 * rng
    return pd.DataFrame({"long": long, "short": short,
                         "state": _state(upswing & (c > up618), downswing & (c < dn618))})


def _volume(df):
    vr = ind.volume_ratio(df["volume"], 20)
    green, red = df["close"] > df["open"], df["close"] < df["open"]
    spike = vr > 2.0
    return pd.DataFrame({"long": spike & green & ~(spike.shift(1, fill_value=False)),
                         "short": spike & red & ~(spike.shift(1, fill_value=False)),
                         "state": _state(green & (vr > 1), red & (vr > 1)), "vr": vr})


INDICATORS = {
    # ad: (fonksiyon, açıklama)
    "ema_cross": (_ema_cross, "EMA 9/21 kesişimi"),
    "ema200": (_ema200, "Fiyat EMA200'ü kesiyor"),
    "macd": (_macd, "MACD sinyal çizgisi kesişimi"),
    "rsi": (_rsi, "RSI 30'dan yukarı / 70'ten aşağı dönüş"),
    "stoch": (_stoch, "Stokastik 20/80 bölgesinde K/D kesişimi"),
    "bollinger": (_bollinger, "Bollinger bandına değip içeri dönüş"),
    "bollinger_break": (_bollinger_break, "Bollinger bandını kırma (trend)"),
    "dmi": (_dmi, "+DI / -DI kesişimi (ADX > 20)"),
    "supertrend": (_supertrend, "Supertrend yön değişimi"),
    "psar": (_psar, "Parabolic SAR yön değişimi"),
    "ichimoku": (_ichimoku, "Ichimoku Tenkan/Kijun kesişimi, bulut onaylı"),
    "donchian": (_donchian, "Donchian 20 mum kırılımı"),
    "cci": (_cci, "CCI -100'den yukarı / +100'den aşağı dönüş"),
    "mfi": (_mfi, "MFI 20'den yukarı / 80'den aşağı dönüş"),
    "obv": (_obv, "OBV kendi EMA20'sini kesiyor"),
    "vwap": (_vwap, "Fiyat 24 saatlik VWAP'ı kesiyor"),
    "fibonacci": (_fibonacci, "Fibonacci 0.5-0.786 altın bölgeden tepki"),
    "volume": (_volume, "Hacim patlaması (2x) + mum yönü"),
}

# ONAY rolleri: ad -> (açıklama, fonksiyon(df) -> (long_ok, short_ok))
def _confirm_from_state(name):
    def f(df, **p):
        s = _run(name, df, p)["state"]
        return s == 1, s == -1
    return f


def _run(name: str, df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """İndikatör kuralını çalıştırır; ayarlanabilir parametreleri iletir."""
    params = params or {}
    if name == "fibonacci":
        return _fibonacci(df, int(params.get("fib_lookback", 72)))
    return INDICATORS[name][0](df)


CONFIRMS = {name: (desc, _confirm_from_state(name)) for name, (_, desc) in INDICATORS.items()}
CONFIRMS.update({
    "rsi": ("RSI 50'nin doğru tarafında", _confirm_from_state("rsi")),
    "adx": ("ADX > 25 (güçlü trend)",
            lambda df, **_: ((a := ind.adx(df)) > 25, a > 25)),
    "adx_weak": ("ADX < 20 (yatay piyasa — tepki stratejileri için)",
                 lambda df, **_: ((a := ind.adx(df)) < 20, a < 20)),
    "volume": ("Hacim ortalamanın 1.5 katı",
               lambda df, **_: ((v := ind.volume_ratio(df["volume"], 20)) > 1.5, v > 1.5)),
    "bollinger": ("Son 3 mumda Bollinger alt/üst bandına değmiş",
                  lambda df, **_: ((b := _bollinger(df))["touch_low"], b["touch_high"])),
    "rsi_extreme": ("RSI son 3 mumda 30 altı / 70 üstü",
                    lambda df, **_: ((r := ind.rsi(df["close"], 14)).rolling(3, min_periods=1).min() < 30,
                                r.rolling(3, min_periods=1).max() > 70)),
})

# Piyasada yaygın olarak birlikte önerilen eşler (giriş indikatörü -> onay adayları)
COMMON_PAIRS = {
    "ema_cross": ["rsi", "adx", "macd", "volume", "ema200"],
    "ema200": ["rsi", "macd", "adx", "volume"],
    "macd": ["rsi", "ema200", "adx", "volume"],
    "rsi": ["bollinger", "macd", "ema200", "adx_weak"],
    "stoch": ["ema200", "macd", "rsi", "adx_weak"],
    "bollinger": ["rsi_extreme", "stoch", "adx_weak", "volume"],
    "bollinger_break": ["adx", "volume", "rsi", "ema200"],
    "dmi": ["ema200", "macd", "volume"],
    "supertrend": ["ema200", "rsi", "adx", "macd"],
    "psar": ["adx", "ema200", "macd"],
    "ichimoku": ["rsi", "macd", "volume", "adx"],
    "donchian": ["adx", "volume", "ema200"],
    "cci": ["ema200", "adx_weak", "rsi"],
    "mfi": ["ema200", "bollinger", "adx_weak"],
    "obv": ["ema200", "macd", "rsi"],
    "vwap": ["rsi", "volume", "ema200"],
    "fibonacci": ["rsi", "macd", "ema200", "volume"],
    "volume": ["ema200", "macd", "rsi"],
}


TREND_DEFAULTS = {
    "use_regime": False, "adx_min": 20, "adx_period": 14,       # 4s ADX >= adx_min değilse işlem yok
    "use_htf": False, "htf": "4h", "htf_fast": 21, "htf_slow": 55,
    "htf_mode": "loose",                                         # strict: fiyat>EMA21>EMA55 | loose: EMA21>EMA55
    "use_btc_filter": False,                                     # altcoinde BTC'nin 4s yönü ters olmamalı
}


def htf_trend(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Üst zaman diliminde (geleceği görmeden) yön ve ADX. dir: +1 / -1 / 0."""
    def fn(h):
        c = h["close"]
        f, s = ind.ema(c, p["htf_fast"]), ind.ema(c, p["htf_slow"])
        if p.get("htf_mode", "loose") == "strict":
            d = np.where((c > f) & (f > s), 1, np.where((c < f) & (f < s), -1, 0))
        else:
            d = np.where(f > s, 1, np.where(f < s, -1, 0))
        d = np.where(s.notna(), d, 0)
        return pd.DataFrame({"dir": d, "adx": ind.adx(h, p["adx_period"])}, index=h.index)
    t = ind.higher_timeframe(df, p["htf"], fn)
    t["dir"] = t["dir"].fillna(0)
    return t


class SignalStrategy(Strategy):
    """Tek bir indikatörün klasik kuralıyla işlem; isteğe bağlı olarak bir onay indikatörü ve
    trend filtreleri (ADX rejimi, 4 saatlik yön, BTC yönü) ile."""

    def __init__(self, primary: str = "fibonacci", confirm: str | None = "volume", atr_period: int = 14,
                 fib_lookback: int = 72, **params):
        super().__init__(primary=primary, confirm=confirm, atr_period=atr_period, fib_lookback=fib_lookback,
                         **{**TREND_DEFAULTS, **params})
        self.uses_btc = bool(self.params["use_btc_filter"])
        if primary not in INDICATORS:
            raise ValueError(f"Bilinmeyen indikatör: {primary}. Seçenekler: {list(INDICATORS)}")
        if confirm and confirm not in CONFIRMS:
            raise ValueError(f"Bilinmeyen onay indikatörü: {confirm}. Seçenekler: {list(CONFIRMS)}")
        self.primary, self.confirm = primary, confirm

    @property
    def warmup(self) -> int:
        return 260                                   # EMA200 ve Ichimoku (52 + 26) için

    def generate(self, df, btc=None):
        out = df.copy()
        out["atr"] = ind.atr(out, self.params["atr_period"])
        sig = _run(self.primary, out, self.params)
        long, short = sig["long"].fillna(False), sig["short"].fillna(False)
        if self.confirm:
            lok, sok = CONFIRMS[self.confirm][1](out, **self.params)
            long &= lok.fillna(False).astype(bool)
            short &= sok.fillna(False).astype(bool)
        p = self.params
        if p["use_regime"] or p["use_htf"]:
            tr = htf_trend(out, p)
            out["htf_dir"], out["htf_adx"] = tr["dir"], tr["adx"]
            if p["use_regime"]:
                strong = (tr["adx"] >= p["adx_min"]).fillna(False)
                long &= strong
                short &= strong
            if p["use_htf"]:
                long &= tr["dir"] == 1
                short &= tr["dir"] == -1
        if p["use_btc_filter"] and btc is not None:
            bd = htf_trend(btc.reindex(out.index, method="ffill"), p)["dir"]
            out["btc_dir"] = bd
            long &= bd != -1
            short &= bd != 1
        ready = out["atr"].notna()
        out["long_entry"] = (long & ready).astype(bool)
        out["short_entry"] = (short & ready & ~out["long_entry"]).astype(bool)
        out["long_exit"] = sig["short"].fillna(False).astype(bool)   # ters sinyal -> çık
        out["short_exit"] = sig["long"].fillna(False).astype(bool)
        return out

    def describe(self) -> str:
        p = self.params
        d = INDICATORS[self.primary][1]
        if self.primary == "fibonacci":
            d += f" ({p['fib_lookback']} mum)"
        d += f" + onay: {CONFIRMS[self.confirm][0]}" if self.confirm else ""
        f = []
        if p["use_regime"]:
            f.append(f"ADX({p['htf']})>={p['adx_min']}")
        if p["use_htf"]:
            f.append(f"{p['htf']} yön ({'fiyat>EMA21>EMA55' if p['htf_mode'] == 'strict' else 'EMA21>EMA55'})")
        if p["use_btc_filter"]:
            f.append("BTC yönü")
        return d + (" | filtreler: " + ", ".join(f) if f else "")
