"""Teknik göstergeler (yalnızca pandas ile, harici bağımlılık yok).

Her gösterge, yalnızca o ana kadarki veriyi kullanır (geleceği görmez).
Gruplar — her biri farklı bir şeyi ölçer:
  trend      : sma, ema, adx, macd
  momentum   : rsi
  oynaklık   : atr
  hacim      : volume_ratio
  yapı       : fib_retracement (son salınımın ne kadarı geri alındı)
  zaman dil. : higher_timeframe (ör. 1s veriden 4s göstergeleri, geleceği görmeden)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, 1e-12)
    return 100 - 100 / (1 + rs)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(),
                      (df["low"] - prev_close).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Trend GÜCÜ (yön değil). ~20 altı: yatay piyasa, 25 üstü: belirgin trend."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def volume_ratio(volume: pd.Series, n: int = 20) -> pd.Series:
    """Son mumun hacmi / son n mumun ortalama hacmi. >1: ortalamanın üstünde ilgi."""
    return volume / volume.rolling(n, min_periods=n).mean()


def fib_retracement(df: pd.DataFrame, lookback: int = 72, pullback_bars: int = 8) -> pd.DataFrame:
    """Fibonacci geri çekilme seviyesi.

    Son `lookback` mumdaki salınımın (en yüksek - en düşük) ne kadarının, son `pullback_bars`
    mumdaki geri çekilmede geri alındığını ölçer:
      long_level : yükselişten sonraki düşüşün derinliği  = (tepe - son dip) / salınım
      short_level: düşüşten sonraki tepkinin yüksekliği    = (son tepe - dip) / salınım
    0.382 – 0.618 arası klasik "altın bölge"dir; 0.786'dan derin geri çekilme trendin bozulduğuna işaret eder.
    """
    hi = df["high"].rolling(lookback, min_periods=lookback).max()
    lo = df["low"].rolling(lookback, min_periods=lookback).min()
    rng = (hi - lo).replace(0, np.nan)
    recent_lo = df["low"].rolling(pullback_bars, min_periods=pullback_bars).min()
    recent_hi = df["high"].rolling(pullback_bars, min_periods=pullback_bars).max()
    return pd.DataFrame({"long_level": (hi - recent_lo) / rng, "short_level": (recent_hi - lo) / rng})


TF_RULE = {"1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h", "1d": "1D"}


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule, label="left", closed="left").agg(agg).dropna()


def higher_timeframe(df: pd.DataFrame, rule: str, fn) -> pd.DataFrame:
    """Küçük zaman dilimi verisinden büyük zaman dilimi göstergesi üretir, GELECEĞİ GÖRMEDEN.

    `fn(htf_df) -> DataFrame` büyük zaman dilimindeki göstergeleri hesaplar. Sonuç, büyük mum
    tamamen kapandıktan sonra (o mumun son küçük mumunun kapanışında) kullanılabilir hale gelir
    ve küçük zaman dilimine ileriye doğru taşınır (ffill)."""
    base_step = df.index.to_series().diff().median()
    htf = resample_ohlcv(df, rule)
    out = fn(htf)
    # büyük mum [T, T+R) -> son küçük mumun açılışı T+R-base_step'te bilgisi kullanılabilir
    out.index = out.index + pd.Timedelta(rule) - base_step
    return out.reindex(df.index, method="ffill")


# ============================================================ ek göstergeler (v0.7)
def stochastic(df: pd.DataFrame, k: int = 14, smooth: int = 3, d: int = 3) -> pd.DataFrame:
    lo = df["low"].rolling(k, min_periods=k).min()
    hi = df["high"].rolling(k, min_periods=k).max()
    raw = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    kk = raw.rolling(smooth, min_periods=smooth).mean()
    return pd.DataFrame({"k": kk, "d": kk.rolling(d, min_periods=d).mean()})


def bollinger(close: pd.Series, n: int = 20, mult: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    return pd.DataFrame({"mid": mid, "upper": mid + mult * sd, "lower": mid - mult * sd})


def dmi(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr
    return pd.DataFrame({"plus": pdi, "minus": mdi, "adx": adx(df, n)})


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.Series:
    """+1 yükseliş, -1 düşüş (Supertrend yönü)."""
    a = atr(df, n).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    c = df["close"].to_numpy()
    up, dn = hl2 - mult * a, hl2 + mult * a
    fu, fd = up.copy(), dn.copy()
    direction = np.zeros(len(c))
    for i in range(1, len(c)):
        if np.isnan(a[i]):
            continue
        fu[i] = up[i] if (np.isnan(fu[i - 1]) or c[i - 1] <= fu[i - 1]) else max(up[i], fu[i - 1])
        fd[i] = dn[i] if (np.isnan(fd[i - 1]) or c[i - 1] >= fd[i - 1]) else min(dn[i], fd[i - 1])
        prev = direction[i - 1] if direction[i - 1] != 0 else 1
        if prev == 1:
            direction[i] = -1 if c[i] < fu[i] else 1
        else:
            direction[i] = 1 if c[i] > fd[i] else -1
    s = pd.Series(direction, index=df.index)
    return s.where(~np.isnan(a), 0)


def parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """+1 fiyat SAR'ın üstünde (yükseliş), -1 altında."""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(h)
    out = np.zeros(n)
    if n < 3:
        return pd.Series(out, index=df.index)
    bull, af = True, step
    ep, sar = h[0], l[0]
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if bull:
            sar = min(sar, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < sar:
                bull, sar, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_step)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > sar:
                bull, sar, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, max_step)
        out[i] = 1 if bull else -1
    return pd.Series(out, index=df.index)


def ichimoku(df: pd.DataFrame, t: int = 9, k: int = 26, s: int = 52) -> pd.DataFrame:
    """Bulut, 26 mum ÖNCE hesaplanan değerlerdir (shift ileri) -> geleceği görmez. Chikou kullanılmaz."""
    mid = lambda n: (df["high"].rolling(n, min_periods=n).max() + df["low"].rolling(n, min_periods=n).min()) / 2
    tenkan, kijun = mid(t), mid(k)
    span_a = ((tenkan + kijun) / 2).shift(k)
    span_b = mid(s).shift(k)
    return pd.DataFrame({"tenkan": tenkan, "kijun": kijun,
                         "cloud_top": pd.concat([span_a, span_b], axis=1).max(axis=1, skipna=False),
                         "cloud_bot": pd.concat([span_a, span_b], axis=1).min(axis=1, skipna=False)})


def donchian(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Önceki n mumun en yüksek/en düşüğü (bu mum hariç)."""
    return pd.DataFrame({"upper": df["high"].rolling(n, min_periods=n).max().shift(1),
                         "lower": df["low"].rolling(n, min_periods=n).min().shift(1)})


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    m = tp.rolling(n, min_periods=n).mean()
    md = tp.rolling(n, min_periods=n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - m) / (0.015 * md.replace(0, np.nan))


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    flow = tp * df["volume"]
    pos = flow.where(tp > tp.shift(1), 0.0).rolling(n, min_periods=n).sum()
    neg = flow.where(tp < tp.shift(1), 0.0).rolling(n, min_periods=n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()


def rolling_vwap(df: pd.DataFrame, n: int = 24) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).rolling(n, min_periods=n).sum() / df["volume"].rolling(n, min_periods=n).sum()
