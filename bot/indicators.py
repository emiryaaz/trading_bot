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
