"""Geçmiş veri: vadeli işlem mumları + funding geçmişi indirme, CSV önbellek, sentetik veri."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import ROOT, futures_symbol

DATA_DIR = ROOT / "data"

TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


def _paths(cfg: dict):
    ex = cfg["exchange"]
    days = cfg["backtest"]["days"]
    tag = f"{ex['name']}_{ex['symbol'].replace('/', '')}_{ex['timeframe']}_{days}d"
    return DATA_DIR / f"{tag}.csv", DATA_DIR / f"{tag}_funding.csv"


def load_history(cfg: dict, refresh: bool = False) -> tuple[pd.DataFrame, pd.Series | None]:
    """Binance Futures'tan mumları ve funding oranlarını indirir; data/ klasöründe saklar."""
    from .exchange import FuturesExchange

    DATA_DIR.mkdir(exist_ok=True)
    p_ohlcv, p_fund = _paths(cfg)
    if p_ohlcv.exists() and not refresh:
        df = pd.read_csv(p_ohlcv, index_col=0, parse_dates=True)
        fund = pd.read_csv(p_fund, index_col=0, parse_dates=True).iloc[:, 0] if p_fund.exists() else None
        return df, fund

    ex = cfg["exchange"]
    client = FuturesExchange(ex["name"])
    sym = futures_symbol(ex["symbol"])
    since = int((time.time() - cfg["backtest"]["days"] * 86400) * 1000)
    df = client.fetch_ohlcv_range(sym, ex["timeframe"], since)
    df.to_csv(p_ohlcv)
    fund = None
    try:
        fund = client.fetch_funding_range(sym, since)
        fund.to_csv(p_fund)
    except Exception as e:  # funding verisi alınamazsa varsayılan oran kullanılır
        print(f"Uyarı: funding geçmişi indirilemedi ({e}); varsayılan oran kullanılacak.")
    return df, fund


def synthetic_ohlcv(n: int = 5000, timeframe: str = "1h", seed: int = 42,
                    start_price: float = 30000.0) -> pd.DataFrame:
    """Rejim değiştiren (yükseliş / düşüş / yatay) rastgele fiyat serisi. Sadece kodun çalıştığını
    test etmek içindir; sonuçları gerçek piyasa hakkında hiçbir şey söylemez."""
    rng = np.random.default_rng(seed)
    drift = np.zeros(n)
    i = 0
    while i < n:
        length = rng.integers(100, 600)
        drift[i:i + length] = rng.choice([-0.0006, 0.0, 0.0008])
        i += length
    rets = drift + rng.normal(0, 0.008, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    spread = np.abs(rng.normal(0, 0.004, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = rng.uniform(50, 500, n)
    idx = pd.date_range("2024-01-01", periods=n, freq=f"{TF_MINUTES[timeframe]}min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
