"""Geçmiş veri: vadeli işlem mumları + funding geçmişi indirme, CSV önbellek, sentetik veri."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import ROOT, futures_symbol

DATA_DIR = ROOT / "data"

TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


def _paths(cfg: dict, symbol: str):
    ex = cfg["exchange"]
    days = cfg["backtest"]["days"]
    tag = f"{ex['name']}_{symbol.replace('/', '')}_{ex['timeframe']}_{days}d"
    return DATA_DIR / f"{tag}.csv", DATA_DIR / f"{tag}_funding.csv"


def load_history(cfg: dict, refresh: bool = False) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series | None]]:
    """Her coin için Binance Futures mumlarını ve funding oranlarını indirir; data/ klasöründe saklar.
    Dönüş: ({coin: mum verisi}, {coin: funding serisi veya None})"""
    DATA_DIR.mkdir(exist_ok=True)
    ex = cfg["exchange"]
    client = None
    since = int((time.time() - cfg["backtest"]["days"] * 86400) * 1000)
    data, fundings = {}, {}
    for symbol in ex["symbols"]:
        p_ohlcv, p_fund = _paths(cfg, symbol)
        if p_ohlcv.exists() and not refresh:
            data[symbol] = pd.read_csv(p_ohlcv, index_col=0, parse_dates=True)
            fundings[symbol] = (pd.read_csv(p_fund, index_col=0, parse_dates=True).iloc[:, 0]
                                if p_fund.exists() else None)
            continue
        if client is None:
            from .exchange import FuturesExchange
            client = FuturesExchange(ex["name"])
        sym = futures_symbol(symbol)
        print(f"  {symbol} indiriliyor...", flush=True)
        try:
            df = client.fetch_ohlcv_range(sym, ex["timeframe"], since)
        except Exception as e:
            print(f"  UYARI: {symbol} verisi indirilemedi ({e}); bu coin atlanıyor.")
            continue
        df.to_csv(p_ohlcv)
        data[symbol] = df
        fundings[symbol] = None
        try:
            f = client.fetch_funding_range(sym, since)
            f.to_csv(p_fund)
            fundings[symbol] = f
        except Exception as e:
            print(f"  Uyarı: {symbol} funding geçmişi indirilemedi ({e}); varsayılan oran kullanılacak.")
    return data, fundings


def synthetic_portfolio(symbols: list[str], n: int = 5000, timeframe: str = "1h") -> dict[str, pd.DataFrame]:
    """Her coin için farklı tohumla sentetik veri (sadece kod testi için)."""
    return {s: synthetic_ohlcv(n=n, timeframe=timeframe, seed=10 + k, start_price=100.0 * (k + 1))
            for k, s in enumerate(symbols)}


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
