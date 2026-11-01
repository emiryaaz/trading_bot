"""Ayar dosyası ve ortam değişkenlerini yükler."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict:
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["secrets"] = {
        "api_key": os.getenv("BINANCE_API_KEY", ""),
        "api_secret": os.getenv("BINANCE_API_SECRET", ""),
        "allow_live": os.getenv("ALLOW_LIVE", "no").strip().lower() == "yes",
        "telegram_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    }
    normalize(cfg)
    validate(cfg)
    return cfg


def normalize(cfg: dict) -> dict:
    """Eski ayar dosyalarıyla uyumluluk ve varsayılanlar."""
    ex = cfg.setdefault("exchange", {})
    if "taker_fee" not in ex:
        ex["taker_fee"] = ex.get("fee_rate", 0.0005)
    ex.setdefault("maker_fee", ex["taker_fee"])
    ex["fee_rate"] = ex["taker_fee"]
    exe = cfg.setdefault("execution", {})
    exe.setdefault("entry_order", "market")
    exe.setdefault("limit_offset_atr", 0.0)
    exe.setdefault("limit_ttl_bars", 1)
    exe.setdefault("tp_order", "market")
    return cfg


def backtest_kwargs(cfg: dict) -> dict:
    """run_backtest'e config'den gelen ücret / emir tipi parametreleri."""
    ex, exe, fut = cfg["exchange"], cfg["execution"], cfg["futures"]
    return {"fee_rate": ex["taker_fee"], "maker_fee": ex["maker_fee"], "slippage": ex["slippage"],
            "allow_short": fut["allow_short"], "funding_fallback": fut["funding_rate_fallback"],
            "entry_order": exe["entry_order"], "limit_offset_atr": exe["limit_offset_atr"],
            "limit_ttl_bars": exe["limit_ttl_bars"], "tp_order": exe["tp_order"]}


def futures_symbol(symbol: str) -> str:
    """'BTC/USDT' -> 'BTC/USDT:USDT' (ccxt'de USDⓈ-M sürekli kontrat sembolü)."""
    if ":" in symbol:
        return symbol
    quote = symbol.split("/")[1]
    return f"{symbol}:{quote}"


def validate(cfg: dict) -> None:
    mode = cfg.get("mode")
    if mode not in ("paper", "testnet", "live"):
        raise ValueError(f"Geçersiz mode: {mode!r} (paper / testnet / live olmalı)")

    exe = cfg.get("execution", {})
    if exe.get("entry_order", "market") not in ("market", "limit"):
        raise ValueError("execution.entry_order 'market' veya 'limit' olmalı")
    if exe.get("tp_order", "market") not in ("market", "limit"):
        raise ValueError("execution.tp_order 'market' veya 'limit' olmalı")
    if exe.get("limit_ttl_bars", 1) < 1:
        raise ValueError("execution.limit_ttl_bars en az 1 olmalı")

    fut = cfg["futures"]
    if not 1 <= fut["leverage"] <= 5:
        raise ValueError("leverage 1 ile 5 arasında olmalı (güvenlik sınırı)")
    if fut["margin_mode"] != "isolated":
        raise ValueError("Bu bot yalnızca isolated marjin ile çalışır")

    syms = cfg.get("exchange", {}).get("symbols")
    if syms is not None and (not syms or len(set(syms)) != len(syms)):
        raise ValueError("exchange.symbols boş olamaz ve tekrar eden coin içeremez")

    r = cfg["risk"]
    if r.get("max_open_positions", 1) < 1:
        raise ValueError("max_open_positions en az 1 olmalı")
    if not 0 < r.get("max_total_margin_pct", 1) <= 1:
        raise ValueError("max_total_margin_pct 0 ile 1 arasında olmalı")
    if not 0 < r["margin_per_trade_pct"] <= 1:
        raise ValueError("margin_per_trade_pct 0 ile 1 arasında olmalı")
    if r["stop_mode"] not in ("roe", "atr"):
        raise ValueError("stop_mode 'roe' veya 'atr' olmalı")
    if r["stop_mode"] == "roe" and not 0 < r["stop_loss_roe"] < 0.8:
        # 0.8: likidasyondan önce stop'un mutlaka çalışması için güvenli üst sınır
        raise ValueError("stop_loss_roe 0 ile 0.8 arasında olmalı (stop-loss zorunlu)")
    if r["stop_mode"] == "atr" and r["stop_atr_mult"] <= 0:
        raise ValueError("stop_atr_mult pozitif olmalı")
    if not 0 < r["max_drawdown_pct"] < 1:
        raise ValueError("max_drawdown_pct 0 ile 1 arasında olmalı")
