"""Borsa bağlantı katmanı — Binance global USDⓈ-M Futures (ccxt: binanceusdm).

Botun geri kalanı yalnızca `FuturesClient` arayüzünü bilir:
  * FuturesExchange : gerçek borsa (live / testnet)
  * PaperFutures    : sanal bakiye; fiyatı gerçek borsadan alır, emirleri ve borsadaki
                      stop/kâr-al emirlerini simüle eder. Borsaya hiçbir emir gönderilmez.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import ccxt
import pandas as pd

from .config import futures_symbol


@dataclass
class Fill:
    side: int          # 1 = long yönlü (buy), -1 = short yönlü (sell)
    qty: float
    price: float
    fee: float         # USDT


@dataclass
class Position:
    side: int          # 1 long, -1 short
    qty: float
    entry: float
    unrealized: float = 0.0


@dataclass
class ClosedResult:
    exit_price: float
    pnl: float         # ücretler ve funding dahil net K/Z (USDT)
    fees: float
    funding: float


class FuturesClient(ABC):
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame: ...
    @abstractmethod
    def fetch_price(self, symbol: str) -> float: ...
    @abstractmethod
    def setup(self, symbol: str, leverage: int, margin_mode: str) -> None: ...
    @abstractmethod
    def available_balance(self) -> float: ...
    @abstractmethod
    def get_position(self, symbol: str) -> Position | None: ...

    def fetch_prices(self, symbols: list[str]) -> dict[str, float]:
        return {s: self.fetch_price(s) for s in symbols}

    def get_positions(self, symbols: list[str]) -> dict[str, Position]:
        out = {}
        for s in symbols:
            p = self.get_position(s)
            if p:
                out[s] = p
        return out
    @abstractmethod
    def open_position(self, symbol: str, side: int, qty: float) -> Fill: ...
    @abstractmethod
    def close_position(self, symbol: str, pos: Position) -> Fill: ...
    @abstractmethod
    def place_protection(self, symbol: str, pos: Position, stop: float, take_profit: float) -> None: ...
    @abstractmethod
    def cancel_protection(self, symbol: str) -> None: ...
    @abstractmethod
    def closed_result(self, symbol: str, side: int, since_ms: int, entry_fee: float) -> ClosedResult | None: ...

    def round_qty(self, symbol: str, qty: float) -> float:
        return qty

    def round_price(self, symbol: str, price: float) -> float:
        return price

    def min_notional(self, symbol: str) -> float:
        return 5.0

    def min_qty(self, symbol: str) -> float:
        return 0.0


def ohlcv_to_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").astype(float)


def _side_str(side: int) -> str:
    return "buy" if side == 1 else "sell"


class FuturesExchange(FuturesClient):
    def __init__(self, name: str = "binanceusdm", api_key: str = "", api_secret: str = "",
                 testnet: bool = False, testnet_kind: str = "testnet"):
        cls = getattr(ccxt, name)
        self.ex = cls({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"adjustForTimeDifference": True},
        })
        if testnet:
            if testnet_kind == "demo":
                self.ex.enable_demo_trading(True)   # demo.binance.com anahtarları
            else:
                self.ex.set_sandbox_mode(True)      # testnet.binancefuture.com anahtarları
        self.ex.load_markets()

    # ------------------------------------------------------------ veri
    def fetch_ohlcv(self, symbol, timeframe, limit=500):
        return ohlcv_to_df(self.ex.fetch_ohlcv(symbol, timeframe, limit=limit))

    def fetch_ohlcv_range(self, symbol, timeframe, since_ms: int) -> pd.DataFrame:
        rows, since = [], since_ms
        while True:
            batch = self.ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not batch:
                break
            rows += batch
            since = batch[-1][0] + 1
            if len(batch) < 1000:
                break
        df = ohlcv_to_df(rows)
        return df[~df.index.duplicated()]

    def fetch_funding_range(self, symbol, since_ms: int) -> pd.Series:
        rows, since = [], since_ms
        while True:
            batch = self.ex.fetch_funding_rate_history(symbol, since=since, limit=1000)
            if not batch:
                break
            rows += batch
            since = batch[-1]["timestamp"] + 1
            if len(batch) < 1000:
                break
        s = pd.Series({pd.Timestamp(r["timestamp"], unit="ms", tz="UTC"): float(r["fundingRate"]) for r in rows})
        s.name = "funding_rate"
        return s.sort_index()

    def fetch_price(self, symbol):
        return float(self.ex.fetch_ticker(symbol)["last"])

    # ------------------------------------------------------------ hesap ayarları
    def setup(self, symbol, leverage, margin_mode):
        for fn, args in ((self.ex.set_position_mode, (False, symbol)),     # tek yönlü (one-way) mod
                         (self.ex.set_margin_mode, (margin_mode, symbol)),
                         (self.ex.set_leverage, (leverage, symbol))):
            try:
                fn(*args)
            except ccxt.ExchangeError as e:
                if "No need to change" not in str(e):      # zaten ayarlıysa Binance hata döner
                    raise

    def available_balance(self):
        bal = self.ex.fetch_balance()
        return float((bal.get("free") or {}).get("USDT", 0) or 0)

    def fetch_prices(self, symbols):
        tick = self.ex.fetch_tickers(symbols)                 # tek istekte tüm coinler
        return {s: float(tick[s]["last"]) for s in symbols if s in tick}

    # ------------------------------------------------------------ pozisyon
    def get_positions(self, symbols):
        out = {}
        for p in self.ex.fetch_positions(symbols):            # tek istekte tüm coinler
            qty = float(p.get("contracts") or 0)
            if p.get("symbol") in symbols and qty > 0:
                side = 1 if p.get("side") == "long" else -1
                out[p["symbol"]] = Position(side, qty, float(p.get("entryPrice") or 0),
                                            float(p.get("unrealizedPnl") or 0))
        return out

    def get_position(self, symbol):
        return self.get_positions([symbol]).get(symbol)

    def _fill(self, symbol: str, order: dict, side: int) -> Fill:
        qty = float(order.get("filled") or order.get("amount") or 0)
        price = float(order.get("average") or order.get("price") or 0)
        if not price and order.get("cost") and qty:
            price = float(order["cost"]) / qty
        if not price:
            price = self.fetch_price(symbol)
        fee = sum(float(f["cost"]) for f in (order.get("fees") or []) if f and f.get("cost")
                  and f.get("currency") == "USDT")
        if not fee:
            fee = qty * price * 0.0005   # borsa ücreti bildirmediyse tahmini taker ücreti
        return Fill(side, qty, price, fee)

    def open_position(self, symbol, side, qty):
        o = self.ex.create_order(symbol, "market", _side_str(side), self.round_qty(symbol, qty))
        return self._fill(symbol, o, side)

    def close_position(self, symbol, pos):
        o = self.ex.create_order(symbol, "market", _side_str(-pos.side), self.round_qty(symbol, pos.qty),
                                 None, {"reduceOnly": True})
        return self._fill(symbol, o, -pos.side)

    def place_protection(self, symbol, pos, stop, take_profit):
        """Stop-loss ve kâr al emirlerini BORSAYA koyar. Bot kapansa bile çalışırlar."""
        common = {"reduceOnly": True, "workingType": "MARK_PRICE"}
        close_side = _side_str(-pos.side)
        qty = self.round_qty(symbol, pos.qty)
        self.ex.create_order(symbol, "market", close_side, qty, None,
                             {**common, "stopLossPrice": self.round_price(symbol, stop)})
        if take_profit:
            self.ex.create_order(symbol, "market", close_side, qty, None,
                                 {**common, "takeProfitPrice": self.round_price(symbol, take_profit)})

    def cancel_protection(self, symbol):
        for params in ({"trigger": True}, {}):          # koşullu (algo) emirler + normal emirler
            try:
                self.ex.cancel_all_orders(symbol, params)
            except ccxt.BaseError:
                pass

    def closed_result(self, symbol, side, since_ms, entry_fee):
        """Pozisyon borsada (stop / kâr-al / likidasyon) kapandıysa gerçek sonucu borsadan okur."""
        trades = self.ex.fetch_my_trades(symbol, since=since_ms)
        closing = [t for t in trades if t["side"] == _side_str(-side)]
        if not closing:
            return None
        qty = sum(float(t["amount"]) for t in closing)
        exit_px = sum(float(t["price"]) * float(t["amount"]) for t in closing) / qty
        realized = sum(float((t.get("info") or {}).get("realizedPnl") or 0) for t in trades)
        fees = sum(float(t["fee"]["cost"]) for t in trades
                   if t.get("fee") and t["fee"].get("cost") and t["fee"].get("currency") == "USDT")
        funding = 0.0
        try:
            funding = -sum(float(f["amount"]) for f in self.ex.fetch_funding_history(symbol, since=since_ms))
        except ccxt.BaseError:
            pass
        return ClosedResult(exit_px, realized - fees - funding, fees, funding)

    # ------------------------------------------------------------ hassasiyet / limitler
    def round_qty(self, symbol, qty):
        return float(self.ex.amount_to_precision(symbol, qty))

    def round_price(self, symbol, price):
        return float(self.ex.price_to_precision(symbol, price))

    def min_notional(self, symbol):
        lim = self.ex.market(symbol).get("limits", {}).get("cost", {}) or {}
        return float(lim.get("min") or 5.0)

    def min_qty(self, symbol):
        lim = self.ex.market(symbol).get("limits", {}).get("amount", {}) or {}
        return float(lim.get("min") or 0.0)


class PaperFutures(FuturesClient):
    """Sanal vadeli işlem hesabı (çoklu coin). Borsadaki stop / kâr-al emirlerini her fiyat okumasında
    simüle eder. Borsaya hiçbir emir gönderilmez."""

    def __init__(self, data_source, balance: float, fee_rate: float, slippage: float, leverage: float):
        self.src = data_source
        self.fee_rate, self.slippage, self.lev = fee_rate, slippage, leverage
        self.balance = float(balance)                       # cüzdan (gerçekleşmiş K/Z dahil)
        self.pos: dict[str, Position] = {}
        self.protect: dict[str, tuple[float, float]] = {}
        self.last_closed: dict[str, ClosedResult] = {}
        self._entry_fee: dict[str, float] = {}

    # veri
    def fetch_ohlcv(self, symbol, timeframe, limit=500):
        return self.src.fetch_ohlcv(symbol, timeframe, limit)

    def fetch_price(self, symbol):
        return self.src.fetch_price(symbol)

    def fetch_prices(self, symbols):
        return self.src.fetch_prices(symbols)

    def setup(self, symbol, leverage, margin_mode):
        pass

    def round_qty(self, symbol, qty):
        return self.src.round_qty(symbol, qty)

    def round_price(self, symbol, price):
        return self.src.round_price(symbol, price)

    def min_notional(self, symbol):
        return self.src.min_notional(symbol)

    def min_qty(self, symbol):
        return self.src.min_qty(symbol)

    # hesap
    def available_balance(self):
        used = sum(p.qty * p.entry / self.lev for p in self.pos.values())
        return self.balance - used

    def _check_protection(self, symbol, price):
        p, prot = self.pos.get(symbol), self.protect.get(symbol)
        if not (p and prot):
            return
        stop, tp = prot
        hit_stop = price <= stop if p.side == 1 else price >= stop
        hit_tp = tp and (price >= tp if p.side == 1 else price <= tp)
        if hit_stop or hit_tp:
            self._close(symbol, stop if hit_stop else tp)

    def get_positions(self, symbols):
        prices = self.fetch_prices([s for s in symbols if s in self.pos]) if self.pos else {}
        out = {}
        for s in symbols:
            if s in self.pos and s in prices:
                self._check_protection(s, prices[s])
            if s in self.pos:
                p = self.pos[s]
                p.unrealized = p.side * (prices.get(s, p.entry) - p.entry) * p.qty
                out[s] = p
        return out

    def get_position(self, symbol):
        return self.get_positions([symbol]).get(symbol)

    def open_position(self, symbol, side, qty):
        if symbol in self.pos:
            raise RuntimeError(f"Paper: {symbol} için zaten açık pozisyon var")
        price = self.fetch_price(symbol) * (1 + side * self.slippage)
        fee = qty * price * self.fee_rate
        if qty * price / self.lev + fee > self.available_balance() + 1e-9:
            raise ValueError("Paper: yetersiz teminat")
        self.balance -= fee
        self.pos[symbol] = Position(side, qty, price)
        self._entry_fee[symbol] = fee
        return Fill(side, qty, price, fee)

    def _close(self, symbol, raw_price) -> Fill:
        p = self.pos.pop(symbol)
        self.protect.pop(symbol, None)
        price = raw_price * (1 - p.side * self.slippage)
        fee = p.qty * price * self.fee_rate
        gross = max(p.side * (price - p.entry) * p.qty, -p.qty * p.entry / self.lev)  # isolated sınırı
        self.balance += gross - fee
        ef = self._entry_fee.pop(symbol, 0.0)
        self.last_closed[symbol] = ClosedResult(price, gross - fee - ef, fee + ef, 0.0)
        return Fill(-p.side, p.qty, price, fee)

    def close_position(self, symbol, pos):
        return self._close(symbol, self.fetch_price(symbol))

    def place_protection(self, symbol, pos, stop, take_profit):
        self.protect[symbol] = (stop, take_profit)

    def cancel_protection(self, symbol):
        self.protect.pop(symbol, None)

    def closed_result(self, symbol, side, since_ms, entry_fee):
        return self.last_closed.pop(symbol, None)

    def to_state(self) -> dict:
        return {"balance": self.balance,
                "pos": {s: p.__dict__ for s, p in self.pos.items()},
                "protect": {s: list(v) for s, v in self.protect.items()},
                "entry_fee": self._entry_fee}

    def load_state(self, st: dict) -> None:
        self.balance = st["balance"]
        self.pos = {s: Position(**p) for s, p in (st.get("pos") or {}).items()}
        self.protect = {s: tuple(v) for s, v in (st.get("protect") or {}).items()}
        self._entry_fee = dict(st.get("entry_fee") or {})


def build_exchange(cfg: dict) -> FuturesClient:
    ex_cfg, sec, mode = cfg["exchange"], cfg["secrets"], cfg["mode"]
    if mode == "paper":
        src = FuturesExchange(ex_cfg["name"])          # sadece halka açık fiyat verisi
        return PaperFutures(src, cfg["live"]["capital"], ex_cfg["fee_rate"], ex_cfg["slippage"],
                            cfg["futures"]["leverage"])
    if not sec["api_key"]:
        raise RuntimeError(f"{mode} modu için .env içine API anahtarı girin")
    if mode == "testnet":
        return FuturesExchange(ex_cfg["name"], sec["api_key"], sec["api_secret"], testnet=True,
                               testnet_kind=ex_cfg.get("testnet_kind", "testnet"))
    if mode == "live":
        if not sec["allow_live"]:
            raise RuntimeError("GERÇEK PARA modu kilitli. Bilinçli olarak açmak için .env: ALLOW_LIVE=yes")
        return FuturesExchange(ex_cfg["name"], sec["api_key"], sec["api_secret"])
    raise ValueError(mode)


def now_ms() -> int:
    return int(time.time() * 1000)
