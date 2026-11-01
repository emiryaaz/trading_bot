"""Canlı / paper / testnet işlem döngüsü (vadeli işlemler, isolated).

Her turda:
  1. Bot devre kesiciyle durdurulmuşsa çalışmaz (önce rapor incelenir, sonra `python main.py resume`).
  2. STOP dosyası varsa açık pozisyon kapatılır ve bot durur.
  3. Borsadaki pozisyon kontrol edilir: stop / kâr-al emri borsada tetiklendiyse sonuç kaydedilir.
  4. DEVRE KESİCİ: bot bakiyesi son başlatmaya göre %10 düştüyse pozisyon kapatılır,
     rapor yazılır ve bot durur.
  5. Yeni mum KAPANDIYSA strateji çalışır; çıkış / giriş kararları uygulanır.
     Her yeni pozisyonla birlikte stop-loss ve kâr-al emirleri hemen BORSAYA konur.
  6. Durum data/state.json'a kaydedilir.

Bot yalnızca KENDİ açtığı pozisyonu ve kendisine ayrılan `capital` kadar bakiyeyi yönetir.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, futures_symbol
from .exchange import FuturesClient, PaperFutures, Position, now_ms
from .logger import Notifier, get_logger
from .report import write_report
from .risk import LONG, RiskManager
from .strategy import Strategy


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Trader:
    def __init__(self, cfg: dict, exchange: FuturesClient, strategy: Strategy,
                 risk: RiskManager, notifier: Notifier | None = None, state_path: Path | None = None):
        self.cfg, self.ex, self.strategy, self.risk = cfg, exchange, strategy, risk
        self.symbol = futures_symbol(cfg["exchange"]["symbol"])
        self.tf = cfg["exchange"]["timeframe"]
        self.capital = float(cfg["live"]["capital"])
        self.allow_short = cfg["futures"].get("allow_short", True)
        self.log = get_logger()
        self.notify = notifier or Notifier()
        self.state_path = state_path or ROOT / cfg["live"]["state_file"]
        self.state = self._load_state()

    # ================================================================ durum
    def _load_state(self) -> dict:
        default = {"position": None, "last_candle": None, "day": None, "day_start_equity": None,
                   "realized": 0.0, "trades": [], "halted": False, "halt_reason": None,
                   "last_report": None, "session": {"start": _now(), "baseline": self.capital},
                   "paper": None, "warned_foreign_position": False}
        if self.state_path.exists():
            st = {**default, **json.loads(self.state_path.read_text(encoding="utf-8"))}
            if isinstance(self.ex, PaperFutures) and st.get("paper"):
                self.ex.load_state(st["paper"])
            self.log.info(f"Kayıtlı durum yüklendi. Açık pozisyon: {st['position']}")
            return st
        return default

    def save(self) -> None:
        if isinstance(self.ex, PaperFutures):
            self.state["paper"] = self.ex.to_state()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.state_path)

    # ================================================================ hesaplar
    def equity(self, ex_pos: Position | None = None) -> float:
        """Bot bakiyesi = ayrılan sermaye + gerçekleşmiş K/Z + açık pozisyonun anlık K/Z'si."""
        unreal = ex_pos.unrealized if (ex_pos and self.state["position"]) else 0.0
        return self.capital + self.state["realized"] + unreal

    def _record(self, side: int, entry: float, exit_px: float, qty: float, pnl: float,
                fees: float, funding: float, reason: str) -> None:
        p = self.state["position"]
        margin = p["margin"] if p else qty * entry / self.cfg["futures"]["leverage"]
        t = {"side": "LONG" if side == LONG else "SHORT", "entry": entry, "exit": exit_px, "qty": qty,
             "margin": margin, "pnl": round(pnl, 6), "roe": pnl / margin if margin else 0.0,
             "fees": round(fees, 6), "funding": round(funding, 6), "reason": reason,
             "opened": p["opened"] if p else _now(), "closed": _now()}
        self.state["trades"].append(t)
        self.state["realized"] += pnl
        self.state["position"] = None
        self.notify.send(f"KAPANDI ({reason}) {t['side']} {qty} @ {exit_px:.2f} | "
                         f"K/Z {pnl:+.2f} USDT (ROE {t['roe'] * 100:+.1f}%) | bot bakiyesi {self.equity():.2f}")

    # ================================================================ işlemler
    def _close(self, reason: str) -> None:
        p = self.state["position"]
        self.ex.cancel_protection(self.symbol)
        ex_pos = self.ex.get_position(self.symbol)
        fill = self.ex.close_position(self.symbol, ex_pos) if ex_pos else None
        time.sleep(0.5 if not isinstance(self.ex, PaperFutures) else 0)
        res = self.ex.closed_result(self.symbol, p["side"], p["opened_ms"], p["entry_fee"])
        if res:
            self._record(p["side"], p["entry"], res.exit_price, p["qty"], res.pnl, res.fees, res.funding, reason)
        else:  # borsa sonucu okunamadıysa yaklaşık hesap
            px = fill.price if fill else self.ex.fetch_price(self.symbol)
            fee = (fill.fee if fill else 0) + p["entry_fee"]
            pnl = p["side"] * (px - p["entry"]) * p["qty"] - fee
            self._record(p["side"], p["entry"], px, p["qty"], pnl, fee, 0.0, reason + " (yaklaşık)")

    def _on_exchange_close(self, price: float) -> None:
        """Pozisyon borsada kapanmış (stop-loss / kâr-al / likidasyon). Kalan emirleri temizle, kaydet."""
        p = self.state["position"]
        self.ex.cancel_protection(self.symbol)
        res = self.ex.closed_result(self.symbol, p["side"], p["opened_ms"], p["entry_fee"])
        exit_px = res.exit_price if res else price
        if p["take_profit"] and abs(exit_px - p["take_profit"]) < abs(exit_px - p["stop"]):
            reason = "kâr-al"
        else:
            reason = "stop-loss"
        if res:
            self._record(p["side"], p["entry"], exit_px, p["qty"], res.pnl, res.fees, res.funding, reason)
        else:
            pnl = p["side"] * (exit_px - p["entry"]) * p["qty"] - 2 * p["entry_fee"]
            self._record(p["side"], p["entry"], exit_px, p["qty"], pnl, 2 * p["entry_fee"], 0.0,
                         reason + " (yaklaşık)")

    def _open(self, side: int, price: float, atr: float) -> None:
        eq = self.equity()
        available = min(self.ex.available_balance(), eq)
        plan = self.risk.plan_entry(eq, available, price, side, atr)
        if not plan:
            self.log.info("Risk planı oluşturulamadı (bakiye / stop kontrolü), işlem atlandı.")
            return
        qty = self.ex.round_qty(self.symbol, plan.qty)
        notional, min_n = qty * price, self.ex.min_notional(self.symbol)
        if qty <= 0 or qty < self.ex.min_qty(self.symbol) or notional < min_n:
            self.log.warning(f"Pozisyon büyüklüğü {notional:.2f} USDT, borsanın minimumu {min_n:.2f} USDT. "
                             f"İşlem atlandı → capital'i veya margin_per_trade_pct'i artırın ya da başka sembol seçin.")
            return

        opened_ms = now_ms() - 2000
        fill = self.ex.open_position(self.symbol, side, qty)
        stop, tp = self.risk.protection_prices(fill.price, side, atr)
        stop = self.ex.round_price(self.symbol, stop)
        tp = self.ex.round_price(self.symbol, tp) if tp else 0.0
        self.state["position"] = {"side": side, "qty": fill.qty, "entry": fill.price, "entry_fee": fill.fee,
                                  "margin": fill.qty * fill.price / self.cfg["futures"]["leverage"],
                                  "stop": stop, "take_profit": tp, "opened": _now(), "opened_ms": opened_ms}
        self.save()   # pozisyon açıldı bilgisini hemen diske yaz
        try:
            self.ex.place_protection(self.symbol, Position(side, fill.qty, fill.price), stop, tp)
        except Exception as e:
            # Korumasız pozisyon bırakmak kabul edilemez: hemen kapat.
            self.log.exception("Stop / kâr-al emri konamadı!")
            self.notify.send(f"UYARI: Koruma emri konamadı ({e}). Pozisyon güvenlik için kapatılıyor.")
            self._close("koruma emri hatası")
            return
        name = "LONG" if side == LONG else "SHORT"
        self.notify.send(f"AÇILDI {name} {fill.qty} @ {fill.price:.2f} | teminat "
                         f"{self.state['position']['margin']:.2f} USDT | stop {stop:.2f}"
                         + (f" | kâr-al {tp:.2f}" if tp else "") + f" | likidasyon ~{plan.liquidation:.0f}")

    def _halt(self, reason: str, equity: float) -> None:
        self.state["halted"], self.state["halt_reason"] = True, reason
        path = write_report(self.state, self.cfg, reason, equity)
        self.state["last_report"] = str(path)
        self.save()
        self.notify.send(f"BOT DURDU: {reason}. Rapor: {path}. İnceleyip güncelledikten sonra: python main.py resume")

    # ================================================================ tek tur
    def step(self) -> bool:
        """Bir kontrol turu. Bot durmalıysa False döner."""
        st = self.state
        if st["halted"]:
            self.notify.send(f"Bot durdurulmuş durumda ({st['halt_reason']}). Rapor: {st['last_report']}. "
                             "Yeniden başlatmak için: python main.py resume")
            return False

        if self.risk.kill_switch():
            self.notify.send("STOP dosyası bulundu: açık pozisyon kapatılıp bot durduruluyor.")
            if st["position"]:
                self._close("acil durdurma")
            self.save()
            return False

        price = self.ex.fetch_price(self.symbol)
        ex_pos = self.ex.get_position(self.symbol)

        # --- borsa ile senkronizasyon
        if st["position"] and ex_pos is None:
            self._on_exchange_close(price)
        foreign = (not st["position"]) and ex_pos is not None
        if foreign and not st["warned_foreign_position"]:
            self.notify.send(f"UYARI: Hesapta {self.symbol} için botun açmadığı bir pozisyon var. "
                             "Bot bu pozisyona dokunmaz ve o kapanana kadar yeni işlem açmaz.")
            st["warned_foreign_position"] = True
        if not foreign:
            st["warned_foreign_position"] = False

        # --- gün başı ve devre kesici
        eq = self.equity(ex_pos if st["position"] else None)
        today = datetime.now(timezone.utc).date().isoformat()
        if st["day"] != today:
            st["day"], st["day_start_equity"] = today, eq

        if self.risk.drawdown_hit(st["session"]["baseline"], eq):
            if st["position"]:
                self._close("devre kesici")
            eq = self.equity()
            pct = self.cfg["risk"]["max_drawdown_pct"] * 100
            self._halt(f"%{pct:.0f} kayıp sınırına ulaşıldı "
                       f"({st['session']['baseline']:.2f} → {eq:.2f} USDT)", eq)
            return False

        # --- yeni kapanmış mum?
        df = self.ex.fetch_ohlcv(self.symbol, self.tf, limit=self.strategy.warmup + 100)
        closed = df.iloc[:-1]                         # son mum henüz kapanmadı
        last_ts = str(closed.index[-1])
        if last_ts != st["last_candle"]:
            st["last_candle"] = last_ts
            sig = self.strategy.generate(closed).iloc[-1]
            atr = float(sig["atr"])
            self.log.info(f"Yeni mum {last_ts} | fiyat {price:.2f} | L+ {bool(sig['long_entry'])} "
                          f"S+ {bool(sig['short_entry'])} | bakiye {eq:.2f}")

            p = st["position"]
            if p and ((p["side"] == LONG and sig["long_exit"]) or (p["side"] != LONG and sig["short_exit"])):
                self._close("sinyal")

            want = LONG if sig["long_entry"] else (-1 if (sig["short_entry"] and self.allow_short) else 0)
            if want and not st["position"] and not foreign:
                if self.risk.daily_limit_hit(st["day_start_equity"], self.equity()):
                    self.notify.send("Günlük zarar limitine ulaşıldı: bugün yeni işlem açılmayacak.")
                else:
                    self._open(want, self.ex.fetch_price(self.symbol), atr)

        self.save()
        return True

    # ================================================================ döngü
    def run(self) -> None:
        if not isinstance(self.ex, PaperFutures):
            self.ex.setup(self.symbol, self.cfg["futures"]["leverage"], self.cfg["futures"]["margin_mode"])
        self.notify.send(f"Bot başladı | mod: {self.cfg['mode']} | {self.symbol} {self.tf} | "
                         f"{self.cfg['futures']['leverage']}x isolated | bot bakiyesi {self.equity():.2f} USDT | "
                         f"devre kesici: {self.state['session']['baseline'] * (1 - self.cfg['risk']['max_drawdown_pct']):.2f}")
        errors = 0
        while True:
            try:
                if not self.step():
                    break
                errors = 0
            except KeyboardInterrupt:
                self.notify.send("Bot elle durduruldu (Ctrl+C). Açık pozisyon ve borsadaki stop emirleri korunuyor.")
                self.save()
                break
            except Exception as e:
                errors += 1
                self.log.exception(f"Hata ({errors}. kez): {e}")
                if errors >= 5:
                    self.notify.send(f"Üst üste 5 hata, bot durduruluyor (borsadaki stop emirleri yerinde). Son hata: {e}")
                    self.save()
                    break
                time.sleep(min(30 * errors, 300))
                continue
            time.sleep(self.cfg["live"]["poll_seconds"])

    # ================================================================ yeniden başlatma
    def resume(self) -> float:
        """Devre kesiciden sonra yeniden başlatma: yeni dönem, güncel bakiye yeni referans olur."""
        eq = self.equity()
        self.state.update({"halted": False, "halt_reason": None,
                           "session": {"start": _now(), "baseline": eq}})
        self.save()
        return eq
