"""Canlı / paper / testnet işlem döngüsü — çoklu coin, vadeli işlemler, isolated.

Her turda:
  1. Bot devre kesiciyle durdurulmuşsa çalışmaz (önce rapor incelenir, sonra `python main.py resume`).
  2. STOP dosyası varsa TÜM açık pozisyonlar kapatılır ve bot durur.
  3. Tüm coinlerin fiyatı ve pozisyonları tek istekte okunur; borsada stop / kâr-al ile kapanan
     pozisyonların sonucu kaydedilir.
  4. DEVRE KESİCİ: toplam bot bakiyesi son başlatmaya göre %10 düştüyse tüm pozisyonlar kapatılır,
     rapor yazılır ve bot durur.
  5. Her coin için yeni bir mum KAPANDIYSA strateji çalışır. Önce çıkışlar, sonra girişler
     (config sırasına göre öncelik, en fazla `max_open_positions` pozisyon, toplam teminat sınırı).
     Her yeni pozisyonla birlikte stop-loss ve kâr-al emirleri hemen BORSAYA konur.
  6. Durum data/state.json'a kaydedilir.

Bot yalnızca KENDİ açtığı pozisyonları ve kendisine ayrılan `capital` kadar bakiyeyi yönetir.
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


def _name(side: int) -> str:
    return "LONG" if side == LONG else "SHORT"


class Trader:
    def __init__(self, cfg: dict, exchange: FuturesClient, strategy: Strategy,
                 risk: RiskManager, notifier: Notifier | None = None, state_path: Path | None = None):
        self.cfg, self.ex, self.strategy, self.risk = cfg, exchange, strategy, risk
        self.symbols = [futures_symbol(s) for s in cfg["exchange"]["symbols"]]
        self.tf = cfg["exchange"]["timeframe"]
        self.capital = float(cfg["live"]["capital"])
        self.lev = cfg["futures"]["leverage"]
        self.allow_short = cfg["futures"].get("allow_short", True)
        self.max_pos = cfg["risk"].get("max_open_positions", 1)
        self.max_margin = cfg["risk"].get("max_total_margin_pct", 1.0)
        self.log = get_logger()
        self.notify = notifier or Notifier()
        self.state_path = state_path or ROOT / cfg["live"]["state_file"]
        self.state = self._load_state()
        self._ex_pos: dict[str, Position] = {}

    # ================================================================ durum
    def _load_state(self) -> dict:
        default = {"positions": {}, "last_candle": {}, "day": None, "day_start_equity": None,
                   "realized": 0.0, "trades": [], "halted": False, "halt_reason": None,
                   "last_report": None, "session": {"start": _now(), "baseline": self.capital},
                   "paper": None, "warned_foreign": []}
        if self.state_path.exists():
            st = {**default, **json.loads(self.state_path.read_text(encoding="utf-8"))}
            if "position" in st:                       # v0.3 (tek coin) durum dosyasından geçiş
                old = st.pop("position")
                if old:
                    st["positions"][futures_symbol(self.cfg["exchange"]["symbols"][0])] = old
                if not isinstance(st.get("last_candle"), dict):
                    st["last_candle"] = {}
            if isinstance(self.ex, PaperFutures) and st.get("paper") and isinstance(st["paper"].get("pos"), dict):
                self.ex.load_state(st["paper"])
            self.log.info(f"Kayıtlı durum yüklendi. Açık pozisyonlar: {list(st['positions'])}")
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
    @property
    def positions(self) -> dict:
        return self.state["positions"]

    def equity(self) -> float:
        """Bot bakiyesi = ayrılan sermaye + gerçekleşmiş K/Z + botun açık pozisyonlarının anlık K/Z'si."""
        unreal = sum(self._ex_pos[s].unrealized for s in self.positions if s in self._ex_pos)
        return self.capital + self.state["realized"] + unreal

    def used_margin(self) -> float:
        return sum(p["margin"] for p in self.positions.values())

    def _record(self, sym: str, exit_px: float, pnl: float, fees: float, funding: float, reason: str) -> None:
        p = self.positions.pop(sym)
        self._ex_pos.pop(sym, None)
        t = {"symbol": sym.split(":")[0], "side": _name(p["side"]), "entry": p["entry"], "exit": exit_px,
             "qty": p["qty"], "margin": p["margin"], "pnl": round(pnl, 6),
             "roe": pnl / p["margin"] if p["margin"] else 0.0, "fees": round(fees, 6),
             "funding": round(funding, 6), "reason": reason, "opened": p["opened"], "closed": _now()}
        self.state["trades"].append(t)
        self.state["realized"] += pnl
        self.notify.send(f"KAPANDI {t['symbol']} ({reason}) {t['side']} @ {exit_px:g} | "
                         f"K/Z {pnl:+.2f} USDT (ROE {t['roe'] * 100:+.1f}%) | bot bakiyesi {self.equity():.2f}")

    # ================================================================ işlemler
    def _close(self, sym: str, reason: str) -> None:
        p = self.positions[sym]
        self.ex.cancel_protection(sym)
        ex_pos = self.ex.get_position(sym)
        fill = self.ex.close_position(sym, ex_pos) if ex_pos else None
        if not isinstance(self.ex, PaperFutures):
            time.sleep(0.5)                            # borsanın işlem kaydını oluşturması için
        res = self.ex.closed_result(sym, p["side"], p["opened_ms"], p["entry_fee"])
        if res:
            self._record(sym, res.exit_price, res.pnl, res.fees, res.funding, reason)
        else:
            px = fill.price if fill else self.ex.fetch_price(sym)
            fee = (fill.fee if fill else 0) + p["entry_fee"]
            pnl = p["side"] * (px - p["entry"]) * p["qty"] - fee
            self._record(sym, px, pnl, fee, 0.0, reason + " (yaklaşık)")

    def _on_exchange_close(self, sym: str, price: float) -> None:
        """Pozisyon borsada kapanmış (stop-loss / kâr-al / likidasyon). Kalan emirleri temizle, kaydet."""
        p = self.positions[sym]
        self.ex.cancel_protection(sym)
        res = self.ex.closed_result(sym, p["side"], p["opened_ms"], p["entry_fee"])
        exit_px = res.exit_price if res else price
        near_tp = p["take_profit"] and abs(exit_px - p["take_profit"]) < abs(exit_px - p["stop"])
        reason = "kâr-al" if near_tp else "stop-loss"
        if res:
            self._record(sym, exit_px, res.pnl, res.fees, res.funding, reason)
        else:
            pnl = p["side"] * (exit_px - p["entry"]) * p["qty"] - 2 * p["entry_fee"]
            self._record(sym, exit_px, pnl, 2 * p["entry_fee"], 0.0, reason + " (yaklaşık)")

    def _open(self, sym: str, side: int, price: float, atr: float) -> None:
        eq = self.equity()
        used = self.used_margin()
        room = max(0.0, eq * self.max_margin - used)
        available = min(self.ex.available_balance(), eq - used, room)
        plan = self.risk.plan_entry(eq, available, price, side, atr)
        if not plan:
            self.log.info(f"{sym}: risk planı oluşturulamadı (teminat sınırı / stop kontrolü), atlandı.")
            return
        qty = self.ex.round_qty(sym, plan.qty)
        notional, min_n = qty * price, self.ex.min_notional(sym)
        if qty <= 0 or qty < self.ex.min_qty(sym) or notional < min_n:
            self.log.warning(f"{sym}: pozisyon {notional:.2f} USDT, borsa minimumu {min_n:.2f} USDT → atlandı. "
                             "capital'i / margin_per_trade_pct'i artırın veya bu coini listeden çıkarın.")
            return

        opened_ms = now_ms() - 2000
        fill = self.ex.open_position(sym, side, qty)
        stop, tp = self.risk.protection_prices(fill.price, side, atr)
        stop = self.ex.round_price(sym, stop)
        tp = self.ex.round_price(sym, tp) if tp else 0.0
        self.positions[sym] = {"side": side, "qty": fill.qty, "entry": fill.price, "entry_fee": fill.fee,
                               "margin": fill.qty * fill.price / self.lev, "stop": stop, "take_profit": tp,
                               "opened": _now(), "opened_ms": opened_ms}
        self._ex_pos[sym] = Position(side, fill.qty, fill.price)
        self.save()                                    # pozisyon bilgisini hemen diske yaz
        try:
            self.ex.place_protection(sym, Position(side, fill.qty, fill.price), stop, tp)
        except Exception as e:
            self.log.exception(f"{sym}: stop / kâr-al emri konamadı!")
            self.notify.send(f"UYARI: {sym} koruma emri konamadı ({e}). Pozisyon güvenlik için kapatılıyor.")
            self._close(sym, "koruma emri hatası")
            return
        self.notify.send(f"AÇILDI {sym.split(':')[0]} {_name(side)} {fill.qty} @ {fill.price:g} | teminat "
                         f"{self.positions[sym]['margin']:.2f} USDT | stop {stop:g}"
                         + (f" | kâr-al {tp:g}" if tp else "")
                         + f" | açık pozisyon {len(self.positions)}/{self.max_pos}")

    def _close_all(self, reason: str) -> None:
        for sym in list(self.positions):
            self._close(sym, reason)

    def _halt(self, reason: str) -> None:
        eq = self.equity()
        self.state["halted"], self.state["halt_reason"] = True, reason
        path = write_report(self.state, self.cfg, reason, eq)
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
            self.notify.send("STOP dosyası bulundu: tüm açık pozisyonlar kapatılıp bot durduruluyor.")
            self._close_all("acil durdurma")
            self.save()
            return False

        prices = self.ex.fetch_prices(self.symbols)
        self._ex_pos = self.ex.get_positions(self.symbols)

        # --- borsa ile senkronizasyon
        for sym in list(self.positions):
            if sym not in self._ex_pos:
                self._on_exchange_close(sym, prices.get(sym, self.positions[sym]["entry"]))
        foreign = {s for s in self._ex_pos if s not in self.positions}
        for s in foreign - set(st["warned_foreign"]):
            self.notify.send(f"UYARI: Hesapta {s} için botun açmadığı bir pozisyon var. "
                             "Bot ona dokunmaz ve o kapanana kadar bu coinde işlem açmaz.")
        st["warned_foreign"] = sorted(foreign)

        # --- gün başı ve devre kesici
        eq = self.equity()
        today = datetime.now(timezone.utc).date().isoformat()
        if st["day"] != today:
            st["day"], st["day_start_equity"] = today, eq

        if self.risk.drawdown_hit(st["session"]["baseline"], eq):
            base = st["session"]["baseline"]
            self._close_all("devre kesici")
            pct = self.cfg["risk"]["max_drawdown_pct"] * 100
            self._halt(f"%{pct:.0f} kayıp sınırına ulaşıldı ({base:.2f} → {self.equity():.2f} USDT)")
            return False

        # --- her coin için yeni kapanmış mum?
        signals = {}
        for sym in self.symbols:
            df = self.ex.fetch_ohlcv(sym, self.tf, limit=self.strategy.warmup + 100)
            closed = df.iloc[:-1]                      # son mum henüz kapanmadı
            if len(closed) < self.strategy.warmup:
                continue
            ts = str(closed.index[-1])
            if ts == st["last_candle"].get(sym):
                continue
            st["last_candle"][sym] = ts
            signals[sym] = self.strategy.generate(closed).iloc[-1]

        if signals:
            self.log.info("Yeni mum | " + " ".join(
                f"{s.split('/')[0]}:{'L' if r['long_entry'] else 'S' if r['short_entry'] else '-'}"
                for s, r in signals.items()) + f" | bakiye {self.equity():.2f} | açık {len(self.positions)}")

            # önce çıkışlar
            for sym, r in signals.items():
                p = self.positions.get(sym)
                if p and ((p["side"] == LONG and r["long_exit"]) or (p["side"] != LONG and r["short_exit"])):
                    self._close(sym, "sinyal")

            # sonra girişler (config sırası öncelik)
            wants = [(s, LONG if r["long_entry"] else -1) for s, r in signals.items()
                     if r["long_entry"] or (r["short_entry"] and self.allow_short)]
            wants = [(s, side) for s, side in wants if s not in self.positions and s not in foreign]
            if wants:
                if self.risk.daily_limit_hit(st["day_start_equity"], self.equity()):
                    self.notify.send("Günlük zarar limitine ulaşıldı: bugün yeni işlem açılmayacak.")
                else:
                    for sym, side in wants:
                        if len(self.positions) >= self.max_pos:
                            self.log.info(f"{sym}: {_name(side)} sinyali var ama pozisyon sınırı dolu "
                                          f"({self.max_pos}).")
                            continue
                        self._open(sym, side, self.ex.fetch_price(sym), float(signals[sym]["atr"]))

        self.save()
        return True

    # ================================================================ döngü
    def run(self) -> None:
        if not isinstance(self.ex, PaperFutures):
            for sym in self.symbols:
                self.ex.setup(sym, self.lev, self.cfg["futures"]["margin_mode"])
        lim = self.state["session"]["baseline"] * (1 - self.cfg["risk"]["max_drawdown_pct"])
        self.notify.send(f"Bot başladı | mod: {self.cfg['mode']} | {len(self.symbols)} coin {self.tf} | "
                         f"{self.lev}x isolated | en fazla {self.max_pos} pozisyon | bot bakiyesi "
                         f"{self.equity():.2f} USDT | devre kesici {lim:.2f}")
        errors = 0
        while True:
            try:
                if not self.step():
                    break
                errors = 0
            except KeyboardInterrupt:
                self.notify.send("Bot elle durduruldu (Ctrl+C). Açık pozisyonlar ve borsadaki stop emirleri korunuyor.")
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
