"""Komut satırı.

    python main.py backtest              # Binance Futures geçmiş verisiyle strateji testi (%70 eğitim / %30 test)
    python main.py backtest --compare    # ayar varyasyonlarını karşılaştır (ROE vs ATR, long+short vs long)
    python main.py backtest --synthetic  # internet olmadan, sahte veriyle (sadece kod testi)
    python main.py info                  # sembolün borsa limitleri ve gereken minimum sermaye
    python main.py run                   # config.yaml'daki moda göre botu başlat (paper/testnet/live)
    python main.py status                # durum, açık pozisyon ve son işlemler
    python main.py report                # şu anki dönemin raporunu üret
    python main.py resume                # devre kesiciden sonra botu yeniden etkinleştir
"""
from __future__ import annotations

import argparse
import copy
import json

import pandas as pd

from bot.backtest import run_backtest
from bot.config import ROOT, futures_symbol, load_config
from bot.data import TF_MINUTES, load_history, synthetic_ohlcv
from bot.risk import RiskManager
from bot.strategy import build_strategy


def _bt(cfg, df, funding):
    ex, fut = cfg["exchange"], cfg["futures"]
    return run_backtest(df, build_strategy(cfg), RiskManager(cfg["risk"], fut["leverage"]),
                        cfg["backtest"]["initial_balance"], ex["fee_rate"], ex["slippage"],
                        allow_short=fut["allow_short"], funding=funding,
                        funding_fallback=fut["funding_rate_fallback"],
                        bars_per_year=525600 / TF_MINUTES[ex["timeframe"]])


def _print(name, res):
    print(f"\n=== {name} ===")
    for k, v in res.metrics().items():
        print(f"  {k:<30} {v}")


def cmd_backtest(cfg, synthetic, refresh, split, compare):
    if synthetic:
        df, funding = synthetic_ohlcv(timeframe=cfg["exchange"]["timeframe"]), None
    else:
        df, funding = load_history(cfg, refresh)
    print(f"{len(df)} mum | {df.index[0]} -> {df.index[-1]}"
          + ("  (SENTETİK VERİ)" if synthetic else "")
          + ("" if funding is not None else "  | funding: varsayılan oran"))
    f = cfg["futures"]
    r = cfg["risk"]
    print(f"Ayarlar: {f['leverage']}x isolated | teminat/işlem %{r['margin_per_trade_pct'] * 100:.0f} | "
          f"stop {r['stop_mode']} | short {'açık' if f['allow_short'] else 'kapalı'}")

    if compare:
        variants = {
            "Mevcut ayarlar": {},
            "Sadece long": {"futures": {"allow_short": False}},
            "ATR stop (2x / 4x ATR)": {"risk": {"stop_mode": "atr"}},
            "ROE %5 stop, %10 kâr-al": {"risk": {"take_profit_roe": 0.10}},
            "4 saatlik mumlar": {"exchange": {"timeframe": "4h"}},
        }
        rows = []
        for name, over in variants.items():
            c = copy.deepcopy(cfg)
            for sec, vals in over.items():
                c[sec].update(vals)
            d = df
            if c["exchange"]["timeframe"] != cfg["exchange"]["timeframe"]:
                d = df.resample(c["exchange"]["timeframe"]).agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
            cut = int(len(d) * split) if split else 0
            m_all = _bt(c, d, funding).metrics()
            m_test = _bt(c, d.iloc[cut:], funding).metrics() if split else m_all
            rows.append({"Varyasyon": name, "Getiri %": m_all["Toplam getiri %"],
                         "Maks DD %": m_all["Maks. düşüş (drawdown) %"],
                         "İşlem": m_all["İşlem sayısı (long/short)"], "Kazanma %": m_all["Kazanma oranı %"],
                         "Kâr faktörü": m_all["Kâr faktörü"], "Test dönemi getiri %": m_test["Toplam getiri %"],
                         "Test KF": m_test["Kâr faktörü"],
                         "Devre kesici": m_all["Devre kesici (%10) tetiklenme"].split(",")[0]})
        pd.set_option("display.width", 200)
        print("\n" + pd.DataFrame(rows).set_index("Varyasyon").to_string())
        return

    parts = [("Tüm veri", df)]
    if split:
        cut = int(len(df) * split)
        parts += [(f"Eğitim (ilk %{split * 100:.0f})", df.iloc[:cut]),
                  (f"Test (son %{(1 - split) * 100:.0f})", df.iloc[cut:])]
    res_all = None
    for name, part in parts:
        res = _bt(cfg, part, funding)
        res_all = res_all or res
        _print(name, res)
        for ev in res.breaker_events[:5]:
            print(f"    devre kesici: {ev['time']}  {ev['baseline']} → {ev['equity']}")

    out = ROOT / "data" / "backtest_trades.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame([t.as_dict() for t in res_all.trades]).to_csv(out, index=False)
    print(f"\nİşlem listesi (tüm veri): {out}")


def _trader(cfg):
    from bot.exchange import build_exchange
    from bot.live import Trader
    from bot.logger import Notifier
    sec = cfg["secrets"]
    return Trader(cfg, build_exchange(cfg), build_strategy(cfg),
                  RiskManager(cfg["risk"], cfg["futures"]["leverage"]),
                  Notifier(sec["telegram_token"], sec["telegram_chat_id"]))


def cmd_run(cfg):
    if cfg["mode"] == "live":
        print(f"!!! GERÇEK PARA MODU !!!  Bot {cfg['live']['capital']} USDT ile "
              f"{cfg['futures']['leverage']}x isolated işlem yapacak. Devam etmek için 'EVET' yazın:")
        if input().strip() != "EVET":
            print("İptal edildi.")
            return
    _trader(cfg).run()


def cmd_info(cfg):
    from bot.exchange import FuturesExchange
    ex = FuturesExchange(cfg["exchange"]["name"])
    sym = futures_symbol(cfg["exchange"]["symbol"])
    price = ex.fetch_price(sym)
    min_n, min_q = ex.min_notional(sym), ex.min_qty(sym)
    r, lev = cfg["risk"], cfg["futures"]["leverage"]
    need = max(min_n, min_q * price) / (r["margin_per_trade_pct"] * lev)
    print(f"{sym}  fiyat {price:.2f}")
    print(f"  Borsa minimum pozisyon: {min_n} USDT, minimum miktar: {min_q}")
    print(f"  Mevcut ayarlarla (teminat %{r['margin_per_trade_pct'] * 100:.0f}, {lev}x) gereken minimum capital: "
          f"~{need * 1.05:.0f} USDT")
    print(f"  config.yaml → live.capital = {cfg['live']['capital']} USDT "
          + ("✓" if cfg["live"]["capital"] >= need * 1.05 else "✗ YETERSİZ: capital'i artırın veya başka sembol seçin"))


def cmd_status(cfg):
    p = ROOT / cfg["live"]["state_file"]
    if not p.exists():
        print("Henüz kayıtlı durum yok.")
        return
    st = json.loads(p.read_text(encoding="utf-8"))
    eq = cfg["live"]["capital"] + st.get("realized", 0)
    base = st["session"]["baseline"]
    print(f"Durum: {'DURDURULDU - ' + str(st['halt_reason']) if st.get('halted') else 'aktif'}")
    print(f"Bot bakiyesi (gerçekleşmiş): {eq:.2f} USDT | dönem başı {base:.2f} | "
          f"devre kesici seviyesi {base * (1 - cfg['risk']['max_drawdown_pct']):.2f}")
    print("Açık pozisyon:", st.get("position"))
    trades = st.get("trades", [])
    print(f"Toplam işlem: {len(trades)} | Toplam K/Z: {sum(t['pnl'] for t in trades):+.2f} USDT")
    for t in trades[-10:]:
        print(f"  {t['closed'][:16]} {t['side']:<5} {t['entry']:.2f} → {t['exit']:.2f}  "
              f"{t['pnl']:+.2f} ({t['roe'] * 100:+.1f}%)  {t['reason']}")
    if st.get("last_report"):
        print("Son rapor:", st["last_report"])


def cmd_report(cfg):
    from bot.report import write_report
    t = _trader(cfg)
    print("Rapor:", write_report(t.state, cfg, "Kullanıcı isteğiyle ara rapor", t.equity()))


def cmd_resume(cfg):
    t = _trader(cfg)
    if not t.state["halted"]:
        print("Bot zaten durdurulmuş değil.")
        return
    print(f"Durma nedeni: {t.state['halt_reason']}\nRapor: {t.state['last_report']}")
    print("Ayarları güncelleyip backtest ile test ettiniz mi? Yeniden etkinleştirmek için 'EVET' yazın:")
    if input().strip() != "EVET":
        print("İptal edildi.")
        return
    eq = t.resume()
    print(f"Bot yeniden etkin. Yeni referans bakiye {eq:.2f} USDT, devre kesici "
          f"{eq * (1 - cfg['risk']['max_drawdown_pct']):.2f} USDT. Başlatmak için: python main.py run")


def main():
    ap = argparse.ArgumentParser(description="Finans İşlem Botu (Binance Futures)")
    ap.add_argument("command", choices=["backtest", "info", "run", "status", "report", "resume"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true", help="sahte veriyle backtest")
    ap.add_argument("--refresh", action="store_true", help="geçmiş veriyi yeniden indir")
    ap.add_argument("--compare", action="store_true", help="ayar varyasyonlarını karşılaştır")
    ap.add_argument("--split", type=float, default=0.7, help="eğitim/test ayrımı (0 = ayırma)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"backtest": lambda: cmd_backtest(cfg, a.synthetic, a.refresh, a.split, a.compare),
     "info": lambda: cmd_info(cfg), "run": lambda: cmd_run(cfg), "status": lambda: cmd_status(cfg),
     "report": lambda: cmd_report(cfg), "resume": lambda: cmd_resume(cfg)}[a.command]()


if __name__ == "__main__":
    main()
