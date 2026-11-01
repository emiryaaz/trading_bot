"""Komut satırı.

    python main.py backtest              # 10 coin, Binance Futures geçmiş verisiyle portföy testi (+ coin bazında)
    python main.py backtest --compare    # ayar varyasyonlarını karşılaştır (ROE vs ATR, long+short vs long)
    python main.py backtest --synthetic  # internet olmadan, sahte veriyle (sadece kod testi)
    python main.py layers               # her indikatör katmanının katkısı (eğitim / test ayrı)
    python main.py layers --tf 4h       # aynı test 4 saatlik mumlarla
    python main.py collect              # open interest + long/short oranlarını topla (son 30 gün)
    python main.py collect --archive 365  # + arşivden geçmişi doldurmayı dene
    python main.py analyze              # OI / long-short metrikleri gelecekteki getiriyi öngörüyor mu?
    python main.py info                  # her coinin borsa limitleri ve gereken minimum sermaye
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
from bot.config import ROOT, backtest_kwargs, futures_symbol, load_config
from bot.data import TF_MINUTES, load_history, synthetic_portfolio
from bot.risk import RiskManager
from bot.strategy import build_strategy


def _bt(cfg, df, funding, strategy=None, tf=None):
    kw = backtest_kwargs(cfg)
    return run_backtest(df, strategy or build_strategy(cfg), RiskManager(cfg["risk"], cfg["futures"]["leverage"]),
                        cfg["backtest"]["initial_balance"], kw.pop("fee_rate"), kw.pop("slippage"),
                        funding=funding, bars_per_year=525600 / TF_MINUTES[tf or cfg["exchange"]["timeframe"]], **kw)


def _print(name, res):
    print(f"\n=== {name} ===")
    for k, v in res.metrics().items():
        print(f"  {k:<30} {v}")


def _resample(data: dict, tf: str) -> dict:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return {k: d.resample(tf).agg(agg).dropna() for k, d in data.items()}


def _slice(data: dict, start=None, end=None) -> dict:
    return {k: d.loc[start:end] for k, d in data.items() if len(d.loc[start:end])}


def cmd_backtest(cfg, synthetic, refresh, split, compare):
    ex = cfg["exchange"]
    if synthetic:
        data, funding = synthetic_portfolio(ex["symbols"], timeframe=ex["timeframe"]), {}
    else:
        print("Veri yükleniyor...")
        data, funding = load_history(cfg, refresh)
    if not data:
        print("Hiç veri yok.")
        return
    idx = sorted(set().union(*[d.index for d in data.values()]))
    print(f"{len(data)} coin | {idx[0]} -> {idx[-1]}" + ("  (SENTETİK VERİ)" if synthetic else ""))
    for k, d in data.items():
        has_f = funding.get(k) is not None
        print(f"  {k:<11} {len(d):>6} mum, başlangıç {d.index[0]:%Y-%m-%d}" + ("" if has_f else "  (funding: varsayılan)"))
    f, r = cfg["futures"], cfg["risk"]
    print(f"Ayarlar: {f['leverage']}x isolated | teminat/işlem %{r['margin_per_trade_pct'] * 100:.0f} | "
          f"en fazla {r['max_open_positions']} pozisyon, toplam teminat ≤ %{r['max_total_margin_pct'] * 100:.0f} | "
          f"stop {r['stop_mode']} | short {'açık' if f['allow_short'] else 'kapalı'}")

    cut_time = idx[int(len(idx) * split)] if split else None

    if compare:
        variants = {
            "Mevcut ayarlar": {},
            "Sadece long": {"futures": {"allow_short": False}},
            "ATR stop 2x/4x (risk %1)": {"risk": {"stop_mode": "atr", "risk_per_trade_pct": 0.01}},
            "ROE %5 stop, %10 kâr-al": {"risk": {"take_profit_roe": 0.10}},
            "En fazla 1 pozisyon": {"risk": {"max_open_positions": 1}},
            "En fazla 5 pozisyon": {"risk": {"max_open_positions": 5, "max_total_margin_pct": 1.0}},
        }
        rows = []
        for name, over in variants.items():
            c = copy.deepcopy(cfg)
            for sec, vals in over.items():
                c[sec].update(vals)
            res = _bt(c, data, funding)
            m_all = res.metrics()
            m_test = res.subset(cut_time).metrics() if split else m_all
            rows.append({"Varyasyon": name, "Getiri %": m_all["Toplam getiri %"],
                         "Maks DD %": m_all["Maks. düşüş (drawdown) %"],
                         "İşlem": m_all["İşlem sayısı (long/short)"], "Kazanma %": m_all["Kazanma oranı %"],
                         "KF": m_all["Kâr faktörü"], "Test getiri %": m_test["Toplam getiri %"],
                         "Test KF": m_test["Kâr faktörü"],
                         "Devre kesici": m_all["Devre kesici tetiklenme"].split(",")[0]})
        pd.set_option("display.width", 220)
        print("\n" + pd.DataFrame(rows).set_index("Varyasyon").to_string())
        return

    # Tek simülasyon tüm veride; eğitim / test dönemleri bu simülasyondan kesilir
    # (göstergeler önceki veriyle ısınmış olur — test dönemi "soğuk" başlamaz).
    res_all = _bt(cfg, data, funding)
    parts = [("Tüm veri", res_all)]
    if split:
        parts += [(f"Eğitim (ilk %{split * 100:.0f}, {idx[0]:%Y-%m-%d} → {cut_time:%Y-%m-%d})", res_all.subset(None, cut_time)),
                  (f"Test (son %{(1 - split) * 100:.0f}, {cut_time:%Y-%m-%d} →)", res_all.subset(cut_time))]
    for name, res in parts:
        _print(name, res)
        for ev in res.breaker_events[:5]:
            print(f"    devre kesici: {ev['time']}  {ev['baseline']} → {ev['equity']}")

    pd.set_option("display.width", 220)
    print("\n=== Coin bazında (tüm veri) ===")
    print(res_all.per_symbol().to_string())

    out = ROOT / "data" / "backtest_trades.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame([t.as_dict() for t in res_all.trades]).to_csv(out, index=False)
    print(f"\nİşlem listesi (tüm veri): {out}")


LAYER_OFF = dict(use_regime=False, use_htf=False, use_btc_filter=False, trigger="cross",
                 use_macd=False, use_fib=False, use_volume=False)


def cmd_layers(cfg, split, tf):
    """Her katmanın katkısını ayrı ayrı ölçer. Kararlar EĞİTİM dönemine göre verilmeli; TEST dönemi
    yalnızca en sonda doğrulama içindir (test sonucuna bakıp ayar değiştirmek ezberlemeye yol açar)."""
    from bot import indicators as ind
    data, funding = load_history(cfg)
    base_tf = cfg["exchange"]["timeframe"]
    if tf and tf != base_tf:
        data = {k: ind.resample_ohlcv(d, tf) for k, d in data.items()}
    tf = tf or base_tf
    htf = {"1h": {"htf": "4h"}, "4h": {"htf": "1D", "htf_fast": 10, "htf_slow": 30, "fib_lookback": 30,
                                         "pull_window": 6}}.get(tf, {})
    idx = sorted(set().union(*[d.index for d in data.values()]))
    cut = idx[int(len(idx) * split)]
    ex = cfg["exchange"]
    print(f"{len(data)} coin, {tf} mumlar | eğitim: {idx[0]:%Y-%m-%d} → {cut:%Y-%m-%d} | test: {cut:%Y-%m-%d} → {idx[-1]:%Y-%m-%d}")
    print(f"Çıkış kuralı (config): stop {cfg['risk']['stop_mode']}\n")
    on = lambda **kw: {**LAYER_OFF, **htf, **kw}
    L3 = dict(use_regime=True, use_htf=True, use_btc_filter=True)
    MKT = {"entry_order": "market", "tp_order": "market"}          # sinyal katmanları piyasa emriyle ölçülür
    LIM = {"entry_order": "limit", "tp_order": "limit"}
    has_taker = all("taker_buy_volume" in d for d in data.values())
    stages = [
        ("0 Eski: SMA kesişimi", "sma_cross_rsi", {"fast": 20, "slow": 50, "rsi_period": 14, "rsi_max_entry": 70,
                                                    "rsi_min_entry": 30, "atr_period": 14}, MKT),
        ("1 + Rejim (ADX)", "multi_factor", on(use_regime=True), MKT),
        ("2 + Üst zaman dilimi yönü", "multi_factor", on(use_regime=True, use_htf=True), MKT),
        ("3 + BTC filtresi", "multi_factor", on(**L3), MKT),
        ("4 Giriş: geri çekilme (RSI)", "multi_factor", on(**L3, trigger="pullback"), MKT),
        ("5 + MACD", "multi_factor", on(**L3, trigger="pullback", use_macd=True), MKT),
        ("6 + Fibonacci", "multi_factor", on(**L3, trigger="pullback", use_macd=True, use_fib=True), MKT),
        ("7 + Hacim", "multi_factor", on(**L3, trigger="pullback", use_macd=True, use_fib=True, use_volume=True), MKT),
        ("F  1-3 + Funding filtresi", "multi_factor", on(**L3, use_funding=True), MKT),
    ]
    if has_taker:
        stages.append(("T  1-3 + Alıcı/satıcı baskısı", "multi_factor", on(**L3, use_taker_flow=True), MKT))
    else:
        print("Not: mum verisinde alıcı/satıcı (taker) hacmi yok → 'T' katmanı atlandı. "
              "Eklemek için bir kez: python main.py backtest --refresh\n")
    stages += [
        ("L  1-3 + Limit giriş ve kâr-al", "multi_factor", on(**L3), LIM),
        ("LF 1-3 + Funding + Limit", "multi_factor", on(**L3, use_funding=True), LIM),
    ]
    from bot.strategy import STRATEGIES
    rows = []
    for name, strat, params, exec_over in stages:
        c = copy.deepcopy(cfg)
        c["execution"].update(exec_over)
        res = _bt(c, data, funding, STRATEGIES[strat](**params), tf)
        tr, te = res.subset(None, cut).metrics(), res.subset(cut).metrics()
        rows.append({"Aşama": name, "Eğitim getiri %": tr["Toplam getiri %"], "Eğitim DD %": tr["Maks. düşüş (drawdown) %"],
                     "Eğitim işlem": tr["İşlem sayısı (long/short)"].split()[0], "Eğitim KF": tr["Kâr faktörü"],
                     "| Test getiri %": te["Toplam getiri %"], "Test işlem": te["İşlem sayısı (long/short)"].split()[0],
                     "Test KF": te["Kâr faktörü"]})
        print(f"  {name} tamam", flush=True)
    pd.set_option("display.width", 220)
    print("\n" + pd.DataFrame(rows).set_index("Aşama").to_string())
    print("\nNot: KF = kâr faktörü (kazanç / kayıp). 1'in üstü kârlı. 30'dan az işlemle verilen sonuçlar güvenilir değildir.")
    print("0-7, F ve T satırları piyasa emriyle; L satırları limit giriş + limit kâr-al ile ölçülür.")


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


def cmd_collect(cfg, archive_days):
    from bot.collector import METRICS_DIR, backfill_archive, collect_recent
    print(f"Open interest ve long/short oranları toplanıyor → {METRICS_DIR}")
    summary = collect_recent(cfg)
    rows = {}
    for (coin, metric), v in summary.items():
        rows.setdefault(coin, {})[metric] = v
    pd.set_option("display.width", 220)
    print("\nEklenen yeni satır sayısı (1 saatlik kayıt):")
    print(pd.DataFrame(rows).T.to_string())
    if archive_days:
        print(f"\nArşivden son {archive_days} gün deneniyor (data.binance.vision)...")
        backfill_archive(cfg, archive_days)
    print("\nNot: Binance bu veriyi sadece son 30 gün için veriyor. Bot sürekli çalışmıyorsa bu komutu "
          "en az 3 haftada bir çalıştırın.")


def cmd_analyze(cfg, split, horizon):
    """Open interest / long-short / taker metriklerinin sonraki `horizon` saatlik getiriyle ilişkisi.
    Sadece EĞİTİM döneminde ölçülür. Tüm coinler birlikte değerlendirilir (her coin kendi içinde sıralanır)."""
    import numpy as np
    from bot.collector import METRICS_DIR
    data, _ = load_history(cfg)
    names = ["oi", "oi_value", "top_acct_ratio", "top_pos_ratio", "global_ratio", "taker_ratio"]
    pooled, coins = [], []
    for sym, px in data.items():
        path = METRICS_DIR / f"{sym.replace('/', '')}_archive.csv"
        if not path.exists():
            continue
        a = pd.read_csv(path, index_col=0, parse_dates=True)
        a.columns = names[:len(a.columns)]
        df = px.join(a, how="inner")
        cut = df.index[int(len(df) * split)]
        tr = df.loc[:cut]
        c = tr["close"]
        feats = pd.DataFrame({
            "OI 24s değişim": tr["oi"].pct_change(24),
            "OI değişimi x fiyat yönü": tr["oi"].pct_change(24) * np.sign(c.pct_change(24)),
            "Büyük trader pozisyon L/S": tr["top_pos_ratio"],
            "Büyük trader hesap L/S": tr["top_acct_ratio"],
            "Tüm hesaplar L/S": tr["global_ratio"],
            "Taker alış/satış (24s ort.)": tr["taker_ratio"].rolling(24).mean(),
            "(referans) son 24s getiri": c.pct_change(24),
        })
        ranked = feats.rank(pct=True)                   # her coin kendi içinde sıralanır
        ranked["fwd"] = c.shift(-horizon) / c - 1
        pooled.append(ranked.iloc[::horizon].dropna())  # örtüşmeyen pencereler
        coins.append(sym)
    if not pooled:
        print("Arşiv verisi yok. Önce: python main.py collect --archive 365")
        return
    d = pd.concat(pooled)
    n = len(d)
    rows = []
    for col in [c for c in d.columns if c != "fwd"]:
        q = pd.qcut(d[col], 5, labels=False, duplicates="drop")
        g = d.groupby(q)["fwd"].mean() * 100
        rows.append({"Metrik": col, "Sıra korelasyonu": round(d[col].corr(d["fwd"], method="spearman"), 3),
                     f"En düşük %20 → {horizon}s getiri %": round(g.iloc[0], 2),
                     f"En yüksek %20 → {horizon}s getiri %": round(g.iloc[-1], 2)})
    pd.set_option("display.width", 220)
    print(f"{len(coins)} coin ({', '.join(c.split('/')[0] for c in coins)}) | eğitim dönemi | {n} gözlem\n")
    print(pd.DataFrame(rows).set_index("Metrik").to_string())
    print(f"\nKaba anlamlılık eşiği: |korelasyon| > {2 / np.sqrt(n):.3f}. Bunun altındakiler büyük olasılıkla gürültüdür.")


def cmd_info(cfg):
    from bot.exchange import FuturesExchange
    ex = FuturesExchange(cfg["exchange"]["name"])
    r, lev, cap = cfg["risk"], cfg["futures"]["leverage"], cfg["live"]["capital"]
    syms = [futures_symbol(s) for s in cfg["exchange"]["symbols"]]
    prices = ex.fetch_prices([s for s in syms if s in ex.ex.markets])
    print(f"Ayarlar: capital {cap} USDT | teminat/işlem %{r['margin_per_trade_pct'] * 100:.0f} | {lev}x "
          f"→ işlem başına pozisyon ~{cap * r['margin_per_trade_pct'] * lev:.2f} USDT\n")
    print(f"{'Coin':<16}{'Fiyat':>12}{'Min. pozisyon':>16}{'Gereken capital':>18}  Durum")
    worst = 0.0
    for s in syms:
        if s not in ex.ex.markets:
            print(f"{s:<16}{'-':>12}{'-':>16}{'-':>18}  ✗ Binance Futures'ta bulunamadı")
            continue
        px = prices.get(s) or ex.fetch_price(s)
        need_notional = max(ex.min_notional(s), ex.min_qty(s) * px)
        need = need_notional / (r["margin_per_trade_pct"] * lev) * 1.05
        worst = max(worst, need)
        ok = "✓" if cap >= need else "✗ capital yetersiz"
        print(f"{s:<16}{px:>12g}{need_notional:>13.2f} $ {need:>15.0f} $  {ok}")
    print(f"\nTüm coinlerde işlem yapabilmek için gereken minimum capital: ~{worst:.0f} USDT")


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
    pos = st.get("positions") or {}
    print(f"Açık pozisyonlar ({len(pos)}):")
    for sym, p in pos.items():
        print(f"  {sym.split(':')[0]:<10} {'LONG' if p['side'] == 1 else 'SHORT':<5} {p['qty']} @ {p['entry']:g} | "
              f"stop {p['stop']:g} | kâr-al {p['take_profit']:g} | teminat {p['margin']:.2f}")
    trades = st.get("trades", [])
    print(f"Toplam işlem: {len(trades)} | Toplam K/Z: {sum(t['pnl'] for t in trades):+.2f} USDT")
    for t in trades[-10:]:
        print(f"  {t['closed'][:16]} {t.get('symbol', ''):<10} {t['side']:<5} {t['entry']:g} → {t['exit']:g}  "
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
    ap.add_argument("command", choices=["backtest", "layers", "collect", "analyze", "info", "run", "status", "report",
                                        "resume"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true", help="sahte veriyle backtest")
    ap.add_argument("--refresh", action="store_true", help="geçmiş veriyi yeniden indir")
    ap.add_argument("--compare", action="store_true", help="ayar varyasyonlarını karşılaştır")
    ap.add_argument("--split", type=float, default=0.7, help="eğitim/test ayrımı (0 = ayırma)")
    ap.add_argument("--tf", default=None, help="layers için zaman dilimi (ör. 4h; 1s veriden üretilir)")
    ap.add_argument("--archive", type=int, default=0, help="collect: arşivden kaç günlük geçmiş denensin")
    ap.add_argument("--horizon", type=int, default=24, help="analyze: kaç saatlik ileri getiri")
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"backtest": lambda: cmd_backtest(cfg, a.synthetic, a.refresh, a.split, a.compare),
     "layers": lambda: cmd_layers(cfg, a.split or 0.7, a.tf),
     "collect": lambda: cmd_collect(cfg, a.archive),
     "analyze": lambda: cmd_analyze(cfg, a.split or 0.7, a.horizon),
     "info": lambda: cmd_info(cfg), "run": lambda: cmd_run(cfg), "status": lambda: cmd_status(cfg),
     "report": lambda: cmd_report(cfg), "resume": lambda: cmd_resume(cfg)}[a.command]()


if __name__ == "__main__":
    main()
