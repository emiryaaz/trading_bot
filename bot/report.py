"""Devre kesici (veya istek üzerine) işlem raporu üretir: reports/rapor_YYYYmmdd_HHMM.md"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import ROOT



def _stats(trades: list[dict]) -> dict:
    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gw, gl = sum(t["pnl"] for t in wins), -sum(t["pnl"] for t in losses)
    by_reason: dict[str, int] = {}
    for t in trades:
        by_reason[t["reason"]] = by_reason.get(t["reason"], 0) + 1
    longs = [t for t in trades if t["side"] == "LONG"]
    shorts = [t for t in trades if t["side"] == "SHORT"]
    # en uzun kayıp serisi
    streak = best = 0
    for t in trades:
        streak = streak + 1 if t["pnl"] <= 0 else 0
        best = max(best, streak)
    return {
        "n": n, "wins": len(wins), "win_rate": len(wins) / n * 100 if n else 0.0,
        "pnl": sum(t["pnl"] for t in trades), "fees": sum(t.get("fees", 0) for t in trades),
        "funding": sum(t.get("funding", 0) for t in trades),
        "pf": gw / gl if gl > 0 else float("inf") if gw else 0.0,
        "avg_win": gw / len(wins) if wins else 0.0, "avg_loss": -gl / len(losses) if losses else 0.0,
        "long_n": len(longs), "long_pnl": sum(t["pnl"] for t in longs),
        "short_n": len(shorts), "short_pnl": sum(t["pnl"] for t in shorts),
        "reasons": by_reason, "max_loss_streak": best,
    }


def write_report(state: dict, cfg: dict, reason: str, equity: float) -> Path:
    report_dir = Path(os.getenv("BOT_REPORT_DIR", ROOT / "reports"))   # testler ayrı klasöre yazar
    report_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    sess = state["session"]
    trades = [t for t in state["trades"] if t["closed"] >= sess["start"]]
    s = _stats(trades)
    base = sess["baseline"]
    change = (equity - base) / base * 100 if base else 0.0

    safe_cfg = {k: v for k, v in cfg.items() if k != "secrets"}   # API anahtarları rapora yazılmaz
    lines = [
        f"# Bot Raporu: {now:%Y-%m-%d %H:%M} UTC",
        "",
        f"**Durma nedeni:** {reason}",
        "",
        "## Özet (son başlatmadan bu yana)",
        "",
        "| | |",
        "|---|---|",
        f"| Mod | {cfg['mode']} |",
        f"| Sembol / zaman dilimi | {cfg['exchange']['symbol']} / {cfg['exchange']['timeframe']} |",
        f"| Dönem | {sess['start']} → {now.isoformat(timespec='seconds')} |",
        f"| Başlangıç bakiyesi | {base:.2f} USDT |",
        f"| Güncel bakiye | {equity:.2f} USDT |",
        f"| Değişim | {change:+.2f}% |",
        f"| İşlem sayısı | {s['n']} (kazanan {s['wins']}, oran %{s['win_rate']:.1f}) |",
        f"| Long | {s['long_n']} işlem, {s['long_pnl']:+.2f} USDT |",
        f"| Short | {s['short_n']} işlem, {s['short_pnl']:+.2f} USDT |",
        f"| Kâr faktörü | {s['pf']:.2f} |",
        f"| Ort. kazanç / ort. kayıp | {s['avg_win']:+.2f} / {s['avg_loss']:+.2f} USDT |",
        f"| En uzun kayıp serisi | {s['max_loss_streak']} işlem |",
        f"| Ödenen ücret / funding | {s['fees']:.2f} / {s['funding']:.2f} USDT |",
        f"| Kapanış nedenleri | {s['reasons']} |",
        "",
        "## İşlemler",
        "",
        "| # | Yön | Açılış | Kapanış | Giriş | Çıkış | K/Z (USDT) | ROE % | Neden |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, t in enumerate(trades, 1):
        lines.append(f"| {i} | {t['side']} | {t['opened'][:16]} | {t['closed'][:16]} | {t['entry']:.2f} | "
                     f"{t['exit']:.2f} | {t['pnl']:+.2f} | {t['roe'] * 100:+.1f} | {t['reason']} |")
    lines += [
        "",
        "## Değerlendirme için sorular",
        "",
        "- Kayıplar çoğunlukla stop-loss'tan mı geliyor? Stop çok mu dar (stop_loss_roe / timeframe)?",
        "- Long ve short taraflarından biri belirgin şekilde kötü mü? (allow_short)",
        "- Kayıplar belirli saatlerde / piyasa koşullarında mı yoğunlaşıyor?",
        "- Değişiklikten sonra `python main.py backtest` ile yeni ayarları test edip ancak sonra `python main.py resume` çalıştırın.",
        "",
        "## O anki ayarlar",
        "",
        "```yaml",
        yaml.safe_dump(safe_cfg, allow_unicode=True, sort_keys=False).strip(),
        "```",
    ]
    path = report_dir / f"rapor_{now:%Y%m%d_%H%M}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
