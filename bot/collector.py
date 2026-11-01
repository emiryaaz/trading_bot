"""Vadeli piyasa metrikleri kayıt programı: open interest ve long/short oranları.

Binance bu verilerin geçmişini REST API'de yalnızca SON 30 GÜN için verir; daha eskisi API'den silinir.
Bu yüzden veriyi düzenli toplayıp kendi diskimizde biriktiriyoruz:

  python main.py collect                 # her coin için son ~30 günü indir, data/metrics/ altına ekle
  python main.py collect --archive 365   # (deneysel) data.binance.vision arşivinden geçmişi doldurmayı dene

`python main.py run` çalışırken bu toplama günde bir kez otomatik yapılır. Bot sürekli çalışmıyorsa
`collect` komutunu en az ~3 haftada bir çalıştırın (Windows Görev Zamanlayıcı ile otomatikleştirilebilir),
yoksa aradaki veri kaybolur.

Toplanan metrikler (1 saatlik):
  open_interest      : açık pozisyon miktarı (kontrat ve USDT)
  top_position_ratio : en büyük trader'ların long/short POZİSYON oranı
  top_account_ratio  : en büyük trader'ların long/short HESAP oranı
  global_ratio       : tüm hesapların long/short oranı
  taker_ratio        : agresif alıcı / agresif satıcı hacim oranı
"""
from __future__ import annotations

import io
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .config import ROOT, futures_symbol

METRICS_DIR = ROOT / "data" / "metrics"

# metrik adı -> (ccxt implicit metodu, {kaynak kolon: hedef kolon})
ENDPOINTS = {
    "open_interest": ("fapiDataGetOpenInterestHist",
                      {"sumOpenInterest": "oi", "sumOpenInterestValue": "oi_value"}),
    "top_position_ratio": ("fapiDataGetTopLongShortPositionRatio",
                           {"longShortRatio": "ratio", "longAccount": "long", "shortAccount": "short"}),
    "top_account_ratio": ("fapiDataGetTopLongShortAccountRatio",
                          {"longShortRatio": "ratio", "longAccount": "long", "shortAccount": "short"}),
    "global_ratio": ("fapiDataGetGlobalLongShortAccountRatio",
                     {"longShortRatio": "ratio", "longAccount": "long", "shortAccount": "short"}),
    "taker_ratio": ("fapiDataGetTakerlongshortRatio",
                    {"buySellRatio": "ratio", "buyVol": "buy_vol", "sellVol": "sell_vol"}),
}
PERIOD = "1h"
PAGE = 500                         # Binance sayfa başına en fazla 500 kayıt
WINDOW_DAYS = 29                   # API'nin verdiği pencere (~30 gün), güvenlik payıyla


def metric_path(symbol: str, metric: str) -> Path:
    return METRICS_DIR / f"{symbol.split(':')[0].replace('/', '')}_{metric}.csv"


def _merge_save(path: Path, new: pd.DataFrame) -> int:
    """Yeni kayıtları mevcut dosyayla birleştirir (tekrarları atar). Eklenen satır sayısını döndürür."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, index_col=0, parse_dates=True)
        before = len(old)
        both = pd.concat([old, new])
    else:
        before, both = 0, new
    both = both[~both.index.duplicated(keep="last")].sort_index()
    both.to_csv(path)
    return len(both) - before


def _rows_to_df(rows: list[dict], cols: dict) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(cols.values()))
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms", utc=True)
    df = df.set_index("timestamp")[[c for c in cols if c in df]].rename(columns=cols)
    return df.astype(float)


def fetch_metric(ex, symbol: str, metric: str, since_ms: int) -> pd.DataFrame:
    """Bir metriği bugünden geriye doğru `since_ms`'e kadar sayfa sayfa indirir.
    (Binance bu uçlarda yalnız startTime verilince son 500 kaydı döndürüyor; bu yüzden endTime ile geriye gidilir.)"""
    method, cols = ENDPOINTS[metric]
    sym_id = ex.market(symbol)["id"]
    fn = getattr(ex, method)
    rows, end = [], int(time.time() * 1000)
    for _ in range(20):                             # 20 x 500 saat >> 30 gün; sonsuz döngü güvencesi
        batch = fn({"symbol": sym_id, "period": PERIOD, "limit": PAGE, "endTime": int(end)})
        if not batch:
            break
        rows += batch
        first = min(int(r["timestamp"]) for r in batch)
        if first <= since_ms or len(batch) < PAGE:
            break
        end = first - 1
        time.sleep(0.15)                            # istek limitlerine saygı
    rows = [r for r in rows if int(r["timestamp"]) >= since_ms]
    return _rows_to_df(rows, cols)


def collect_recent(cfg: dict, ex=None, log=print) -> dict:
    """Tüm coinler ve metrikler için son ~30 günü (veya kaldığı yerden) indirip diske ekler."""
    if ex is None:
        from .exchange import FuturesExchange
        ex = FuturesExchange(cfg["exchange"]["name"]).ex
    earliest = int((datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).timestamp() * 1000)
    summary = {}
    for s in cfg["exchange"]["symbols"]:
        sym = futures_symbol(s)
        for metric in ENDPOINTS:
            path = metric_path(sym, metric)
            since = earliest                         # her seferinde tüm pencere: aradaki boşluklar da dolar
                                                     # (tekrarlar birleştirmede atılır)
            try:
                df = fetch_metric(ex, sym, metric, since)
                added = _merge_save(path, df) if len(df) else 0
                summary[(s, metric)] = added
            except Exception as e:                  # tek bir metriğin hatası diğerlerini durdurmasın
                summary[(s, metric)] = f"hata: {e}"
                log(f"  {s} {metric}: {e}")
    return summary


def load_metrics(symbol: str) -> pd.DataFrame | None:
    """Bir coin için toplanmış tüm metrikleri tek tabloda birleştirir (kolon: metrik_kolon)."""
    frames = []
    for metric in ENDPOINTS:
        path = metric_path(symbol, metric)
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            frames.append(df.add_prefix(f"{metric}_"))
    arch = METRICS_DIR / f"{symbol.split(':')[0].replace('/', '')}_archive.csv"
    if arch.exists():
        frames.append(pd.read_csv(arch, index_col=0, parse_dates=True).add_prefix("archive_"))
    if not frames:
        return None
    return pd.concat(frames, axis=1).sort_index()


# ------------------------------------------------------------------------- arşiv (deneysel)
ARCHIVE_URL = "https://data.binance.vision/data/futures/um/daily/metrics/{id}/{id}-metrics-{day}.zip"


def parse_archive_csv(raw: bytes) -> pd.DataFrame:
    """data.binance.vision 'metrics' zip içeriğini okur. Beklenen kolonlar: create_time, symbol,
    sum_open_interest, sum_open_interest_value, count_toptrader_long_short_ratio,
    sum_toptrader_long_short_ratio, count_long_short_ratio, sum_taker_long_short_vol_ratio."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = z.namelist()[0]
        text = z.read(name).decode("utf-8")
    df = pd.read_csv(io.StringIO(text))
    if "create_time" not in df.columns:            # başlıksız dosya ihtimali
        df = pd.read_csv(io.StringIO(text), header=None, names=[
            "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
            "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
            "count_long_short_ratio", "sum_taker_long_short_vol_ratio"])
    df["create_time"] = pd.to_datetime(df["create_time"], utc=True)
    df = df.drop(columns=[c for c in ("symbol",) if c in df]).set_index("create_time")
    return df.apply(pd.to_numeric, errors="coerce")


def _download_day(session, url):
    try:
        r = session.get(url, timeout=30)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        return parse_archive_csv(r.content), None
    except Exception as e:
        return None, str(e)


def backfill_archive(cfg: dict, days: int, log=print, workers: int = 8) -> dict:
    """data.binance.vision günlük 'metrics' dosyalarından geçmişi doldurur (1 saatlik örnekleme).
    Paralel indirir; daha önce indirilen günleri atlar (yarıda kalırsa tekrar çalıştırmak yeterli)."""
    from concurrent.futures import ThreadPoolExecutor

    import requests

    summary = {}
    today = datetime.now(timezone.utc).date()
    session = requests.Session()
    for s in cfg["exchange"]["symbols"]:
        sid = s.replace("/", "")
        out = METRICS_DIR / f"{sid}_archive.csv"
        have = set()
        if out.exists():
            have = {d.date() for d in pd.read_csv(out, index_col=0, parse_dates=True).index}
        todo = [today - timedelta(days=k) for k in range(days, 0, -1)]
        todo = [d for d in todo if d not in have]
        urls = [ARCHIVE_URL.format(id=sid, day=d.isoformat()) for d in todo]
        frames, ok, missing = [], 0, 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for (df, err), day in zip(pool.map(lambda u: _download_day(session, u), urls), todo):
                if df is None:
                    missing += 1
                    if missing <= 2 and err and not err.startswith("HTTP 404"):
                        log(f"  {s} {day}: {err}")
                else:
                    frames.append(df)
                    ok += 1
        if frames:
            df = pd.concat(frames)
            df = df[~df.index.duplicated()].sort_index().resample("1h").last().dropna(how="all")
            _merge_save(out, df)
        summary[s] = {"indirilen_gün": ok, "bulunamayan_gün": missing}
        log(f"  {s}: {ok} gün indirildi, {missing} gün bulunamadı")
    return summary
