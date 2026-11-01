# Finans İşlem Botu (Binance Futures, v0.4)

Binance **global USDⓈ-M Futures** üzerinde, **isolated marjin** ve düşük kaldıraçla (varsayılan 2x)
**long ve short** yönlü, **10 coin** üzerinde otomatik işlem yapan bot:
BTC, ETH, BNB, XRP, SOL, TRX, HYPE, ZEC, DOGE, LINK (`config.yaml → exchange.symbols`).
Aşamalar: **backtest** → **paper** (sanal para) → **testnet** → **live**.

> Bu yazılım eğitim ve geliştirme amaçlıdır; yatırım tavsiyesi değildir. Örnek stratejinin kârlı olduğu varsayılmamalıdır.
> Kaldıraçlı işlemler, yatırdığınız teminatın tamamını kaybetme riski taşır.

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate             # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env             # macOS/Linux: cp .env.example .env
python -m pytest -q                # testler (internet gerekmez)
```

## Komutlar

| Komut | Ne yapar |
|---|---|
| `python main.py backtest` | 10 coinin futures verisini (mum + funding) indirir, ortak bakiyeyle portföy testi yapar + coin bazında tablo |
| `python main.py backtest --compare` | Ayar varyasyonlarını yan yana karşılaştırır |
| `python main.py backtest --refresh` | Veriyi yeniden indirir |
| `python main.py info` | Her coinin borsa limitleri ve mevcut ayarlarla gereken **minimum sermaye** |
| `python main.py run` | `config.yaml`'daki moda göre botu başlatır |
| `python main.py status` | Bot bakiyesi, devre kesici seviyesi, açık pozisyon, son işlemler |
| `python main.py report` | Ara rapor üretir (`reports/` klasörüne) |
| `python main.py resume` | Devre kesiciyle duran botu yeniden etkinleştirir |

## Risk kuralları (config.yaml → risk)

| Kural | Varsayılan | Açıklama |
|---|---|---|
| Kaldıraç / marjin | 2x isolated | Bir işlemde en fazla o işlemin teminatı kaybedilebilir |
| Teminat / işlem | bot bakiyesinin %20'si | 2x ile pozisyon = bakiyenin %40'ı |
| Stop-loss | teminatın %5'i (ROE) | 2x'te fiyatın %2.5 aleyhe hareketi → **bot bakiyesinin %1'i** |
| Kâr al | teminatın %5'i (ROE) | 2x'te fiyatın %2.5 lehe hareketi |
| Aynı anda açık pozisyon | en fazla 3 | Coinler birlikte hareket ettiği için tek bir piyasa düşüşünde riski sınırlar |
| Toplam teminat | bakiyenin en fazla %60'ı | Tüm açık pozisyonların teminat toplamı |
| **Devre kesici** | %10 | Son başlatmadaki bakiyeye göre %10 kayıpta: pozisyon kapanır, rapor yazılır, bot durur |
| Günlük limit | %3 | Gün içinde %3 kayıpta o gün yeni işlem açılmaz (0 = kapalı) |
| Acil durdurma | `STOP` dosyası | Klasöre `STOP` adlı dosya koyunca pozisyon kapanır, bot durur |

Stop-loss ve kâr al emirleri pozisyon açılır açılmaz **borsaya** konur; bot kapalı olsa bile çalışırlar.
Koruma emri konamazsa bot pozisyonu derhal kapatır. Likidasyon fiyatı (2x'te ~%50 uzakta) stop'tan çok daha uzaktadır.

Aynı mumda birden fazla coinde sinyal oluşursa `symbols` listesindeki sıra önceliklidir.
Bot her coinde en fazla bir pozisyon tutar.

Bot yalnızca `live.capital` kadar bakiyeyi kendi bakiyesi sayar ve yalnızca kendi açtığı pozisyonu yönetir.
Hesapta elle açılmış bir pozisyon varsa ona dokunmaz ve o kapanana kadar o coinde işlem açmaz.

## Devre kesici akışı

1. Bot bakiyesi, son başlatmadaki değerinin %10 altına düşer.
2. Bot açık pozisyonu kapatır, `reports/rapor_TARİH.md` raporunu yazar, (varsa) Telegram'a bildirir ve durur.
3. Raporu incele → `config.yaml`'ı / stratejiyi güncelle → `python main.py backtest` ile test et.
4. `python main.py resume` (onay ister) → `python main.py run`. Güncel bakiye yeni referans olur.

## Modlar ve API anahtarı

1. **paper** (varsayılan): Gerçek fiyat, sanal bakiye. API anahtarı gerekmez.
2. **testnet**: `testnet.binancefuture.com` adresinden anahtar al (`testnet_kind: testnet`)
   veya `demo.binance.com` → API Management (`testnet_kind: demo`). `.env`'ye yaz.
3. **live**: Gerçek para. `.env` içinde `ALLOW_LIVE=yes` ve başlangıçta `EVET` onayı gerekir.

Gerçek anahtar (binance.com → API Management):
- ✅ Enable Reading, ✅ **Enable Futures**
- ❌ **Enable Withdrawals KAPALI**, ❌ Spot & Margin Trading kapalı (gerekmiyor)
- 🔒 **IP kısıtlaması** açık (botun çalıştığı bilgisayarın IP'si)
- Futures cüzdanına (USDⓈ-M) en az `capital` kadar USDT aktar.

## Minimum sermaye uyarısı

Binance her kontrat için minimum pozisyon büyüklüğü uygular ve bu coinden coine değişir.
Bot bunun altındaki işlemleri atlar. Başlamadan önce **`python main.py info`** çalıştır; her coin için
gereken minimum `capital`'i gösterir. Yetersizse `capital`'i artır, `margin_per_trade_pct`'i yükselt
ya da o coini `symbols` listesinden çıkar.

## Yapı

```
bot/
  config.py      ayarlar + .env + doğrulama (kaldıraç ≤5, sadece isolated, stop zorunlu)
  exchange.py    FuturesExchange (Binance USDⓈ-M, borsada SL/TP), PaperFutures (sanal)
  data.py        mum + funding geçmişi indirme/önbellek, sentetik veri
  indicators.py  SMA, EMA, RSI, ATR
  strategy.py    stratejiler (long_entry / short_entry / long_exit / short_exit)
  risk.py        teminat, ROE/ATR stop & kâr-al, likidasyon, devre kesici, günlük limit
  backtest.py    ücret + kayma + funding + isolated likidasyon + devre kesici dahil backtest
  live.py        canlı döngü, borsa senkronizasyonu, durum kaydı, resume
  report.py      devre kesici raporu (markdown)
main.py          komut satırı
tests/           birim testleri (internet gerekmez)
```

## Bilinen sınırlamalar

- Paper modunda funding simüle edilmez ve stop/kâr-al kontrolü her turda (poll_seconds) yapılır.
- Borsa ücreti BNB ile ödenirse live K/Z'de ücret kısmı eksik görünebilir (realizedPnl doğrudur).
- Örnek strateji (SMA kesişimi + RSI) gerçek BTC verisinde henüz kârlı değil; strateji geliştirme sürüyor.
- Sabit %2.5 fiyat stop'u (ROE %5 / 2x), HYPE ve ZEC gibi oynak coinlerde daha sık tetiklenir.
  Coin bazında tablo bunu gösterir; gerekirse `stop_mode: atr` dene.
