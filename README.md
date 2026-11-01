# Finans İşlem Botu (Binance Futures, v0.8.1)

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
| `python main.py indicators` | 18 indikatörü tek tek, aynı koşullarla test eder (eğitim / test ayrı) |
| `python main.py indicators --pairs` | Eğitimde en iyi 5 indikatörü, yaygın önerilen eşleriyle (onay) ikili test eder |
| `python main.py layers` | Her indikatör katmanının katkısını eğitim ve test dönemi için ayrı ayrı gösterir |
| `python main.py layers --tf 4h` | Aynı katman testini 4 saatlik mumlarla yapar |
| `python main.py backtest --refresh` | Veriyi yeniden indirir |
| `python main.py collect` | Open interest ve long/short oranlarını indirip `data/metrics/` altında biriktirir |
| `python main.py collect --archive 365` | Binance veri arşivinden (data.binance.vision) 1 yıllık saatlik geçmişi paralel indirir; yarıda kalırsa tekrar çalıştırmak yeterli |
| `python main.py analyze` | OI / long-short / taker metrikleri sonraki 24 saatlik getiriyi öngörüyor mu? (sadece eğitim dönemi, tüm coinler) |
| `python main.py info` | Her coinin borsa limitleri ve mevcut ayarlarla gereken **minimum sermaye** |
| `python main.py run` | `config.yaml`'daki moda göre botu başlatır |
| `python main.py status` | Bot bakiyesi, devre kesici seviyesi, açık pozisyon, son işlemler |
| `python main.py report` | Ara rapor üretir (`reports/` klasörüne) |
| `python main.py resume` | Devre kesiciyle duran botu yeniden etkinleştirir |

## Varsayılan strateji (v0.8): Fibonacci + hacim onayı + 4 saatlik trend filtresi (`strategy.name: signal`)

- **Giriş (Fibonacci):** Son 72 mumun salınımında fiyat 0.5–0.786 "altın bölgeye" geri çekilip 0.5 seviyesinin
  üstüne (yükselişte) / altına (düşüşte) geri döndüğünde.
- **Onay (hacim):** Sinyal mumunun hacmi 20 mumluk ortalamanın 1.5 katı olmalı.
- **Trend filtresi (v0.8):** LONG sadece 4 saatlikte fiyat > EMA21 > EMA55 iken, SHORT sadece fiyat < EMA21 < EMA55 iken.

v0.8 trend filtresi testi (10 coin, 1 yıl; eğitim %70 / test %30):

| | Eğitim KF | Test KF | Yıllık getiri | Maks. düşüş | İşlem başı ort. |
|---|---|---|---|---|---|
| Fib + hacim (v0.7) | 1.28 | 1.03 | +32.5% | −18.2% | 0.94 USDT |
| **+ 4s yön, katı (v0.8)** | **1.38** | **1.33** | +23.3% | **−9.5%** | **1.44 USDT** |
| + 4s yön, gevşek (EMA21>EMA55) | 1.15 | 0.98 | +9.8% | −13.3% | |
| + ADX ≥ 20 | 1.09 | 1.70 | +21.4% | −11.9% | |
| + BTC yönü | 1.20 | 0.90 | +11.8% | −20.0% | |
| + ADX + 4s katı + BTC | 0.97 | 1.94 | +6.6% | −9.9% | |

Sağlamlık: Eğitim KF her ayarda (EMA 13/34–34/89, 2s–8s, Fib 48–120) 1.32–1.55 (hep v0.7'nin üstünde);
test KF ise ayara duyarlı (sadece 21/55 ve 20/50, 4s'te 1.33; diğerlerinde 0.85–0.93 — test döneminde sadece ~47
işlem var). Piyasa emriyle bile her iki dönem kârlı (1.16 / 1.19). Kâr ağırlıkla long'dan: long +291, short −57 USDT.
Getiri / düşüş oranı 1.79 → 2.47. Aynı düşüşte daha fazla getiri için `margin_per_trade_pct` artırılabilir:
%30 teminat (+ `max_total_margin_pct: 0.90`) → yıllık +36%, maks. düşüş −14%.
- **Çıkış:** Stop-loss / kâr-al (borsada) veya Fibonacci'nin ters yönde sinyali.

Nasıl seçildi (`python main.py indicators --pairs`, 10 coin, 1s, Kas 2025 – Kas 2026, limit emirle):
18 indikatör tek başına test edildi; hiçbiri iki dönemde birden kârlı değildi. Eğitimde en iyi 5'i (Fibonacci,
hacim, Bollinger kırılımı, EMA 9/21, Supertrend) yaygın önerilen eşleriyle denendi. Fibonacci'nin 4 eşinden 3'ü
her iki dönemde de kârlıydı; eğitimde en iyisi hacim onayıydı:

| | Eğitim KF | Test KF | Yıllık getiri | Maks. düşüş |
|---|---|---|---|---|
| Fibonacci tek başına | 1.05 | 0.87 | | |
| **Fibonacci + hacim** | **1.28** | **1.03** | **+32.5%** | **−18%** |
| Fibonacci + MACD | 1.13 | 1.29 | +28.5% | −21% |
| Fibonacci + RSI | 1.15 | 1.04 | | |

Sağlamlık: Fibonacci + hacim, salınım süresi 72 / 96 / 120 mumda her iki dönemde kârlı (48'de bozuluyor),
10 coinin 8'inde kârlı, hem long hem short kazandırıyor. **Uyarı:** piyasa emriyle girilirse test KF 0.95'e
düşüyor (kârın bir kısmı limit emir tasarrufundan); 24 kombinasyon denendiği için şans payı var.
Gerçek doğrulama paper / demo hesapta ileriye dönük çalıştırmadır.

Başka bir kombinasyon denemek için `config.yaml → strategy.params` içinde `primary` ve `confirm` değiştirin.

## Alternatif strateji: çok katmanlı karar sistemi (`strategy.name: multi_factor`)

Bir işlem açılması için açık olan **tüm** katmanların onay vermesi gerekir:

| Katman | Gösterge | Kural | Varsayılan |
|---|---|---|---|
| 1. Piyasa rejimi | ADX (4s) | ADX ≥ 20 değilse piyasa yatay sayılır, işlem açılmaz | **açık** |
| 2. Yön | EMA21 / EMA55 (4s) | Fiyat > EMA21 > EMA55 → sadece LONG; tersi → sadece SHORT | **açık** |
| 3. BTC filtresi | BTC'nin 4s yönü | BTC düşüşteyken altcoinde LONG, yükselişteyken SHORT yok | **açık** |
| Giriş | SMA20 / SMA50 (1s) | Kesişim, yukarıdaki filtreler onaylıyorsa işleme dönüşür | `trigger: cross` |
| (4) Geri çekilme | RSI (1s) | Trend yönünde RSI geri çekilip toparlanınca gir | kapalı (`trigger: pullback`) |
| (5) Momentum | MACD histogramı | Giriş yönüne dönmüş olmalı | kapalı |
| (6) Yapı | Fibonacci | Geri çekilme son salınımın 0.382–0.786 bölgesinde olmalı | kapalı |
| (7) Onay | Hacim | Son mumun hacmi 20 mumluk ortalamanın üstünde | kapalı |
| 5. Funding | Funding rate | Funding aşırı pozitifken LONG, aşırı negatifken SHORT açılmaz (kalabalığa katılmama) | **açık** |
| (T) Akış | Taker alış / toplam hacim | Son 6 mumda agresif alıcı/satıcı payı giriş yönünü desteklemeli | kapalı* |

\* Alıcı/satıcı hacmi için mum verisinin yeniden indirilmesi gerekir: `python main.py backtest --refresh`

Çıkış: borsadaki stop-loss / kâr-al, ya da ters kesişim sinyali.

Kapalı katmanlar 1 yıllık veride (Kas 2025 – Kas 2026, 10 coin) eğitim döneminde katkı sağlamadı ve
test döneminde sonucu kötüleştirdi. Kodda duruyorlar; `config.yaml`'dan açılıp `python main.py layers` ile
yeniden değerlendirilebilirler. Katman eklerken kural: **karar eğitim dönemine göre verilir**, test dönemi
yalnızca en sonda doğrulama içindir.

Backtest notu: Eğitim/test dönemleri tek bir simülasyondan kesilir; böylece test dönemi başında
göstergeler (ör. 4s EMA55, ADX) önceki veriyle ısınmış olur.

## Paper (sanal para) canlı test planı — v0.8.1

Ayarlar: `mode: paper`, `live.capital: 1000` (sanal), teminat/işlem %30, toplam teminat ≤ %90, stop ve kâr-al
teminatın %5'i (2x'te fiyatın %2.5'i), stop başına kayıp ≈ bakiyenin %1.5'i, devre kesici %10.

Backtest beklentisi (aynı ayarlarla, 1 yıl): yıllık +36%, maks. düşüş −14%, **ayda ~14 işlem**, kazanma %56,
kâr faktörü 1.35, ort. kazanç +15.2 / ort. kayıp −14.5 USDT. Canlı bot gerçek veri üzerinde tekrar oynatıldığında
verdiği emirlerin tamamı (15/15) backtest sinyalleriyle aynı çıktı.

Süre ve değerlendirme (en az 3–4 hafta, tercihen ≥ 30 işlem):

| Ölçüt | İyi | Dikkat | Dur ve incele |
|---|---|---|---|
| Kâr faktörü | ≥ 1.2 | 1.0 – 1.2 | < 1.0 |
| Kazanma oranı | ≥ %52 | %45 – 52 | < %45 |
| Düşüş | < %8 | %8 – 10 | devre kesici |
| Limit emir dolma oranı | ≥ %70 | %50 – 70 | < %50 |

İzleme: `python main.py status` (günlük), `python main.py report` (haftalık), `logs/bot.log`.
Kısa vadede (20–30 işlem) sonuçlar şansa çok bağlıdır; 3 kötü gün stratejinin bozulduğu anlamına gelmez.

## Kademeli izleyen stop (config.yaml → risk.trailing) — varsayılan KAPALI

Fiyat her `step_r` x R lehe gittiğinde stop, ulaşılan seviyenin `gap_r` x R gerisine çekilir (R = ilk stop mesafesi);
`move_tp` açıksa kâr-al da aynı miktar ileri taşınır. Örnek (long, 2x): giriş 80k, stop 78k, kâr-al 83k →
82k'de stop 80k (başa baş) ve kâr-al 85k → 84k'de stop 82k ve kâr-al 87k ... Canlıda stop emri borsada güncellenir:
önce yeni emir konur, sonra eskisi iptal edilir. Backtest temkinlidir: bir mum içinde önce eski stop kontrol edilir.

Fibonacci + hacim stratejisinde (1 yıl, 10 coin) sonucu **kötüleştirdi**:

| Çıkış | Eğitim KF | Test KF | Yıllık getiri |
|---|---|---|---|
| Kâr-al 1R, izleyen yok (varsayılan) | 1.28 | 1.03 | +32.5% |
| Kâr-al 1.5R + her 1R'de stop/kâr-al 1R yukarı | 1.13 | 0.88 | +7.8% |
| Kâr-al yok + her 1R'de stop 1R yukarı | 1.07 | 0.96 | +5.0% |

Neden: fiyat 1R gidip stop girişe çekildikten sonra sık sık girişe geri dönüyor; eskiden +1R kazanç olan işlemler
"başa baş" kapanıyor. Fibonacci tepkisi kısa bir sıçrama yakalıyor; trend takip eden stratejilerde işe yarayabilir.

## Kısmi kâr-al (config.yaml → risk.partial) — sadece backtest, varsayılan KAPALI

Fiyat `at_r` x R lehe gidince pozisyonun `fraction` kadarı (limit emirle) kapanır; kalan kısmın stop'u girişin
`lock_r` x R ÜSTÜNE çekilir (kalan kısım artık zararla kapanamaz) ve fiyat her `step_r` x R ilerledikçe stop,
ulaşılan seviyenin `gap_r` x R gerisine taşınır; kalan kısımda kâr-al yoktur.
Örnek (long, 2x): giriş 80k, stop 78k → 82k'de yarısı kapanır, stop 81k → 84k'de stop 82k → 86k'de 84k ...

Fibonacci + hacimde (1 yıl, 10 coin) mevcut "tamamı 1R'de" kuralından kötü çıktı. Backtest bir mum içinde
tepe/dip sırasını bilemediği için iki varsayımla ölçüldü:

| Kural | Eğitim KF (temkinli / iyimser) | Test KF (temkinli / iyimser) |
|---|---|---|
| Tamamı 1R'de (varsayılan) | 1.28 | 1.03 |
| Yarısı 1R; kalan stop +0.5R, 0.5R adım | 1.02 / 1.19 | 0.96 / 1.03 |
| Yarısı 1R; kalan stop +0.25R | 1.06 / 1.18 | 1.00 / 1.05 |
| Yarısı 1.5R; kalan stop +0.5R | 1.08 / 1.13 | 1.00 / 1.04 |

Kalan yarı ortalama ~+5 USDT kazandırıyor; bu, o yarıyı 1R'de kapatmakla aynı. Canlı bot bu özellik açıkken
başlamayı reddeder (backtest ile canlı farklı davranmasın diye).

## Emir tipleri ve ücretler (config.yaml → execution)

Binance normal kullanıcı ücretleri: limit emir (maker) %0.02, piyasa emri (taker) %0.05.

| Emir | Varsayılan | Açıklama |
|---|---|---|
| Giriş | **limit (post-only)** | Sinyal mumunun kapanış fiyatına konur; sadece maker olarak dolabilir. `limit_ttl_bars` mum içinde dolmazsa iptal edilir |
| Kâr-al | **limit** | Deftere yazılı reduce-only limit emir (maker, kaymasız) |
| Stop-loss | piyasa (stop-market) | Her zaman garantili çıkış için piyasa emri |
| Sinyal / devre kesici çıkışı | piyasa | |

Backtest, limit emri sadece fiyat limit seviyesini **geçerse** dolu sayar (dokunması yetmez), böylece kuyruk
önceliği ve "fiyat kaçınca dolmama" etkisi hesaba katılır. 1 yıllık veride emirlerin ~%82'si doldu.

## Open interest ve long/short verisi

Binance bu verileri API'de **yalnızca son 30 gün** için verir. `python main.py collect` her coin için
open interest, büyük trader'ların long/short oranları, tüm hesapların long/short oranı ve taker alış/satış
oranını `data/metrics/` altına ekler. `python main.py run` çalışırken bu toplama günde bir kez otomatik yapılır.
Bot sürekli çalışmıyorsa `collect`'i en az 3 haftada bir çalıştırın; Windows Görev Zamanlayıcı ile
otomatikleştirmek için (klasör yolunu kendi yolunuzla değiştirin):

```
schtasks /create /tn "trading_bot_collect" /sc daily /st 09:00 /tr "cmd /c cd /d C:\Users\emiry\Documents\trading_bot && .venv\Scripts\python.exe main.py collect"
```

Arşiv (data.binance.vision) bu metriklerin 1 saatlik geçmişini de içeriyor; `collect --archive 365` ile
indirilir. `python main.py analyze` hangi metriğin gerçekten bilgi taşıdığını gösterir; sadece anlamlı çıkanlar
stratejiye katman olarak eklenmelidir.

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
- Strateji 1 yıllık veride henüz kârlı değil (kâr faktörü ~0.96); canlıya alınmamalı. Strateji geliştirme sürüyor.
- Sabit %2.5 fiyat stop'u (ROE %5 / 2x), HYPE ve ZEC gibi oynak coinlerde daha sık tetiklenir.
  Coin bazında tablo bunu gösterir; gerekirse `stop_mode: atr` dene.
