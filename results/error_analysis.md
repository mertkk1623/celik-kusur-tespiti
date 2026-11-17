# Hata Analizi — Final Test Değerlendirmesi

**Proje:** Çelik Yüzey Kusuru Tespiti (NEU-DET)
**Model:** `models/best.pt` (Aşama 3'te eğitilmiş, sabit model)
**Değerlendirme:** Yalnızca test spliti (270 görüntü)
**Tarih:** 18 Kasım 2026

---

## 1. Amaç ve Yöntem

### Amaç

Bu analizin amacı, final test değerlendirmesinde elde edilen sayısal metriklerin
**arkasında hangi somut hataların yattığını** görmektir. Metrikler (Precision,
Recall, mAP) bir modelin ne kadar iyi olduğunu söyler; ancak **nasıl** hata
yaptığını söylemez. Bu doküman o boşluğu doldurmak için hazırlandı.

Analiz **tanı amaçlıdır**. Buradaki bulgulara dayanarak model, ağırlık veya
hiperparametre değiştirilmedi. Test sonuçlarına bakıp model değiştirmek
değerlendirmeyi geçersiz kılacağı için bu aşamada yalnızca gözlem yapıldı.

### Yöntem

1. Final test değerlendirmesi sırasında Ultralytics tarafından otomatik olarak
   kaydedilen batch görselleri görsel olarak incelendi.
2. Her batch için **etiket** görseli ile **tahmin** görseli yan yana
   karşılaştırıldı. Böylece modelin neyi kaçırdığı, neyi fazladan işaretlediği
   ve kutuları nereye koyduğu doğrudan görülebildi.
3. Belirgin hata içeren **9 görüntü** seçildi ve her biri için hata türü
   kaydedildi.
4. Gözlemler, confusion matrix ve sınıf bazında metriklerle birlikte
   değerlendirildi.

### Kullanılan hata türleri

| Hata türü | Tanım |
|---|---|
| **False Negative (FN)** | Gerçek etiketli kusur var, model hiç tahmin üretmemiş |
| **False Positive (FP)** | Gerçek etiket yok, model kutu üretmiş |
| **Çoklu tespit** | Tek kusur için birden fazla örtüşen kutu üretilmiş |
| **Zayıf konumlandırma** | Kutu doğru sınıfta ama konumu/boyutu gerçek kusuru tam kapsamıyor |

---

## 2. İncelenen 9 Örnek ve Gözlenen Hatalar

| # | Görüntü | Gözlenen hata | Hata türü |
|---|---|---|---|
| 1 | `crazing_183.jpg` | Gerçek etiketlerde kusur kutuları mevcut ancak model tahmin üretememiş. | False Negative |
| 2 | `crazing_158.jpg` | Gerçek kusur bulunmasına rağmen model tespit yapamamış. | False Negative |
| 3 | `crazing_205.jpg` | Etiketli kusur bölgesi model tarafından kaçırılmış. | False Negative |
| 4 | `crazing_23.jpg` | İki gerçek Crazing bölgesi bulunuyor ancak model bu bölgelerin hiçbirini tespit edememiş. | False Negative (2 kusur) |
| 5 | `crazing_285.jpg` | Gerçek etiket kutusu görünmüyor ancak model Crazing tahminleri üretmiş. | False Positive |
| 6 | `crazing_51.jpg` | Gerçek etiket kutusu görünmemesine rağmen model tahmin kutusu oluşturmuş. | False Positive |
| 7 | `crazing_300.jpg` | Tek gerçek kusur bölgesine karşılık birden fazla, örtüşen tahmin kutusu oluşturulmuş. | Çoklu tespit |
| 8 | `crazing_92.jpg` | Birden fazla tahmin kutusu birbirinin üzerine geliyor; kutu konumları ve kapsamları gerçek etiketlerle tam örtüşmüyor. | Çoklu tespit + zayıf konum |
| 9 | `inclusion_110.jpg` | Inclusion sınıfında tahminler mevcut ancak kutuların konumu ve boyutları gerçek etiketlerle tam eşleşmiyor. | Zayıf konumlandırma |

### Örneklerin dağılımı

| Hata türü | Örnek sayısı | Görüntüler |
|---|---|---|
| False Negative | 4 (5 kusur bölgesi) | `crazing_183`, `crazing_158`, `crazing_205`, `crazing_23` |
| False Positive | 2 | `crazing_285`, `crazing_51` |
| Çoklu tespit | 1 (+1 birlikte) | `crazing_300`, `crazing_92` |
| Zayıf konumlandırma | 1 (+1 birlikte) | `inclusion_110`, `crazing_92` |

9 örneğin **8'i Crazing**, 1'i Inclusion sınıfına aittir. Bu dağılım,
aşağıdaki metrik bulgusuyla tutarlıdır: modelin en zayıf olduğu sınıf Crazing'dir.

---

## 3. Genel Hata Analizi

Görsel inceleme, modelin temel sorununun **üç başlıkta** toplandığını
gösteriyor: kusurları kaçırma, gereksiz tahmin üretme ve sınır kutularını doğru
konumlandıramama.

En sık görülen hata **False Negative**, yani kaçırılan kusurlardır. Dört ayrı
Crazing görüntüsünde gerçek etiketlerde kusur açıkça işaretli olmasına rağmen
model hiç kutu üretmemiş; `crazing_23.jpg` örneğinde iki ayrı kusur bölgesinin
ikisi de kaçırılmıştır. Bu durum, modelin düşük Recall değeriyle doğrudan
örtüşür: kaçırılan her kusur, Recall'ı düşürür.

İkinci sorun **False Positive**, yani olmayan kusurların işaretlenmesidir. İki
görüntüde gerçek etiket bulunmamasına rağmen model Crazing kutusu üretmiştir.
Bu hatalar Precision değerini düşürür ve pratikte gereksiz inceleme yükü
yaratır — üretim hattında operatörün "kusur var" diye bakıp bir şey bulamadığı
durumlar bu türden doğar.

Üçüncü sorun **çoklu tespit ve zayıf konumlandırma**dır. `crazing_300.jpg`
görüntüsünde tek kusur için birden fazla örtüşen kutu üretilmiş; `crazing_92.jpg`
ve `inclusion_110.jpg` görüntülerinde ise kutular doğru sınıfta olmasına rağmen
konum ve boyut olarak gerçek kusurla tam örtüşmüyor. Bu tür hatalar, IoU
eşiğinin üzerinde kalan ama tam oturmayan kutular anlamına gelir. Bu nedenle
kutu konumlandırma hatalarının mAP50-95 performansını olumsuz etkilemesi
beklenir; çünkü mAP50-95, mAP50'ye göre çok daha sıkı IoU eşiklerinde
hesaplanır ve tam oturmayan kutular bu eşiklerde geçerli sayılmaz.

Ancak bu değerlendirme, dokuz örnek üzerinden yapılan **niteliksel** bir görsel
incelemedir. Söz konusu hataların mAP50 ve mAP50-95 değerlerine ne ölçüde katkı
yaptığı ayrı ayrı ölçülmemiştir; her hata türünün metriğe etkisi izole
edilmemiştir. Bu nedenle konumlandırma hatası, düşük mAP50-95'in **başlıca
nedeni** olarak kesin biçimde ileri sürülemez. Yalnızca katkıda bulunması
muhtemel etkenlerden biri olarak değerlendirilmelidir.

Confusion matrix incelendiğinde **sınıflar arası karışıklığın görece sınırlı**
olduğu görülüyor. Yani model inclusion ile scratches'i, ya da patches ile
pitted_surface'ı sık sık birbirine karıştırmıyor. En belirgin sorun, gerçek
arka planla eşleşmeyen tahminlerdir: model ya kusuru hiç görmüyor (arka plan
olarak değerlendiriyor) ya da arka planı kusur sanıyor. Bu, sınıflandırma
hatasından çok **nesne bulma (localization) ve varlık tespiti** hatası
olduğunu gösterir.

---

## 4. Crazing Sınıfının Düşük Performansı

Crazing sınıfı, test setinde **en düşük performansı** gösteren sınıftır:

| Metrik | Değer |
|---|---|
| **Recall** | **%31,90** |
| **mAP50** | **%45,44** |

Recall'ın %31,90 olması, test setindeki gerçek Crazing kusurlarının yalnızca
yaklaşık **üçte birinin** bulunabildiği anlamına gelir. Bu, diğer sınıflara
göre belirgin bir gerilemedir ve 9 örnek görüntünün 8'inin Crazing olmasıyla
tutarlıdır.

### Neden bu kadar düşük?

Gözlemler, bunun birkaç iç içe geçmiş nedenden kaynaklandığını gösteriyor:

1. **Düşük kontrast.** Crazing (yüzeydeki ince çatlak ağı) çelik yüzeyinde
   çoğunlukla çok hafif bir doku farkı olarak görünür. Kusur ile arka plan
   arasındaki parlaklık farkı azdır.
2. **Doku benzerliği.** Crazing deseni, çelik yüzeyin kendi dokusuyla ve
   aydınlatmadan kaynaklanan desenlerle kolayca karışır. Modelin ayırt etmesi
   gereken ipucu çok incedir.
3. **Belirgin sınır yokluğu.** Diğer sınıflar (ör. inclusion, patches) daha
   keskin kenarlı, daha "nesne gibi" bölgelerdir. Crazing ise yayılmış, sınırı
   belirsiz bir dokudur; bu da hem bulmayı hem de kutu yerleştirmeyi
   zorlaştırır.
4. **Kaçırma eğilimi.** Model, düşük kontrastlı bu bölgeleri arka plan olarak
   değerlendirme eğilimindedir. FN örneklerinin tamamının Crazing olması bu
   yorumu destekler.

mAP50'nin %45,44'te kalması ise iki şeyi birlikte anlatır: model bulduğu
Crazing bölgelerinde bile kutuyu tam doğru yere oturtamıyor ve/veya bulduğu
kutuların güven skoru düşük olduğu için precision-recall eğrisi erken
düşüyor.

> **Not:** Bu bölümdeki değerler final test değerlendirmesinin gerçek
> çıktısıdır. Bu analiz onları değiştirmez, yalnızca yorumlar.

---

## 5. İncelemenin Kapsamı ve Sınırları

Bu analiz, **tüm test setinin manuel incelemesi değildir**. Yalnızca görsel
olarak incelenen hata örneklerini kapsar.

### Neyle sınırlı?

İnceleme, final test değerlendirmesi sırasında Ultralytics tarafından
otomatik olarak oluşturulan **üç kaydedilmiş test batch görseliyle**
sınırlıdır. Her batch için iki dosya vardır: gerçek etiketler ve model
tahminleri.

| Batch | Etiket görseli | Tahmin görseli |
|---|---|---|
| 0 | `val_batch0_labels.jpg` | `val_batch0_pred.jpg` |
| 1 | `val_batch1_labels.jpg` | `val_batch1_pred.jpg` |
| 2 | `val_batch2_labels.jpg` | `val_batch2_pred.jpg` |

> **Dosya adları hakkında:** Ultralytics, değerlendirme sırasında batch
> görsellerini `val_batch*.jpg` biçiminde adlandırır. Bu adlar korunmuştur;
> ancak bu dosyalar **final test değerlendirmesi** (test spliti) çalıştırılırken
> üretilmiştir, eğitim sırasındaki validation aşamasına ait değildir.

### Bu sınırlar ne anlama geliyor?

- Bir batch tipik olarak 16 görüntü içerir. Üç batch, 270 görüntülük test
  setinin **yalnızca küçük bir kısmını** temsil eder.
- Buradaki 9 örnek, **hatalı olduğu gözlemlenen** görüntülerdir. Test setindeki
  hata dağılımının istatistiksel bir örneği değildir; hata oranı hakkında sayı
  vermez.
- Bu nedenle bulgular **niteliksel** (hangi tür hatalar oluyor?) soruları
  yanıtlar, **niceliksel** (hataların yüzde kaçı şu türden?) soruları
  yanıtlamaz. Niceliksel dağılım için tüm test setinin sistematik incelemesi
  gerekir.
- Crazing'e dair yorumlar, hem metriklerden (Recall %31,90) hem de bu 9
  örnekten beslendiği için görece güvenilirdir. Diğer sınıflar için benzer
  yorum yapılamaz; yeterli sayıda örnek incelenmemiştir.

---

## Özet

| Bulgu | Durum |
|---|---|
| En sık hata türü | False Negative (kaçırılan kusur) |
| En zayıf sınıf | Crazing (Recall %31,90, mAP50 %45,44) |
| Sınıflar arası karışıklık | Görece sınırlı |
| Baskın sorun | Arka planla eşleşmeyen tahminler (varlık tespiti) |
| İkincil sorun | Sınır kutusu konumlandırma hassasiyeti |
| İnceleme kapsamı | 3 test batch görseli, 9 örnek — tüm test seti değil |

Bu doküman **yalnızca tanı** amaçlıdır. Model, ağırlık veya hiperparametre
değiştirilmedi; test sonuçlarına göre yeni deney başlatılmadı.
