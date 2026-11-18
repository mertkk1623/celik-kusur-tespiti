# Çelik Yüzey Kusuru Tespiti

## Proje Amacı

Bu proje, çelik yüzeylerdeki kusurların görüntü işleme ve derin öğrenme yöntemleriyle
otomatik olarak tespit edilmesini amaçlar. Çalışma kapsamında NEU-DET veri seti
kullanılarak bir nesne tespiti modeli eğitilmiş ve sonuçlar bir Streamlit arayüzünde
sunulmuştur.

Proje; makine öğrenmesi ve bilgisayarlı görü alanında çalışan, uçtan uca
(veri hazırlama → eğitim → değerlendirme → arayüz) bir örnek oluşturmayı hedefler.

## Kullanılan Teknolojiler

- **Python** – proje dili
- **PyTorch** – derin öğrenme altyapısı
- **Ultralytics YOLO** – nesne tespiti modeli
- **NEU-DET** – çelik yüzey kusuru veri seti
- **Google Colab Pro** – model eğitimi (GPU)
- **Streamlit** – sonuçların sunulduğu arayüz

## Veri Bölünmesi

- Eğitim / Doğrulama / Test oranı: **70 / 15 / 15**
- Rastgelelik tohumu (random seed): **42**

## Sonuçlar

Final test değerlendirmesi 270 görüntü üzerinde yapılmıştır:

| Metrik | Değer |
| --- | --- |
| Precision | 78,58 |
| Recall | 73,42 |
| mAP50 | 79,56 |
| mAP50-95 | 46,63 |

Sınıf bazında en zayıf sonuç **crazing** sınıfındadır (Recall 31,90 · mAP50 45,44);
ince ve düşük kontrastlı çatlaklar modelin kaçırma eğiliminde olduğu sınıftır.
Ayrıntılı hata analizi `results/error_analysis.md` dosyasındadır.

## Arayüzü Çalıştırma

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Uygulama, `models/best.pt` ağırlık dosyasını bekler. Dosya yoksa arayüz yine açılır,
ancak tahmin yapamaz ve yükleme yönergesini gösterir.

Arayüzün tamamı tek dosyadadır (`app/app.py`). Tema CSS'i, Streamlit 1.67.0'in DOM
yapısına göre yazılmıştır; Streamlit sürümü yükseltilirse görsel stillerin gözden
geçirilmesi gerekebilir.

## Kısa Yol Haritası

1. **Aşama 0 – İskelet:** Klasör yapısı, README, .gitignore ve requirements dosyalarının hazırlanması.
2. **Aşama 1 – Veri:** NEU-DET veri setinin incelenmesi, etiket formatının dönüştürülmesi ve
   eğitim/doğrulama/test bölünmesinin (`splits/`) oluşturulması.
3. **Aşama 2 – Eğitim:** Google Colab Pro üzerinde YOLO modelinin eğitilmesi.
4. **Aşama 3 – Değerlendirme:** Test seti metriklerinin ve görsel sonuçların (`results/`) üretilmesi.
5. **Aşama 4 – Arayüz:** Eğitilen modelin Streamlit uygulaması (`app/`) ile sunulması.

## Proje Durumu

> **V1 tamamlandı.** Model eğitimi, test değerlendirmesi ve Streamlit arayüzü hazırdır.
> Arayüz bir mühendislik demosudur ve insan değerlendirmesinin yerini almaz; üretim
> hattında otomatik kalite kararı için ek doğrulama ve kalibrasyon gerekir.

## Depo Yapısı

```
.
├── app/            # Streamlit arayüzü
├── data/           # Veri seti (Git'e dahil edilmez)
├── models/         # Eğitilmiş model ağırlıkları (best.pt)
├── notebooks/      # Keşif ve deney not defterleri
├── results/        # Değerlendirme çıktıları ve metrikler
├── splits/         # Eğitim/doğrulama/test bölünme dosyaları
├── src/            # Kaynak kod (veri inceleme, eğitim, değerlendirme)
├── tests/          # Testler
├── LICENSE
├── requirements.txt
└── README.md
```

## Veri Seti Atfı

Bu proje **NEU-DET** (NEU Surface Defect Database) veri setini kullanır. Veri setinin
görselleri ve anotasyonları bu depoya dahil edilmemiştir (bkz. `.gitignore` → `data/`);
veri seti aşağıdaki resmî kaynaktan temin edilmelidir.

- **Kaynak sayfa:** <https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/>
- **Atıf:**

  K. Song and Y. Yan, “A noise robust method based on completed local binary patterns
  for hot-rolled steel strip surface defects,” *Applied Surface Science*, vol. 285,
  pp. 858–864, 2013.

NEU-DET, Northeastern University (Çin) tarafından akademik araştırma amacıyla
yayımlanmıştır. Veri setinin kullanım koşulları için lütfen yukarıdaki resmî kaynak
sayfasına başvurun. Bu depo veri setinin kendisini barındırmaz veya yeniden dağıtmaz.

## Lisans

Bu proje **GNU Affero General Public License v3.0 (AGPL-3.0)** altında lisanslanmıştır.
Tam lisans metni kök dizindeki [`LICENSE`](LICENSE) dosyasındadır.

Bu lisans seçimi büyük ölçüde bir zorunluluktur: proje, **AGPL-3.0** ile lisanslanan
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) kütüphanesini kullanır.
Ultralytics, kendi koduyla üretilen modellerin de aynı lisans kapsamında olduğunu
belirtir; bu nedenle eğitilmiş ağırlıklar dahil tüm proje AGPL-3.0 altında
yayımlanmaktadır. Uygulama ağ üzerinden sunulduğunda AGPL-3.0'ın 13. maddesi gereği
kullanıcılara kaynak koduna erişim imkânı sağlanmalıdır; bu depo o kaynağı sağlar.

Ticari veya kapalı kaynaklı kullanım planlanıyorsa AGPL-3.0 yeterli olmaz; Ultralytics'in
ticari lisansı gerekir: <https://www.ultralytics.com/license>

Telif hakkı (C) 2026 — proje sahibi. Ayrıntı için `LICENSE` dosyasına bakın.
