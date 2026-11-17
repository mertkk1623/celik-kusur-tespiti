# Çelik Yüzey Kusuru Tespiti

## Proje Amacı

Bu proje, çelik yüzeylerdeki kusurların görüntü işleme ve derin öğrenme yöntemleriyle
otomatik olarak tespit edilmesini amaçlar. Çalışma kapsamında NEU-DET veri seti
kullanılarak bir nesne tespiti modeli eğitilecek ve sonuçlar görselleştirilecektir.

Proje; makine öğrenmesi ve bilgisayarlı görü alanında çalışan, uçtan uca
(veri hazırlama → eğitim → değerlendirme → arayüz) bir örnek oluşturmayı hedefler.

## Kullanılacak Teknolojiler

- **Python** – proje dili
- **PyTorch** – derin öğrenme altyapısı
- **Ultralytics YOLO** – nesne tespiti modeli
- **NEU-DET** – çelik yüzey kusuru veri seti
- **Google Colab Pro** – model eğitimi (GPU)
- **Streamlit** – sonuçların sunulacağı arayüz

## Veri Bölünmesi

- Eğitim / Doğrulama / Test oranı: **70 / 15 / 15**
- Rastgelelik tohumu (random seed): **42**

## Kısa Yol Haritası

1. **Aşama 0 – İskelet:** Klasör yapısı, README, .gitignore ve requirements dosyalarının hazırlanması.
2. **Aşama 1 – Veri:** NEU-DET veri setinin incelenmesi, etiket formatının dönüştürülmesi ve
   eğitim/doğrulama/test bölünmesinin (`splits/`) oluşturulması.
3. **Aşama 2 – Eğitim:** Google Colab Pro üzerinde YOLO modelinin eğitilmesi.
4. **Aşama 3 – Değerlendirme:** Test seti metriklerinin ve görsel sonuçların (`results/`) üretilmesi.
5. **Aşama 4 – Arayüz:** Eğitilen modelin Streamlit uygulaması (`app/`) ile sunulması.

## Proje Durumu

> **Bu proje henüz geliştirme aşamasındadır.**
> Depodaki içerik, yukarıdaki yol haritasına göre aşamalı olarak tamamlanacaktır.
> Kod, veri ve sonuçlar hazır olmadan bu depo çalışan bir uygulama olarak değerlendirilmemelidir.

## Depo Yapısı

```
.
├── app/          # Streamlit arayüzü (henüz boş)
├── data/         # Veri seti (Git'e dahil edilmez)
├── models/       # Eğitilmiş model ağırlıkları (Git'e dahil edilmez)
├── notebooks/    # Keşif ve deney not defterleri (henüz boş)
├── results/      # Değerlendirme çıktıları ve metrikler
├── splits/       # Eğitim/doğrulama/test bölünme dosyaları
├── src/          # Kaynak kod (henüz boş)
├── tests/        # Testler (henüz boş)
├── requirements.txt
└── README.md
```
