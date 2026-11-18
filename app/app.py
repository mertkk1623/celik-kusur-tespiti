"""Çelik Yüzey Kusuru Tespiti – V1 Streamlit arayüzü.

NE YAPAR?
    Kullanıcının yüklediği tek bir çelik yüzey görüntüsünü, Aşama 3'te
    eğitilmiş sabit model (models/best.pt) ile analiz eder. Tespit edilen
    kusurları görüntü üzerinde kutu + sınıf adı + güven skoru olarak çizer,
    sayısal özet ve tespit tablosu sunar.

NE YAPMAZ?
    - Modeli eğitmez, ince ayar yapmaz.
    - Test seti metriklerini yeniden hesaplamaz (sabit değerler gösterilir).
    - Sonuçları kalıcı olarak kaydetmez; sayfa yenilendiğinde kaybolur.
    - Video, webcam veya toplu işleme yapmaz (V2/V3 kapsamı).

ÇALIŞTIRMA
    streamlit run app/app.py

GEREKSİNİM
    models/best.pt dosyası proje kökündeki models/ klasöründe bulunmalıdır.
    Bu dosya Git'e dahil değildir; Google Drive'daki eğitim çıktısından
    indirilmesi gerekir.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# =============================================================================
# 1) SABİTLER
# =============================================================================

# Bu dosya app/ içinde çalışır; proje kökü bir üst klasördür.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

# Modelin eğitildiği görüntü boyutu. Eğitimle aynı tutulmalıdır.
IMG_SIZE = 640

# Projenin kaynak kodu. AGPL-3.0'ın 13. maddesi, ağ üzerinden sunulan
# değiştirilmiş sürümlerde kullanıcıya kaynağa erişim imkânı verilmesini
# istediği için arayüzde bu bağlantı gösterilir.
SOURCE_URL = "https://github.com/mertkk1623/celik-kusur-tespiti"
SOURCE_LABEL = "mertkk1623/celik-kusur-tespiti"

# Eğitimde kullanılan sınıf sırası (Aşama 2 / Aşama 3 ile birebir aynı).
EXPECTED_CLASS_NAMES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)

# Sınıf renkleri: src/data_inspection.py içindeki CLASS_COLORS ile aynı.
# Notebook grafiklerinde bir sınıf hangi renkteyse arayüzde de o renkte.
CLASS_COLORS = {
    "crazing": "#e6194b",
    "inclusion": "#3cb44b",
    "patches": "#4363d8",
    "pitted_surface": "#f58231",
    "rolled-in_scale": "#911eb4",
    "scratches": "#008080",
}
DEFAULT_CLASS_COLOR = "#8B9AAB"

# Altı sınıfın kısa Türkçe tanımları. Arayüzde yalnızca açıklama amaçlıdır;
# modelin davranışını değiştirmez.
CLASS_DESCRIPTIONS = {
    "crazing": "İnce ve düzensiz yüzey çatlakları.",
    "inclusion": (
        "Çelik içerisinde veya yüzeyinde görülen metalik olmayan safsızlıklar."
    ),
    "patches": "Düzensiz biçimli bölgesel yüzey lekeleri.",
    "pitted_surface": "Küçük çukur ve oyuk benzeri yüzey kusurları.",
    "rolled-in_scale": (
        "Haddeleme sırasında yüzeye taşınan veya gömülen oksit tabakası izleri."
    ),
    "scratches": "Çizgisel yüzey çizikleri.",
}

# Varsayılan eşikler.
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

# Kaydırıcı aralıkları. Hem widget tanımı hem dolgu oranı hesabı bu sabitleri
# kullanır; iki yerde ayrı sayı tutulmaz.
CONF_RANGE = (0.05, 0.95)
IOU_RANGE = (0.10, 0.90)
CONF_STEP = 0.05
IOU_STEP = 0.05

# Kabul edilen görsel formatları.
ALLOWED_EXTENSIONS = ("jpg", "jpeg", "png", "bmp")

# Tablodaki güven çubuğunun ipucu metni (title özniteliği). Modelin ürettiği
# skorun kalibre edilmiş bir olasılık olmadığını belirtir.
CONFIDENCE_HELP = (
    "Modelin bu kutuyu kusur olarak işaretlerken verdiği 0–1 aralığındaki "
    "skor; kalibre edilmiş olasılık değildir"
)

# Final test değerlendirmesinden gelen sabit metrikler (Aşama 4).
# Bunlar bu arayüzde hesaplanmaz; yalnızca bağlam olarak gösterilir.
TEST_METRICS = {
    "precision": "78,58",
    "recall": "73,42",
    "mAP50": "79,56",
    "mAP50_95": "46,63",
    "test_images": 270,
}
CRAZING_METRICS = {
    "recall": "31,90",
    "mAP50": "45,44",
}

# Model dosyası bulunamadığında gösterilecek yönerge.
MODEL_MISSING_MESSAGE = """
Model dosyası bulunamadı: `models/best.pt`

Bu dosya Git deposuna dahil **değildir** (`.gitignore` içinde `*.pt` satırı var;
model ağırlıkları sürüm kontrolüne alınmaz). Google Drive'daki eğitim
çıktısından indirip proje kökündeki `models/` klasörüne `best.pt` adıyla
kaydetmeniz gerekiyor:

```
/content/drive/MyDrive/celik-kusur-tespiti/models/best.pt
```

Dosyayı yerleştirdikten sonra bu sayfayı yenileyin. Uygulama, model
olmadan da açılır; yalnızca tahmin yapamaz.
"""


# =============================================================================
# 2) TEMA (CSS)
# =============================================================================
# Tüm stiller bilinçli olarak tek dosyada tutulur. Ayrı bir CSS dosyası veya
# .streamlit/config.toml eklenmedi; böylece uygulama tek dosyayla taşınabilir
# kalır. Seçiciler streamlit 1.67.0 DOM yapısına göre yazılmıştır.

THEME_CSS = """
<style>
/* ===========================================================================
   2.1 Tasarım belirteçleri
   =========================================================================== */
:root {
  --bg-deep:      #0B0F14;
  --bg-card:      #141A21;
  --bg-raised:    #1C242E;
  --border:       #2A3542;
  --border-soft:  #202932;

  --text-primary: #E8EDF2;
  --text-body:    #C3CFDB;
  --text-muted:   #8B9AAB;
  --text-faint:   #64748B;

  --accent:       #38BDF8;
  --accent-soft:  rgba(56, 189, 248, 0.12);
  --warning:      #F59E0B;
  --warning-soft: rgba(245, 158, 11, 0.10);
  --success:      #22C55E;
  --success-soft: rgba(34, 197, 94, 0.10);

  /* 8px tabanlı boşluk ölçeği */
  --space-1: 8px;
  --space-2: 16px;
  --space-3: 24px;
  --space-4: 32px;
  --space-5: 48px;
  --space-6: 64px;

  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 14px;

  --font-sans: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
  --font-mono: "JetBrains Mono", "Consolas", "SF Mono", ui-monospace, monospace;
}

/* ===========================================================================
   2.2 Uygulama gövdesi ve varsayılan Streamlit görünümünün bastırılması
   =========================================================================== */
.stApp {
  background: var(--bg-deep);
  color: var(--text-body);
  font-family: var(--font-sans);
}

/* Streamlit'in varsayılan üst çubuğu, menüsü ve alt imzası gizlenir. */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] {
  background: transparent;
  height: 0;
}

/* Ana içerik genişliği ve dikey boşluklar */
.block-container {
  max-width: 1280px;
  padding-top: var(--space-4);
  padding-bottom: var(--space-5);
  padding-left: var(--space-3);
  padding-right: var(--space-3);
}

/* ===========================================================================
   2.3 Tipografi
   =========================================================================== */
h1, h2, h3, h4, h5, h6 { color: var(--text-primary); }

/* Bölüm başlıkları (ADIM 01 · GÖRÜNTÜ gibi) */
.section-head {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin: var(--space-5) 0 var(--space-3) 0;
  padding-top: var(--space-2);
  border-top: 1px solid var(--border-soft);
}
.section-head .step {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
  white-space: nowrap;
}
.section-head .title {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-primary);
  letter-spacing: -0.01em;
}
.section-head .hint {
  margin-left: auto;
  font-size: 0.78rem;
  color: var(--text-faint);
}

/* ===========================================================================
   2.4 Başlık alanı (hero)
   =========================================================================== */
.hero {
  padding: var(--space-4) 0 var(--space-3) 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: var(--space-2);
}
.hero-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-2);
  flex-wrap: wrap;
  margin-bottom: var(--space-3);
}
.eyebrow {
  display: inline-flex;
  align-items: center;
  gap: var(--space-1);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-muted);
}
.eyebrow .dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}
.hero h1 {
  font-size: 2.6rem;
  font-weight: 650;
  letter-spacing: -0.025em;
  line-height: 1.12;
  margin: 0 0 var(--space-2) 0;
  color: var(--text-primary);
}
.hero .lede {
  font-size: 0.95rem;
  line-height: 1.65;
  color: var(--text-muted);
  max-width: 72ch;
  margin: 0;
}

/* Durum etiketi (V1 · Active) */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 11px;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  border: 1px solid transparent;
  white-space: nowrap;
}
.badge-active {
  color: var(--accent);
  background: var(--accent-soft);
  border-color: rgba(56, 189, 248, 0.32);
}
.badge-planned {
  color: var(--text-faint);
  background: rgba(139, 154, 171, 0.06);
  border-color: var(--border);
}
.badge-test {
  color: var(--success);
  background: var(--success-soft);
  border-color: rgba(34, 197, 94, 0.30);
  text-transform: none;
  letter-spacing: 0.02em;
}

/* ===========================================================================
   2.5 Kartlar ve paneller
   =========================================================================== */
.panel {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: var(--space-3);
}
.panel-title {
  font-size: 0.78rem;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: var(--space-2);
}

/* Metrik kartları */
.stat-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: var(--space-2);
  margin-bottom: var(--space-3);
}
.stat {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: var(--space-3);
}
.stat-value {
  font-family: var(--font-mono);
  font-size: 2rem;
  font-weight: 600;
  line-height: 1.1;
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
}
.stat-value.accent { color: var(--accent); }
.stat-value.warn { color: var(--warning); }
.stat-label {
  margin-top: var(--space-1);
  font-size: 0.78rem;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-muted);
}
.stat-sub {
  margin-top: 4px;
  font-size: 0.78rem;
  color: var(--text-faint);
}

/* ===========================================================================
   2.6 Sınıf çipleri
   =========================================================================== */
.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1);
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 5px 11px;
  border-radius: var(--radius-sm);
  background: var(--bg-raised);
  border: 1px solid var(--border);
  font-size: 0.82rem;
  color: var(--text-body);
}
.chip .swatch {
  width: 8px;
  height: 8px;
  border-radius: 2px;
  flex-shrink: 0;
}
.chip .count {
  font-family: var(--font-mono);
  font-size: 0.78rem;
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
}

/* ===========================================================================
   2.7 Dosya yükleyici
   =========================================================================== */
[data-testid="stFileUploader"] section {
  background: var(--bg-card) !important;
  border: 1px dashed var(--border) !important;
  border-radius: var(--radius-md) !important;
  padding: var(--space-4) var(--space-3) !important;
  transition: border-color 120ms ease;
}
[data-testid="stFileUploader"] section:hover {
  border-color: rgba(56, 189, 248, 0.45) !important;
}
[data-testid="stFileUploader"] section button {
  background: var(--bg-raised) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
}
[data-testid="stFileUploader"] section button:hover {
  border-color: rgba(56, 189, 248, 0.45) !important;
  color: var(--accent) !important;
}
/* İngilizce "Upload" düğme etiketi, widget API'siyle değiştirilemiyor
   (streamlit 1.67.0 etiketi ön yüzde sabit tutuyor). Etiket metni
   gizlenip yerine Türkçesi yazılıyor; DOM değişirse yalnızca İngilizce
   metin geri gelir, işlev bozulmaz. */
[data-testid="stFileUploaderDropzone"] button {
  font-size: 0 !important;
  gap: 0 !important;
}
[data-testid="stFileUploaderDropzone"] button::after {
  content: "Dosya seç";
  font-size: 0.86rem;
  font-weight: 500;
  letter-spacing: 0.01em;
}
/* Biçim ve boyut bilgisinin kontrastı: soluk metin yerine okunur ton. */
[data-testid="stFileUploaderDropzoneInstructions"] div {
  color: var(--text-body) !important;
  font-size: 0.8rem !important;
}
[data-testid="stFileUploader"] label {
  font-size: 0.9rem !important;
  color: var(--text-muted) !important;
}

/* ===========================================================================
   2.8 Görsel önizleme kartları
   =========================================================================== */
/* Karşılaştırma sütunları kart görünümü alır. st.columns yalnızca burada
   kullanıldığı için genel bir yan etki oluşmaz. */
[data-testid="stColumn"] {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: var(--space-2);
}
.image-card-label {
  font-size: 0.78rem;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: var(--space-2);
}
.image-card-caption {
  margin-top: var(--space-1);
  font-size: 0.78rem;
  color: var(--text-faint);
  font-family: var(--font-mono);
}

/* Streamlit görselleri köşe yumuşatması */
[data-testid="stImage"] img {
  border-radius: var(--radius-sm);
  display: block;
}

/* ===========================================================================
   2.9 Tespit tablosu
   =========================================================================== */
/* Tablo `st.dataframe` yerine semantik HTML olarak çizilir (bkz.
   build_results_table). Böylece açık renkli ızgara yerine tema belirteçleri
   doğrudan uygulanır. Sarmalayıcı yatay taşmayı kendi içinde kaydırır;
   sayfa gövdesi küçük ekranlarda yana kaymaz. */
.results-table-wrap {
  width: 100%;
  overflow-x: auto;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
}
.results-table {
  width: 100%;
  min-width: 560px;
  border-collapse: collapse;
  font-size: 0.85rem;
  color: var(--text-body);
}
.results-table thead th {
  background: var(--bg-raised);
  color: var(--text-muted);
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  text-align: left;
  white-space: nowrap;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
}
.results-table tbody td {
  padding: 9px 12px;
  border-bottom: 1px solid var(--border-soft);
  vertical-align: middle;
}
.results-table tbody tr:last-child td {
  border-bottom: none;
}
.results-table tbody tr:hover td {
  background: rgba(56, 189, 248, 0.05);
}
.results-table .cell-class {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  color: var(--text-primary);
  font-weight: 500;
  white-space: nowrap;
}
.results-table .swatch {
  width: 8px;
  height: 8px;
  border-radius: 2px;
  flex-shrink: 0;
}
.results-table .cell-confidence {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  min-width: 150px;
}
.confidence-bar {
  flex: 1 1 auto;
  min-width: 70px;
  height: 6px;
  border-radius: 999px;
  background: var(--bg-raised);
  overflow: hidden;
}
.confidence-fill {
  display: block;
  height: 100%;
  border-radius: 999px;
  background: var(--accent);
}
.confidence-value {
  font-family: var(--font-mono);
  font-size: 0.78rem;
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
  min-width: 46px;
  text-align: right;
}
.results-table .cell-num {
  font-family: var(--font-mono);
  font-size: 0.78rem;
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

/* ===========================================================================
   2.10 Bilgi / uyarı panelleri
   =========================================================================== */
.notice {
  display: flex;
  gap: var(--space-2);
  padding: var(--space-3);
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
  background: var(--bg-card);
  font-size: 0.9rem;
  line-height: 1.6;
}
.notice .icon {
  font-size: 1rem;
  line-height: 1.5;
  flex-shrink: 0;
}
.notice .body { min-width: 0; }
.notice .body strong { color: var(--text-primary); }
.notice-warn {
  background: var(--warning-soft);
  border-color: rgba(245, 158, 11, 0.32);
  color: #F2D8A8;
}
.notice-warn .icon { color: var(--warning); }
.notice-info {
  background: var(--bg-card);
  border-color: var(--border);
  color: var(--text-body);
}
.notice-info .icon { color: var(--accent); }
.notice-title {
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 4px;
}

/* ===========================================================================
   2.11 Yol haritası
   =========================================================================== */
.roadmap {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: var(--space-2);
}
.roadmap-item {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: var(--space-3);
}
.roadmap-item.active {
  border-color: rgba(56, 189, 248, 0.34);
  background: linear-gradient(
    180deg,
    rgba(56, 189, 248, 0.06) 0%,
    var(--bg-card) 100%
  );
}
.roadmap-head {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  margin-bottom: var(--space-2);
}
.roadmap-version {
  font-family: var(--font-mono);
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text-primary);
}
.roadmap-body {
  font-size: 0.86rem;
  line-height: 1.6;
  color: var(--text-muted);
}
.roadmap-body ul {
  margin: 0;
  padding-left: 18px;
}
.roadmap-body li { margin-bottom: 3px; }

/* ===========================================================================
   2.12 Kenar çubuğu
   =========================================================================== */
[data-testid="stSidebar"] {
  background: #0E141A;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container {
  padding-top: var(--space-3);
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  font-size: 0.78rem;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-muted);
}
[data-testid="stSidebar"] label {
  color: var(--text-body) !important;
  font-size: 0.86rem !important;
}
.sidebar-block {
  padding: var(--space-2) 0;
  border-top: 1px solid var(--border-soft);
  margin-top: var(--space-2);
}
.sidebar-key {
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-faint);
  margin-bottom: 4px;
}
.sidebar-val {
  font-family: var(--font-mono);
  font-size: 0.82rem;
  color: var(--text-body);
  word-break: break-all;
}

/* ===========================================================================
   2.13 Kaydırıcılar (güven ve IoU eşikleri)
   =========================================================================== */
/* streamlit 1.67.0 kaydırıcıyı baseweb ile değil React Aria ile çiziyor;
   bu yüzden eski `[data-baseweb="slider"]` seçicisi hiçbir şeye denk
   gelmiyordu. Sürümdeki gerçek yapı (Slider.*.js):
     kök (stSlider) > etiket + sarmalayıcı > iz (track)
     iz > ray (ilk çocuk) + düğme(ler) + stSliderTickBar
     düğme > input + stSliderThumbValue
   Yani düğme, içinde doğrudan `stSliderThumbValue` barındıran tek öğedir;
   iz ise içinde doğrudan `stSliderTickBar` barındıran tek öğedir. Seçiciler
   yalnızca bu iki test id'sine ve `stSlider` kapsamına dayanır; emotion
   hash'leri veya baseweb gibi sürüme bağlı adlar kullanılmaz. Düğmenin
   odak/sürükleme halkası da aynı kuraldan cyan gölge alır. */
[data-testid="stSlider"] div:has(> [data-testid="stSliderThumbValue"]) {
  background-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
  color: var(--accent) !important;
}
/* Rayın dolu kısmı: oran, slider_fill_css() tarafından yayınlanan
   `--slider-fill` değişkeninden gelir; boş kısım tema kenarlık rengidir. */
[data-testid="stSlider"] div:has(> [data-testid="stSliderTickBar"]) > div:first-child {
  background: linear-gradient(
    to right,
    var(--accent) 0 var(--slider-fill, 0%),
    var(--border) var(--slider-fill, 0%) 100%
  ) !important;
}

/* ===========================================================================
   2.14 Kullanım kılavuzu ve sınıf tanımları
   =========================================================================== */
.howto {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: var(--space-2);
}
.howto-step {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: var(--space-3);
}
.howto-num {
  font-family: var(--font-mono);
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  color: var(--accent);
  margin-bottom: var(--space-1);
}
.howto-title {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 4px;
}
.howto-body {
  font-size: 0.86rem;
  line-height: 1.6;
  color: var(--text-muted);
}

.class-guide {
  display: grid;
  gap: 1px;
  background: var(--border-soft);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  overflow: hidden;
}
.class-guide-row {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-2) var(--space-3);
  background: var(--bg-card);
}
.class-guide-row .swatch {
  width: 8px;
  height: 8px;
  border-radius: 2px;
  flex-shrink: 0;
}
.class-guide-name {
  font-family: var(--font-mono);
  font-size: 0.84rem;
  color: var(--text-primary);
  min-width: 132px;
  flex-shrink: 0;
}
.class-guide-desc {
  font-size: 0.86rem;
  line-height: 1.55;
  color: var(--text-muted);
}

/* ===========================================================================
   2.15 Alt bilgi
   =========================================================================== */
.footer-note {
  margin-top: var(--space-5);
  padding-top: var(--space-3);
  border-top: 1px solid var(--border);
  font-size: 0.82rem;
  line-height: 1.65;
  color: var(--text-faint);
}
.footer-note strong { color: var(--text-muted); }
.footer-note a {
  color: var(--accent);
  text-decoration: none;
  border-bottom: 1px solid rgba(56, 189, 248, 0.32);
}
.footer-note a:hover {
  border-bottom-color: var(--accent);
}

/* ===========================================================================
   2.16 Duyarlı yerleşim
   =========================================================================== */
@media (max-width: 768px) {
  .block-container {
    padding-left: var(--space-2);
    padding-right: var(--space-2);
    padding-top: var(--space-3);
  }
  .hero h1 { font-size: 1.9rem; }
  .hero .lede { font-size: 0.9rem; }
  .block-container { max-width: 100%; }
  .stat-row { grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
  .stat-value { font-size: 1.6rem; }
  .section-head { flex-wrap: wrap; gap: var(--space-1); }
  .section-head .hint { margin-left: 0; width: 100%; }
  /* Sınıf tanımı satırında açıklama alt satıra iner */
  .class-guide-row { flex-wrap: wrap; }
  .class-guide-name { min-width: 0; }
  .class-guide-desc { flex: 1 1 100%; }
  /* Yan yana sütunlar mobilde alt alta */
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
  [data-testid="stHorizontalBlock"] > [data-testid="column"] {
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }
}
</style>
"""


# =============================================================================
# 3) HTML PARÇA ÜRETİCİLERİ
# =============================================================================


def section_head(step, title, hint=""):
    """Bölüm başlığı üretir: 'ADIM 01 · GÖRÜNTÜ'."""
    hint_html = f'<span class="hint">{hint}</span>' if hint else ""
    return (
        f'<div class="section-head">'
        f'<span class="step">{step}</span>'
        f'<span class="title">{title}</span>'
        f"{hint_html}"
        f"</div>"
    )


def stat_card(value, label, sub="", style=""):
    """Tek bir metrik kartı üretir."""
    sub_html = f'<div class="stat-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="stat">'
        f'<div class="stat-value {style}">{value}</div>'
        f'<div class="stat-label">{label}</div>'
        f"{sub_html}"
        f"</div>"
    )


def stat_row(*cards):
    """Metrik kartlarını duyarlı bir ızgarada birleştirir."""
    return f'<div class="stat-row">{"".join(cards)}</div>'


def notice(title, body, kind="info", icon="›"):
    """Bilgi veya uyarı paneli üretir."""
    title_html = f'<div class="notice-title">{title}</div>' if title else ""
    return (
        f'<div class="notice notice-{kind}">'
        f'<span class="icon">{icon}</span>'
        f'<div class="body">{title_html}{body}</div>'
        f"</div>"
    )


def class_chips(counts):
    """Sınıf adı + adet çipleri üretir.

    Sınıf adları modelden (`model.names`) geldiği için güvenilmeyen kabul
    edilir; HTML'e yazılmadan önce kaçışlanır.
    """
    chips = []
    for name, count in counts.items():
        color = CLASS_COLORS.get(name, DEFAULT_CLASS_COLOR)
        chips.append(
            f'<span class="chip">'
            f'<span class="swatch" style="background:{color}"></span>'
            f"{html.escape(str(name))}"
            f'<span class="count">{count}</span>'
            f"</span>"
        )
    return f'<div class="chip-row">{"".join(chips)}</div>'


def class_guide_rows():
    """Altı sınıfı renk + ad + Türkçe tanımla listeler.

    Tanımlar sabit metindir; sınıf adları ise modelden gelebilecek değerlere
    karşı kaçışlanır. Sıra, eğitimdeki sabit sınıf sırasıdır.
    """
    rows = []
    for name in EXPECTED_CLASS_NAMES:
        color = CLASS_COLORS.get(name, DEFAULT_CLASS_COLOR)
        description = html.escape(
            CLASS_DESCRIPTIONS.get(name, "Tanım bulunmuyor.")
        )
        rows.append(
            f'<div class="class-guide-row">'
            f'<span class="swatch" style="background:{color}"></span>'
            f'<span class="class-guide-name">{html.escape(name)}</span>'
            f'<span class="class-guide-desc">{description}</span>'
            f"</div>"
        )
    return f'<div class="class-guide">{"".join(rows)}</div>'


# =============================================================================
# 4) MODEL YÜKLEME VE DOĞRULAMA
# =============================================================================


@st.cache_resource(show_spinner=False)
def load_model(model_path_str):
    """Modeli yalnızca bir kez yükler ve Streamlit önbelleğinde tutar.

    Neden cache_resource?
        Model yüklemesi birkaç saniye sürer. Streamlit her etkileşimde betiği
        baştan çalıştırdığı için, önbellek olmadan her slider hareketinde model
        yeniden yüklenirdi. cache_resource nesneyi paylaşır.

    Args:
        model_path_str: Model dosyasının tam yolu (hash'lenebilir olsun diye
            string alınıyor).

    Returns:
        tuple: (model, hata_mesaji). Başarıda hata_mesaji None olur.
    """
    from ultralytics import YOLO

    try:
        model = YOLO(model_path_str)
    except Exception as error:  # bozuk dosya, uyumsuz sürüm, eksik bağımlılık
        return None, f"{type(error).__name__}: {error}"

    return model, None


def check_class_names(model):
    """Modelin sınıf listesini beklenen sırayla karşılaştırır.

    Neden?
        Arayüzdeki sınıf adları modelin kendi `names` alanından gelir. Ancak
        yanlış bir ağırlık dosyası yüklenirse sınıf sırası kaymış olabilir.
        Sessizce yanlış etiket göstermek yerine kullanıcıyı uyarıyoruz.

    Args:
        model: Yüklenmiş Ultralytics modeli.

    Returns:
        tuple: (sozluk {sinif_id: ad}, uyari_mesaji_veya_None)
    """
    raw_names = getattr(model, "names", None) or {}
    model_names = {int(key): str(value) for key, value in raw_names.items()}
    ordered = [model_names[key] for key in sorted(model_names)]

    if tuple(ordered) != EXPECTED_CLASS_NAMES:
        expected = ", ".join(EXPECTED_CLASS_NAMES)
        found = ", ".join(ordered) if ordered else "(boş)"
        return model_names, (
            f"Modelin sınıf listesi beklenen sırayla uyuşmuyor. "
            f"Beklenen: {expected}. "
            f"Modelde: {found}. "
            f"Tahminler çalışır, ancak sınıf adları yanlış görünebilir. "
            f"Doğru best.pt dosyasını kullandığınızdan emin olun."
        )

    return model_names, None


# =============================================================================
# 5) TAHMİN YARDIMCILARI
# =============================================================================


def prepare_inference_image(image):
    """Modelin eğitildiği gri tonlamalı dağılıma uygun çıkarım görüntüsü üretir.

    Neden ayrı bir görüntü?
        NEU-DET görüntüleri tek kanallı (PIL modu "L") gri tonlamalıdır.
        Eğitim bu dağılım üzerinde yapıldı. Arayüzde göstermek için orijinal
        görüntüyü olduğu gibi tutuyoruz; çıkarıma ise ayrı bir gri tonlamalı
        kopya veriyoruz. Böylece renkli bir JPEG yüklendiğinde modelin
        görmediği renk bilgisi tahmine karışmaz.

        Akış: mod "L" (tek kanal) → mod "RGB" (aynı gri değer üç kanala
        kopyalanır). Kopyalama, bilgi eklemez; yalnızca Ultralytics'in
        beklediği 3 kanallı girdi biçimini üretir. Görüntü yeniden
        örneklenmez, boyut korunur.

    Args:
        image: Kullanıcının yüklediği, RGB'ye dönüştürülmüş PIL görüntüsü.

    Returns:
        PIL.Image.Image: RGB modunda, gri tonlamalı içerikli yeni görüntü.
    """
    grayscale = image.convert("L")
    return grayscale.convert("RGB")


def run_prediction(model, image, conf, iou):
    """Görüntü üzerinde tahmin çalıştırır.

    Args:
        model: Yüklenmiş Ultralytics modeli.
        image: Çıkarım için hazırlanmış PIL görüntüsü (RGB modunda).
        conf: Güven eşiği.
        iou: NMS IoU eşiği.

    Returns:
        Tek bir görüntüye ait Ultralytics Results nesnesi.
    """
    # PIL görüntüsü doğrudan veriliyor. NumPy dizisi verilseydi Ultralytics
    # onu BGR kabul ederdi; oysa PIL yolu renk sırasını kendi yönetir.
    results = model.predict(
        source=image,
        conf=float(conf),
        iou=float(iou),
        imgsz=IMG_SIZE,
        verbose=False,
    )
    return results[0]


def extract_detections(result, class_names):
    """Results nesnesinden tespit listesi çıkarır.

    Args:
        result: Ultralytics Results nesnesi.
        class_names: {sinif_id: ad} sözlüğü.

    Returns:
        list[dict]: Güvene göre azalan sırada tespitler.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    classes = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    coordinates = boxes.xyxy.cpu().numpy()

    detections = []
    for index in range(len(classes)):
        class_id = int(classes[index])
        x1, y1, x2, y2 = (float(value) for value in coordinates[index])
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_names.get(class_id, f"sınıf_{class_id}"),
                "confidence": float(confidences[index]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections


def plot_result(result):
    """Kutuları çizilmiş görüntüyü RGB olarak döndürür.

    Neden kanal çeviriyoruz?
        Ultralytics `plot()` çıktısı OpenCV geleneğine uygun olarak BGR'dir.
        Streamlit ise RGB bekler. `[:, :, ::-1]` ile mavi ve kırmızı kanalları
        yer değiştiriyoruz; aksi hâlde görüntü renkleri ve kutular ters görünür.

    Neden çizgi kalınlığı ve yazı boyutu veriliyor?
        `plot()` bu değerleri vermezsek görüntü boyutuna göre seçiyor:
        çizgi kalınlığı `max(round(ortalama_kenar * 0.003), 2)`, yazı ölçeği
        ise kalınlığa bağlı (`font_scale = line_width / 3`). 200×200'lük
        NEU-DET görüntülerinde bu, 2 px kutu ve `0.67` yazı ölçeği demek;
        sınıf adı + skor etiketleri görüntünün önemli bir bölümünü kaplayıp
        birbiriyle çakışıyor. `line_width=1` hem kutuyu inceltiyor hem yazı
        ölçeğini `0.33`'e düşürüyor. `font_size` yalnızca PIL ile metin çizen
        yolda (örneğin sınıflandırma çıktısı) kullanılır; orada da varsayılan
        12 yerine 10 veriliyor.

        `color_mode="class"` korunuyor: kutular sınıf başına aynı rengi alır,
        yani sınıf renkleri değişmez. `conf=True` ve `labels=True` ile sınıf
        adı ve güven skoru etiket üzerinde kalır.

    Args:
        result: Ultralytics Results nesnesi.

    Returns:
        numpy.ndarray: RGB uint8 görüntü.
    """
    bgr_image = result.plot(
        conf=True,
        labels=True,
        boxes=True,
        line_width=1,
        font_size=10,
        color_mode="class",
    )
    rgb_image = bgr_image[:, :, ::-1]
    return np.ascontiguousarray(rgb_image)


def build_results_table(detections):
    """Tespitleri koyu temalı, semantik bir HTML tablosuna dönüştürür.

    Neden `st.dataframe` değil?
        `st.dataframe` ızgarası açık renkli gelir ve bu dosyadaki tema
        belirteçlerine bağlanamaz. Semantik tablo ile arka plan, kenarlık,
        yazı rengi ve güven çubuğu doğrudan CSS'ten (bkz. 2.9) gelir; yatay
        taşma da sarmalayıcı içinde kalır.

    Güvenlik:
        Sınıf adları modelden (`model.names`) gelir ve güvenilmeyen metin
        sayılır; hücreye yazılmadan önce kaçışlanır. Sayısal hücreler
        biçimlendirilmiş metne çevrilir, ardından yine kaçışlanır.

    Args:
        detections: `extract_detections()` çıktısı.

    Returns:
        str: `<table>` içeren HTML parçası.
    """
    header = (
        "<thead><tr>"
        "<th>Kusur sınıfı</th>"
        "<th>Güven</th>"
        "<th>Skor</th>"
        "<th>Sol üst (x, y)</th>"
        "<th>Sağ alt (x, y)</th>"
        "<th>Boyut (g × y)</th>"
        "</tr></thead>"
    )

    help_text = html.escape(CONFIDENCE_HELP, quote=True)

    rows = []
    for detection in detections:
        name = html.escape(str(detection["class_name"]))
        color = CLASS_COLORS.get(detection["class_name"], DEFAULT_CLASS_COLOR)
        percent = min(max(detection["confidence"] * 100.0, 0.0), 100.0)
        width = detection["x2"] - detection["x1"]
        height = detection["y2"] - detection["y1"]

        cells = {
            "score": html.escape(f"{detection['confidence']:.3f}"),
            "top_left": html.escape(
                f"{detection['x1']:.0f}, {detection['y1']:.0f}"
            ),
            "bottom_right": html.escape(
                f"{detection['x2']:.0f}, {detection['y2']:.0f}"
            ),
            "size": html.escape(f"{width:.0f} × {height:.0f}"),
            "percent": html.escape(f"{percent:.1f}%"),
        }

        rows.append(
            "<tr>"
            f'<td><span class="cell-class">'
            f'<span class="swatch" style="background:{color}"></span>'
            f"{name}</span></td>"
            f'<td><span class="cell-confidence" title="{help_text}">'
            f'<span class="confidence-bar">'
            f'<span class="confidence-fill" style="width:{percent:.1f}%">'
            f"</span></span>"
            f'<span class="confidence-value">{cells["percent"]}</span>'
            f"</span></td>"
            f'<td class="cell-num">{cells["score"]}</td>'
            f'<td class="cell-num">{cells["top_left"]}</td>'
            f'<td class="cell-num">{cells["bottom_right"]}</td>'
            f'<td class="cell-num">{cells["size"]}</td>'
            "</tr>"
        )

    return (
        '<div class="results-table-wrap">'
        f'<table class="results-table">{header}<tbody>'
        f'{"".join(rows)}</tbody></table></div>'
    )


def slider_fill_css(value, minimum, maximum, key):
    """Kaydırıcının dolu çizgi oranını CSS değişkeni olarak yayınlar.

    Neden gerekli?
        React Aria kaydırıcısının dolgu gradyanı emotion ile satır içi
        stillenir ve rengi `theme.colors.primary`'den (kırmızı) gelir. Dolgu
        yüzdesi DOM'da ayrı bir öznitelik olarak durmadığı için CSS'ten
        okunamaz; ancak Python tarafında değer bilinir. `--slider-fill`
        değişkeni, kaydırıcıyı saran `st-key-<key>` kapsayıcısına yazılır,
        aşağıya doğru miras alınır ve 2.13'teki gradyan kuralı tarafından
        okunur. Böylece düğme, değer etiketi ve dolu çizgi aynı cyan rengi
        paylaşır.

    Args:
        value: Kaydırıcının geçerli değeri.
        minimum: Alt sınır.
        maximum: Üst sınır.
        key: `st.slider(key=...)` ile verilen anahtar.

    Returns:
        str: `<style>` etiketi içeren HTML parçası.
    """
    span = float(maximum) - float(minimum)
    ratio = 0.0 if span <= 0 else (float(value) - float(minimum)) / span
    percent = min(max(ratio * 100.0, 0.0), 100.0)
    return (
        "<style>"
        f".st-key-{key} {{ --slider-fill: {percent:.2f}%; }}"
        "</style>"
    )


# =============================================================================
# 6) SAYFA BÖLÜMLERİ
# =============================================================================


def render_header():
    """Başlık alanını çizer."""
    st.markdown(
        '<div class="hero">'
        '<div class="hero-top">'
        '<span class="eyebrow"><span class="dot"></span>'
        "Yüzey Kalite Kontrolü</span>"
        '<span class="badge badge-active">V1 · Active</span>'
        "</div>"
        "<h1>Çelik Yüzey Kusuru Tespiti</h1>"
        '<p class="lede">NEU-DET veri seti üzerinde eğitilmiş YOLO26n modeliyle '
        "tek görüntü çıkarımı. Altı kusur sınıfı, 1800 görüntü, 4189 etiketli "
        "sınır kutusu. Görüntü yükleyin; model kutuları, sınıf adlarını ve "
        "güven skorlarını üretsin.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Kenar çubuğundaki sabit bilgileri çizer."""
    st.sidebar.markdown(
        '<div class="sidebar-block">'
        '<div class="sidebar-key">Model</div>'
        '<div class="sidebar-val">YOLO26n · 640 px</div>'
        "</div>"
        '<div class="sidebar-block">'
        '<div class="sidebar-key">Ağırlık dosyası</div>'
        f'<div class="sidebar-val">{html.escape(MODEL_PATH.name)}</div>'
        "</div>"
        '<div class="sidebar-block">'
        '<div class="sidebar-key">Çıktı</div>'
        '<div class="sidebar-val">Kaydedilmez · oturumluk</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_model_dataset():
    """Model ve veri seti bilgi kartlarını çizer."""
    st.markdown(
        section_head("BİLGİ", "Model ve veri seti", "Eğitim: Aşama 3"),
        unsafe_allow_html=True,
    )
    st.markdown(
        stat_row(
            stat_card("YOLO26n", "Mimari", "ultralytics 8.4.x"),
            stat_card("6", "Kusur sınıfı", "NEU-DET"),
            stat_card("1.800", "Görüntü", "1260 / 270 / 270"),
            stat_card("4.189", "Sınır kutusu", "etiketli"),
        ),
        unsafe_allow_html=True,
    )


def render_test_metrics():
    """Final test metriklerini doğru bağlamıyla gösterir."""
    st.markdown(
        section_head(
            "PERFORMANS",
            "Final test değerlendirmesi",
            f"{TEST_METRICS['test_images']} görüntülük test seti",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        stat_row(
            stat_card(
                f"%{TEST_METRICS['precision']}",
                "Precision",
                "Üretilen tahminlerin ne kadarı doğru",
                style="accent",
            ),
            stat_card(
                f"%{TEST_METRICS['recall']}",
                "Recall",
                "Duyarlılık · kusur yakalama",
                style="accent",
            ),
            stat_card(
                f"%{TEST_METRICS['mAP50']}",
                "mAP50",
                "IoU 0,50 eşiğindeki mAP",
            ),
            stat_card(
                f"%{TEST_METRICS['mAP50_95']}",
                "mAP50-95",
                "IoU 0,50–0,95 arası AP ortalaması",
            ),
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        notice(
            "Bu metrikler neyi ölçer?",
            f"Değerler, {TEST_METRICS['test_images']} görüntülük <strong>final "
            "test setinin tamamı</strong> üzerinde yapılan tek seferlik "
            "değerlendirmeden gelir. Tek bir görüntü üzerindeki tahmin "
            "performansını göstermez; yüklediğiniz görüntüdeki sonuç bu "
            "ortalamanın altında veya üstünde olabilir."
            "<br><br>"
            "<strong>Precision</strong>, üretilen kutuların ne kadarının "
            "gerçekten kusur olduğunu; <strong>Recall</strong>, gerçek "
            "kusurların ne kadarının bulunabildiğini gösterir."
            "<br><br>"
            "<strong>AP</strong> (average precision), bir sınıf için "
            "Precision–Recall eğrisinin altındaki alanı ifade eder. "
            "<strong>mAP</strong>, AP değerlerinin sınıflar üzerindeki "
            "ortalamasıdır. <strong>mAP50</strong>, IoU 0,50 eşiğindeki mAP "
            "değeridir. <strong>mAP50-95</strong> ise IoU 0,50–0,95 aralığındaki "
            "eşiklerde hesaplanan AP değerlerinin ortalamasıdır; bu nedenle "
            "daha zorlayıcı ve her zaman daha düşük bir metriktir.",
            kind="info",
            icon="›",
        ),
        unsafe_allow_html=True,
    )


def render_crazing_warning():
    """Crazing sınıfının zayıf performansını açıkça gösterir."""
    body = (
        f"<strong>Recall %{CRAZING_METRICS['recall']}</strong> · "
        f"<strong>mAP50 %{CRAZING_METRICS['mAP50']}</strong>"
        "<br><br>"
        "Crazing (ince yüzey çatlağı) sınıfı, düşük kontrast ve arka plan "
        "dokusuna benzerliği nedeniyle modelin en zayıf olduğu sınıftır. "
        "Test setindeki gerçek crazing kusurlarının yaklaşık üçte biri "
        "bulunabilmiştir."
        "<br><br>"
        "Güven eşiğini düşürmek bu sınıfta daha fazla tespit üretebilir; "
        "ancak <strong>yanlış pozitifleri de artırır</strong>. Eşiği "
        "düşürdüğünüzde sonuçları görsel olarak doğrulamanız önerilir."
    )
    st.markdown(
        notice("Crazing sınıfında bilinen zayıflık", body, kind="warn", icon="!"),
        unsafe_allow_html=True,
    )


def render_roadmap():
    """V1 / V2 / V3 yol haritasını çizer."""
    st.markdown(
        section_head("YOL HARİTASI", "Sürüm planı", "V2 ve V3 uygulanmadı"),
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="roadmap">'
        '<div class="roadmap-item active">'
        '<div class="roadmap-head">'
        '<span class="roadmap-version">V1</span>'
        '<span class="badge badge-active">Active</span>'
        "</div>"
        '<div class="roadmap-body"><ul>'
        "<li>Tek görüntü çıkarımı</li>"
        "<li>Güven ve IoU eşiği denetimi</li>"
        "<li>Tespit tablosu ve görsel karşılaştırma</li>"
        "</ul></div></div>"
        '<div class="roadmap-item">'
        '<div class="roadmap-head">'
        '<span class="roadmap-version">V2</span>'
        '<span class="badge badge-planned">Planned</span>'
        "</div>"
        '<div class="roadmap-body"><ul>'
        "<li>Toplu görüntü işleme</li>"
        "<li>Sonuçları dosyaya aktarma</li>"
        "<li>Sınıf bazında rapor özeti</li>"
        "</ul></div></div>"
        '<div class="roadmap-item">'
        '<div class="roadmap-head">'
        '<span class="roadmap-version">V3</span>'
        '<span class="badge badge-planned">Planned</span>'
        "</div>"
        '<div class="roadmap-body"><ul>'
        "<li>Video / akış çıkarımı</li>"
        "<li>Model karşılaştırma paneli</li>"
        "<li>Kalite kararı için eşik profilleri</li>"
        "</ul></div></div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_howto():
    """Sade bir 'Nasıl kullanılır?' bölümü çizer."""
    st.markdown(
        section_head("KILAVUZ", "Nasıl kullanılır?", "Üç adım"),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="howto">'
        '<div class="howto-step">'
        '<div class="howto-num">01</div>'
        '<div class="howto-title">Görseli yükle</div>'
        '<div class="howto-body">JPG, JPEG, PNG veya BMP formatında bir çelik '
        "yüzey görüntüsü seçin veya sürükleyip bırakın. Görüntü yüklendiği anda "
        "tahmin otomatik çalışır; ayrıca bir düğmeye basmanız gerekmez.</div>"
        "</div>"
        '<div class="howto-step">'
        '<div class="howto-num">02</div>'
        '<div class="howto-title">Eşikleri ayarla</div>'
        '<div class="howto-body"><strong>Confidence</strong>, bir kutunun '
        "gösterilmesi için gereken en düşük güven skorudur; eşiği düşürmek daha "
        "çok kutu üretir.<br><strong>IoU</strong>, aynı sınıfa ait örtüşen "
        "kutuları eleyen NMS eşiğidir; düşük değer bu kutuları daha agresif "
        "eler.</div>"
        "</div>"
        '<div class="howto-step">'
        '<div class="howto-num">03</div>'
        '<div class="howto-title">Sonuçları incele</div>'
        '<div class="howto-body">Karşılaştırma panelinde kutuların görüntüde '
        "doğru yerde olup olmadığına bakın. Ardından sonuç tablosundaki sınıf "
        "adı, güven oranı ve kutu koordinatlarını kontrol edin.</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="panel-title" style="margin-top:var(--space-4)">'
        f"Kusur sınıfları</div>"
        f"{class_guide_rows()}",
        unsafe_allow_html=True,
    )


def render_footer():
    """Teknik sorumluluk reddi notunu ve kaynak kodu bağlantısını çizer."""
    st.markdown(
        '<div class="footer-note">'
        "<strong>Teknik not.</strong> Bu arayüz bir mühendislik demosudur ve "
        "insan değerlendirmesinin yerini almaz. Model çıktıları olasılıksaldır; "
        "güven skoru yüksek bir kutu bile hatalı olabilir, güven skoru düşük bir "
        "kusur ise gerçek olabilir. Üretim hattında otomatik kalite kararı "
        "vermek için ek doğrulama, kalibrasyon ve sınıf bazlı eşik stratejisi "
        "gerekir. Sonuçlar bu oturum dışında saklanmaz."
        "<br>"
        f'Kaynak kodu: <a href="{SOURCE_URL}" target="_blank" '
        f'rel="noopener noreferrer">{SOURCE_LABEL}</a> · '
        "AGPL-3.0 ile lisanslanmıştır."
        "</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# 7) ANA AKIŞ
# =============================================================================


def main():
    """Arayüzü kurar ve çalıştırır."""
    st.set_page_config(
        page_title="Çelik Yüzey Kusuru Tespiti",
        page_icon="◆",
        layout="wide",
    )

    # Tema CSS'i en başta enjekte edilir.
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    render_header()
    render_howto()

    # --- Kenar çubuğu: ayarlar -------------------------------------------
    st.sidebar.markdown("### Ayarlar")

    conf = st.sidebar.slider(
        "Güven eşiği",
        min_value=CONF_RANGE[0],
        max_value=CONF_RANGE[1],
        value=DEFAULT_CONF,
        step=CONF_STEP,
        key="conf_slider",
        help=(
            "Bu değerin altındaki tahminler gösterilmez. Düşük değer daha çok "
            "kutu üretir (yanlış alarm artar), yüksek değer daha az kutu "
            "üretir (kusur kaçırma artar)."
        ),
    )

    iou = st.sidebar.slider(
        "Örtüşme eşiği (IoU)",
        min_value=IOU_RANGE[0],
        max_value=IOU_RANGE[1],
        value=DEFAULT_IOU,
        step=IOU_STEP,
        key="iou_slider",
        help=(
            "NMS (non-maximum suppression) adımında örtüşen tahmin kutularını "
            "elemek için kullanılan eşik. Varsayılan ayarda eleme, sınıf bazlı "
            "NMS kapsamında yalnızca aynı sınıfa ait örtüşen kutular arasında "
            "uygulanır; farklı sınıflardan örtüşen kutular birbirini elemez. "
            "Bir kutunun, aynı sınıftan kendisinden daha yüksek skorlu bir "
            "kutuyla IoU değeri bu eşiği aşarsa o kutu elenir. Düşük eşikte "
            "örtüşen kutular daha agresif elenir (daha az, daha temiz kutu "
            "kalır). Yüksek eşikte eleme gevşer; aynı kusur için örtüşen birden "
            "fazla kutu kalabilir."
        ),
    )

    # Dolu çizgi oranı CSS'ten okunamadığı için değerden hesaplanıp yayınlanır.
    st.markdown(
        slider_fill_css(conf, *CONF_RANGE, "conf_slider")
        + slider_fill_css(iou, *IOU_RANGE, "iou_slider"),
        unsafe_allow_html=True,
    )

    render_sidebar()

    # --- Model dosyası kontrolü ------------------------------------------
    if not MODEL_PATH.is_file():
        st.markdown(
            section_head("ADIM 00", "Model kontrolü"),
            unsafe_allow_html=True,
        )
        st.markdown(
            notice(
                "Model dosyası bulunamadı",
                "Uygulama çalışıyor, ancak tahmin yapabilmek için ağırlık "
                "dosyası gerekiyor. Aşağıdaki yönergeyi izleyin.",
                kind="warn",
                icon="!",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(MODEL_MISSING_MESSAGE)
        render_model_dataset()
        render_test_metrics()
        render_crazing_warning()
        render_roadmap()
        render_footer()
        st.stop()

    # --- Model yükleme ----------------------------------------------------
    with st.spinner("Model yükleniyor..."):
        model, load_error = load_model(str(MODEL_PATH))

    if model is None:
        st.markdown(
            notice(
                "Model yüklenemedi",
                "Dosya bozuk olabilir veya kurulu Ultralytics sürümü modelle "
                "uyumsuz olabilir. `requirements.txt` içindeki sürümün kurulu "
                "olduğunu kontrol edin.",
                kind="warn",
                icon="!",
            ),
            unsafe_allow_html=True,
        )
        with st.expander("Teknik ayrıntı"):
            st.code(load_error or "bilinmeyen hata", language="text")
        st.stop()

    class_names, names_warning = check_class_names(model)
    if names_warning:
        # Uyarı metni modelin sınıf adlarını içerir; HTML'e gömülmeden önce
        # kaçışlanır.
        st.markdown(
            notice(
                "Sınıf sırası uyuşmazlığı",
                html.escape(names_warning),
                kind="warn",
                icon="!",
            ),
            unsafe_allow_html=True,
        )

    # --- ADIM 01: Görüntü yükleme ----------------------------------------
    st.markdown(
        section_head(
            "ADIM 01",
            "Görüntü",
            "JPG · JPEG · PNG · BMP",
        ),
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Bir çelik yüzey görüntüsü yükleyin",
        type=list(ALLOWED_EXTENSIONS),
        help="Desteklenen formatlar: JPG, JPEG, PNG, BMP",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(
            notice(
                "",
                "Başlamak için yukarıdan bir görüntü seçin veya sürükleyip "
                "bırakın. Tahmin, görüntü yüklendiği anda otomatik çalışır.",
                kind="info",
                icon="›",
            ),
            unsafe_allow_html=True,
        )
        render_model_dataset()
        render_test_metrics()
        render_crazing_warning()
        render_roadmap()
        render_footer()
        st.stop()

    # --- Görüntüyü açma ---------------------------------------------------
    try:
        image = Image.open(uploaded_file)
        image.load()  # dosyanın gerçekten okunabildiğini doğrular
    except UnidentifiedImageError:
        st.markdown(
            notice(
                "Görüntü okunamadı",
                "Dosya bir görüntü olarak açılamadı. Dosyanın bozuk olmadığından "
                "ve JPG, JPEG, PNG veya BMP formatlarından birinde olduğundan "
                "emin olun.",
                kind="warn",
                icon="!",
            ),
            unsafe_allow_html=True,
        )
        st.stop()
    except Exception as error:
        st.markdown(
            notice(
                "Görüntü açılırken hata",
                f"Beklenmeyen bir hata oluştu: {html.escape(str(error))}",
                kind="warn",
                icon="!",
            ),
            unsafe_allow_html=True,
        )
        st.stop()

    # Orijinal görüntü arayüzde gösterilir; çıkarım için ayrı bir gri tonlamalı
    # kopya hazırlanır (NEU-DET eğitim dağılımı ile tutarlılık).
    image = image.convert("RGB")
    inference_image = prepare_inference_image(image)

    # --- Tahmin -----------------------------------------------------------
    try:
        with st.spinner("Tahmin yapılıyor..."):
            result = run_prediction(model, inference_image, conf, iou)
    except Exception as error:
        st.markdown(
            notice(
                "Tahmin sırasında hata",
                f"Model çalıştırılamadı: {html.escape(str(error))}",
                kind="warn",
                icon="!",
            ),
            unsafe_allow_html=True,
        )
        st.stop()

    detections = extract_detections(result, class_names)

    # --- ADIM 02: Karşılaştırma ------------------------------------------
    st.markdown(
        section_head(
            "ADIM 02",
            "Karşılaştırma",
            f"conf ≥ {conf:.2f} · IoU {iou:.2f}",
        ),
        unsafe_allow_html=True,
    )

    left_column, right_column = st.columns(2)

    with left_column:
        st.markdown(
            '<div class="image-card-label">Orijinal</div>',
            unsafe_allow_html=True,
        )
        st.image(image, width="stretch")
        st.markdown(
            f'<div class="image-card-caption">'
            f"{html.escape(uploaded_file.name)} · "
            f"{image.width}×{image.height} px · yüklediğiniz görüntü"
            f"</div>",
            unsafe_allow_html=True,
        )

    with right_column:
        st.markdown(
            '<div class="image-card-label">Tespit</div>',
            unsafe_allow_html=True,
        )
        st.image(plot_result(result), width="stretch")
        st.markdown(
            f'<div class="image-card-caption">'
            f"{len(detections)} tespit · gri tonlamalı girdi üzerinde"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        notice(
            "İki görsel arasındaki fark",
            "NEU-DET veri seti <strong>gri tonlamalı</strong> görüntülerden "
            "oluşur ve model bu dağılım üzerinde eğitilmiştir. Bu yüzden "
            "tahmin, yüklediğiniz görüntünün gri tonlamalı bir kopyası üzerinde "
            "çalıştırılır: görüntü önce tek kanala (<code>L</code>) çevrilir, "
            "sonra aynı gri değer üç kanala kopyalanarak (<code>RGB</code>) "
            "modele verilir. Kopyalama yeni bilgi eklemez; yalnızca modelin "
            "beklediği 3 kanallı biçimi üretir ve görüntü yeniden ölçeklenmez. "
            "Soldaki görsel <strong>orijinal yüklemeniz</strong>, sağdaki ise "
            "bu gri tonlamalı girdi üzerinde üretilen tespitlerdir; bu nedenle "
            "renkli bir görüntü yüklediyseniz iki görsel arasında renk farkı "
            "görmeniz normaldir.",
            kind="info",
            icon="›",
        ),
        unsafe_allow_html=True,
    )

    # --- ADIM 03: Bulgular ------------------------------------------------
    st.markdown(
        section_head("ADIM 03", "Bulgular"),
        unsafe_allow_html=True,
    )

    if not detections:
        st.markdown(
            notice(
                "Kusur tespit edilmedi",
                "Ayarlanan güven eşiğinin üzerinde kusur bulunamadı. Bu geçerli "
                "bir sonuçtur; hata değildir. Eşiği düşürerek tekrar "
                "deneyebilirsiniz — ancak bu, yanlış pozitifleri de artırır. "
                "İnce çatlaklarda (crazing) modelin kusur kaçırma eğilimi "
                "bilinmektedir.",
                kind="info",
                icon="›",
            ),
            unsafe_allow_html=True,
        )
    else:
        # Sınıf bazında sayım
        counts = {}
        for detection in detections:
            name = detection["class_name"]
            counts[name] = counts.get(name, 0) + 1

        highest = max(item["confidence"] for item in detections)

        st.markdown(
            stat_row(
                stat_card(str(len(detections)), "Tespit", style="accent"),
                stat_card(str(len(counts)), "Farklı sınıf"),
                stat_card(f"{highest:.2f}", "En yüksek skor"),
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="panel-title">Sınıf dağılımı</div>'
            f"{class_chips(counts)}",
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            build_results_table(detections),
            unsafe_allow_html=True,
        )

        st.caption(
            "Kutu koordinatları yüklenen görüntünün piksel ölçeğindedir. "
            "Güven skoru, modelin tahmine verdiği 0–1 aralığındaki skordur; "
            "kalibre edilmiş bir olasılık veya doğruluk garantisi değildir."
        )

    # --- Alt bölümler -----------------------------------------------------
    render_model_dataset()
    render_test_metrics()
    render_crazing_warning()
    render_roadmap()
    render_footer()


if __name__ == "__main__":
    main()
