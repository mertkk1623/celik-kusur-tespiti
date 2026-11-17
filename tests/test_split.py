"""Aşama 1B – split kesişim testi.

Tek bir şeyi kontrol eder: train, validation ve test gruplarındaki görüntü
kümelerinin birbiriyle kesişimi sıfır mı?

Neden önemli?
    Aynı görüntü iki grupta yer alırsa test seti eğitime sızar. Bu durumda
    ölçülen başarım gerçek performansı yansıtmaz. Bu test o hatayı yakalar.

Test hem CSV dosyasından hem de split fonksiyonundan okunan veriyle çalışır:
    - CSV varsa: gerçek ayrım dosyası kontrol edilir.
    - CSV yoksa: aynı fonksiyonla küçük sentetik veri üretilip kontrol edilir.
      Böylece test, Colab'da veri seti olmadan da çalışır.

Çalıştırma:
    python -m pytest tests/test_split.py -v
    veya
    python tests/test_split.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import split_dataset  # noqa: E402  (sys.path ayarından sonra içe aktarılmalı)

SPLIT_CSV = PROJECT_ROOT / "splits" / "neu_det_split.csv"


def _read_images_per_split_from_csv(csv_path):
    """CSV'yi okuyup her split için görüntü yollarını döndürür."""
    images_per_split = {name: set() for name in split_dataset.SPLIT_NAMES}
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            images_per_split[row["split"]].add(row["image_rel_path"])
    return images_per_split


def _images_per_split_from_function():
    """Split fonksiyonunu küçük sentetik kayıt kümesiyle çalıştırır."""
    records = [
        {
            "image_rel_path": f"images/img_{index:04d}.jpg",
            "annotation_rel_path": f"annotations/img_{index:04d}.xml",
            "defect_classes": ["crazing"],
        }
        for index in range(60)
    ]
    rows = split_dataset.split_records(records, seed=split_dataset.DEFAULT_SEED)
    images_per_split = {name: set() for name in split_dataset.SPLIT_NAMES}
    for row in rows:
        images_per_split[row["split"]].add(row["image_rel_path"])
    return images_per_split


def _assert_no_overlap(images_per_split, source_label):
    """Üç grup arasındaki ikili kesişimlerin boş olduğunu doğrular."""
    pairs = (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    )
    for first, second in pairs:
        overlap = images_per_split[first] & images_per_split[second]
        assert not overlap, (
            f"[{source_label}] '{first}' ve '{second}' grupları {len(overlap)} görüntüde "
            f"kesişiyor. Örnek: {sorted(overlap)[:5]}"
        )


def test_split_groups_do_not_overlap():
    """Train / validation / test görüntü kümeleri birbirinden tamamen ayrı olmalı."""
    if SPLIT_CSV.is_file():
        images_per_split = _read_images_per_split_from_csv(SPLIT_CSV)
        source_label = str(SPLIT_CSV)
    else:
        images_per_split = _images_per_split_from_function()
        source_label = "sentetik kayıtlar (CSV bulunamadı)"

    _assert_no_overlap(images_per_split, source_label)


if __name__ == "__main__":
    test_split_groups_do_not_overlap()
    print("Test geçti: split grupları arasında görüntü kesişimi yok.")
