"""NEU-DET veri setini görüntü bazında train / validation / test olarak ayırır.

Aşama 1B kapsamı:
    - Her görüntü için tek bir kayıt oluşturmak (görüntüdeki tüm sınıflar korunur)
    - Sabit seed ile tekrarlanabilir rastgele ayrım yapmak
    - splits/neu_det_split.csv dosyasını yazmak

Bu modül **yalnızca ayrım yapar**. XML → YOLO dönüşümü yapmaz, model eğitmez.

Neden görüntü bazında ayrım?
    Bir görüntüde birden fazla kusur sınıfı ve birden fazla bounding box olabilir.
    Anotasyonlar görüntüye bağlı olduğu için görüntüyü bölmek, kutuları bölmekle
    aynı şeydir. Böylece aynı görüntünün bir kutusu train'de, diğeri test'te
    kalamaz (veri sızıntısı olmaz).

Kullanım (Colab / notebook):
    import split_dataset
    result = split_dataset.create_split(
        dataset_root="/content/neu_det_raw",
        output_csv="/content/celik-kusur-tespiti/splits/neu_det_split.csv",
    )
"""

from __future__ import annotations

import csv
import random
from itertools import combinations
from pathlib import Path

import data_inspection

# --- Ayrım oranları ve hedef sayılar ---------------------------------------

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

DEFAULT_SEED = 42

# NEU-DET 1800 görüntüden oluşur: 1800 * 0.70 = 1260, 1800 * 0.15 = 270.
TARGET_TOTAL_IMAGES = 1800
TARGET_COUNTS = {"train": 1260, "validation": 270, "test": 270}

SPLIT_NAMES = ("train", "validation", "test")

# CSV sütunları. Sıra bilinçli olarak sabit: dosya elle açıldığında okunaklı olsun.
CSV_COLUMNS = ("image_rel_path", "annotation_rel_path", "defect_classes", "split")

# defect_classes hücresinde sınıfları ayıran karakter.
# Virgül kullanılmıyor çünkü CSV'de sütun ayırıcısı virgül.
CLASS_SEPARATOR = ";"


# ---------------------------------------------------------------------------
# 1) Görüntü kayıtlarını hazırlama
# ---------------------------------------------------------------------------


def build_image_records(dataset_root):
    """Veri setindeki her görüntü için tek bir kayıt üretir.

    Ne yapar?
        - data_inspection.discover_dataset_pairs() ile görüntü/XML çiftlerini bulur.
        - data_inspection.parse_voc_annotation() ile her XML'i okur
          (aynı XML okuma kodu ikinci kez yazılmadı).
        - Görüntüdeki **tüm** sınıfları tek bir listede toplar.
        - Yolları dataset_root'a göre göreli hale getirir.

    Neden tek kayıt?
        Aynı görüntü CSV'de iki satırda görünürse ayrım sırasında iki farklı
        gruba düşebilir. Tek kayıt kuralı bunu baştan engeller.

    Args:
        dataset_root: Veri setinin çıkarıldığı kök klasör.

    Returns:
        tuple:
            records (list): Her görüntü için bir sözlük.
            warnings (list): Hatalı XML / sınıfı olmayan görüntü uyarıları.
    """
    dataset_root = Path(dataset_root)
    discovered = data_inspection.discover_dataset_pairs(dataset_root)

    records = []
    warnings = []

    for pair in discovered["matched_pairs"]:
        image_path = Path(pair["image"])
        annotation_path = Path(pair["annotation"])
        parsed = data_inspection.parse_voc_annotation(annotation_path)

        class_names = [obj["name"] for obj in parsed["objects"]]
        defect_classes = _order_classes(class_names)

        if parsed["errors"]:
            warnings.append(
                {
                    "file": str(annotation_path),
                    "message": "XML hataları var: " + "; ".join(parsed["errors"]),
                }
            )
        if not defect_classes:
            warnings.append(
                {
                    "file": str(annotation_path),
                    "message": "Geçerli bounding box / sınıf bulunamadı; "
                    "defect_classes boş kalacak.",
                }
            )

        records.append(
            {
                "image_rel_path": _relative_posix(image_path, dataset_root),
                "annotation_rel_path": _relative_posix(annotation_path, dataset_root),
                "defect_classes": defect_classes,
            }
        )

    # Girdi sırası işletim sistemine göre değişebilir; sıralayıp sabitliyoruz.
    # Böylece aynı seed her makinede aynı ayrımı üretir.
    records.sort(key=lambda record: record["image_rel_path"])

    return records, warnings


# ---------------------------------------------------------------------------
# 2) Ayrım (split)
# ---------------------------------------------------------------------------


def target_counts_for(total_images):
    """İstenen train/validation/test sayılarını hesaplar.

    Ne yapar?
        - Toplam 1800 ise doğrudan 1260 / 270 / 270 döndürür.
        - Toplam farklıysa aynı oranları (70/15/15) uygular.

    Neden böyle?
        Beklenen veri 1800 görüntü. Veri eksik/fazla çıkarsa sabit sayıları
        zorlamak hata verirdi. Bu durumda oran korunur ve çağıran taraf
        uyarılır (bkz. create_split -> total_images_matches_target).

    Args:
        total_images: Ayrılacak görüntü sayısı.

    Returns:
        dict: {"train": n, "validation": n, "test": n}
    """
    if total_images <= 0:
        raise ValueError("Ayrım için en az bir görüntü gerekiyor.")

    if total_images == TARGET_TOTAL_IMAGES:
        return dict(TARGET_COUNTS)

    train_count = int(round(total_images * TRAIN_RATIO))
    validation_count = int(round(total_images * VALIDATION_RATIO))
    test_count = total_images - train_count - validation_count

    if test_count < 0:
        # Çok küçük veri setlerinde yuvarlama test'i negatife düşürebilir.
        raise ValueError(
            f"Toplam {total_images} görüntü için 70/15/15 ayrımı uygulanamıyor."
        )

    return {"train": train_count, "validation": validation_count, "test": test_count}


def split_records(records, seed=DEFAULT_SEED, counts=None):
    """Görüntü kayıtlarını karıştırıp train / validation / test olarak böler.

    Ne yapar?
        - Kayıtları yola göre sıralar (girdi sırasından bağımsız olsun diye).
        - random.Random(seed) ile karıştırır: global rastgelelik durumunu
          değiştirmez, aynı seed her zaman aynı sonucu verir.
        - İstenen sayılarda keserek her kayda "split" alanı ekler.

    Görüntü kesişimi neden olmaz?
        Her kayıt listeye bir kez girer ve dilimler ardışık olduğu için
        aynı kayıt iki gruba düşemez.

    Args:
        records: build_image_records() çıktısındaki kayıt listesi.
        seed: Rastgelelik tohumu (varsayılan 42).
        counts: {"train", "validation", "test"} sayıları. None ise
                target_counts_for(len(records)) kullanılır.

    Returns:
        list: "split" alanı eklenmiş yeni kayıt listesi.
    """
    if not records:
        raise ValueError("Ayrılacak kayıt yok.")

    if counts is None:
        counts = target_counts_for(len(records))

    total = len(records)
    if sum(counts.values()) != total:
        raise ValueError(
            f"Split sayıları toplamı ({sum(counts.values())}) "
            f"görüntü sayısına ({total}) eşit değil."
        )
    if any(counts[name] < 0 for name in SPLIT_NAMES):
        raise ValueError(f"Negatif split sayısı olamaz: {counts}")

    ordered = sorted(records, key=lambda record: record["image_rel_path"])
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)

    train_end = counts["train"]
    validation_end = train_end + counts["validation"]

    slices = (
        ("train", shuffled[:train_end]),
        ("validation", shuffled[train_end:validation_end]),
        ("test", shuffled[validation_end:]),
    )

    rows = []
    for split_name, split_records_slice in slices:
        for record in split_records_slice:
            row = dict(record)  # girdi kayıtlarını değiştirmiyoruz
            row["split"] = split_name
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# 3) Raporlama yardımcıları
# ---------------------------------------------------------------------------


def check_split_overlap(rows):
    """Split gruplarının görüntü kümelerini karşılaştırır.

    Ne yapar?
        - Her grup için görüntü yollarından bir küme oluşturur.
        - İkili kesişimleri hesaplar.
        - Birleşimdeki benzersiz görüntü sayısını verir.

    Neden gerekli?
        Aynı görüntü iki grupta olursa test seti eğitime sızar ve ölçülen
        başarım gerçekçi olmaz. Bu kontrol bunu yakalar.

    Args:
        rows: "split" alanı olan kayıt listesi.

    Returns:
        dict: Grup sayıları, kesişimler ve toplam benzersiz görüntü sayısı.
    """
    image_sets = {name: set() for name in SPLIT_NAMES}
    for row in rows:
        split_name = row["split"]
        if split_name not in image_sets:
            raise ValueError(f"Beklenmeyen split adı: {split_name}")
        image_sets[split_name].add(row["image_rel_path"])

    counts = {name: len(images) for name, images in image_sets.items()}

    overlaps = {}
    for first, second in combinations(SPLIT_NAMES, 2):
        overlap = image_sets[first] & image_sets[second]
        overlaps[f"{first}__{second}"] = len(overlap)

    all_images = set().union(*image_sets.values())

    return {
        "counts": counts,
        "overlaps": overlaps,
        "total_unique_images": len(all_images),
        "total_rows": len(rows),
        "has_overlap": any(value > 0 for value in overlaps.values()),
    }


def class_distribution_by_split(rows):
    """Her split için sınıf başına **görüntü** sayısını hesaplar.

    Önemli:
        Bir görüntü birden fazla sınıf içerebilir. Bu yüzden bir split'in
        sınıf sayıları toplamı, o split'teki görüntü sayısından büyük olabilir.
        Bu bir hata değil, veri yapısının doğal sonucudur.

    Args:
        rows: "split" ve "defect_classes" alanı olan kayıt listesi.

    Returns:
        dict: split adı -> {sınıf adı: görüntü sayısı}
    """
    distribution = {name: {class_name: 0 for class_name in data_inspection.EXPECTED_CLASSES}
                    for name in SPLIT_NAMES}

    for row in rows:
        split_name = row["split"]
        for class_name in row["defect_classes"]:
            if class_name not in distribution[split_name]:
                distribution[split_name][class_name] = 0
            distribution[split_name][class_name] += 1

    return distribution


def write_split_csv(rows, output_csv):
    """Ayrım sonucunu CSV dosyasına yazar.

    Ne yapar?
        - Sütun sırasını CSV_COLUMNS ile sabitler.
        - defect_classes listesini CLASS_SEPARATOR (';') ile birleştirir.
        - UTF-8 yazar (Türkçe karakter sorunu olmasın diye).
        - Hedef klasör yoksa oluşturur.

    Args:
        rows: "split" alanı olan kayıt listesi.
        output_csv: Yazılacak CSV yolu.

    Returns:
        Path: Yazılan dosyanın yolu.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_rel_path": row["image_rel_path"],
                    "annotation_rel_path": row["annotation_rel_path"],
                    "defect_classes": CLASS_SEPARATOR.join(row["defect_classes"]),
                    "split": row["split"],
                }
            )

    return output_csv


# ---------------------------------------------------------------------------
# 4) Tek giriş noktası
# ---------------------------------------------------------------------------


def create_split(dataset_root, output_csv, seed=DEFAULT_SEED):
    """Veri setini okuyup ayrımı yapar, CSV'yi yazar ve özet döndürür.

    Args:
        dataset_root: Veri setinin çıkarıldığı kök klasör (/content/neu_det_raw).
        output_csv: Yazılacak CSV yolu (splits/neu_det_split.csv).
        seed: Rastgelelik tohumu (varsayılan 42).

    Returns:
        dict: Kayıtlar, sayılar, sınıf dağılımı, kesişim kontrolü, uyarılar.
    """
    records, warnings = build_image_records(dataset_root)
    if not records:
        raise RuntimeError(
            f"Hiç görüntü/XML çifti bulunamadı: {dataset_root}. "
            "Aşama 1A çıktısını kontrol edin."
        )

    counts = target_counts_for(len(records))
    rows = split_records(records, seed=seed, counts=counts)
    csv_path = write_split_csv(rows, output_csv)

    overlap = check_split_overlap(rows)
    distribution = class_distribution_by_split(rows)

    if overlap["has_overlap"]:
        raise RuntimeError(
            "Split grupları arasında görüntü kesişimi var, bu bir hatadır: "
            f"{overlap['overlaps']}"
        )

    multi_class_images = sum(1 for row in rows if len(row["defect_classes"]) > 1)

    return {
        "seed": seed,
        "dataset_root": str(dataset_root),
        "csv_path": str(csv_path),
        "total_images": len(records),
        "counts": counts,
        "total_images_matches_target": len(records) == TARGET_TOTAL_IMAGES,
        "class_distribution": distribution,
        "overlap": overlap,
        "multi_class_image_count": multi_class_images,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Küçük yardımcılar
# ---------------------------------------------------------------------------


def _order_classes(class_names):
    """Sınıf adlarını tekilleştirip sabit bir sıraya koyar.

    Neden?
        CSV'de aynı görüntü için her zaman aynı metin sırası çıksın diye.
        EXPECTED_CLASSES sırası korunur, bilinmeyen sınıflar sona alfabetik eklenir.
    """
    unique = set(class_names)
    ordered = [name for name in data_inspection.EXPECTED_CLASSES if name in unique]
    unexpected = sorted(unique - set(data_inspection.EXPECTED_CLASSES))
    return ordered + unexpected


def _relative_posix(path, root):
    """Yolu kök klasöre göre göreli ve '/' ayırıcılı hale getirir."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        # Kök dışında bir dosya (beklenmez). Sessizce atlamak yerine hata veriyoruz.
        raise ValueError(f"Dosya veri seti kökünün dışında: {path} (kök: {root})")


__all__ = [
    "TRAIN_RATIO",
    "VALIDATION_RATIO",
    "TEST_RATIO",
    "DEFAULT_SEED",
    "TARGET_TOTAL_IMAGES",
    "TARGET_COUNTS",
    "SPLIT_NAMES",
    "CSV_COLUMNS",
    "CLASS_SEPARATOR",
    "build_image_records",
    "target_counts_for",
    "split_records",
    "check_split_overlap",
    "class_distribution_by_split",
    "write_split_csv",
    "create_split",
]
