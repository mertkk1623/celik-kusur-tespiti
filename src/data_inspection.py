"""NEU-DET veri setini incelemek için yardımcı fonksiyonlar.

Bu modül yalnızca veriyi **okur ve raporlar**. Veri dönüştürmez, veri setini
bölmez ve model eğitmez.

Aşama 1A kapsamı:
    - Görüntü ve XML (Pascal VOC) dosyalarını bulup eşleştirmek
    - XML içeriğini okuyup doğrulamak
    - Özet istatistikler üretmek
    - Sınıf dağılımı ve örnek anotasyon görsellerini çizmek

Kullanım örneği (Colab / notebook):
    import data_inspection
    summary = data_inspection.inspect_dataset("/content/neu_det_raw")
    data_inspection.plot_class_distribution(summary, "results/class_distribution.png")
    pairs = data_inspection.discover_dataset_pairs("/content/neu_det_raw")["matched_pairs"]
    data_inspection.plot_annotation_samples(pairs, "results/annotation_samples.png")

Not: Matplotlib yalnızca çizim fonksiyonları çağrıldığında içe aktarılır.
Böylece veri inceleme kısmı grafik kütüphanesi olmadan da çalışabilir.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

# Veri setinde kabul edilen görüntü uzantıları.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# NEU-DET veri setinde beklenen altı kusur sınıfı.
# Sıra bilinçli olarak sabittir: rapor ve grafiklerde aynı sıra kullanılır.
EXPECTED_CLASSES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)

# NEU-DET görüntülerinin beklenen boyutu.
EXPECTED_IMAGE_SIZE = (200, 200)

# Grafiklerde her sınıf için sabit renk. Aynı sınıf her görselde aynı renkte görünür.
CLASS_COLORS = {
    "crazing": "#e6194b",
    "inclusion": "#3cb44b",
    "patches": "#4363d8",
    "pitted_surface": "#f58231",
    "rolled-in_scale": "#911eb4",
    "scratches": "#008080",
}
DEFAULT_COLOR = "#ffd700"


# ---------------------------------------------------------------------------
# 1) Dosya bulma ve eşleştirme
# ---------------------------------------------------------------------------


def discover_dataset_pairs(dataset_root):
    """Klasörün altındaki görüntü ve XML dosyalarını bulup isimlerine göre eşleştirir.

    Ne yapar?
        - Alt klasörler dahil tüm görüntü ve XML dosyalarını tarar.
        - Dosya adını (uzantısız isim) anahtar kabul edip görüntü ile XML'i eşleştirir.
        - Eşleşmeyen dosyaları ayrı ayrı listeler.
        - Aynı isimde birden fazla dosya varsa bunu "duplicate" olarak raporlar.

    Neden dosya adına göre?
        NEU-DET'te her görüntü için aynı adlı bir VOC XML dosyası bulunur.
        Klasör yapısı sürümden sürüme değişebildiği için sabit bir klasör yolu
        varsaymak yerine dosya adı eşleştirmesi kullanılıyor.

    Args:
        dataset_root: Veri setinin çıkarıldığı kök klasör.

    Returns:
        dict: Dosya listeleri, eşleşen çiftler ve sorunlu dosyalar.
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Veri seti klasörü bulunamadı: {dataset_root}")

    # rglob("*") alt klasörleri de tarar. Sıralama, çıktının her çalıştırmada
    # aynı olmasını sağlar (tekrarlanabilirlik).
    images = sorted(
        (p for p in dataset_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: str(p).lower(),
    )
    annotations = sorted(
        (p for p in dataset_root.rglob("*") if p.is_file() and p.suffix.lower() == ".xml"),
        key=lambda p: str(p).lower(),
    )

    images_by_stem = defaultdict(list)
    for path in images:
        images_by_stem[path.stem.lower()].append(path)

    annotations_by_stem = defaultdict(list)
    for path in annotations:
        annotations_by_stem[path.stem.lower()].append(path)

    # Aynı isim birden fazla klasörde varsa eşleştirme belirsizleşir.
    # Sessizce birini seçmek yerine bunu rapora ekliyoruz.
    duplicate_image_stems = {stem: paths for stem, paths in images_by_stem.items() if len(paths) > 1}
    duplicate_annotation_stems = {
        stem: paths for stem, paths in annotations_by_stem.items() if len(paths) > 1
    }

    matched_pairs = []
    unmatched_images = []
    unmatched_annotations = []

    for stem in sorted(set(images_by_stem) | set(annotations_by_stem)):
        stem_images = images_by_stem.get(stem, [])
        stem_annotations = annotations_by_stem.get(stem, [])

        if stem_images and stem_annotations:
            matched_pairs.append({"image": stem_images[0], "annotation": stem_annotations[0]})
            # Aynı isimde fazladan dosya varsa onlar da eşleşmemiş sayılır.
            unmatched_images.extend(stem_images[1:])
            unmatched_annotations.extend(stem_annotations[1:])
        elif stem_images:
            unmatched_images.extend(stem_images)
        else:
            unmatched_annotations.extend(stem_annotations)

    return {
        "dataset_root": str(dataset_root),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "images": images,
        "annotations": annotations,
        "matched_pairs": matched_pairs,
        "unmatched_images": sorted(unmatched_images, key=lambda p: str(p).lower()),
        "unmatched_annotations": sorted(unmatched_annotations, key=lambda p: str(p).lower()),
        "duplicate_image_stems": duplicate_image_stems,
        "duplicate_annotation_stems": duplicate_annotation_stems,
    }


# ---------------------------------------------------------------------------
# 2) XML (Pascal VOC) okuma ve doğrulama
# ---------------------------------------------------------------------------


def parse_voc_annotation(annotation_path):
    """Bir Pascal VOC XML dosyasını okur ve içeriğini doğrular.

    Ne yapar?
        - Görüntü genişliği/yüksekliğini okur.
        - Her nesnenin sınıf adını ve bounding box koordinatlarını çıkarır.
        - Eksik, sayı olmayan veya mantıksız alanları hata listesine yazar.

    Önemli davranış:
        Hatalı bir nesne kutusu sessizce atılmaz; hata mesajı üretilir ve o
        nesne geçerli kutular listesine eklenmez. Böylece rapor gerçeği yansıtır.

    Args:
        annotation_path: XML dosyasının yolu.

    Returns:
        dict: Dosya yolu, görüntü adı, boyut, geçerli nesneler ve hata listesi.
    """
    annotation_path = Path(annotation_path)
    result = {
        "path": str(annotation_path),
        "image_filename": None,
        "width": None,
        "height": None,
        "objects": [],
        "errors": [],
    }

    if not annotation_path.is_file():
        result["errors"].append("XML dosyası bulunamadı.")
        return result

    try:
        root = ET.parse(annotation_path).getroot()
    except ET.ParseError as exc:
        result["errors"].append(f"XML ayrıştırılamadı: {exc}")
        return result
    except OSError as exc:
        result["errors"].append(f"XML dosyası okunamadı: {exc}")
        return result

    if root.tag != "annotation":
        result["errors"].append(
            f"Beklenmeyen kök etiket: '{root.tag}' (Pascal VOC için 'annotation' beklenir)."
        )

    filename_text = root.findtext("filename")
    result["image_filename"] = filename_text.strip() if filename_text else None

    # --- Görüntü boyutu ---
    width = height = None
    size_element = root.find("size")
    if size_element is None:
        result["errors"].append("'size' alanı yok; görüntü boyutu okunamadı.")
    else:
        width = _parse_int(size_element.findtext("width"))
        height = _parse_int(size_element.findtext("height"))
        if width is None:
            result["errors"].append("'size/width' alanı eksik veya geçerli bir tam sayı değil.")
        if height is None:
            result["errors"].append("'size/height' alanı eksik veya geçerli bir tam sayı değil.")
        if width is not None and width <= 0:
            result["errors"].append(f"'size/width' pozitif olmalı, bulunan: {width}.")
            width = None
        if height is not None and height <= 0:
            result["errors"].append(f"'size/height' pozitif olmalı, bulunan: {height}.")
            height = None
    result["width"] = width
    result["height"] = height

    # --- Nesneler ve bounding box'lar ---
    for index, obj in enumerate(root.findall("object"), start=1):
        label = f"object[{index}]"

        name_text = obj.findtext("name")
        if not name_text or not name_text.strip():
            result["errors"].append(f"{label}: 'name' alanı yok veya boş.")
            continue
        class_name = name_text.strip()

        bndbox = obj.find("bndbox")
        if bndbox is None:
            result["errors"].append(f"{label} ('{class_name}'): 'bndbox' alanı yok.")
            continue

        coordinates = {}
        coordinates_ok = True
        for field in ("xmin", "ymin", "xmax", "ymax"):
            value = _parse_float(bndbox.findtext(field))
            if value is None:
                result["errors"].append(
                    f"{label} ('{class_name}'): '{field}' alanı eksik veya sayı değil."
                )
                coordinates_ok = False
            else:
                coordinates[field] = value
        if not coordinates_ok:
            continue

        # Koordinat mantığı: sol üst, sağ alt köşeden küçük olmalı.
        if coordinates["xmin"] >= coordinates["xmax"]:
            result["errors"].append(
                f"{label} ('{class_name}'): xmin >= xmax "
                f"({coordinates['xmin']} >= {coordinates['xmax']})."
            )
            coordinates_ok = False
        if coordinates["ymin"] >= coordinates["ymax"]:
            result["errors"].append(
                f"{label} ('{class_name}'): ymin >= ymax "
                f"({coordinates['ymin']} >= {coordinates['ymax']})."
            )
            coordinates_ok = False

        # Kutu görüntü sınırlarının dışına çıkmamalı.
        if min(coordinates.values()) < 0:
            result["errors"].append(f"{label} ('{class_name}'): negatif koordinat var.")
            coordinates_ok = False
        if width is not None and coordinates["xmax"] > width:
            result["errors"].append(
                f"{label} ('{class_name}'): xmax ({coordinates['xmax']}) görüntü genişliğini "
                f"({width}) aşıyor."
            )
            coordinates_ok = False
        if height is not None and coordinates["ymax"] > height:
            result["errors"].append(
                f"{label} ('{class_name}'): ymax ({coordinates['ymax']}) görüntü yüksekliğini "
                f"({height}) aşıyor."
            )
            coordinates_ok = False

        if not coordinates_ok:
            continue

        result["objects"].append(
            {
                "name": class_name,
                "xmin": coordinates["xmin"],
                "ymin": coordinates["ymin"],
                "xmax": coordinates["xmax"],
                "ymax": coordinates["ymax"],
            }
        )

    return result


# ---------------------------------------------------------------------------
# 3) Veri seti incelemesi
# ---------------------------------------------------------------------------


def inspect_dataset(dataset_root):
    """Veri setinin tamamını inceleyip özet bir rapor (dict) üretir.

    Ne hesaplar?
        - Toplam görüntü ve XML sayısı
        - Sınıf başına **görüntü** sayısı (bir görüntü o sınıfı en az bir kez içeriyorsa)
        - Sınıf başına **bounding box** sayısı (kutu sayısı, görüntü sayısı değil)
        - Görüntü boyutları ve beklenen 200x200 dışındaki görüntüler
        - Eksik XML'ler, eşleşmeyen dosyalar, hatalı XML'ler, beklenmeyen sınıflar

    Neden görüntü ve kutu sayıları ayrı?
        Bir görüntüde aynı sınıftan birden fazla kusur olabilir. Bu durumda
        görüntü sayısı 1 artarken kutu sayısı 2+ artar. İki metriği
        karıştırmamak için ayrı ayrı tutuluyor.

    Not:
        "Eksik XML" ve "eşleşmeyen görüntü" pratikte aynı kümeyi anlatır:
        XML'i olmayan görüntü dosyaları. İstendiği için ikisi de raporda yer alır.

    Args:
        dataset_root: Veri setinin çıkarıldığı kök klasör.

    Returns:
        dict: JSON'a doğrudan yazılabilir özet rapor.
    """
    dataset_root = Path(dataset_root)
    discovered = discover_dataset_pairs(dataset_root)

    class_image_counts = {name: 0 for name in EXPECTED_CLASSES}
    class_bbox_counts = {name: 0 for name in EXPECTED_CLASSES}
    unexpected_class_names = set()

    invalid_annotations = []
    unreadable_images = []
    images_without_objects = []
    images_with_multiple_classes = 0

    size_counter = Counter()
    non_standard_size_images = []
    all_widths = []
    all_heights = []

    for pair in discovered["matched_pairs"]:
        image_path = pair["image"]
        parsed = parse_voc_annotation(pair["annotation"])

        # Bu görüntüde geçen sınıflar (tekrar etmeyen küme).
        classes_in_image = set()
        for obj in parsed["objects"]:
            class_name = obj["name"]
            if class_name not in class_image_counts:
                unexpected_class_names.add(class_name)
                class_image_counts[class_name] = 0
                class_bbox_counts[class_name] = 0
            class_bbox_counts[class_name] += 1
            classes_in_image.add(class_name)

        for class_name in classes_in_image:
            class_image_counts[class_name] += 1

        if len(classes_in_image) > 1:
            images_with_multiple_classes += 1

        if parsed["errors"]:
            invalid_annotations.append(
                {"file": str(pair["annotation"]), "errors": parsed["errors"]}
            )
        elif not parsed["objects"]:
            # Geçerli XML ama hiç kutu yok: bu bir veri sorunudur, sessizce geçilmez.
            images_without_objects.append(str(image_path))

        # Gerçek görüntü boyutunu diskteki dosyadan okuyoruz (XML'e güvenmiyoruz).
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as exc:  # bozuk/okunamayan görüntü
            unreadable_images.append({"file": str(image_path), "error": str(exc)})
            continue

        all_widths.append(width)
        all_heights.append(height)
        size_counter[f"{width}x{height}"] += 1
        if (width, height) != EXPECTED_IMAGE_SIZE:
            non_standard_size_images.append({"file": str(image_path), "size": f"{width}x{height}"})

    missing_xml_files = [str(p) for p in discovered["unmatched_images"]]
    unmatched_xml_files = [str(p) for p in discovered["unmatched_annotations"]]

    summary = {
        "dataset_root": str(dataset_root),
        "total_images": discovered["image_count"],
        "total_xml_annotations": discovered["annotation_count"],
        "matched_pairs": len(discovered["matched_pairs"]),
        "class_image_counts": dict(class_image_counts),
        "class_bbox_counts": dict(class_bbox_counts),
        "image_dimensions": {
            "expected": f"{EXPECTED_IMAGE_SIZE[0]}x{EXPECTED_IMAGE_SIZE[1]}",
            "most_common": size_counter.most_common(1)[0][0] if size_counter else None,
            "unique_size_counts": dict(size_counter.most_common()),
            "min_width": min(all_widths) if all_widths else None,
            "max_width": max(all_widths) if all_widths else None,
            "min_height": min(all_heights) if all_heights else None,
            "max_height": max(all_heights) if all_heights else None,
            "non_standard_size_count": len(non_standard_size_images),
            "non_standard_size_images": non_standard_size_images,
        },
        "missing_xml_count": len(missing_xml_files),
        "missing_xml_files": missing_xml_files,
        "unmatched_image_count": len(missing_xml_files),
        "unmatched_image_files": missing_xml_files,
        "unmatched_xml_count": len(unmatched_xml_files),
        "unmatched_xml_files": unmatched_xml_files,
        "invalid_annotation_count": len(invalid_annotations),
        "invalid_annotations": invalid_annotations,
        "unreadable_image_count": len(unreadable_images),
        "unreadable_images": unreadable_images,
        "images_without_objects_count": len(images_without_objects),
        "images_without_objects": images_without_objects,
        "images_with_multiple_classes": images_with_multiple_classes,
        "unexpected_class_names": sorted(unexpected_class_names),
        "duplicate_image_stems": {
            stem: [str(p) for p in paths]
            for stem, paths in discovered["duplicate_image_stems"].items()
        },
        "duplicate_annotation_stems": {
            stem: [str(p) for p in paths]
            for stem, paths in discovered["duplicate_annotation_stems"].items()
        },
    }
    return summary


# ---------------------------------------------------------------------------
# 4) Görsel çıktılar
# ---------------------------------------------------------------------------


def plot_class_distribution(summary, output_path):
    """Altı sınıfın görüntü sayılarını bar grafik olarak kaydeder.

    Grafikte her sınıfın yanında kaç **görüntüde** göründüğü yazar
    (kutu sayısı değil). Sınıf renkleri diğer görsellerle aynıdır.

    Args:
        summary: inspect_dataset() çıktısı.
        output_path: Kaydedilecek PNG yolu.

    Returns:
        Path: Kaydedilen dosyanın yolu.
    """
    import matplotlib

    matplotlib.use("Agg")  # Colab'da pencere açmaya çalışmadan dosyaya çizmek için.
    import matplotlib.pyplot as plt

    class_names = list(EXPECTED_CLASSES)
    counts = [int(summary["class_image_counts"].get(name, 0)) for name in class_names]

    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(class_names, counts, color=[CLASS_COLORS[name] for name in class_names])
    axis.set_title("NEU-DET: Sınıf Başına Görüntü Sayısı")
    axis.set_xlabel("Kusur sınıfı")
    axis.set_ylabel("Görüntü sayısı")
    axis.bar_label(bars, padding=2, fontsize=9)
    axis.set_ylim(0, max(counts + [1]) * 1.15)
    plt.setp(axis.get_xticklabels(), rotation=20, ha="right")
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def select_sample_items(annotated_items, max_samples=8, seed=42):
    """Örnek görsel için hangi görüntülerin çizileceğine karar verir.

    Ne yapar?
        - Her beklenen sınıftan bir görüntü seçmeye çalışır (6 sınıf -> 6 görsel).
        - Kalan yerleri rastgele görüntülerle doldurur.
        - Sonucu her çalıştırmada aynı olacak şekilde (seed) döndürür.

    Neden ayrı fonksiyon?
        Çizim kütüphanesi (matplotlib) olmadan da seçim mantığı test edilebilsin
        ve kod okunur kalsın diye.

    Args:
        annotated_items: Her biri {"image", "parsed"} içeren liste.
        max_samples: En fazla kaç görüntü seçileceği.
        seed: Rastgele seçim tohumu.

    Returns:
        list: Seçilen öğeler.
    """
    import random

    if not annotated_items:
        return []
    if max_samples < 6:
        raise ValueError("En az 6 örnek gerekiyor (max_samples >= 6).")

    # Sınıf adı -> o sınıfı içeren görüntülerin indeksleri.
    by_class = defaultdict(list)
    for index, item in enumerate(annotated_items):
        for class_name in {obj["name"] for obj in item["parsed"]["objects"]}:
            by_class[class_name].append(index)

    rng = random.Random(seed)
    selected_indices = []

    # 1. adım: her beklenen sınıftan bir örnek seç (6 sınıf -> 6 görsel).
    for class_name in EXPECTED_CLASSES:
        candidates = [i for i in by_class.get(class_name, []) if i not in selected_indices]
        if candidates:
            selected_indices.append(rng.choice(candidates))

    # 2. adım: kalan yerleri rastgele görüntülerle doldur.
    remaining = [i for i in range(len(annotated_items)) if i not in selected_indices]
    rng.shuffle(remaining)
    target = min(max_samples, len(annotated_items))
    while len(selected_indices) < target and remaining:
        selected_indices.append(remaining.pop())

    return [annotated_items[i] for i in selected_indices[:max_samples]]


def plot_annotation_samples(matched_pairs, output_path, max_samples=8, seed=42):
    """Gerçek görüntüler üzerine XML kutularını ve sınıf adlarını çizer.

    Ne yapar?
        - Örnek görüntüleri select_sample_items() ile seçer.
        - Görüntüyü olduğu gibi gösterir; kutu koordinatları XML'deki
          piksellerle aynı ölçekte çizilir.

    Args:
        matched_pairs: discover_dataset_pairs()["matched_pairs"] listesi.
        output_path: Kaydedilecek PNG yolu.
        max_samples: En fazla kaç görüntü gösterileceği (6-8 arası).
        seed: Rastgele seçimin tekrarlanabilir olması için tohum.

    Returns:
        Path: Kaydedilen dosyanın yolu.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not matched_pairs:
        raise ValueError("Çizim için eşleşmiş görüntü/XML çifti bulunamadı.")
    if max_samples < 6:
        raise ValueError("En az 6 örnek gerekiyor (max_samples >= 6).")

    # Kutuları çizmek için XML'leri yeniden okuyoruz (yalnızca seçilen görüntüler için).
    annotated = []
    for pair in matched_pairs:
        parsed = parse_voc_annotation(pair["annotation"])
        if parsed["objects"]:
            annotated.append({"image": Path(pair["image"]), "parsed": parsed})
    if not annotated:
        raise ValueError("Geçerli bounding box içeren görüntü bulunamadı; örnek görsel üretilemedi.")

    selected = select_sample_items(annotated, max_samples=max_samples, seed=seed)

    columns = min(4, len(selected))
    rows = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")

    for axis, item in zip(axes.flat, selected):
        with Image.open(item["image"]) as loaded:
            # RGB'ye çevirmek bozuk/grayscale görüntülerde de çizim yapmayı sağlar.
            # Pikseller yeniden örneklenmez, görüntü olduğu gibi gösterilir.
            image = loaded.convert("RGB")
        axis.imshow(image)
        for obj in item["parsed"]["objects"]:
            color = CLASS_COLORS.get(obj["name"], DEFAULT_COLOR)
            xmin, ymin = obj["xmin"], obj["ymin"]
            width = obj["xmax"] - obj["xmin"]
            height = obj["ymax"] - obj["ymin"]
            axis.add_patch(
                Rectangle((xmin, ymin), width, height, fill=False, edgecolor=color, linewidth=1.5)
            )
            axis.text(
                xmin,
                max(ymin - 3, 2),
                obj["name"],
                color="white",
                fontsize=7,
                va="bottom",
                bbox={"facecolor": color, "alpha": 0.85, "pad": 1.2, "edgecolor": "none"},
            )
        axis.set_title(item["image"].name, fontsize=8)

    figure.suptitle("NEU-DET: Örnek Görüntüler ve VOC Kutuları")
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


# ---------------------------------------------------------------------------
# Küçük yardımcılar
# ---------------------------------------------------------------------------


def _parse_int(raw_value):
    """Metni tam sayıya çevirir; boş veya geçersizse None döndürür."""
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        return int(round(float(text)))
    except (TypeError, ValueError):
        return None


def _parse_float(raw_value):
    """Metni ondalık sayıya çevirir; boş veya geçersizse None döndürür."""
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


__all__ = [
    "IMAGE_EXTENSIONS",
    "EXPECTED_CLASSES",
    "EXPECTED_IMAGE_SIZE",
    "CLASS_COLORS",
    "discover_dataset_pairs",
    "parse_voc_annotation",
    "inspect_dataset",
    "select_sample_items",
    "plot_class_distribution",
    "plot_annotation_samples",
]
