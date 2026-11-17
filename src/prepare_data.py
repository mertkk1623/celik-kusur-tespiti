"""NEU-DET Pascal VOC XML etiketlerini YOLO formatına dönüştürür.

Aşama 2 kapsamı:
    - Split CSV'sini okumak (Aşama 1B çıktısı)
    - Her XML nesnesini YOLO satırına çevirmek
    - Ultralytics klasör yapısını oluşturmak (images/labels x train/val/test)
    - data.yaml yazmak
    - Dönüşümü görsel olarak doğrulamak

Bu modül **model eğitmez**, model indirmez, augmentation eklemez.

Neden XML okuma kodu tekrar yazılmadı?
    Aşama 1A'daki data_inspection.parse_voc_annotation() XML'i okuyup
    genişlik/yükseklik, sınıf adları ve kutuları doğrulayarak veriyor.
    Aynı kodu ikinci kez yazmak iki farklı davranış riski doğurur.

Kullanım (Colab / notebook):
    import prepare_data
    result = prepare_data.prepare_yolo_dataset(
        dataset_root="/content/neu_det_raw",
        split_csv="/content/celik-kusur-tespiti/splits/neu_det_split.csv",
        output_dir="/content/celik-kusur-tespiti/data/processed/neu_det_yolo",
        yaml_path="/content/celik-kusur-tespiti/data/processed/neu_det_yolo/data.yaml",
    )
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from PIL import Image

import data_inspection

# ---------------------------------------------------------------------------
# Sınıf eşleştirmesi (SABİT - değiştirilmemeli)
# ---------------------------------------------------------------------------

CLASS_NAMES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)

CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

# Aşama 1A incelemesinde ölçülen toplam bounding box sayısı.
# Dönüşüm sonucu bu sayıyla karşılaştırılır.
EXPECTED_TOTAL_BOXES = 4189

# Ultralytics'te doğrulama klasörünün adı "val" olarak beklenir.
# Aşama 1B CSV'sinde bu grup "validation" adıyla geçiyor; eşleme burada yapılır.
SPLIT_DIR_NAMES = {"train": "train", "validation": "val", "test": "test"}

CSV_COLUMNS = ("image_rel_path", "annotation_rel_path", "defect_classes", "split")


# ---------------------------------------------------------------------------
# 1) Split CSV okuma
# ---------------------------------------------------------------------------


def load_split_csv(csv_path):
    """Aşama 1B'nin ürettiği split CSV'sini okur ve doğrular.

    Ne yapar?
        - Sütunların beklenen isimlerde olduğunu kontrol eder.
        - split değerlerinin train / validation / test olduğunu kontrol eder.
        - Aynı görüntünün iki kez geçmediğini kontrol eder.

    Neden doğrulama?
        CSV elle düzenlenmiş veya eski bir sürüm olabilir. Hatalı girdi ile
        dönüşüm yapmak, sessizce bozuk bir veri seti üretmek demektir.

    Args:
        csv_path: split CSV dosyasının yolu.

    Returns:
        list: Her satır için sözlük (image_rel_path, split, ...).
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Split CSV bulunamadı: {csv_path}\n"
            "Aşama 1B notebook hücrelerini çalıştırın veya Drive'daki "
            "/content/drive/MyDrive/celik-kusur-tespiti/outputs/neu_det_split.csv "
            "dosyasını kullanın."
        )

    rows = []
    seen_images = set()
    seen_annotations = set()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = [column for column in CSV_COLUMNS if column not in (reader.fieldnames or [])]
        if missing_columns:
            raise ValueError(
                f"CSV'de eksik sütun(lar) var: {missing_columns}. "
                f"Bulunan sütunlar: {reader.fieldnames}"
            )

        for line_number, row in enumerate(reader, start=2):
            image_rel_path = (row.get("image_rel_path") or "").strip()
            annotation_rel_path = (row.get("annotation_rel_path") or "").strip()
            split_name = (row.get("split") or "").strip()

            if not image_rel_path or not annotation_rel_path:
                raise ValueError(f"CSV satır {line_number}: dosya yolu boş.")
            if split_name not in SPLIT_DIR_NAMES:
                raise ValueError(
                    f"CSV satır {line_number}: beklenmeyen split değeri '{split_name}'. "
                    f"Beklenen: {sorted(SPLIT_DIR_NAMES)}"
                )
            if image_rel_path in seen_images:
                raise ValueError(
                    f"CSV satır {line_number}: görüntü birden fazla satırda geçiyor "
                    f"({image_rel_path}). Her görüntü tek satırda olmalı."
                )
            if annotation_rel_path in seen_annotations:
                raise ValueError(
                    f"CSV satır {line_number}: anotasyon birden fazla satırda geçiyor "
                    f"({annotation_rel_path})."
                )

            seen_images.add(image_rel_path)
            seen_annotations.add(annotation_rel_path)
            rows.append(
                {
                    "image_rel_path": image_rel_path,
                    "annotation_rel_path": annotation_rel_path,
                    "split": split_name,
                }
            )

    if not rows:
        raise ValueError(f"Split CSV boş: {csv_path}")

    return rows


# ---------------------------------------------------------------------------
# 2) Tek XML → YOLO satırları
# ---------------------------------------------------------------------------


def convert_annotation_to_yolo(annotation_path, image_path=None):
    """Bir XML dosyasını YOLO satırlarına çevirir.

    Ne yapar?
        - data_inspection.parse_voc_annotation() ile XML'i okur (kod tekrarı yok).
        - Sınıf adını CLASS_TO_ID ile sayıya çevirir.
        - Koordinatları görüntü boyutuna bölerek 0-1 aralığına normalize eder.
        - Bilinmeyen sınıf, hatalı kutu veya boyut uyuşmazlığında hata verir.

    Formüller:
        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        width    = (xmax - xmin) / image_width
        height   = (ymax - ymin) / image_height

    Koordinat konvansiyonu:
        Pascal VOC'de xmin/ymin sol-üst, xmax/ymax sağ-alt köşedir (dahil).
        NEU-DET etiketleri bu konvansiyonda. Herhangi bir kaydırma (+1/-1)
        uygulanmıyor; doğrulanmamış bir düzeltme yapılmıyor.

    Args:
        annotation_path: XML dosyasının yolu.
        image_path: Görüntü dosyasının yolu. Verilirse XML'deki boyut ile
                    gerçek görüntü boyutu karşılaştırılır.

    Returns:
        dict: lines (YOLO metin satırları), boxes (piksel kutuları), width, height.

    Raises:
        ValueError: XML hatalıysa, sınıf bilinmiyorsa veya boyutlar uyuşmuyorsa.
    """
    annotation_path = Path(annotation_path)
    parsed = data_inspection.parse_voc_annotation(annotation_path)

    if parsed["errors"]:
        raise ValueError(
            f"Hatalı XML: {annotation_path}\n  - " + "\n  - ".join(parsed["errors"])
        )

    xml_width = parsed["width"]
    xml_height = parsed["height"]
    if not xml_width or not xml_height:
        raise ValueError(f"XML'de geçerli görüntü boyutu yok: {annotation_path}")

    # XML'deki boyut ile gerçek görüntü boyutu aynı olmalı.
    # Farklıysa koordinatlar yanlış ölçeklenir; sessizce devam etmiyoruz.
    if image_path is not None:
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")
        with Image.open(image_path) as image:
            real_width, real_height = image.size
        if (real_width, real_height) != (xml_width, xml_height):
            raise ValueError(
                f"Boyut uyuşmazlığı: {image_path.parent.name}/{image_path.name}\n"
                f"  XML : {xml_width}x{xml_height}\n"
                f"  Gerçek: {real_width}x{real_height}\n"
                "Koordinat düzeltmesi otomatik uygulanmaz; veri kontrol edilmeli."
            )

    if not parsed["objects"]:
        raise ValueError(f"XML'de hiç nesne yok: {annotation_path}")

    lines = []
    boxes = []
    for obj in parsed["objects"]:
        class_name = obj["name"]
        if class_name not in CLASS_TO_ID:
            raise ValueError(
                f"Bilinmeyen sınıf: '{class_name}' ({annotation_path}). "
                f"Beklenen sınıflar: {list(CLASS_NAMES)}"
            )
        class_id = CLASS_TO_ID[class_name]

        xmin, ymin, xmax, ymax = obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]

        # parse_voc_annotation bu kontrolleri yapıyor; yine de savunma amaçlı.
        if xmin < 0 or ymin < 0 or xmax > xml_width or ymax > xml_height:
            raise ValueError(
                f"Kutu görüntü sınırları dışında: {annotation_path} -> "
                f"({xmin}, {ymin}, {xmax}, {ymax}) / {xml_width}x{xml_height}"
            )

        x_center = ((xmin + xmax) / 2.0) / xml_width
        y_center = ((ymin + ymax) / 2.0) / xml_height
        box_width = (xmax - xmin) / xml_width
        box_height = (ymax - ymin) / xml_height

        for value_name, value in (
            ("x_center", x_center),
            ("y_center", y_center),
            ("width", box_width),
            ("height", box_height),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Normalize değer 0-1 dışında: {value_name}={value} ({annotation_path})"
                )
        if box_width <= 0 or box_height <= 0:
            raise ValueError(f"Sıfır boyutlu kutu: {annotation_path}")

        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )
        boxes.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return {
        "lines": lines,
        "boxes": boxes,
        "width": xml_width,
        "height": xml_height,
    }


def yolo_line_to_pixels(line, image_width, image_height):
    """YOLO satırını tekrar piksel koordinatlarına çevirir (görsel doğrulama için).

    Ne yapar?
        - class_id, x_center, y_center, width, height değerlerini okur.
        - Normalize değerleri görüntü boyutuyla çarpar.
        - xmin/ymin/xmax/ymax (sol-üst, sağ-alt) olarak döndürür.

    Neden gerekli?
        Görsel doğrulamada kutuları XML'den değil, yazdığımız TXT'den çiziyoruz.
        Böylece TXT dosyasının gerçekten doğru olduğu görülür.

    Args:
        line: "class_id x_center y_center width height" metni.
        image_width: Görüntü genişliği (piksel).
        image_height: Görüntü yüksekliği (piksel).

    Returns:
        dict: class_id, xmin, ymin, xmax, ymax
    """
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"YOLO satırı 5 alan içermeli, bulunan: {len(parts)} -> '{line}'")

    class_id = int(parts[0])
    x_center, y_center, box_width, box_height = (float(value) for value in parts[1:])

    pixel_width = box_width * image_width
    pixel_height = box_height * image_height
    center_x = x_center * image_width
    center_y = y_center * image_height

    return {
        "class_id": class_id,
        "xmin": center_x - pixel_width / 2.0,
        "ymin": center_y - pixel_height / 2.0,
        "xmax": center_x + pixel_width / 2.0,
        "ymax": center_y + pixel_height / 2.0,
    }


# ---------------------------------------------------------------------------
# 3) Veri setini hazırlama
# ---------------------------------------------------------------------------


def prepare_yolo_dataset(dataset_root, split_csv, output_dir, yaml_path=None):
    """Tüm veri setini YOLO klasör yapısına dönüştürür.

    Ne yapar?
        - CSV'yi okur.
        - images/{train,val,test} ve labels/{train,val,test} klasörlerini oluşturur.
        - Her görüntüyü **orijinal haliyle** kopyalar (yeniden boyutlandırma yok).
        - Her görüntü için aynı isimli .txt etiketi yazar.
        - data.yaml dosyasını üretir.

    "Aynı isimli dosyanın üzerine yazma" kuralı:
        Hedef dosya zaten varsa hata verilir. Sessizce üzerine yazmak,
        iki farklı görüntünün aynı ada sahip olması durumunda veri kaybı
        demektir ve fark edilmez.

    Args:
        dataset_root: Ham veri seti kökü (/content/neu_det_raw).
        split_csv: Aşama 1B split CSV yolu.
        output_dir: Çıktı kökü (data/processed/neu_det_yolo).
        yaml_path: data.yaml yolu. None ise output_dir/data.yaml.

    Returns:
        dict: Sayımlar, data.yaml yolu, sınıf dağılımı, uyarılar.
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    yaml_path = Path(yaml_path) if yaml_path else output_dir / "data.yaml"

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Ham veri seti klasörü bulunamadı: {dataset_root}")

    rows = load_split_csv(split_csv)

    # Klasör yapısını oluştur.
    for split_name, directory_name in SPLIT_DIR_NAMES.items():
        (output_dir / "images" / directory_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / directory_name).mkdir(parents=True, exist_ok=True)

    split_image_counts = {name: 0 for name in SPLIT_DIR_NAMES}
    split_box_counts = {name: 0 for name in SPLIT_DIR_NAMES}
    class_box_counts = {name: 0 for name in CLASS_NAMES}
    class_image_counts = {name: 0 for name in CLASS_NAMES}
    multi_class_images = 0
    total_boxes = 0

    for row in rows:
        image_source = dataset_root / row["image_rel_path"]
        annotation_source = dataset_root / row["annotation_rel_path"]

        if not image_source.is_file():
            raise FileNotFoundError(
                f"CSV'de geçen görüntü diskte yok: {image_source}\n"
                "Ham veri seti eksik çıkarılmış olabilir."
            )
        if not annotation_source.is_file():
            raise FileNotFoundError(f"CSV'de geçen XML diskte yok: {annotation_source}")

        converted = convert_annotation_to_yolo(annotation_source, image_source)

        directory_name = SPLIT_DIR_NAMES[row["split"]]
        image_target = output_dir / "images" / directory_name / image_source.name
        label_target = output_dir / "labels" / directory_name / f"{image_source.stem}.txt"

        for target in (image_target, label_target):
            if target.exists():
                raise FileExistsError(
                    f"Hedef dosya zaten var, üzerine yazılmıyor: {target}\n"
                    "Aynı isimde iki farklı görüntü olabilir. Çıktı klasörünü "
                    "temizleyip yeniden çalıştırın."
                )

        shutil.copy2(image_source, image_target)
        label_target.write_text("\n".join(converted["lines"]) + "\n", encoding="utf-8")

        split_image_counts[row["split"]] += 1
        split_box_counts[row["split"]] += len(converted["lines"])
        total_boxes += len(converted["lines"])

        image_classes = set()
        for box in converted["boxes"]:
            class_box_counts[box["class_name"]] += 1
            image_classes.add(box["class_name"])
        for class_name in image_classes:
            class_image_counts[class_name] += 1
        if len(image_classes) > 1:
            multi_class_images += 1

    class_dirs = write_data_yaml(yaml_path, output_dir)

    return {
        "output_dir": str(output_dir),
        "data_yaml": str(yaml_path),
        "class_dirs": class_dirs,
        "split_image_counts": split_image_counts,
        "split_box_counts": split_box_counts,
        "total_images": sum(split_image_counts.values()),
        "total_boxes": total_boxes,
        "expected_total_boxes": EXPECTED_TOTAL_BOXES,
        "boxes_match_expected": total_boxes == EXPECTED_TOTAL_BOXES,
        "class_box_counts": class_box_counts,
        "class_image_counts": class_image_counts,
        "multi_class_images": multi_class_images,
        "rows": rows,
    }


def write_data_yaml(yaml_path, output_dir):
    """Ultralytics uyumlu data.yaml dosyasını yazar.

    Neden elle yazılıyor?
        Yeni kütüphane (PyYAML) eklememek için. İçerik basit ve sabit.

    Args:
        yaml_path: Yazılacak data.yaml yolu.
        output_dir: Veri seti kök klasörü (path alanına yazılır).

    Returns:
        dict: {"path": ..., "train": ..., "val": ..., "test": ...}
    """
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    class_dirs = {
        "path": Path(output_dir).resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
    }

    lines = [
        "# NEU-DET YOLO veri seti (Aşama 2 tarafından üretildi)",
        f"path: {class_dirs['path']}",
        f"train: {class_dirs['train']}",
        f"val: {class_dirs['val']}",
        f"test: {class_dirs['test']}",
        "",
        "names:",
    ]
    for class_name, class_id in CLASS_TO_ID.items():
        lines.append(f"  {class_id}: {class_name}")
    lines.append("")

    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return class_dirs


# ---------------------------------------------------------------------------
# 4) Görsel doğrulama
# ---------------------------------------------------------------------------


def plot_yolo_samples(dataset_root, split_csv, output_dir, output_png, max_samples=8, seed=42):
    """YOLO TXT kutularını gerçek görüntülerin üzerine çizer.

    Ne yapar?
        - Aşama 1B'deki select_sample_items() ile en az 6, en fazla 8 görüntü seçer
          (her sınıftan en az bir tane).
        - Kutuları **XML'den değil, yazdığımız TXT dosyasından** okur.
        - Normalize değerleri tekrar piksele çevirip çizer.
        - Ek olarak XML kutularıyla karşılaştırıp maksimum sapmayı ölçer.

    Neden TXT'den çiziyoruz?
        Amaç YOLO dönüşümünün doğru olduğunu görmek. XML'den çizmek
        dönüşümü test etmez.

    Args:
        dataset_root: Ham veri seti kökü.
        split_csv: Split CSV yolu.
        output_dir: YOLO çıktı klasörü (TXT'ler buradan okunur).
        output_png: Kaydedilecek PNG yolu.
        max_samples: En fazla kaç görüntü (6-8).
        seed: Seçim tohumu.

    Returns:
        dict: Kaydedilen yol, seçilen görüntüler, maksimum koordinat sapması.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)

    rows = load_split_csv(split_csv)

    # select_sample_items() için {"image", "parsed"} yapısı gerekiyor.
    # Aynı seçim mantığını yeniden yazmamak için onu kullanıyoruz.
    annotated_items = []
    for row in rows:
        annotation_path = dataset_root / row["annotation_rel_path"]
        parsed = data_inspection.parse_voc_annotation(annotation_path)
        if not parsed["objects"]:
            continue
        annotated_items.append(
            {
                "image": dataset_root / row["image_rel_path"],
                "parsed": parsed,
                "split": row["split"],
                "annotation": annotation_path,
            }
        )

    selected = data_inspection.select_sample_items(annotated_items, max_samples=max_samples, seed=seed)

    max_deviation = 0.0
    comparison_records = []

    columns = min(4, len(selected))
    rows_count = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows_count, columns, figsize=(4 * columns, 4 * rows_count), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")

    for axis, item in zip(axes.flat, selected):
        image_path = Path(item["image"])
        directory_name = SPLIT_DIR_NAMES[item["split"]]
        label_path = output_dir / "labels" / directory_name / f"{image_path.stem}.txt"

        if not label_path.is_file():
            raise FileNotFoundError(
                f"YOLO etiketi bulunamadı: {label_path}\n"
                "Önce prepare_yolo_dataset() çalıştırılmalı."
            )

        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
        image_width, image_height = image.size

        axis.imshow(image)

        label_lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        xml_objects = item["parsed"]["objects"]
        if len(label_lines) != len(xml_objects):
            raise ValueError(
                f"Kutu sayısı uyuşmuyor: {label_path} ({len(label_lines)}) vs "
                f"XML ({len(xml_objects)})"
            )

        for line, xml_object in zip(label_lines, xml_objects):
            box = yolo_line_to_pixels(line, image_width, image_height)
            class_name = CLASS_NAMES[box["class_id"]]
            color = data_inspection.CLASS_COLORS.get(class_name, "#ffd700")

            # Dönüşümün geri alınması XML koordinatlarını vermeli.
            deviations = (
                abs(box["xmin"] - xml_object["xmin"]),
                abs(box["ymin"] - xml_object["ymin"]),
                abs(box["xmax"] - xml_object["xmax"]),
                abs(box["ymax"] - xml_object["ymax"]),
            )
            local_deviation = max(deviations)
            max_deviation = max(max_deviation, local_deviation)
            comparison_records.append(
                {
                    "image": str(image_path),
                    "class_name": class_name,
                    "max_deviation_px": local_deviation,
                }
            )

            box_width = box["xmax"] - box["xmin"]
            box_height = box["ymax"] - box["ymin"]
            axis.add_patch(
                Rectangle(
                    (box["xmin"], box["ymin"]),
                    box_width,
                    box_height,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.5,
                )
            )
            axis.text(
                box["xmin"],
                max(box["ymin"] - 3, 2),
                f"{class_name} ({box['class_id']})",
                color="white",
                fontsize=7,
                va="bottom",
                bbox={"facecolor": color, "alpha": 0.85, "pad": 1.2, "edgecolor": "none"},
            )

        axis.set_title(f"{image_path.name}\n{item['split']}", fontsize=8)

    figure.suptitle("NEU-DET: YOLO TXT Kutuları (TXT'den geri çizim)")
    figure.tight_layout()

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return {
        "output_png": str(output_png),
        "sample_count": len(selected),
        "sample_images": [str(item["image"]) for item in selected],
        "max_deviation_px": max_deviation,
        "comparisons": comparison_records,
    }


# ---------------------------------------------------------------------------
# 5) Drive'a kopyalama
# ---------------------------------------------------------------------------


def mirror_to_drive(source_dir, drive_dir):
    """Hazır veri setini Drive'a kopyalar ve dosya sayılarını doğrular.

    Ne yapar?
        - Hedef klasörü oluşturur.
        - Klasör ağacını kopyalar (images, labels, data.yaml).
        - Kaynak ve hedefteki dosya sayılarını karşılaştırır.

    Args:
        source_dir: Yerel YOLO veri seti klasörü.
        drive_dir: Drive hedef klasörü.

    Returns:
        dict: Kaynak/hedef dosya sayıları ve doğrulama sonucu.
    """
    source_dir = Path(source_dir)
    drive_dir = Path(drive_dir)

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Kaynak klasör yok: {source_dir}")

    drive_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for source_file in sorted(source_dir.rglob("*")):
        if not source_file.is_file():
            continue
        target_file = drive_dir / source_file.relative_to(source_dir)
        target_file.parent.mkdir(parents=True, exist_ok=True)

        # Boyut ve değişiklik zamanı aynıysa gereksiz kopyalama yapmıyoruz.
        if target_file.is_file() and target_file.stat().st_size == source_file.stat().st_size:
            skipped += 1
            continue

        shutil.copy2(source_file, target_file)
        copied += 1

    source_files = sum(1 for path in source_dir.rglob("*") if path.is_file())
    drive_files = sum(1 for path in drive_dir.rglob("*") if path.is_file())

    return {
        "drive_dir": str(drive_dir),
        "source_files": source_files,
        "drive_files": drive_files,
        "copied": copied,
        "skipped": skipped,
        "verified": source_files == drive_files,
    }


__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_ID",
    "EXPECTED_TOTAL_BOXES",
    "SPLIT_DIR_NAMES",
    "load_split_csv",
    "convert_annotation_to_yolo",
    "yolo_line_to_pixels",
    "prepare_yolo_dataset",
    "write_data_yaml",
    "plot_yolo_samples",
    "mirror_to_drive",
]
