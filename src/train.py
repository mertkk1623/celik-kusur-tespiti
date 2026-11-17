"""Aşama 3 – NEU-DET için YOLO model eğitimi.

Bu modül **tek bir eğitim denemesi** çalıştırır ve sonuçları raporlar.
Hiperparametre arama, otomatik yeniden deneme veya final değerlendirme yapmaz.

## En önemli kural: TEST SETİ KORUNUR

Eğitim, `data.yaml` içindeki test yolunu **kullanmayan** ayrı bir YAML ile yapılır.
Bu YAML dosyasında `test:` anahtarı hiç yoktur. Böylece:

    - Eğitim sırasında test görüntüleri okunmaz.
    - Test etiketleri kullanılmaz.
    - Test üzerinde tahmin veya değerlendirme yapılmaz.

Test seti yalnızca ilerideki final değerlendirme aşamasında kullanılacak.

## Kullanım (Colab / notebook)

    import train
    train.check_environment()                     # ortam + GPU kontrolü
    train.ensure_ultralytics_installed()          # eksikse kur
    train.verify_dataset("/content/celik-kusur-tespiti/data/processed/neu_det_yolo")
    train.write_training_yaml("/content/.../neu_det_yolo")   # test'siz YAML
    train.train_yolo(...)
    train.read_training_results(run_dir)
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Sabitler (V1 kapsamı – değiştirilmemeli)
# ---------------------------------------------------------------------------

MODEL_NAME = "yolo26n.pt"
EXPERIMENT_NAME = "yolo26n_exp01"

# Sınıf sıralaması Aşama 2 ile aynı olmalı. Değiştirilmemeli.
CLASS_NAMES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)

# Veri setinde beklenen görüntü sayıları.
EXPECTED_TRAIN_IMAGES = 1260
EXPECTED_VAL_IMAGES = 270

# Eğitim ayarları. Ultralytics'in doğrulanmış varsayılanları kullanılıyor;
# sadece bu listede açıkça verilen değerler değiştiriliyor.
TRAIN_SETTINGS = {
    "epochs": 50,
    "imgsz": 640,
    "batch": -1,
    "patience": 15,
    "device": 0,
    "workers": 2,
    "seed": 42,
    "deterministic": True,
    "pretrained": True,
    "amp": True,
    "plots": True,
    "save": True,
    "val": True,
}

# Üretilecek eğitim YAML'inin adı. Test yolu içermez.
TRAINING_YAML_NAME = "data_train_only.yaml"


# ---------------------------------------------------------------------------
# 1) Ortam kontrolü
# ---------------------------------------------------------------------------


def check_environment(require_cuda=True):
    """Python, PyTorch, Ultralytics ve GPU bilgilerini toplar.

    Ne yapar?
        - Sürümleri toplar.
        - CUDA ve GPU var mı bakar.
        - VRAM miktarını GB cinsinden döndürür.

    Neden gerekli?
        Eğitim GPU ister. CUDA yoksa eğitimi başlatmak saatler kaybettirir
        (CPU'da 50 epoch pratikte bitmez). Bu yüzden en başta kontrol ediyoruz.

    Args:
        require_cuda: True ise CUDA yoksa hata verir.

    Returns:
        dict: Sürümler ve GPU bilgileri.
    """
    import torch

    ultralytics_version = None
    ultralytics_error = None
    try:
        import ultralytics

        ultralytics_version = getattr(ultralytics, "__version__", "bilinmiyor")
    except Exception as error:  # kurulu değilse burada yakalanır
        ultralytics_error = str(error)

    cuda_available = torch.cuda.is_available()
    info = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "ultralytics_version": ultralytics_version,
        "cuda_available": cuda_available,
        "gpu_name": None,
        "vram_gb": None,
    }

    if cuda_available:
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        info["gpu_name"] = torch.cuda.get_device_name(device_index)
        info["vram_gb"] = round(properties.total_memory / (1024 ** 3), 2)
    elif require_cuda:
        raise RuntimeError(
            "CUDA bulunamadı, eğitim başlatılmıyor.\n"
            "Colab menüsünden GPU seçin: "
            "Runtime -> Change runtime type -> Hardware accelerator -> GPU\n"
            f"PyTorch: {torch.__version__} | CUDA derlemesi: {torch.version.cuda}"
        )

    if ultralytics_error is not None:
        # Kurulum çağıran tarafın kararı; burada sadece bilgi veriyoruz.
        info["ultralytics_error"] = ultralytics_error

    return info


def ensure_ultralytics_installed():
    """Ultralytics kurulu değilse kurar.

    Neden böyle?
        Colab'da ultralytics genelde kurulu gelir. Kurulu ise hiçbir şey
        yapmıyoruz. Eksikse sadece ultralytics kuruluyor; PyTorch/CUDA
        paketlerine dokunulmuyor.

    Returns:
        dict: Kuruldu mu, sürüm ne, çıktı ne.
    """
    try:
        import ultralytics

        return {
            "installed_now": False,
            "version": getattr(ultralytics, "__version__", "bilinmiyor"),
            "output": "Ultralytics zaten kurulu, kurulum yapılmadı.",
        }
    except ImportError:
        pass

    # --upgrade-strategy only-if-needed varsayılan; PyTorch'u zorla güncellemez.
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "ultralytics"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Ultralytics kurulamadı.\n"
            f"pip çıktısı:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )

    import ultralytics  # kurulumdan sonra tekrar dene

    return {
        "installed_now": True,
        "version": getattr(ultralytics, "__version__", "bilinmiyor"),
        "output": completed.stdout[-500:],
    }


# ---------------------------------------------------------------------------
# 2) Veri seti hazırlığı ve doğrulaması
# ---------------------------------------------------------------------------


def prepare_local_dataset(drive_dataset_dir, local_dataset_dir):
    """Drive'daki hazır veri setini yerel kopya yoksa /content altına kopyalar.

    Ne yapar?
        - Yerel klasör doluysa **hiçbir şey yapmaz** (üzerine yazmaz).
        - Yerel klasör yoksa/boşsa Drive'dan kopyalar.

    Neden?
        Drive üzerinden eğitim yapmak çok yavaştır. Yerel diske kopyalamak
        eğitimi belirgin şekilde hızlandırır. Ama mevcut veriyi yeniden
        oluşturmak veya üzerine yazmak yasak.

    Args:
        drive_dataset_dir: Drive'daki veri seti klasörü.
        local_dataset_dir: Yerel hedef klasör.

    Returns:
        dict: Kullanılan klasör, kopyalandı mı, dosya sayısı.
    """
    drive_dataset_dir = Path(drive_dataset_dir)
    local_dataset_dir = Path(local_dataset_dir)

    local_has_content = local_dataset_dir.is_dir() and any(
        path.is_file() for path in local_dataset_dir.rglob("*")
    )

    if local_has_content:
        return {
            "dataset_dir": str(local_dataset_dir),
            "copied": False,
            "message": "Yerel veri seti zaten mevcut, üzerine yazılmadı.",
        }

    if not drive_dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Veri seti ne yerelde ne Drive'da bulundu.\n"
            f"  Yerel : {local_dataset_dir}\n"
            f"  Drive : {drive_dataset_dir}\n"
            "Önce Aşama 2 hücrelerini çalıştırın."
        )

    local_dataset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(drive_dataset_dir, local_dataset_dir, dirs_exist_ok=True)

    file_count = sum(1 for path in local_dataset_dir.rglob("*") if path.is_file())
    return {
        "dataset_dir": str(local_dataset_dir),
        "copied": True,
        "file_count": file_count,
        "message": f"Drive'dan yerel klasöre kopyalandı ({file_count} dosya).",
    }


def verify_dataset(dataset_dir):
    """Veri setinin eğitime hazır olduğunu kontrol eder.

    Neleri kontrol eder?
        1. data.yaml var mı?
        2. images/train klasöründe tam 1260 görüntü var mı?
        3. images/val klasöründe tam 270 görüntü var mı?
        4. Her görüntü için aynı isimli .txt etiketi var mı?

    Test klasörü **sayılmaz** ve okunmaz. Test seti bu aşamada kullanılmıyor.
    (Yalnızca data.yaml içinde tanımlı olup olmadığı bilgi olarak raporlanır.)

    Args:
        dataset_dir: YOLO veri seti klasörü.

    Returns:
        dict: Sayılar, eksik dosyalar ve doğrulama sonucu.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Veri seti klasörü bulunamadı: {dataset_dir}")

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml bulunamadı: {data_yaml}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    def _count_and_check(split_dir_name):
        image_dir = dataset_dir / "images" / split_dir_name
        label_dir = dataset_dir / "labels" / split_dir_name
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Görüntü klasörü yok: {image_dir}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Etiket klasörü yok: {label_dir}")

        images = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        )
        missing_labels = [
            image.name for image in images
            if not (label_dir / f"{image.stem}.txt").is_file()
        ]
        return len(images), missing_labels

    train_count, train_missing = _count_and_check("train")
    val_count, val_missing = _count_and_check("val")

    problems = []
    if train_count != EXPECTED_TRAIN_IMAGES:
        problems.append(
            f"Train görüntü sayısı {train_count}, beklenen {EXPECTED_TRAIN_IMAGES}."
        )
    if val_count != EXPECTED_VAL_IMAGES:
        problems.append(
            f"Validation görüntü sayısı {val_count}, beklenen {EXPECTED_VAL_IMAGES}."
        )
    if train_missing:
        problems.append(f"Train için {len(train_missing)} eksik TXT (örnek: {train_missing[:5]}).")
    if val_missing:
        problems.append(f"Validation için {len(val_missing)} eksik TXT (örnek: {val_missing[:5]}).")

    result = {
        "dataset_dir": str(dataset_dir),
        "data_yaml": str(data_yaml),
        "train_images": train_count,
        "val_images": val_count,
        "train_missing_labels": len(train_missing),
        "val_missing_labels": len(val_missing),
        "problems": problems,
        "ok": not problems,
    }

    if problems:
        raise RuntimeError(
            "Veri seti doğrulaması başarısız, eğitim başlatılmıyor:\n  - "
            + "\n  - ".join(problems)
        )

    return result


def write_training_yaml(dataset_dir, output_path=None):
    """Yalnızca train ve val içeren eğitim YAML'ini yazar.

    **Test setini korumanın somut adımı burasıdır.**
    `data.yaml` içinde `test:` anahtarı olabilir. Eğitim için kullandığımız
    YAML'de bu anahtar hiç yazılmaz. Ultralytics eğitim sırasında yalnızca
    `train` ve `val` yollarını kullanır; `test` anahtarı yoksa test verisine
    erişemez.

    Args:
        dataset_dir: YOLO veri seti klasörü.
        output_path: Yazılacak YAML yolu. None ise dataset_dir/data_train_only.yaml.

    Returns:
        dict: YAML yolu ve içeriği.
    """
    dataset_dir = Path(dataset_dir).resolve()
    output_path = Path(output_path) if output_path else dataset_dir / TRAINING_YAML_NAME

    lines = [
        "# EĞİTİM YAML — Aşama 3 tarafından üretildi.",
        "# Bu dosyada 'test' anahtarı bilinçli olarak YOKTUR.",
        "# Test seti eğitimde kullanılmaz; final değerlendirme aşamasında ayrıca ele alınacak.",
        f"path: {dataset_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for class_id, class_name in enumerate(CLASS_NAMES):
        lines.append(f"  {class_id}: {class_name}")
    lines.append("")

    content = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

    # Güvenlik kontrolü: yanlışlıkla test satırı eklenmediğinden emin ol.
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("test:"):
            raise RuntimeError(
                f"HATA: eğitim YAML'inde test yolu var ({output_path}). "
                "Test seti eğitimde kullanılamaz."
            )

    return {"yaml_path": str(output_path), "content": content}


# ---------------------------------------------------------------------------
# 3) Eğitim
# ---------------------------------------------------------------------------


def train_yolo(
    data_yaml,
    project_dir,
    run_name=EXPERIMENT_NAME,
    model_name=MODEL_NAME,
    settings=None,
):
    """Tek bir YOLO eğitim denemesi çalıştırır.

    Ne yapar?
        - Eğitim çıktı klasörü zaten varsa **hata verir** (üzerine yazmaz).
        - `yolo26n.pt` ön eğitimli ağırlıklarını yükler.
        - Verilen ayarlarla `model.train()` çağırır.
        - Doğrulama metrikleri eğitim sırasında Ultralytics tarafından izlenir.

    Neden exist_ok=False?
        Ultralytics varsayılan olarak aynı isimde klasör varsa sonuna sayı
        ekler. Bu, iki denemenin karışmasına yol açar. Bunun yerine açık hata
        veriyoruz; kullanıcı ne yapacağına bilinçli karar verir.

    Args:
        data_yaml: Eğitim YAML yolu (test anahtarı içermeyen).
        project_dir: Eğitim çıktılarının kök klasörü.
        run_name: Deneme adı.
        model_name: Model ağırlık dosyası adı (yolo26n.pt).
        settings: Ek/override ayarlar. None ise TRAIN_SETTINGS kullanılır.

    Returns:
        dict: Çıktı klasörü ve eğitim sonucu özeti.
    """
    data_yaml = Path(data_yaml)
    project_dir = Path(project_dir)
    run_dir = project_dir / run_name

    if not data_yaml.is_file():
        raise FileNotFoundError(f"Eğitim YAML bulunamadı: {data_yaml}")

    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Eğitim klasörü zaten dolu, üzerine yazılmıyor: {run_dir}\n"
            "Önceki denemeyi inceleyin. Yeni deneme için farklı bir run_name "
            "kullanın (deney sınırı: toplam 3)."
        )

    project_dir.mkdir(parents=True, exist_ok=True)

    # Ortam/klasör kontrolleri bittikten sonra ağır kütüphaneyi yüklüyoruz.
    # Böylece hatalı durumda gereksiz model indirme/başlatma olmuyor.
    from ultralytics import YOLO

    train_settings = dict(TRAIN_SETTINGS)
    if settings:
        train_settings.update(settings)

    model = YOLO(model_name)

    results = model.train(
        data=str(data_yaml),
        project=str(project_dir),
        name=run_name,
        exist_ok=False,
        **train_settings,
    )

    return {
        "run_dir": str(run_dir),
        "project_dir": str(project_dir),
        "run_name": run_name,
        "model_name": model_name,
        "settings": train_settings,
        "results_object": results,
    }


# ---------------------------------------------------------------------------
# 4) Sonuç okuma
# ---------------------------------------------------------------------------


def read_training_results(run_dir):
    """Eğitim çıktılarından gerçek metrikleri okur.

    Ne yapar?
        - Son epoch'un validation metriklerini `results.csv` dosyasından okur.
        - En iyi epoch'u belirler.
        - Eğitim süresini `time` sütunundan alır.
        - best.pt / last.pt / grafiklerin varlığını kontrol eder.

    En iyi epoch nasıl belirlenir?
        Ultralytics`best.pt` dosyasını "fitness" değerine göre seçer.
        Detection için fitness = 0.1 * mAP50 + 0.9 * mAP50-95.
        Bu değer results.csv'de hazır sütun olarak yoksa buradan hesaplanır.
        Hesaplama şeffaf olsun diye formül kodda ve raporda belirtiliyor.

    Args:
        run_dir: Ultralytics eğitim çıktı klasörü.

    Returns:
        dict: Son epoch metrikleri, en iyi epoch, dosya yolları.
    """
    run_dir = Path(run_dir)
    results_csv = run_dir / "results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(f"results.csv bulunamadı: {results_csv}")

    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row]

    if not rows:
        raise RuntimeError(f"results.csv boş: {results_csv}")

    def _get(row, *candidate_keys):
        """Ultralytics sürümlerine göre sütun adı değişebilir; esneklik için."""
        for key in candidate_keys:
            if key in row and row[key] not in (None, ""):
                try:
                    return float(row[key])
                except (TypeError, ValueError):
                    continue
        return None

    def _fitness(row):
        map50 = _get(row, "metrics/mAP50(B)")
        map50_95 = _get(row, "metrics/mAP50-95(B)")
        if map50 is None or map50_95 is None:
            return None
        return 0.1 * map50 + 0.9 * map50_95

    fitness_values = [(index, _fitness(row)) for index, row in enumerate(rows)]
    valid_fitness = [(index, value) for index, value in fitness_values if value is not None]

    best_row_index = None
    best_fitness = None
    if valid_fitness:
        best_row_index, best_fitness = max(valid_fitness, key=lambda item: item[1])

    last_row = rows[-1]
    best_row = rows[best_row_index] if best_row_index is not None else last_row

    # Eğitim süresi: 'time' sütunu epoch sonunda kümülatif saniyedir.
    duration_seconds = _get(rows[-1], "time")
    if duration_seconds is None:
        duration_seconds = 0.0

    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    plot_files = {}
    for candidate in (
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "labels.jpg",
        "labels_correlogram.jpg",
        "BoxF1_curve.png",
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "val_batch0_pred.jpg",
        "val_batch0_labels.jpg",
    ):
        path = run_dir / candidate
        if path.is_file():
            plot_files[candidate] = str(path)

    def _metrics_from(row):
        return {
            "epoch": int(float(row.get("epoch", -1))) if row.get("epoch") else None,
            "precision": _get(row, "metrics/precision(B)"),
            "recall": _get(row, "metrics/recall(B)"),
            "mAP50": _get(row, "metrics/mAP50(B)"),
            "mAP50_95": _get(row, "metrics/mAP50-95(B)"),
        }

    return {
        "run_dir": str(run_dir),
        "results_csv": str(results_csv),
        "epochs_completed": len(rows),
        "duration_seconds": duration_seconds,
        "duration_minutes": round(duration_seconds / 60.0, 2),
        "final_epoch_metrics": _metrics_from(last_row),
        "best_epoch": (
            int(float(best_row.get("epoch", -1)))
            if best_row.get("epoch") not in (None, "")
            else None
        ),
        "best_epoch_metrics": _metrics_from(best_row),
        "best_fitness": best_fitness,
        "fitness_formula": "0.1 * mAP50 + 0.9 * mAP50-95 (Ultralytics detection fitness)",
        "best_pt": str(best_pt) if best_pt.is_file() else None,
        "last_pt": str(last_pt) if last_pt.is_file() else None,
        "plot_files": plot_files,
    }


# ---------------------------------------------------------------------------
# 5) Model dosyasını kaydetme
# ---------------------------------------------------------------------------


def save_best_model(best_pt_path, destination):
    """best.pt dosyasını hedef konuma kopyalar; var olanı ezmez.

    Ne yapar?
        - Kaynak dosyanın varlığını ve boyutunu kontrol eder.
        - Hedef dosya zaten varsa **kopyalamaz**, durumu raporlar.
        - Kopyaladıktan sonra boyutları karşılaştırır.

    Args:
        best_pt_path: Eğitim çıktısındaki weights/best.pt yolu.
        destination: Hedef dosya yolu (Drive'daki models/best.pt).

    Returns:
        dict: Durum, kaynak/hedef boyutları.
    """
    best_pt_path = Path(best_pt_path)
    destination = Path(destination)

    if not best_pt_path.is_file():
        raise FileNotFoundError(f"best.pt bulunamadı: {best_pt_path}")

    source_size = best_pt_path.stat().st_size
    if source_size == 0:
        raise RuntimeError(f"best.pt dosyası boş: {best_pt_path}")

    if destination.exists():
        return {
            "status": "skipped_existing",
            "destination": str(destination),
            "source_size_bytes": source_size,
            "destination_size_bytes": destination.stat().st_size,
            "message": (
                "Hedef dosya zaten var, üzerine yazılmadı. "
                "Mevcut dosyayı silmeden yeni modeli kaydetmek için farklı bir ad kullanın."
            ),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt_path, destination)

    destination_size = destination.stat().st_size
    if destination_size != source_size:
        raise RuntimeError(
            f"Kopyalama eksik: kaynak {source_size} bayt, hedef {destination_size} bayt."
        )

    return {
        "status": "copied",
        "destination": str(destination),
        "source_size_bytes": source_size,
        "destination_size_bytes": destination_size,
        "source_size_mb": round(source_size / (1024 ** 2), 2),
        "message": "best.pt başarıyla kopyalandı ve boyutu doğrulandı.",
    }


def write_training_summary(summary, output_path):
    """Eğitim özetini JSON olarak yazar (raporlama ve izlenebilirlik için).

    Args:
        summary: JSON'a yazılabilir sözlük.
        output_path: Yazılacak dosya yolu.

    Returns:
        Path: Yazılan dosyanın yolu.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _default(value):
        # Ultralytics nesneleri JSON'a yazılamaz; metne çeviriyoruz.
        return str(value)

    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_default),
        encoding="utf-8",
    )
    return output_path


__all__ = [
    "MODEL_NAME",
    "EXPERIMENT_NAME",
    "CLASS_NAMES",
    "EXPECTED_TRAIN_IMAGES",
    "EXPECTED_VAL_IMAGES",
    "TRAIN_SETTINGS",
    "TRAINING_YAML_NAME",
    "check_environment",
    "ensure_ultralytics_installed",
    "prepare_local_dataset",
    "verify_dataset",
    "write_training_yaml",
    "train_yolo",
    "read_training_results",
    "save_best_model",
    "write_training_summary",
]
