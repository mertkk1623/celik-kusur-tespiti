"""Aşama 2 – XML → YOLO dönüşüm testi.

Tek bir şeyi doğrular: sentetik bir VOC XML dosyası verildiğinde
- XML koordinatları doğru okunuyor mu,
- sınıf adı doğru ID'ye eşleniyor mu,
- normalize YOLO koordinatları beklenen değerleri veriyor mu?

Neden tek test?
    Bu aşamada dönüşüm mantığı tek bir kritik nokta. Diğer kontroller
    (split kesişimi, kutu sayıları) notebook ve modül içinde hata
    fırlatarak yapılıyor.

Çalıştırma:
    python -m pytest tests/test_xml_to_yolo.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import prepare_data  # noqa: E402  (sys.path ayarından sonra içe aktarılmalı)

# Sentetik test verisi: 200x200 görüntü, iki sınıf, iki kutu.
SYNTHETIC_XML = """<annotation>
  <filename>test_image.jpg</filename>
  <size>
    <width>200</width>
    <height>200</height>
    <depth>3</depth>
  </size>
  <object>
    <name>crazing</name>
    <bndbox>
      <xmin>10</xmin>
      <ymin>20</ymin>
      <xmax>60</xmax>
      <ymax>120</ymax>
    </bndbox>
  </object>
  <object>
    <name>scratches</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>50</ymin>
      <xmax>200</xmax>
      <ymax>150</ymax>
    </bndbox>
  </object>
</annotation>
"""

# Beklenen değerler (elle hesaplandı, 200x200 görüntü):
# crazing:   x_center=(10+60)/2/200=0.175, y_center=(20+120)/2/200=0.35
#            width=(60-10)/200=0.25,      height=(120-20)/200=0.5
# scratches: x_center=(100+200)/2/200=0.75, y_center=(50+150)/2/200=0.5
#            width=(200-100)/200=0.5,      height=(150-50)/200=0.5
EXPECTED_LINES = [
    (0, 0.175, 0.35, 0.25, 0.5),
    (5, 0.75, 0.5, 0.5, 0.5),
]


def test_xml_to_yolo_conversion():
    """Sentetik XML'in doğru YOLO satırlarına dönüştüğünü doğrular."""
    with tempfile.TemporaryDirectory() as temp_directory:
        annotation_path = Path(temp_directory) / "test_image.xml"
        annotation_path.write_text(SYNTHETIC_XML, encoding="utf-8")

        result = prepare_data.convert_annotation_to_yolo(annotation_path)

        # 1) Görüntü boyutu okundu mu?
        assert result["width"] == 200, f"Genişlik yanlış: {result['width']}"
        assert result["height"] == 200, f"Yükseklik yanlış: {result['height']}"

        # 2) Her nesne için bir satır var mı?
        assert len(result["lines"]) == 2, (
            f"Beklenen 2 YOLO satırı, bulunan: {len(result['lines'])}"
        )

        # 3) Sınıf ID eşleştirmesi ve normalize koordinatlar doğru mu?
        for index, (line, expected) in enumerate(zip(result["lines"], EXPECTED_LINES)):
            expected_class_id, expected_x, expected_y, expected_w, expected_h = expected
            parts = line.split()
            assert len(parts) == 5, f"Satır {index + 1} 5 alan içermeli: '{line}'"

            class_id = int(parts[0])
            x_center, y_center, box_width, box_height = (float(value) for value in parts[1:])

            assert class_id == expected_class_id, (
                f"Satır {index + 1}: sınıf ID beklenen {expected_class_id}, bulunan {class_id} "
                f"(satır: '{line}')"
            )
            # 6 ondalık basamakla yazıldığı için tolerans 1e-6.
            assert abs(x_center - expected_x) < 1e-6, (
                f"Satır {index + 1}: x_center beklenen {expected_x}, bulunan {x_center}"
            )
            assert abs(y_center - expected_y) < 1e-6, (
                f"Satır {index + 1}: y_center beklenen {expected_y}, bulunan {y_center}"
            )
            assert abs(box_width - expected_w) < 1e-6, (
                f"Satır {index + 1}: width beklenen {expected_w}, bulunan {box_width}"
            )
            assert abs(box_height - expected_h) < 1e-6, (
                f"Satır {index + 1}: height beklenen {expected_h}, bulunan {box_height}"
            )

        # 4) Kutu sınıf isimleri de doğru okunmuş mu?
        names = [box["class_name"] for box in result["boxes"]]
        assert names == ["crazing", "scratches"], f"Sınıf isimleri yanlış: {names}"

        # 5) Sınıf ID sabit sıralaması doğru mu?
        assert prepare_data.CLASS_TO_ID["crazing"] == 0
        assert prepare_data.CLASS_TO_ID["inclusion"] == 1
        assert prepare_data.CLASS_TO_ID["patches"] == 2
        assert prepare_data.CLASS_TO_ID["pitted_surface"] == 3
        assert prepare_data.CLASS_TO_ID["rolled-in_scale"] == 4
        assert prepare_data.CLASS_TO_ID["scratches"] == 5


if __name__ == "__main__":
    test_xml_to_yolo_conversion()
    print("Test geçti: XML → YOLO dönüşümü beklenen değerleri veriyor.")
