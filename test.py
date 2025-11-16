# %%
import base64
import json
from pathlib import Path

from pipeline import ProductCardGeneration

INPUT = Path("temp.json")
OUTPUT = Path("products_results.json")

with INPUT.open("r", encoding="utf-8") as f:
    data = json.load(f)

products = data if isinstance(data, list) else [data]

pcg = ProductCardGeneration()
results = []

for i, product in enumerate(products, 1):
    if not isinstance(product, dict):
        print(f"⚠️ Пропуск #{i}: элемент не dict (type={type(product).__name__})")
        continue

    sku = product.get("product_sku") or product.get("job_id") or f"#{i}"

    payload = pcg.run(product)
    results.append(payload)

with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

import base64

# %%
import json
import re
from pathlib import Path


def safe_dirname(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9._-]", "_", s)
    return s or "product"


inputs_path = Path("products_formatted.json")
results_path = Path("products_results.json")
out_root = Path("images")
out_root.mkdir(parents=True, exist_ok=True)

products = json.loads(inputs_path.read_text(encoding="utf-8"))
results = json.loads(results_path.read_text(encoding="utf-8"))

for payload in results:
    job_id = payload.get("job_id")
    if not job_id:
        continue

    out_dir = out_root / safe_dirname(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_images = payload.get("generated_images", [])
    if not generated_images:
        continue

    for i, img_data in enumerate(generated_images, start=1):
        if isinstance(img_data, dict):
            b64img = img_data.get("image_base64") or img_data.get("image")
        else:
            b64img = img_data

        if not isinstance(b64img, str):
            continue

        try:
            img_bytes = base64.b64decode(b64img)
        except Exception:
            continue

        (out_dir / f"image_{i}.png").write_bytes(img_bytes)

    print(f"✅ {job_id}: сохранено {len(generated_images)} изображений")

print("\n🎉 Все изображения сохранены в папку 'images/'")
# %%
from pprint import pprint

from pipeline import ProductCardGeneration
from schemas.schema import ProductCardResponse

inp = {
    "job_id": "34dece5bsdag-f4esdag4-440asdsad9-a5b8sdg-7f6d6012343521",
    "product_sku": "5669878",
    "product_name": "Сковорода гриль чугунная Доляна «Квадрат. Гриль», 26x26 см, съёмная деревянная ручка",
    "product_properties": "|Цвет:Чёрный|Форма:Квадратная|Диаметр, см:28|Крышка:Нет|Материал:Чугун|Съёмная ручка:Да|Тип покрытия:Без покрытия|Тип плиты:Для индукционной плиты|Тип плиты:Для электрической плиты|Тип плиты:Для галогенной плиты|Тип плиты:Для газовой плиты|Тип плиты:Для стеклокерамической плиты|Капсульное дно:Нет|Вид сковороды:Сковорода-гриль|Можно мыть в посудомоечной машине:Нет|Особенность:Индукционная плита",
    "product_photos": [
        {
            "image_position": 1,
            "image_url": "https://goods-photos.static1-sima-land.com/items/564932/0/1600.jpg",
        }
    ],
}

gen = ProductCardGeneration()
payload = gen.run(inp)

pprint(payload)
# %%
[
    {
        "job_id": "34dece5b-f4e4-4409-a5b8-7f6d60aa2f73",
        "template": "M",
        "product_sku": "5669545",
        "product_name": "Кастрюля с крышкой из нержавеющей стали Доляна «Классика», 1,5 л, d=17,5 см",
        "product_properties": "|Цвет:Серебристый|Диаметр, см:17.5|Объём, л:1.5|Высота стенки, см:8.5|Крышка:Да|Материал крышки:Стекло|Материал:Нержавеющая сталь|Тип покрытия:Без покрытия|Тип плиты:Для электрической плиты|Тип плиты:Для газовой плиты|Тип плиты:Для стеклокерамической плиты|Тип плиты:Для галогенной плиты|Капсульное дно:Да|Можно мыть в посудомоечной машине:Да",
        "product_photos": [
            {
                "image_id": "52187a69-2e18-4cc7-8400-0b07e0b47b14",
                "image_position": 1,
                "image_url": "https://goods-photos.static1-sima-land.com/items/564941/3/1600.jpg",
            }
        ],
    }
]
