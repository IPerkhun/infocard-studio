#%%
import json
from pathlib import Path
import base64

from pipeline import ProductCardGeneration

INPUT = Path("products_formatted.json")
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
    print(f"⚙️ Обработка товара: {sku}")

    payload = pcg.run(product)
    results.append(payload)

with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ {len(results)} результатов сохранено в {OUTPUT.resolve()}")

#%%
import re
def safe_dirname(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9._-]", "", s)
    return s or "product"

inputs_path = Path("products_input_20.json")
results_path = Path("products_results.json")
out_root = Path("images")
out_root.mkdir(parents=True, exist_ok=True)

products = json.loads(inputs_path.read_text(encoding="utf-8"))
results = json.loads(results_path.read_text(encoding="utf-8"))

name_by_job = {p["job_id"]: p.get("product_name", "product") for p in products}

for payload in results:
    job_id = payload.get("job_id")
    images_b64 = payload.get("generated_images", [])
    product_name = name_by_job.get(job_id, "product")
    out_dir = out_root / safe_dirname(product_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, b64img in enumerate(images_b64, start=1):
        try:
            data = base64.b64decode(b64img)
        except Exception:
            continue
        (out_dir / f"image_{i}.png").write_bytes(data)

print("✅ Все изображения сохранены в папку 'images/'")
# %%
from pipeline import ProductCardGeneration
# %%
pcg = ProductCardGeneration()

job_input = {
    "job_id": "34dece5b-f4e4-4409-a5b8-7f6d60aa2f73",
    "template": "M", 
    "product_sku": "5669545",
    "product_name": "Кастрюля с крышкой из нержавеющей стали Доляна «Классика», 1,5 л, d=17,5 см",
    "product_properties": "|Цвет:Серебристый|Диаметр, см:17.5|Объём, л:1.5|Высота стенки, см:8.5|Крышка:Да|Материал крышки:Стекло|Материал:Нержавеющая сталь|Тип покрытия:Без покрытия|Тип плиты:Для электрической плиты|Тип плиты:Для газовой плиты|Тип плиты:Для стеклокерамической плиты|Тип плиты:Для галогенной плиты|Капсульное дно:Да|Можно мыть в посудомоечной машине:Да",
    "product_photos": [
        {
            "image_id": "52187a69-2e18-4cc7-8400-0b07e0b47b14",
            "image_position": 1,
            "image_url": "https://goods-photos.static1-sima-land.com/items/20392/0/1600.jpg"
        },
        {
            "image_id": "a60b5735-47e0-4fbf-8a03-f9e086269bb1",
            "image_position": 2,
            "image_url": "https://goods-photos.static1-sima-land.com/items/20392/1/1600.jpg"
        },
        {
            "image_id": "e479e8a1-7cd2-493c-aabd-6af864eaf9c3",
            "image_position": 3,
            "image_url": "https://goods-photos.static1-sima-land.com/items/20392/2/1600.jpg"
        },
        {
            "image_id": "ff1854d6-8982-4cfa-8e8b-7fb88f487e84",
            "image_position": 4,
            "image_url": "https://goods-photos.static1-sima-land.com/items/20392/3/1600.jpg"
        }
    ]
}
#%%
# выполняем пайплайн
result = pcg.run(input_data=job_input)

# выводим результат
from pprint import pprint
pprint(result)


# %%
