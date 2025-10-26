# %%
import base64
import json
from pathlib import Path

from pipeline import ProductCardGeneration

INPUT = Path("products_formatted.json")
OUTPUT = Path("products_results.json")

with INPUT.open("r", encoding="utf-8") as f:
    data = json.load(f)

products = data if isinstance(data, list) else [data]
products = products[:1]

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