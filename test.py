#%%
import json
import re
from pathlib import Path
import base64

from pipeline import ProductCardGeneration

with open("products_input_20.json", "r", encoding="utf-8") as f:
    products = json.load(f)

pcg = ProductCardGeneration()
results = []

for product in products[:2]:
    print(f"⚙️ Обработка товара: {product.get('product_sku')}")
    payload = pcg.run(product)           
    results.append(payload)

output_path = Path("products_results.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ Все результаты сохранены в {output_path.resolve()}")
#%%
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
