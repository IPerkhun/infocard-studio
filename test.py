#%%
import json
import pandas as pd
from pipeline import ProductCardGeneration
from tqdm import tqdm

with open("products_input_20.json", "r", encoding="utf-8") as f:
    products = json.load(f)

pcg = ProductCardGeneration()

results = []

for product in tqdm(products):
    print(f"⚙️ Обработка товара: {product['product_sku']}")

    result_state = pcg.run(product)

    results.append(result_state)
# %%
import os
import json
import base64

IN_PATH = "validate_full.json"
OUT_DIR = "images"

os.makedirs(OUT_DIR, exist_ok=True)

with open(IN_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)  
for item in data[:2]:
    job_id = (
        item.get("job_id")
        or item.get("input", {}).get("job_id")
        or "no_id"
    )

    images_b64 = (
        item.get("result", {}).get("generated_images")
        or item.get("generated_images")
        or []
    )

    item_dir = os.path.join(OUT_DIR, job_id)
    os.makedirs(item_dir, exist_ok=True)

    for idx, b64s in enumerate(images_b64, start=1):
        img_bytes = base64.b64decode(b64s)
        out_path = os.path.join(item_dir, f"{job_id}_{idx:02d}.png")
        with open(out_path, "wb") as imgf:
            imgf.write(img_bytes)

pint("✅ Изображения сохранены в папке:", OUT_DIR)

# %%
