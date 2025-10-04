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

IN_PATH = "validate_full.json"   # входной JSON
OUT_DIR = "images"               # корневая папка для результатов

os.makedirs(OUT_DIR, exist_ok=True)

with open(IN_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    # достаём job_id из блока input_data
    job_id = (
        item.get("input_data", {}).get("product_sku")
        or item.get("job_id")
        or "no_id"
    )

    # список base64 изображений
    images_b64 = (
        item.get("result", {}).get("generated_images")
        or item.get("generated_images")
        or []
    )

    # создаём подпапку для конкретного job_id
    job_dir = os.path.join(OUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # сохраняем все картинки из generated_images
    for idx, b64_img in enumerate(images_b64, start=1):
        img_bytes = base64.b64decode(b64_img)
        out_path = os.path.join(job_dir, f"{idx:02d}.png")
        with open(out_path, "wb") as f_out:
            f_out.write(img_bytes)

print(f"✅ Все изображения сохранены в папке: {OUT_DIR}")


# %%
