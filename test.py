# %%
import base64
import json
from pathlib import Path

from pipeline import ProductCardGeneration

INPUT = Path("data/input.json")
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
import base64
import json
from pathlib import Path

INPUT = Path("products_results.json")   # твой файл с результатами
IMAGES_DIR = Path("images")            # куда сохраняем картинки
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

with INPUT.open("r", encoding="utf-8") as f:
    data = json.load(f)

products = data if isinstance(data, list) else [data]

for i, product in enumerate(products, 1):
    if not isinstance(product, dict):
        print(f"⚠️ Пропуск #{i}: элемент не dict (type={type(product).__name__})")
        continue

    job_id = product.get("job_id") or f"item_{i}"
    images = product.get("generated_images") or []

    for img in images:
        if not isinstance(img, dict):
            continue

        pos = img.get("image_position") or 0
        b64 = img.get("image_base64") or ""

        if not b64.strip():
            continue

        # вдруг там "data:image/png;base64,...."
        if "," in b64:
            b64 = b64.split(",", 1)[1]

        try:
            raw = base64.b64decode(b64)
        except Exception as e:
            print(f"⚠️ Ошибка декодирования для job_id={job_id}, position={pos}: {e}")
            continue

        filename = IMAGES_DIR / f"{job_id}_pos{pos}.png"
        with filename.open("wb") as out:
            out.write(raw)

        print(f"✅ Saved: {filename}")

# %%
import pandas as pd
import uuid
import json

# Основной датасет
DATA_FILE = "data/TEST_MODEL/На_вход_Список_товаров_для_теста_на_Озон.xlsx"       # либо data.csv
USE_EXCEL = True

# Файл с маппингом filename -> file_id
MAPPING_FILE = "drive_mapping.csv"   # или .xlsx

# === читаем основной датасет ===
if USE_EXCEL:
    df = pd.read_excel(DATA_FILE)
else:
    df = pd.read_csv(DATA_FILE)

# === читаем mapping filename -> file_id ===
# формат: filename,file_id
mapping_df = pd.read_csv(MAPPING_FILE)   # если xlsx: pd.read_excel(...)
filename_to_id = dict(zip(mapping_df["filename"], mapping_df["file_id"]))

def make_download_url_from_filename(filename: str) -> str:
    """
    Берём имя файла, смотрим в словарь filename -> file_id
    и возвращаем ссылку на скачивание.
    Если не нашли — возвращаем сам filename (чтобы скрипт не падал).
    """
    filename = filename.strip()
    file_id = filename_to_id.get(filename)
    if not file_id:
        # не нашли в маппинге — оставляем как есть
        return filename
    return f"https://drive.usercontent.google.com/u/0/uc?id={file_id}&export=download"

result = []

for _, row in df.iterrows():
    sku = str(row["Артикул товара"]).strip()

    photos = []
    for i in range(1, 5):
        col_name = f"Картинка товара {i}"
        if col_name not in df.columns:
            continue

        value = row[col_name]
        if pd.isna(value):
            continue

        filename = str(value).strip()
        if not filename:
            continue

        download_url = make_download_url_from_filename(filename)

        photos.append({
            "image_id": f"{sku}_{i}",      # можно заменить на uuid если нужно
            "image_position": i,
            "image_url": download_url
        })

    item = {
        "job_id": str(uuid.uuid4()),  # или sku
        "product_sku": sku,
        "product_name": str(row["Название товара"]).strip() if not pd.isna(row["Название товара"]) else "",
        "product_properties": str(row["Характеристики товара"]).strip() if not pd.isna(row["Характеристики товара"]) else "",
        "product_photos": photos
    }

    result.append(item)

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Готово, записано в output.json")



#%%