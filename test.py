#%%
import json
import pandas as pd
from pipeline import ProductCardGeneration

with open("products_input_20.json", "r", encoding="utf-8") as f:
    products = json.load(f)

pcg = ProductCardGeneration()

results = []

for product in products[:2]:
    print(f"⚙️ Обработка товара: {product['product_sku']}")

    result_state = pcg.run(product)

    results.append(result_state)
# %%
