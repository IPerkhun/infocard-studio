#%%
import json
from pathlib import Path
from tqdm import tqdm
from pipeline import ProductCardGeneration

with open("products_input_20.json", "r", encoding="utf-8") as f:
    products = json.load(f)

pcg = ProductCardGeneration()
results = []

for product in tqdm(products, desc="Обработка товаров"):
    print(f"⚙️ Обработка товара: {product.get('product_sku')}")
    state = pcg.run(product)
    results.append({
        "input_data": state.input_data,
        "result": state.result,
    })

output_path = Path("products_results.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ Все результаты сохранены в {output_path.resolve()}")

# %%