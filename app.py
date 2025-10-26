from typing import List, Union

import uvicorn
from fastapi import Body, FastAPI, HTTPException

from pipeline import ProductCardGeneration
from schemas.schema import ProductCardResponse, ProductData

app = FastAPI(title="Product Card Generation API", version="1.0.0")

product_card_generator = ProductCardGeneration()


@app.post(
    "/get_product_card",
    response_model=List[ProductCardResponse],
)

def get_product_card(
    data: Union[ProductData, List[ProductData]] = Body(
        ...,
        examples={
            "single": {
                "summary": "Один товар",
                "value": {
                    "job_id": "34dece5b-f4e4-4409-a5b8-7f6d60aa2f73",
                    "template": "L",  
                    "product_name": "Кастрюля с крышкой из нержавеющей стали",
                    "product_properties": "|Цвет:Серебристый|Диаметр, см:17.5|Объём, л:1.5|...",
                    "product_photos": [
                        {
                            "image_position": 1,
                            "image_url": "https://goods-photos.static1-sima-land.com/items/20392/0/1600.jpg",
                        }
                    ],
                },
            },
            "batch": {
                "summary": "Несколько товаров",
                "value": [
                    {
                        "job_id": "34dece5b-f4e4-4409-a5b8-7f6d60aa2f73",
                        "template": "M",
                        "product_name": "Кастрюля 1.5 л «Классика»",
                        "product_photos": [
                            {
                                "image_position": 1,
                                "image_url": "https://goods-photos.static1-sima-land.com/items/20392/0/1600.jpg",
                            }
                        ],
                    },
                    {
                        "job_id": "34dece5bsdag-f4esdag4-440asdsad9-a5b8sdg-7f6d6012343521",
                        "product_name": "Сковорода гриль «Квадрат. Гриль», 26x26 см",
                        "product_photos": [
                            {
                                "image_position": 2,
                                "image_url": "https://goods-photos.static1-sima-land.com/items/564932/0/1600.jpg",
                            }
                        ],
                    },
                ],
            },
        },
    )
):
    items: List[ProductData] = data if isinstance(data, list) else [data]

    results: List[ProductCardResponse] = []
    for item in items:
        try:
            payload = product_card_generator.run(
                input_data=item.model_dump(mode="json", exclude_none=True)
            )
            results.append(ProductCardResponse(**payload))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"{item.job_id}: {e}")

    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2031)
