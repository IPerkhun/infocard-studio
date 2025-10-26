from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, HttpUrl

Label = Literal["L", "M", "S", "NONE"]


class AgentState(BaseModel):
    input_data: Dict
    result: Dict[str, Any] = Field(default_factory=dict)


class DetectProductInput(BaseModel):
    image_url: HttpUrl
    question: str


class DetectProductOutput(BaseModel):
    label: Label = Field(..., description="Класс габарита посуды")


class ImageGenOutput(BaseModel):
    generated_images: List[str]


class HeadersOutput(BaseModel):
    headers: List[str] = Field(..., min_items=4, max_items=4)


class UTPItem(BaseModel):
    number: int = Field(ge=1, le=8, description="Порядковый номер UTP от 1 до 8.")
    text: str = Field(description="Короткая ёмкая фраза (5–12 слов).")


class OutputLLM(BaseModel):
    title: str = Field(description="Общий заголовок карточки.")
    subtitle: str = Field(description="Подзаголовок карточки.")
    utp: List[UTPItem] = Field(
        min_length=8, max_length=8, description="Ровно 8 преимуществ с номерами 1..8."
    )

    utp_3_continue: str = Field(
        description="Короткое продолжение для пункта 3 (2–8 слов). Пример: «удобно одной рукой»"
    )
    utp_4_continue: str = Field(
        description="Короткое продолжение для пункта 4 (2–8 слов). Пример: «экономит время на готовке»"
    )
    utp_5_continue: str = Field(
        description="Короткое продолжение для пункта 5 (2–8 слов). Пример: «безопасно для всей семьи»"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Сковорода 24 см Current с антипригарным покрытием",
                    "subtitle": "Grano серии Elite",
                    "utp": [
                        {"number": 1, "text": "Усиленный слой против царапин"},
                        {"number": 2, "text": "Для всех видов плит"},
                        {"number": 3, "text": "Click-System ручка с фиксацией"},
                        {"number": 4, "text": "Мгновенный нагрев для экономии энергии"},
                        {"number": 5, "text": "Гранитное покрытие безопасно для детей"},
                        {"number": 6, "text": "Можно мыть в посудомоечной машине"},
                        {"number": 7, "text": "Подходит для духовки до 230 °C"},
                        {"number": 8, "text": "Удобная ручка с надёжной фиксацией"},
                    ],
                    "utp_3_continue": "удобно одной рукой",
                    "utp_4_continue": "экономит время на готовке",
                    "utp_5_continue": "безопасно для всей семьи",
                }
            ]
        }
    }


class SpecsOutput(BaseModel):
    text: str


class DescriptionOutput(BaseModel):
    text: str


class ProductPhoto(BaseModel):
    image_position: Optional[int] = None
    image_url: HttpUrl


class ProductData(BaseModel):
    job_id: str
    template: Optional[Label] = None
    product_sku: Optional[str] = None
    product_name: str
    product_properties: Optional[str] = None
    product_photos: List[Union[str, ProductPhoto]]


class ImageGenInput(BaseModel):
    product_photos: List[Union[str, ProductPhoto]]
    prompt: str


class ProductCardResponse(BaseModel):
    job_id: str = Field(..., description="Идентификатор задачи")
    text: Dict[str, Any] = Field(..., description="Текстовый блок карточки ...")
    generated_images: List[Dict[str, Any]] = Field(
        ...,
        description="Список сгенерированных изображений",
        example=[
            {"image_position": 1, "image_base64": "base64string1"},
            {"image_position": 2, "image_base64": "base64string2"},
            {"image_position": 3, "image_base64": "base64string3"},
            {"image_position": 4, "image_base64": "base64string4"},
        ],
    )
