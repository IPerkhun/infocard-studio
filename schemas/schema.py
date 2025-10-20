# schemas.py
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
    number: int = Field(ge=1, le=8)
    text: str


class OutputLLM(BaseModel):
    title: str
    subtitle: str
    utp: List[UTPItem] = Field(min_length=8, max_length=8)


class SpecsOutput(BaseModel):
    text: str


class DescriptionOutput(BaseModel):
    text: str


class ProductPhoto(BaseModel):
    image_id: Optional[str] = None
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
    text: Dict[str, Any] = Field(
        ...,
        description="Текстовый блок карточки",
        example={
            "m-1-slide-title": "Кастрюля Доляна «Классика» 1,5 л",
            "m-1-slide-subtitle": "Нержавеющая сталь с крышкой из стекла",
            "m-specs": "Объём: 1,5 л, Диаметр: 17,5 см, Материал: Нержавеющая сталь",
            "m-description": "Практичная кастрюля для повседневного приготовления блюд.",
            "m-1-utp": "Подходит для всех типов плит",
            "m-2-utp": "Можно мыть в посудомоечной машине",
            "m-3-utp": "Стильный серебристый цвет",
            "m-4-utp": "Капсульное дно — равномерный нагрев",
            "m-5-utp": "Прочная нержавеющая сталь",
            "m-6-utp": "Компактный размер — удобно хранить",
            "m-7-utp": "Стеклянная крышка сохраняет вкус блюд",
            "m-8-utp": "Идеальный выбор для каждой кухни",
        },
    )
    generated_images: List[Dict[str, Any]] = Field(
        ...,
        description="Список сгенерированных изображений",
        example=[
            {"image_id": "img1", "image_position": 1, "image": "base64string1"},
            {"image_id": "img2", "image_position": 2, "image": "base64string2"},
            {"image_id": "img3", "image_position": 3, "image": "base64string3"},
            {"image_id": "img4", "image_position": 4, "image": "base64string4"},
        ],
    )
