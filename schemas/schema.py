from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    input_data: Dict
    result: Dict[str, Any] = Field(default_factory=dict)


class DetectProductInput(BaseModel):
    image_url: str = Field(..., description="URL изображения для анализа")
    question: str = Field(..., description="Вопрос, который нужно задать модели")


class DetectProductOutput(BaseModel):
    label: Literal["L", "M", "S"] = Field(
        ..., description="Класс габарита посуды: L/M/S"
    )


class ImageGenInput(BaseModel):
    product_photos: List[str]


class ImageGenOutput(BaseModel):
    generated_images: List[str]


class HeadersOutput(BaseModel):
    headers: List[str] = Field(
        ..., min_items=4, max_items=4, description="Ровно 4 заголовка"
    )


class UTPItem(BaseModel):
    number: Annotated[
        int, Field(ge=1, le=8, description="Порядковый номер УТП от 1 до 8")
    ]
    text: str = Field(..., description="Текст характеристики УТП")


class OutputLLM(BaseModel):
    title: str = Field(..., description="Заголовок, например название товара")
    subtitle: str = Field(..., description="Подзаголовок, например серия или модель")
    utp: Annotated[
        List[UTPItem], Field(min_length=8, max_length=8, description="Ровно 8 УТП")
    ]


class SpecsOutput(BaseModel):
    text: str = Field(
        ..., description="Готовый текстовый блок с характеристиками товара"
    )

class DescriptionOutput(BaseModel):
    text: str = Field(...)


class ProductData(BaseModel):
    job_id: str
    product_sku: Optional[str]
    product_name: str
    product_properties: Optional[str]
    product_photos: List[str]


class ResponseFormat(BaseModel):
    job_id: str
    generated_images: List[str]
