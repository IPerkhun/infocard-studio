from typing import Annotated, List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ImageGenInput(BaseModel):
    job_id: str = Field(..., description="ID задачи")
    product_name: str = Field(..., description="Название товара")
    product_photos: List[str] = Field(..., description="Список ссылок на фото товара")

class ImageGenOutput(BaseModel):
    job_id: str
    generated_images: List[str] = Field(..., description="Base64 изображений")

class OutputHeaders(BaseModel):
    headers: List[str]

class UTPItem(BaseModel):
    number: Annotated[int, Field(ge=1, le=8, description="Порядковый номер УТП от 1 до 8")]
    text: str = Field(..., description="Текст характеристики УТП")

class OutputLLM(BaseModel):
    title: str = Field(..., description="Заголовок, например название товара")
    subtitle: str = Field(..., description="Подзаголовок, например серия или модель")
    utp: Annotated[List[UTPItem], Field(min_length=8, max_length=8, description="Ровно 8 УТП")]

class ProductData(BaseModel):
    job_id: str
    product_sku: Optional[str]
    product_name: str
    product_properties: Optional[str]
    product_photos: List[str]


class ResponseFormat(BaseModel):
    job_id: str
    generated_images: List[str] 


class AgentState(BaseModel):
    input_data: Dict
    result: Dict[str, Any] = Field(default_factory=dict)
