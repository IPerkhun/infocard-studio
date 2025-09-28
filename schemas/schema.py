from typing import Annotated, List

from pydantic import BaseModel, Field


class UTPItem(BaseModel):
    number: Annotated[int, Field(ge=1, le=8, description="Порядковый номер УТП от 1 до 8")]
    text: str = Field(..., description="Текст характеристики УТП")

class OutputLLM(BaseModel):
    title: str = Field(..., description="Заголовок, например название товара")
    subtitle: str = Field(..., description="Подзаголовок, например серия или модель")
    utp: Annotated[List[UTPItem], Field(min_length=8, max_length=8, description="Ровно 8 УТП")]
