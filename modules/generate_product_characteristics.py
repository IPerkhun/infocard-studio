from typing import Dict, Type
from pydantic import BaseModel, ValidationError

from models_init.llm_manager import LLMManager
from prompts.prompt import PROMPTS
from schemas.schema import OutputLLM, OutputHeaders


class CharacteristicsGenerate:
    def __init__(self):
        self.client = LLMManager(local_llm=True).client

    def _generate(self, prompt: str, schema: Type[BaseModel]) -> BaseModel:
        try:
            result = self.client(
                prompt,
                schema,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9,
            )
            return result
        except ValidationError as e:
            raise ValueError(f"Ответ модели не прошёл валидацию: {e}")

    def get_characteristics(self, data: Dict) -> OutputLLM:
        prompt = PROMPTS["PROMPT_PRODUCT_DESCRIPTION"].format(
            product_name=data.get("product_name", ""),
            product_properties=data.get("product_properties", ""),
        )
        return self._generate(prompt, OutputLLM)

    def get_headers(self, title: str) -> OutputHeaders:
        prompt = PROMPTS["PROMPT_GENERATE_HEADERS"].format(title=title)
        return self._generate(prompt, OutputHeaders)
