from typing import Dict

from pydantic import ValidationError

from models_init.llm_manager import LLMManager
from prompts.prompt import PROMPT_GENERATE_CHARACTERISTICS
from schemas.schema import OutputLLM


class CharacteristicsGenerate:
    def __init__(self):
        self.client = LLMManager(local_llm=True).client

    def _build_prompt(self, data: Dict) -> str:
        return PROMPT_GENERATE_CHARACTERISTICS.format(
            product_name=data.get("product_name", ""),
            product_properties=data.get("product_properties", "")
        )

    def generate(self, data: Dict) -> OutputLLM:
        prompt = self._build_prompt(data)
        try:
            result: OutputLLM = self.client(
                prompt,
                OutputLLM,         
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9,
            )
            return result
        except ValidationError as e:
            raise ValueError(f"Ответ модели не прошёл валидацию: {e}")