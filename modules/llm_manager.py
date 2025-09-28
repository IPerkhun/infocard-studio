import os

from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()

MODEL_PATH = "models/Qwen_llm"


class LLMManager:
    def __init__(self, local_llm: bool = True):
        self.client = self._init_local_llm() if local_llm else self._init_chat_gpt()

    def _init_local_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype="auto",
            device_map="auto",
        )

        return pipeline("text-generation", model=model, tokenizer=tokenizer)

    def _init_chat_gpt(self):
        return OpenAI(api_key=os.getenv("OPEN_API_KEY"))
        