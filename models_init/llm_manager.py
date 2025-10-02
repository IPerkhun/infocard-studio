import outlines
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/qwen_llm"

load_dotenv()


class LLMManager:
    def __init__(self, local_llm: bool = True):
        self.client = self._init_local_llm() if local_llm else None
        print("Модель подгружена")

    def _init_local_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        return outlines.from_transformers(model, tokenizer)