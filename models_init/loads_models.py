import torch
from diffusers import QwenImageEditPlusPipeline
from transformers import AutoModelForVision2Seq, AutoProcessor

from const import QWEN_EDIT_PATH, QWEN_VL_PATH


class CustomEditQwenPipeline:
    def __init__(self):
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            QWEN_EDIT_PATH,
            torch_dtype=torch.bfloat16,
        )


class QwenVLModel:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(
            QWEN_VL_PATH, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            QWEN_VL_PATH,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        self.model.eval()

    def get_model(self):
        return self.model, self.processor
