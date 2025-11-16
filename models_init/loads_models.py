import torch
from diffusers import QwenImageEditPipeline
from transformers import AutoModelForVision2Seq, AutoProcessor

from const import QWEN_EDIT_PATH, QWEN_VL_PATH


class CustomEditQwenPipeline:
    def __init__(self):
        self.pipeline = QwenImageEditPipeline.from_pretrained(
            QWEN_EDIT_PATH, torch_dtype=torch.bfloat16
        )

        self.pipeline.set_progress_bar_config(disable=None)
        self.pipeline.enable_model_cpu_offload()


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