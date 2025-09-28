import torch
from diffusers import QwenImageEditPipeline

MODEL_PATH = "ovedrive/qwen-image-edit-4bit"

class CustomEditQwenPipeline:
    def __init__(self):
        self.pipeline = QwenImageEditPipeline.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16
        )

        self.pipeline.set_progress_bar_config(disable=None)
        self.pipeline.enable_model_cpu_offload() 