from io import BytesIO
from typing import Dict, List

import requests
import torch
from PIL import Image
from tqdm import tqdm

from configs.parameters_config import ImageEditConfig
from modules.load_pipeline_qwen import CustomEditQwenPipeline
from prompts.prompt import PROMPT_GENERATE_IMAGE

class BackgroundGeneration:
    def __init__(self):
        self.config = ImageEditConfig()
        self.qwen_pipeline = CustomEditQwenPipeline()

    def _fetch_image_rgb(self, url: str) -> Image.Image:
        resp = requests.get(url)
        return Image.open(BytesIO(resp.content)).convert("RGB")

    def _get_images(self, data: Dict) -> List[Image.Image]:
        return [self._fetch_image_rgb(u) for u in data.get("product_photos", [])]

    def generate_images(self, data: Dict) -> List[Image.Image]:
        images_in = self._get_images(data)

        out_images: List[Image.Image] = []
        with torch.inference_mode():
            for img in tqdm(images_in, desc="Генерация изображения"):
                result = self.qwen_pipeline.pipeline(
                    image=img,
                    prompt=PROMPT_GENERATE_IMAGE,
                    num_inference_steps=self.config.num_inference_steps,
                    true_cfg_scale=self.config.true_cfg_scale,
                )
                out_images.append(result.images[0])

        return out_images