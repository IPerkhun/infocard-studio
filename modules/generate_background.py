from io import BytesIO
from typing import Dict, List

import requests
import torch
from PIL import Image
from rembg import remove
from tqdm import tqdm

from configs.parameters_config import ImageEditConfig
from models_init.load_pipeline_qwen import CustomEditQwenPipeline
from prompts.prompt import PROMPT_GENERATE_IMAGE


class BackgroundGeneration:
    def __init__(self):
        self.config = ImageEditConfig()
        self.qwen_pipeline = CustomEditQwenPipeline()

    def _fetch_image_rgb(self, url: str) -> Image.Image:
        resp = requests.get(url)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")

    def _remove_text_and_background(self, img: Image.Image) -> Image.Image:
        return remove(img).convert("RGBA")

    def _get_images(self, data: Dict) -> List[Image.Image]:
        images = []
        for u in data.get("product_photos", []):
            img = self._fetch_image_rgb(u) 
            img = self._remove_text_and_background(img)
            img = img.convert("RGB")  
            images.append(img)
        return images

    def generate_images(self, data: Dict) -> List[Image.Image]:
        images_in = self._get_images(data)
        prompt = PROMPT_GENERATE_IMAGE.format(
            product_name=data.get("product_name", "kitchenware item")
        )

        out_images: List[Image.Image] = []
        with torch.inference_mode():
            for img in tqdm(images_in, desc="Генерация изображения"):
                result = self.qwen_pipeline.pipeline(
                    image=img,
                    prompt=prompt,
                    num_inference_steps=self.config.num_inference_steps,
                    true_cfg_scale=self.config.true_cfg_scale,
                )
                out_images.append(result.images[0])

        return out_images
