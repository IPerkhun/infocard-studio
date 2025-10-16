import base64
import io
from typing import Dict, List, Optional, Tuple

import requests
import torch
from PIL import Image
from torchvision import transforms

from configs.parameters_config import ImageEditConfig
from models_init.loads_models import CustomEditQwenPipeline, RMBGModel


class BackgroundGeneration:
    def __init__(self) -> None:
        self.config = ImageEditConfig()
        self.qwen_pipeline = CustomEditQwenPipeline()
        self.device = "cuda:0"
        self.rmbg_model = RMBGModel().get_model()
        self.rmbg_transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _fetch_image_rgb(self, url: str) -> Image.Image:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def _remove_background(self, img: Image.Image) -> Image.Image:
        inp = self.rmbg_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.rmbg_model(inp)[-1].sigmoid().cpu()
        mask = pred[0].squeeze()
        mask_pil = transforms.ToPILImage()(mask).resize(img.size, Image.BILINEAR)
        rgba = img.convert("RGBA")
        rgba.putalpha(mask_pil)
        return rgba

    def _get_cutouts(self, data: Dict) -> List[Image.Image]:
        urls = data.get("product_photos", [])
        cutouts: List[Image.Image] = []
        for url in urls:
            img = self._fetch_image_rgb(url)
            cutouts.append(self._remove_background(img))
        return cutouts

    def _rgba_to_rgb_with_alpha(
        self, cutout: Image.Image, bg=(255, 255, 255)
    ) -> Image.Image:
        if cutout.mode != "RGBA":
            return cutout.convert("RGB")
        rgb = Image.new("RGB", cutout.size, bg)
        rgb.paste(cutout, mask=cutout.split()[3])
        return rgb

    def generate_images(
        self, data: Dict, prompt: str, output_dir: Optional[str] = None
    ) -> List[Image.Image]:
        cutouts = self._get_cutouts(data)
        prompt = prompt.format(
            product_name=data.get("product_name", "kitchenware item")
        )
        out_images: List[Image.Image] = []
        with torch.inference_mode():
            for cutout in cutouts:
                rgb_for_model = self._rgba_to_rgb_with_alpha(cutout, bg=(255, 255, 255))
                result = self.qwen_pipeline.pipeline(
                    image=rgb_for_model,
                    prompt=prompt,
                    num_inference_steps=self.config.num_inference_steps,
                    true_cfg_scale=self.config.true_cfg_scale,
                )
                out_images.append(result.images[0])
        return out_images

    def get_images(
        self, data: Dict, prompt: str, output_dir: Optional[str] = None
    ) -> Tuple[List[Image.Image], List[str]]:
        out_images = self.generate_images(data, prompt, output_dir)
        encoded_images: List[str] = []
        for img in out_images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            encoded_images.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return out_images, encoded_images
