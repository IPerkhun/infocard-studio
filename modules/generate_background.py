import base64
import io
from typing import Dict, Iterable, List

import requests
import torch
from PIL import Image

from configs.parameters_config import ImageEditConfig
from models_init.loads_models import CustomEditQwenPipeline


_TIMEOUT = 20


def pad_to_size(
    img: Image.Image, target_w: int = 900, target_h: int = 1200
) -> Image.Image:
    img = img.copy()
    w, h = img.size
    ratio = min(target_w / w, target_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    background = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    background.paste(img, offset)
    return background


class BackgroundGeneration:
    def __init__(self) -> None:
        self.cfg = ImageEditConfig()
        self.pipe = CustomEditQwenPipeline().pipeline
        self._session = requests.Session()

    def _fetch_image_rgb(self, url: str) -> Image.Image:
        r = self._session.get(url, timeout=_TIMEOUT)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")

    def _iter_urls(self, items: Iterable) -> Iterable[str]:
        for it in items or []:
            if isinstance(it, str):
                yield it
            else:
                url = it.get("image_url") if isinstance(it, dict) else None
                if url:
                    yield url

    def generate_images(self, data: Dict, prompt: str) -> List[Image.Image]:
        product_name = data.get("product_name", "item")
        prompt = prompt.format(product_name=product_name)

        out_images: List[Image.Image] = []
        urls = list(self._iter_urls(data.get("product_photos")))
        if not urls:
            return out_images

        target_w, target_h = 900, 1200

        negative_prompt = """No people, no faces, no hands, no text, no logos, no labels, no distortions, no reflections, 
        no shadows on the product, no color changes to the product, no extra objects, no artifacts, no blur, no low quality, 
        no watermarks, no decorations covering the product."""

        num_steps = getattr(self.cfg, "num_inference_steps", 35)
        true_cfg_scale = getattr(self.cfg, "true_cfg_scale", 4.0)
        guidance_scale = getattr(self.cfg, "guidance_scale", 1.0)

        with torch.inference_mode():
            for url in urls:
                try:
                    img = self._fetch_image_rgb(url)
                    image = pad_to_size(img, target_w, target_h)

                    inputs = {
                        "image": [image],
                        "prompt": [prompt],
                        "negative_prompt": [negative_prompt],
                        "num_inference_steps": num_steps,
                        "guidance_scale": guidance_scale,
                        "true_cfg_scale": true_cfg_scale,
                    }

                    result = self.pipe(**inputs)
                    out = result.images[0]

                    if out.size != (target_w, target_h):
                        out = out.resize((target_w, target_h), Image.LANCZOS)

                    out_images.append(out)
                except Exception:
                    continue

        return out_images

    def get_images(self, data: Dict, prompt: str):
        pil_images = self.generate_images(data, prompt)
        enc: List[str] = []
        for im in pil_images:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            enc.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return pil_images, enc
