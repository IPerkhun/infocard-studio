import base64
import io
from typing import Dict, Iterable, List, Optional

import requests
import torch
from PIL import Image

from configs.parameters_config import ImageEditConfig
from models_init.loads_models import CustomEditQwenPipeline


_TIMEOUT = 20


def resize_to_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    orig_w, orig_h = img.size
    scale = max(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))
    return img


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

    def _get_generator(self) -> Optional[torch.Generator]:
        seed = getattr(self.cfg, "seed", -1)
        if seed is None or seed < 0:
            return None
        device = str(self.pipe.device) if hasattr(self.pipe, "device") else "cpu"
        return torch.Generator(device=device).manual_seed(seed)

    def generate_images(self, data: Dict, prompt: str) -> List[Image.Image]:
        product_name = data.get("product_name", "item")
        user_prompt = prompt.format(product_name=product_name)

        out_images: List[Image.Image] = []
        urls = list(self._iter_urls(data.get("product_photos")))
        if not urls:
            return out_images

        generator = self._get_generator()

        with torch.inference_mode():
            for url in urls:
                try:
                    image = self._fetch_image_rgb(url)

                    inputs = {
                        "image": image,
                        "prompt": user_prompt,
                        "negative_prompt": " ",
                        "num_inference_steps": self.cfg.num_inference_steps,
                        "true_cfg_scale": self.cfg.true_cfg_scale,
                        "height": self.cfg.height,
                        "width": self.cfg.width,
                        "num_images_per_prompt": 1,
                        "guidance_scale": 1.0,
                    }

                    if generator is not None:
                        inputs["generator"] = generator

                    result = self.pipe(**inputs)
                    out = result.images[0]

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
