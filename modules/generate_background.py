import base64
import io
from typing import Dict, Iterable, List, Tuple

import requests
import torch
from PIL import Image

from configs.parameters_config import ImageEditConfig
from models_init.loads_models import CustomEditQwenPipeline


_TIMEOUT = 20


class BackgroundGeneration:
    def __init__(self) -> None:
        self.cfg = ImageEditConfig()
        self.pipe = CustomEditQwenPipeline()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    @staticmethod
    def _round_to_multiple(x: int, m: int = 8) -> int:
        return x - (x % m)

    @staticmethod
    def _letterbox(
        img: Image.Image, target: Tuple[int, int], bg=(255, 255, 255)
    ) -> Image.Image:
        tw, th = target
        canvas = Image.new("RGB", (tw, th), bg)
        ratio = min(tw / img.width, th / img.height)
        nw, nh = max(1, int(img.width * ratio)), max(1, int(img.height * ratio))
        resized = img.resize((nw, nh), Image.LANCZOS)
        x = (tw - nw) // 2
        y = (th - nh) // 2
        canvas.paste(resized, (x, y))
        return canvas

    def generate_images(self, data: Dict, prompt: str) -> List[Image.Image]:
        product_name = data.get("product_name", "item")
        prompt = prompt.format(product_name=product_name)

        out_images: List[Image.Image] = []
        urls = list(self._iter_urls(data.get("product_photos")))
        if not urls:
            return out_images

        target_w, target_h = 900, 1200

        with torch.inference_mode():
            for url in urls:
                try:
                    img = self._fetch_image_rgb(url)

                    canvas = self._letterbox(
                        img, (target_w, target_h), bg=(255, 255, 255)
                    )

                    pw = self._round_to_multiple(canvas.width, 8)
                    ph = self._round_to_multiple(canvas.height, 8)
                    proc = (
                        canvas
                        if (canvas.width == pw and canvas.height == ph)
                        else canvas.resize((pw, ph), Image.LANCZOS)
                    )

                    result = self.pipe.pipeline(
                        image=proc,
                        prompt=prompt,
                        num_inference_steps=self.cfg.num_inference_steps,
                        true_cfg_scale=self.cfg.true_cfg_scale,
                        negative_prompt="no people, no faces, no text, no logo, no artifacts, no distortions, no blur, no low quality, no shape change of the product.",
                    )

                    out = result.images[0]

                    if out.size != (target_w, target_h):
                        out = out.resize((target_w, target_h), Image.LANCZOS)

                    out_images.append(out)
                except Exception:
                    continue

        return out_images

    def get_images(
        self, data: Dict, prompt: str
    ) -> Tuple[List[Image.Image], List[str]]:
        pil_images = self.generate_images(data, prompt)
        enc: List[str] = []
        for im in pil_images:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            enc.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return pil_images, enc
