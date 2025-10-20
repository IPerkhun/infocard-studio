# background_generation.py
import base64
import io
from typing import Dict, List, Optional, Tuple, Iterable
import requests
import torch
from PIL import Image
from torchvision import transforms

from configs.parameters_config import ImageEditConfig
from models_init.loads_models import CustomEditQwenPipeline, RMBGModel

_TIMEOUT = 20
_MAX_PIX = 1024


class BackgroundGeneration:
    def __init__(self) -> None:
        self.cfg = ImageEditConfig()
        self.pipe = CustomEditQwenPipeline()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.rmbg_model = RMBGModel().get_model().to(self.device).eval()
        self._to_tensor = transforms.Compose(
            [
                transforms.Resize((_MAX_PIX, _MAX_PIX)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self._session = requests.Session()

    def _fetch_image_rgb(self, url: str) -> Image.Image:
        r = self._session.get(url, timeout=_TIMEOUT)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")

    @torch.no_grad()
    def _remove_background(self, img: Image.Image) -> Image.Image:
        inp = self._to_tensor(img).unsqueeze(0).to(self.device)
        pred = self.rmbg_model(inp)[-1].sigmoid().cpu()[0]
        mask = transforms.ToPILImage()(pred).resize(img.size, Image.BILINEAR)
        rgba = img.convert("RGBA")
        rgba.putalpha(mask)
        return rgba

    @staticmethod
    def _rgba_to_rgb(cutout: Image.Image, bg=(255, 255, 255)) -> Image.Image:
        if cutout.mode != "RGBA":
            return cutout.convert("RGB")
        rgb = Image.new("RGB", cutout.size, bg)
        rgb.paste(cutout, mask=cutout.getchannel("A"))
        return rgb

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

        with torch.inference_mode():
            for url in urls:
                try:
                    img = self._fetch_image_rgb(url)
                    cutout = self._remove_background(img)
                    model_input = self._rgba_to_rgb(cutout)
                    result = self.pipe.pipeline(
                        image=model_input,
                        prompt=prompt,
                        num_inference_steps=self.cfg.num_inference_steps,
                        true_cfg_scale=self.cfg.true_cfg_scale,
                        height=self.cfg.height,    
                        width=self.cfg.width,     
                    )
                    out_images.append(result.images[0])
                except Exception:
                    continue


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
