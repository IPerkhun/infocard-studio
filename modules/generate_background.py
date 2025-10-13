# %%
import os
import io
import base64
import requests
from typing import List, Dict
from tqdm import tqdm
from PIL import Image
from typing import Dict, List, Tuple

import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from configs.parameters_config import ImageEditConfig
from models_init.load_pipeline_qwen import CustomEditQwenPipeline



DEVICE = "cuda:0"
MODEL_ID = "briaai/RMBG-2.0"

rmbg_model = (
    AutoModelForImageSegmentation.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
    .to(DEVICE)
    .eval()
)

RMBG_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class BackgroundGeneration:
    def __init__(self) -> None:
        self.config = ImageEditConfig()
        self.qwen_pipeline = CustomEditQwenPipeline()

    def _fetch_image_rgb(self, url: str) -> Image.Image:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def _remove_background_rmbg20(self, img: Image.Image) -> Image.Image:
        inp = RMBG_TRANSFORM(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = rmbg_model(inp)[-1].sigmoid().cpu()
        mask = pred[0].squeeze()
        mask_pil = transforms.ToPILImage()(mask).resize(img.size)
        rgba = img.convert("RGBA")
        rgba.putalpha(mask_pil)
        return rgba

    def _get_images(self, data: Dict) -> List[Image.Image]:
        images: List[Image.Image] = []
        for url in data.get("product_photos", []):
            img = self._fetch_image_rgb(url)
            img = self._remove_background_rmbg20(img).convert("RGB")
            images.append(img)
        return images

    def generate_images(self, data: Dict, prompt: str) -> List[Image.Image]:
        images_in = self._get_images(data)

        prompt = prompt.format(
            product_name=data.get("product_name", "kitchenware item")
        )

        out_images: List[Image.Image] = []
        with torch.inference_mode():
            for img in tqdm(images_in, desc="Генерация изображения QwenEdit"):
                result = self.qwen_pipeline.pipeline(
                    image=img,
                    prompt=prompt,
                    num_inference_steps=self.config.num_inference_steps,
                    true_cfg_scale=self.config.true_cfg_scale,
                )
                out_images.append(result.images[0])
        return out_images

    def get_images(self, data: Dict, prompt) -> Tuple[List[Image.Image], List[str]]:
        out_images = self.generate_images(data, prompt)

        encoded_images: List[str] = []
        for img in out_images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            encoded_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

        return out_images, encoded_images
# %%
# temp = BackgroundGeneration()

# out_images, _ = temp.get_images(
#     {
#         "job_id": "34dece5b-f4e4-4409-a5b8-7f6d60aa2f73",
#         "product_sku": "5669545",
#         "product_name": "Кастрюля с крышкой из нержавеющей стали Доляна «Классика», 1,5 л, d=17,5 см",
#         "product_properties": "|Цвет:Серебристый|Диаметр, см:17.5|Объём, л:1.5|Высота стенки, см:8.5|Крышка:Да|Материал крышки:Стекло|Материал:Нержавеющая сталь|Тип покрытия:Без покрытия|Тип плиты:Для электрической плиты|Тип плиты:Для газовой плиты|Тип плиты:Для стеклокерамической плиты|Тип плиты:Для галогенной плиты|Капсульное дно:Да|Можно мыть в посудомоечной машине:Да",
#         "product_photos": [
#             "https://goods-photos.static1-sima-land.com/items/20392/0/1600.jpg",
#             "https://goods-photos.static1-sima-land.com/items/20392/1/1600.jpg",
#             "https://goods-photos.static1-sima-land.com/items/20392/2/1600.jpg",
#             "https://goods-photos.static1-sima-land.com/items/20392/11/1600.jpg",
#             "https://goods-photos.static1-sima-land.com/items/20392/12/1600.jpg",
#             "https://goods-photos.static1-sima-land.com/items/20392/13/1600.jpg",
#         ],
#     }
# )
# out_images[0]
# %%
