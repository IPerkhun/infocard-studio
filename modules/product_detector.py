# detector.py
import io
import requests
import torch
from PIL import Image
from models_init.loads_models import QwenVLModel 


class QwenVLDetector:
    def __init__(self, device: str = "cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model, self.processor = QwenVLModel().get_model()
        self.model.to(self.device).eval()
        self._session = requests.Session()

    @torch.no_grad()
    def _ask(self, image: Image.Image, question: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(
            self.device
        )
        output_ids = self.model.generate(
            **inputs, max_new_tokens=16, do_sample=False, temperature=0.0
        )
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

    def predict_from_url(self, url: str, question: str) -> str:
        r = self._session.get(url, timeout=20)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        return self._ask(img, question)
