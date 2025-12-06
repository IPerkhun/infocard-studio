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
    def _ask_raw(
        self, image: Image.Image, question: str, max_new_tokens: int = 128
    ) -> str:
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

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=(
                self.processor.tokenizer.pad_token_id
                if self.processor.tokenizer.pad_token_id is not None
                else self.processor.tokenizer.eos_token_id
            ),
        )
        output_ids = self.model.generate(**inputs, **gen_kwargs, temperature=0.1)

        new_tokens = output_ids[:, inputs["input_ids"].shape[-1] :]
        out = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[
            0
        ].strip()

        return out

    def predict_from_url(self, url: str, question: str) -> str:
        r = self._session.get(url, timeout=20)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        return self._ask_raw(img, question)

    def predict_from_image(self, image: Image.Image, question: str) -> str:
        image = image.convert("RGB")
        return self._ask_raw(image, question)
