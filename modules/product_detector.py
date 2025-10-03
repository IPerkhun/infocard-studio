import io, requests, torch
from PIL import Image
from models_init.load_qwen_vl import QwenVLModel


class QwenVLDetector:
    def __init__(self):
        self.device = "cuda:0"
        self.vl_model, self.processor = QwenVLModel().get_model()
        self.vl_model.eval()

    def _vl_ask(self, image: Image.Image, question: str) -> str:
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

        with torch.no_grad():
            output_ids = self.vl_model.generate(
                **inputs, max_new_tokens=16, do_sample=False, temperature=0.0
            )
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

    def predict_from_url(self, url: str, question: str) -> str:
        img = Image.open(io.BytesIO(requests.get(url, timeout=20).content)).convert(
            "RGB"
        )
        return self._vl_ask(img, question)
