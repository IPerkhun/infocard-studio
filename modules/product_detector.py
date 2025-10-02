# import io
# import requests
# import torch
# from PIL import Image
# from models_init import 

# class ProductDetector:
#     def __init__(self):
#         self.processor = qwen_pipeline.det_processor
#         self.model = qwen_pipeline.det_model
#         self.device = self.model.device

#     def predict(self, data: dict) -> str:
#         url = data["product_photos"][0]
#         img = Image.open(io.BytesIO(requests.get(url).content)).convert("RGB")

#         instruction = "Определи, что изображено на фото."

#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": img},
#                     {"type": "text", "text": instruction},
#                 ],
#             }
#         ]

#         # шаблон чата
#         text = self.processor.apply_chat_template(
#             messages, add_generation_prompt=True, tokenize=False
#         )

#         inputs = self.processor(
#             text=[text],
#             images=[img],
#             return_tensors="pt"
#         ).to(self.device)

#         # генерация
#         with torch.no_grad():
#             out_ids = self.model.generate(
#                 **inputs,
#                 max_new_tokens=32,
#             )

#         answer = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
#         return answer.strip()
