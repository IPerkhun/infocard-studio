from typing import List
import base64
from io import BytesIO
from PIL import Image

from langchain_core.tools import tool

from modules.generate_background import BackgroundGeneration
from modules.product_detector import QwenVLDetector
from schemas.schema import ImageGenInput, ImageGenOutput

bg_generator = BackgroundGeneration()
detector = QwenVLDetector()


@tool("detect_product")
def detect_product_tool(image_url: str, question: str) -> str:
    """Возвращает СЫРОЙ ответ VL-модели (строка)."""
    return detector.predict_from_url(image_url, question)


@tool("background_generation", args_schema=ImageGenInput)
def generate_images_tool(product_photos: List[str], prompt: str) -> ImageGenOutput:
    """Генерирует изображение на корректном фоне."""
    _, encoded = bg_generator.get_images(
        {"product_photos": product_photos}, prompt=prompt
    )
    return ImageGenOutput(generated_images=encoded)


@tool("inspect_generated_image")
def inspect_generated_image_tool(image_base64: str, question: str) -> str:
    """
    Возвращает СЫРОЙ ответ VL-модели (строка) по сгенерированному изображению (base64).
    """
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return detector.predict_from_image(image, question)
