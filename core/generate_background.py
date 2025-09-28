from PIL import Image
import torch
from configs.image_edit_config import ImageEditConfig

from modules.load_pipeline_qwen import CustomEditQwenPipeline


class BackgroundGeneration:
    def __init__(self):
        self.config = ImageEditConfig()
        self.qwen_pipeline = CustomEditQwenPipeline()

    def _image_convert_rgb(self):
        return Image.open("photo_2025-05-13_19-46-12.jpg").convert("RGB")

    def generate_image(self):
        image = self._image_convert_rgb()

        inputs = {
            "image": image,
            "prompt": self.config.prompt,
            "num_inference_steps": self.config.num_inference_steps,
            "true_cfg_scale": self.config.true_cfg_scale,
        }

        with torch.inference_mode():
            output = self.qwen_pipeline.pipeline(**inputs)

        return output.images[0]