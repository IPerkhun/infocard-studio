from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageEditConfig:
    prompt: str = "draw as if they are holding a bottle of beer and smoking cigarettes"

    num_inference_steps: int = 20

    true_cfg_scale: float = 4.0
    guidance_rescale: Optional[float] = None

    seed: Optional[int] = -1

    height: Optional[int] = None
    width: Optional[int] = None
