from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageEditConfig:
    num_inference_steps: int = 35

    true_cfg_scale: float = 4.0
    guidance_rescale: Optional[float] = None

    seed: Optional[int] = -1

    height: Optional[int] = 900
    width: Optional[int] = 1200
