from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageEditConfig:
    num_inference_steps: int = 1

    true_cfg_scale: float = 4.0
    guidance_rescale: Optional[float] = None

    seed: Optional[int] = -1

    height: Optional[int] = None
    width: Optional[int] = None
