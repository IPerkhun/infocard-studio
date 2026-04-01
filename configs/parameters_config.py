from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageEditConfig:
    num_inference_steps: int = 50

    true_cfg_scale: float = 6.5

    seed: Optional[int] = -1

    height: Optional[int] = 900
    width: Optional[int] = 1200
