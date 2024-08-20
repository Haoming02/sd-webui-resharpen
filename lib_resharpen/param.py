from dataclasses import dataclass
from torch import Tensor


@dataclass
class ReSharpenParams:

    enable: bool = False
    scaling: str = None
    strength: float = 0.0
    total_step: int = -1
    cache: Tensor = None

    def __bool__(self):
        return self.enable
