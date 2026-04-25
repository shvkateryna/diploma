from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class Detection:
    boxes: np.ndarray
    scores: np.ndarray

class BaseDetector(ABC):
    name: str = "unnamed"

    @abstractmethod
    def predict(self, image_path: str, conf: float) -> Detection:
        ...