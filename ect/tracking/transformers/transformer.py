from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np

@dataclass(slots=True)
class Transformer(ABC):

    @abstractmethod
    def transform(self, inp: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def invert(self, inp: np.ndarray) -> np.ndarray:
        pass


    
