from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np

@dataclass(slots=True)
class Matcher(ABC):
    template: np.ndarray = field(init=False)

    @abstractmethod
    def initialize(self, input: np.ndarray):
        pass

    @abstractmethod
    def match(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, input: np.ndarray, output: np.ndarray):
        pass