from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np

@dataclass(slots=True)
class Matcher(ABC):
    template: np.ndarray = field(init=False)

    @abstractmethod
    def initialize(input: np.ndarray):
        pass

    @abstractmethod
    def match(input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def update(input: np.ndarray, output: np.ndarray):
        pass