from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Iterable

import numpy as np

@dataclass(slots=True)
class Generator(ABC):


    @abstractmethod
    def generate(self) -> np.ndarray: ...


    @abstractmethod
    def images(self) -> Iterable[np.ndarray]: ...