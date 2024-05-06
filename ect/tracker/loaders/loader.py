from numpy import ndarray

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Loader(ABC):

    @abstractmethod
    def load(self, *args, **kwargs) -> ndarray: ...
