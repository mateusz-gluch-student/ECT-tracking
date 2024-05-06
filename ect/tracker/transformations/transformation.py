from numpy import ndarray

from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Transformation(ABC):

    @abstractmethod
    def transform(self, image: ndarray, **params) -> ndarray: ...


    @abstractmethod
    def invert(self, transform: ndarray, **params) -> ndarray: ...