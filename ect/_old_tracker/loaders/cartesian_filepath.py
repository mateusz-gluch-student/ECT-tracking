from numpy import ndarray
from dataclasses import dataclass

from .cartesian import CartesianLoader

@dataclass
class CartesianFilepathLoader(CartesianLoader):

    filepath: str | None = None

    def load(self, filepath: str) -> ndarray:
        self.filepath = filepath
        return super().load()