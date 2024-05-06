import ect
from numpy import ndarray
from dataclasses import dataclass

from .transformation import Transformation

@dataclass
class UnfilteredTransformation(Transformation):

    img_offset: float
    ect_offset: float


    def transform(self, image: ndarray) -> ndarray:

        return ect.fect(
            image, 
            img_offset = self.img_offset, 
            ect_offset = self.ect_offset)
    

    def invert(self, transform: ndarray) -> ndarray:

        return ect.ifect(
            transform, 
            img_offset = self.img_offset,
            ect_offset = self.ect_offset)