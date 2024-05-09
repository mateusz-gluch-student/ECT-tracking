import ect
from numpy import ndarray
from dataclasses import dataclass

from .transformation import Transformation

@dataclass
class FilteredTransformation(Transformation):

    img_offset: float
    ect_offset: float
    radius: int | None = None


    def transform(self, image: ndarray) -> ndarray:
        
        image = ect.fect(
            image, 
            img_offset=self.img_offset, 
            ect_offset=self.ect_offset)

        if self.radius is None:
            radius = image.shape[1]
        else:
            radius = self.radius

        fnf = ect.freqnorm(image.shape[:2], radius)
        
        return image * fnf


    def invert(self, transform: ndarray) -> ndarray:

        image = ect.ifect(
            transform,
            img_offset=self.img_offset,
            ect_offset=self.ect_offset)
        
        if self.radius is None:
            radius = image.shape[1]
        else:
            radius = self.radius

        snf = ect.spacenorm(image.shape[:2], radius)
        # ang = ect.angular_filter(image.shape[:2])

        return image * snf #* ang
