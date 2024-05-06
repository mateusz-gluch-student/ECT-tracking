import cv2
import ect

from numpy import ndarray
from dataclasses import dataclass

from .loader import Loader

@dataclass
class LogpolarLoader(Loader):

    radius: int = 200
    offset: int = 10

    def load(self, image: ndarray) -> ndarray:
        
        image = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
        image = ect.logpolar(image, radius=self.radius, offset=self.offset)

        sidelobe = ect.sidelobe(image.shape[:2], offset=self.offset)

        return image * sidelobe    