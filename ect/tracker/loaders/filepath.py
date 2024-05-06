import cv2
import ect

from numpy import ndarray
from dataclasses import dataclass

from .loader import Loader

@dataclass
class FilepathLoader(Loader):

    filepath: str | None = None
    image: ndarray | None = None
    radius: int = 200
    offset: float = 20

    def load(self, filepath: str) -> ndarray:

        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
        image = ect.logpolar(image, radius=self.radius, offset=self.offset)

        sidelobe = ect.sidelobe(image.shape[:2], offset=self.offset)

        image = image * sidelobe

        return image