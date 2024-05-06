import cv2
import ect
from numpy import ndarray
from dataclasses import dataclass

from .loader import Loader

@dataclass
class ImageLoader(Loader):

    filepath: str
    image: ndarray | None = None
    radius: int = 200
    offset: float = 20

    def load(self) -> ndarray:
        
        if self.image is not None:
            return self.image
        
        self.image = cv2.imread(self.filepath)
        self.image = cv2.cvtColor(self.image, code=cv2.COLOR_BGR2GRAY)
        self.image = ect.logpolar(self.image, radius=self.radius, offset=self.offset)

        sidelobe = ect.sidelobe(self.image.shape[:2], offset=self.offset)

        self.image = self.image * sidelobe

        return self.image
