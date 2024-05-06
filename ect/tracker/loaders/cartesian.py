import cv2
from numpy import ndarray
from dataclasses import dataclass

from .loader import Loader

@dataclass
class CartesianLoader(Loader):

    filepath: str
    image: ndarray | None = None
    
    def load(self) -> ndarray:
        
        if self.image is not None:
            return self.image
        
        self.image = cv2.imread(self.filepath)
        self.image = cv2.cvtColor(self.image, code=cv2.COLOR_BGR2GRAY)
        return self.image