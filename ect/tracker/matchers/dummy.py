import numpy as np
from numpy import ndarray

from .matcher import Matcher

class DummyMatcher(Matcher):

    def match_image(self, image: ndarray, template: ndarray) -> ndarray:
        return np.zeros_like(image)
       
    