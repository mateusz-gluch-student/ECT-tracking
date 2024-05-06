import numpy as np
from numpy import ndarray
from dataclasses import dataclass

from .matcher import Matcher
from ..transformations import Transformation

@dataclass
class CorrelationMatcher(Matcher):

    transformer: Transformation

    def match_image(self, image: ndarray, template: ndarray) -> ndarray:
        
        xcorr = image * np.conjugate(template)
        xcorr /= np.abs(xcorr)

        return xcorr
        # return self.transformer.invert(xcorr)