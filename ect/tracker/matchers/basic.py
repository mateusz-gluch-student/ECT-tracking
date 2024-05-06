import ect
import numpy as np

from numpy import ndarray

from dataclasses import dataclass

from .matcher import Matcher
from ..transformations import Transformation

@dataclass    
class BasicMatcher(Matcher):

    transformer: Transformation
    bp_thresh: float = 0.2
    match_result: ndarray | None = None

    def prepare(self, image: ndarray) -> ndarray:
        return self.transformer.transform(image)

    def match_image(self, image: ndarray, template: ndarray) -> ndarray:
        
        if self.match_result is not None:
            return self.match_result

        xcorr = template * np.conj(image)
        xcorr_norm = xcorr/(xcorr + 1e-12)

        template_abs = ect.norm_minmax(np.abs(template), 0, 1, dtype=np.float64)
        bp_filter = np.zeros_like(template_abs)
        bp_filter[template_abs > self.bp_thresh] = 1

        self.match_result = xcorr_norm * bp_filter
        self.match_result = self.transformer.invert(self.match_result)
        self.match_result = np.abs(self.match_result)
        self.match_result = ect.norm_minmax(self.match_result, 0, 255, dtype=np.uint8)
        self.match_result = ect.ilogpolar(self.match_result, offset=0)
        
        return self.match_result
