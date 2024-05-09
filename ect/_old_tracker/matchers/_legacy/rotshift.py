import cv2
import ect
import numpy as np

from numpy import ndarray

from dataclasses import dataclass

from .matcher import Matcher
from ..transformations import Transformation


@dataclass
class RotShiftMatcher(Matcher):

    transformer: Transformation
    bp_thresh: float = 0.2
    match_result: ndarray | None = None

    
    def prepare(self, image: ndarray) -> ndarray:
        image = self.transformer.transform(image)
        # # image = np.abs(image)
        # pad = np.zeros_like(image)
        # image = np.hstack((pad, image))
        image = np.fft.fft2(image, axes=(0, 1))
        return image

    def match_image(self, image: ndarray, template: ndarray) -> ndarray:
        
        if self.match_result is not None:
            return self.match_result

        # calculate phase shift
        xcorr = template * np.conj(image)
        xcorr_norm = xcorr/(np.abs(xcorr) + 1e-12)

        template_abs = ect.norm_minmax(np.abs(template), 0, 1, dtype=np.float64)
        bp_filter = np.zeros_like(template_abs)
        bp_filter[template_abs > self.bp_thresh] = 1

        P, R = template.shape[:2]
        # return bp_filter
        # return np.exp(1j*np.arctan(phase)) * bp_filter
        self.match_result = xcorr_norm * bp_filter
        self.match_result = np.fft.ifft2(self.match_result, axes=(0, 1))

        left_half, right_half = np.hsplit(self.match_result, 2)
        self.match_result = np.hstack((right_half, left_half))

        upper_half, lower_half = np.vsplit(self.match_result, 2)
        self.match_result = np.vstack((lower_half, upper_half))

        # self.match_result = self.transformer.invert(self.match_result)
        self.match_result = np.abs(self.match_result)
        self.match_result = ect.norm_minmax(self.match_result, 0, 255, dtype=np.uint8)
        self.match_result = cv2.cvtColor(self.match_result, cv2.COLOR_GRAY2BGR)

        return self.match_result
