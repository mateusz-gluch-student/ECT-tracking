from dataclasses import dataclass, field

import numpy as np 
import cv2
from numpy import ndarray
from loguru import logger 

from ...helpers import vectors
from ...configurators import Config
from .matcher import Matcher

@dataclass(slots=True)
class NaiveMatcher(Matcher):
    thresh: float = field(default=0.02)
    template: np.ndarray = field(init=False)
    
    def initialize(self, input: ndarray):
        # #input is already transformed
        # input_abs = np.abs(input)
        # input_phase = input/input_abs

        # input_abs = cv2.normalize(input_abs, None, 0, 1, cv2.NORM_MINMAX)
        # thr = np.zeros_like(input)
        # thr[input_abs > self.thresh] = 1

        # self.template = input_phase * thr
        self.template = input

    def match(self, input: ndarray) -> ndarray:
        xcorr = self.template * np.conj(input)
        xcorr_abs = np.abs(xcorr)
        xcorr_phase = xcorr/xcorr_abs

        xcorr_abs = cv2.normalize(xcorr_abs, None, 0, 1, cv2.NORM_MINMAX)
        thr = np.zeros_like(xcorr)
        thr[xcorr_abs > self.thresh] = 1

        return xcorr_phase * thr

        
    def update(self, input: ndarray, output: ndarray):
        self.initialize(input)