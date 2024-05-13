from dataclasses import dataclass, field

import numpy as np 
from numpy import ndarray
from loguru import logger 

from .matcher import Matcher

class NaiveMatcher(Matcher):
    template: np.ndarray = field(init=False)

    def initialize(self, input: ndarray):
        self.template = input

    def match(self, input: ndarray) -> ndarray:
        return np.conj(self.template) * input
    
    def update(self, input: ndarray, output: ndarray):
        self.template = input