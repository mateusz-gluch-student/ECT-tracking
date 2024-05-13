from dataclasses import dataclass, field

import numpy as np

from ...transforms.utils_new import xcorr
from ..transformers import Transformer
from .matcher import Matcher

@dataclass(slots=True)
class MOSSEMatcher(Matcher):
    t: Transformer
    learn_rate: float
    filter_a: np.ndarray = field(init=False)
    filter_b: np.ndarray = field(init=False)

    def initialize(self, input: np.ndarray):
        self.filter_a = input
        self.filter_b = 1
   
    def match(self, input: np.ndarray) -> np.ndarray:
        filter = self.filter_a/self.filter_b
        filter_t = self.t.transform(filter, logpolar=False)

        return input * np.conj(filter_t)
    
    def update(self, input: np.ndarray, output: np.ndarray):
        '''Minimize 
        $$
        \sum_{ij} | F_{ij} * H_{ij} - G_{ij} |
        $$
        '''
        update_a = self.t.invert(output * np.conj(input), logpolar=False)
        update_b = self.t.invert(input * np.conj(input), logpolar=False) 
        lr = self.learn_rate
        self.filter_a = lr*update_a + (1-lr)*self.filter_a
        self.filter_b = lr*update_b + (1-lr)*self.filter_b

    