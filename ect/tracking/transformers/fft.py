from dataclasses import dataclass, field

import numpy as np

from .transformer import Transformer

from ...configurators import Config

@dataclass(slots=True)
class FFTTransformer(Transformer):

    def transform(self, inp: np.ndarray) -> np.ndarray:
        return np.fft.fft2(inp)
    
    def invert(self, inp: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(inp)