from dataclasses import dataclass, field

from numpy import ndarray

from .transformer import Transformer

@dataclass(slots=True)
class FFTTransformer(Transformer):

    def transform(self, inp: ndarray) -> ndarray:
        return super().transform()
    
    def invert(self, inp: ndarray) -> ndarray:
        return super().invert()