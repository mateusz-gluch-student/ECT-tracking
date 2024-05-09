from dataclasses import dataclass, field

from numpy import ndarray

from .transformer import Transformer

@dataclass(slots=True)
class FFTTransformer(Transformer):

    def transform(inp: ndarray) -> ndarray:
        return super().transform()
    
    def invert(inp: ndarray) -> ndarray:
        return super().invert()