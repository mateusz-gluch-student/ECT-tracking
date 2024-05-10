from dataclasses import dataclass, field

import numpy as np

from .transformer import Transformer

from ...configurators import Config
from ...transforms import fect, ifect
from ...maps import logpolar_new as logpolar, ilogpolar_new as ilogpolar
from ...filters import sidelobe, spacenorm, freqnorm 

@dataclass(slots=True)
class FECTTransformer(Transformer):
    cfg: Config
    dsize: tuple[int, int]
    sidelobe: np.ndarray = field(init=False)

    def __post_init__(self):
        self.sidelobe = sidelobe(self.dsize, self.cfg)
        self.fnf = freqnorm(self.dsize, self.cfg)
        self.snf = spacenorm(self.dsize, self.cfg)


    def transform(self, inp: np.ndarray) -> np.ndarray:
        cx, cy = inp.shape[0]//2, inp.shape[1]//2
        radius = min(inp.shape)//2

        logimg = logpolar(inp, (cx, cy), self.dsize, radius, self.cfg)
        logimg *= self.sidelobe

        out = fect(logimg, self.cfg)
        out *= self.fnf

        return out
    
    def invert(self, inp: np.ndarray) -> np.ndarray:
        dsize = (200, 200)
        radius = 100

        inv = ifect(inp, self.cfg)
        inv *= self.snf

        out = ilogpolar(inv, dsize, radius, self.cfg)

        return out