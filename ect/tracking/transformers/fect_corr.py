from dataclasses import dataclass, field

import numpy as np

from .transformer import Transformer

from ...configurators import Config
from ...transforms import fect, ifect
from ...maps import logpolar_new as logpolar, ilogpolar_new as ilogpolar
from ...filters import sidelobe, spacenorm, freqnorm 

@dataclass(slots=True)
class FECTCorrTransformer(Transformer):
    cfg: Config
    dsize: tuple[int, int]
    sidelobe: np.ndarray = field(init=False)
    snf: np.ndarray = field(init=False)
    fnf: np.ndarray = field(init=False)

    def __post_init__(self):
        self.sidelobe = sidelobe(self.dsize, self.cfg)
        self.fnf = freqnorm(self.dsize, self.cfg)
        self.snf = spacenorm(self.dsize, self.cfg)


    def transform(self, inp: np.ndarray, **kwargs) -> np.ndarray:
        cx, cy = inp.shape[0]//2, inp.shape[1]//2
        radius = min(inp.shape)//2

        lp = kwargs.get("logpolar")
        center = kwargs.get("center")
        if center is None:
            center = (cx, cy)

        if lp is True or lp is None:
            inp = logpolar(inp, center, self.dsize, radius, self.cfg)
            inp *= self.sidelobe

        lp = kwargs.get("ect")
        if lp is True or lp is None:
            inp = fect(inp, self.cfg)
            inp *= self.fnf
    
        out = np.fft.fft2(inp)

        return out
    
    
    def invert(self, inp: np.ndarray, **kwargs) -> np.ndarray:
        inv = np.fft.ifft2(inp)

        dsize = (200, 200)
        radius = 100
        
        lp = kwargs.get("ect")
        if lp is True or lp is None:
            inv = ifect(inv, self.cfg)
            # inv *= self.snf

        lp = kwargs.get("logpolar")
        if lp is True or lp is None:
            inv = ilogpolar(inv, dsize, radius, self.cfg)

        return inv