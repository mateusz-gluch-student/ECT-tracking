from dataclasses import dataclass, field

import numpy as np

from .transformer import Transformer

from ...configurators import Config
from ...transforms import dect
from ...maps import logpolar_new as logpolar, ilogpolar_new as ilogpolar
from ...filters import sidelobe, spacenorm, freqnorm 

@dataclass(slots=True)
class DECTTransformer(Transformer):
    cfg: Config
    dsize: tuple[int, int]
    sidelobe: np.ndarray = field(init=False)

    def __post_init__(self):
        # self.cfg.antialias_params = [0.11, 0.11]
        self.sidelobe = sidelobe(self.dsize, self.cfg)


    def transform(self, inp: np.ndarray, **kwargs) -> np.ndarray:
        cx, cy = inp.shape[0]//2, inp.shape[1]//2
        radius = min(inp.shape)//2

        lp = kwargs.get("logpolar")
        if lp is True or lp is None:
            inp = logpolar(inp, (cx, cy), self.dsize, radius, self.cfg)
            inp *= self.sidelobe

        out = dect(inp, self.cfg)

        return out
    
    def invert(self, inp: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError
        dsize = (200, 200)
        radius = 100

        inv = ifect(inp, self.cfg)

        lp = kwargs.get("logpolar")
        if lp is True or lp is None:
            inv = ilogpolar(inv, dsize, radius, self.cfg)

        return inv