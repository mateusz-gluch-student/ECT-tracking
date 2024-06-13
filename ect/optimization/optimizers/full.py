import numpy as np
import cv2
from dataclasses import dataclass, field
from matplotlib import pyplot as plt


import scipy.optimize as optim
from .optimizer import ECTOptimizer

from dataclasses import dataclass, field
from ect import fect, ifect, Config, AntialiasParameters
from ect import freqnorm, spacenorm, sidelobe


@dataclass
class ECTFullOptimizer(ECTOptimizer):
    offset: float = 10
    ect_offset: float = 10
    n_knots: int = 20
    ect_img: np.ndarray = field(init=False)
    inv: np.ndarray = field(init=False)
    fnf: list = field(init=False)
    snf: list = field(init=False)

    def transform(self, img: np.ndarray, params: list[float]) -> np.ndarray:
    
        aa = params[2*self.n_knots:]
        f = params[:self.n_knots]
        # aa = (0.35, 0.1)
        # f = [2, 0, *np.ones((18,))]
        # f = self.fnf
        s = params[self.n_knots:2*self.n_knots]

        config = Config(
            antialias_factors=aa,
            # antialias_factors=(0.35, 0.1),
            offset_value_px=self.offset,
            ect_offset_value_px=self.ect_offset,
            freqnorm_knots = f,
            spacenorm_knots = s
        )

        dsize = img.shape

        self.ect_img = fect(img, config)
        fnf = freqnorm(self.ect_img.shape, config)
        self.ect_img *= fnf
        
        self.inv = ifect(self.ect_img, config)
        snf = spacenorm(dsize, config)
        self.inv *= snf    

        self.inv = np.real(self.inv)
        self.inv = cv2.normalize(self.inv, None, 0, 1, norm_type=cv2.NORM_MINMAX)

        return self.inv
    

    def optim(self, **kwargs) -> optim.OptimizeResult:

        fnf = kwargs.pop("fnf")
        snf = kwargs.pop("snf")

        result = optim.minimize(
            fun = self.loss,
            x0 = self.start(self.n_knots*2+3, fnf=fnf, snf=snf),
            callback = self.callback,
            **kwargs
        )

        return result
    

    def start(self, N, **kwargs):

        self.fnf = kwargs.get("fnf")
        self.snf = kwargs.get("snf")

        if not self.fnf:
            self.fnf = np.ones((self.n_knots,))
        
        if not self.snf:
            self.snf = np.ones((self.n_knots,))

        return [*self.fnf, *self.snf, 0.49, 0.14]


