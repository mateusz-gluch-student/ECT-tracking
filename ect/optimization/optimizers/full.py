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

        result = optim.minimize(
            fun = self.loss,
            x0 = self.start(self.n_knots*2+3),
            callback = self.callback,
            **kwargs
        )

        return result
    

    def start(self, N): 

        self.fnf = [3.183, -2.054, 1.522, 0.821, 1.511, 0.907, 1.114, 0.934, 1.085, 1.063, 0.979, 1.051, 0.941, 1.005, 0.983, 0.978, 1.079, 1.027, 0.818, 0.608]

        # self.snf = np.ones((N,))
        self.snf = [1.747, 1.743, 1.731, 1.724, 1.708, 1.710, 1.728, 1.723, 1.752, 1.749, 1.762, 1.862, 1.888, 1.894, 1.850, 1.861, 1.720, 1.650, 1.578, 1.202]


        return [*self.fnf, *self.snf, 0.366, 0.117]


