import numpy as np
import ect

import scipy.optimize as optim
from .optimizer import ECTOptimizer

from dataclasses import dataclass, field

@dataclass
class ECTFilterTuner(ECTOptimizer):
    ect_offset: float
    n_knots: int = 20
    params: int = field(init=False)

    def __post_init__(self):
        self.params = 2 * self.n_knots
        super().__post_init__()


    def start(self, N):
        aa_params = np.array([0.27, 0.15, 0.27, 0.15, 0.25, 0.25])
        return np.r_[ect.DEFAULT_FNF, ect.DEFAULT_SNF, aa_params]

    def transform(self, params: list[float]) -> np.ndarray:

        fnf_values = params[:self.params//2]
        snf_values = params[self.params//2:self.params]
        aa_params = params[self.params:]
        ect_aa = aa_params[:2]
        iect_aa = aa_params[2:4]
        ect_slope = aa_params[4]
        iect_slope = aa_params[5]

        dsize = self.image.shape[:2]
        fnf = ect.freqnorm(dsize, self.radius, fnf_values)
        snf = ect.spacenorm(dsize, self.radius, snf_values)
        # ang = ect.angular_filter(dsize)

        self.ect_img = ect.fect(
            self.image, 
            self.img_offset, 
            self.ect_offset,
            aa_factors = ect_aa,
            aa_slope = ect_slope)
        
        self.ect_img = self.ect_img * fnf
        
        self.inv = ect.ifect(
            self.ect_img, 
            self.img_offset, 
            self.ect_offset,
            aa_factors = iect_aa,
            aa_slope = iect_slope)
        
        self.inv = self.inv * snf

        return self.inv
    

    def optim(self, **kwargs) -> optim.OptimizeResult:

        result = optim.minimize(
            fun = self.loss,
            x0 = self.start(self.params),
            callback = self.callback,
            **kwargs
        )

        return result