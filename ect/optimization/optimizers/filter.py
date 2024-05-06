import numpy as np
import ect

import scipy.optimize as optim
from .optimizer import ECTOptimizer

from dataclasses import dataclass, field

@dataclass
class ECTFilterOptimizer(ECTOptimizer):
    ect_offset: float
    n_knots: int
    do_fnf: bool = True
    do_snf: bool = True
    params: int = field(init=False, default=0)

    def __post_init__(self):

        if self.do_fnf:
            self.params += self.n_knots

        if self.do_snf:
            self.params += self.n_knots
        
        super().__post_init__()


    def transform(self, params: list[float]) -> np.ndarray:

        if self.do_fnf and self.do_snf:    
            fnf_values = params[:self.params//2]
            snf_values = params[self.params//2:]    
        elif self.do_fnf:
            fnf_values = params
            snf_values = None
        elif self.do_snf:
            snf_values = params
            fnf_values = None
        else:
            raise Exception("At least one of two must be True.")

        dsize = self.image.shape[:2]
        fnf = ect.freqnorm(dsize, self.radius, fnf_values)
        snf = ect.spacenorm(dsize, self.radius, snf_values)
        ang = ect.angular_filter(dsize)

        self.ect_img = ect.fect(self.image, self.img_offset, self.ect_offset)
        self.ect_img = self.ect_img * fnf
        
        self.inv = ect.ifect(self.ect_img, self.img_offset, self.ect_offset)
        self.inv = self.inv * snf * ang

        return self.inv
    

    def optim(self, **kwargs) -> optim.OptimizeResult:

        result = optim.minimize(
            fun = self.loss,
            x0 = self.start(self.params),
            callback = self.callback,
            **kwargs
        )

        return result