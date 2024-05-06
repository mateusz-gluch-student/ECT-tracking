import numpy as np
import ect

import scipy.optimize as optim
from .optimizer import ECTOptimizer

from dataclasses import dataclass, field

@dataclass
class ECTAngularSNF(ECTOptimizer):
    ect_offset: float
    n_knots: int
    params: int = field(init=False)

    def __post_init__(self):
        self.params = self.n_knots
        return super().__post_init__()


    def transform(self, params: list[float]) -> np.ndarray:

        dsize = self.image.shape[:2]
        fnf = ect.freqnorm(dsize, self.radius)
        snf = ect.spacenorm(dsize, self.radius)

        self.ect_img = ect.fect(self.image, self.img_offset, self.ect_offset)
        self.ect_img = self.ect_img * fnf
        self.inv = ect.ifect(self.ect_img, self.img_offset, self.ect_offset)
        ang = ect.angular_filter(dsize, params)
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