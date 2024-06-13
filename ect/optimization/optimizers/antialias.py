import numpy as np
import cv2

from ect import fect, ifect, Config, AntialiasParameters
from ect.filters import freqnorm, spacenorm, sidelobe

import scipy.optimize as optim
from .optimizer import ECTOptimizer

from dataclasses import dataclass, field

from matplotlib import pyplot as plt

@dataclass
class ECTAntialiasOptimizer(ECTOptimizer):
    offset: int
    ect_offset: int
    ect_img: np.ndarray = field(init=False)
    inv: np.ndarray = field(init=False)


    def transform(self, img: np.ndarray, params: list[float]) -> np.ndarray:
        P, R = img.shape
        ect_config = Config(
            antialias_factors=params[:2],
            offset_value_px=self.offset,
            ect_offset_value_px=self.ect_offset
        )

        self.ect_img = fect(img, ect_config)
        self.inv = ifect(self.ect_img, ect_config)

        self.inv = np.real(self.inv)
        self.inv = cv2.normalize(self.inv, None, 0, 1, norm_type=cv2.NORM_MINMAX)

        return self.inv
    

    def start(self, N: int) -> list[float]:
        return [0.49, 0.14, 0.25]


    def optim(self, **kwargs) -> optim.OptimizeResult:

        result = optim.minimize(
            fun = self.loss,
            x0 = self.start(4),
            callback = self.callback,
            **kwargs
        )

        return result
    

    def callback(self, params: list[float]):
        '''A callback for optimization'''
        loss = self.loss(params)
        print(f"Current {loss=:.3f}")

        # plt.plot(rmsn(self.image.generate())[50, :])
        
        # plt.plot(rmsn(self.inv)[50, :])
        # plt.grid()