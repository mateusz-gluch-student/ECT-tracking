import numpy as np
import ect

import scipy.optimize as optim
from numpy.random import rand
from .optimizer import ECTOptimizer

from dataclasses import dataclass

@dataclass
class ECTOffsetOptimizer(ECTOptimizer):
    '''Optimizer for ECT Offset Optimization'''

    def optim(self, sidelobe: bool= False, **kwargs):
        '''Optimizing function'''
        fnc = self.loss_sidelobe if sidelobe else self.loss

        result = optim.minimize(
            fun = fnc,
            x0 = [self.start(2)],
            bounds = [[0, self.radius]],
            callback=self.callback,
            **kwargs
        )

        return result


    def transform(self, params: list[float]) -> np.ndarray:
        '''ECT transformation for optimization'''
        ect_offset = params[0]
        ect_img = ect.fect(self.image, self.img_offset, ect_offset)
        self.inv = ect.ifect(ect_img, self.img_offset, ect_offset)
        return self.inv


    def loss_sidelobe(self, x) -> float:
        '''Sidelobe filtered loss function'''
        self.inv = self.transform(x)
        self.inv = self.inv * self.filt
        return np.real(self.loss_fcn(self.inv, self.image))


    def callback(self, x):
        '''Optimization callback override'''
        ect_offset = x[0]
        loss = self.loss(x)
        print(f"{ect_offset=:.2f}, {loss=:.2f}")


    def start(self, N):
        '''Seeding optimization'''
        return rand(1)*self.radius/4
    