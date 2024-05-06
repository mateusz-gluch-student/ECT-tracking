import cv2
import numpy as np
import ect

from numpy.random import rand
import matplotlib.pyplot as plt
from typing import Callable

from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from ..generators import Generator
from ...configurators import Config

@dataclass
class ECTOptimizer(ABC):
    '''Base class for ECT optimizers'''
    image: Generator
    # config: Config
    loss_fcn: Callable[[np.ndarray, np.ndarray], float]
    filt: np.ndarray = field(init=False)

    def start(self, N): 
        '''Seeds optimization'''
        return rand(N)


    def loss(self, params: list[float]) -> float:
        '''Calculates loss function'''
        img = self.image.generate()
        P, _ = img.shape
        img = img[:P//2, :]
        return np.real(self.loss_fcn(self.transform(img, params), img))


    def callback(self, params: list[float]):
        '''A callback for optimization'''
        loss = self.loss(params)
        print(f"Current {loss=:.3f}")


    @abstractmethod
    def optim(self): 
        '''Optimization function'''
        pass


    @abstractmethod
    def transform(self, params: list[float]) -> np.ndarray:  
        '''Transformation function'''
        pass


    def show_result(
        self, 
        norm_fcn: Callable[[np.ndarray], np.ndarray],
        *params
        ):
        '''Shows result of optimization'''

        if self.inv is None:
            self.inv = self.transform(params)

        log_img = norm_fcn(self.image)
        inv_img = norm_fcn(self.inv)

        diff = log_img - inv_img

        plt.figure(figsize = (20, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(ect.complex_to_hsv(self.image))

        plt.subplot(1, 3, 2)
        plt.imshow(ect.complex_to_hsv(inv_img))

        plt.subplot(1, 3, 3)
        plt.imshow(ect.complex_to_hsv(diff))